from config import Qwen2Config
from model import Qwen2
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def detect_config(hf: dict, local_dir: str) -> Qwen2Config:
    import json

    cfg = Qwen2Config()

    # Read architecture params directly from config.json — shapes alone are ambiguous
    with open(f"{local_dir}/config.json") as f:
        js = json.load(f)

    cfg.vocab_size = js["vocab_size"]
    cfg.hidden_size = js["hidden_size"]
    cfg.intermediate_size = js["intermediate_size"]
    cfg.num_hidden_layers = js["num_hidden_layers"]
    cfg.num_attention_heads = js["num_attention_heads"]
    cfg.num_key_value_heads = js["num_key_value_heads"]
    cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    cfg.max_position_embeddings = js.get(
        "max_position_embeddings", cfg.max_position_embeddings
    )
    cfg.rms_norm_eps = js.get("rms_norm_eps", cfg.rms_norm_eps)
    cfg.rope_theta = js.get("rope_theta", cfg.rope_theta)
    cfg.sliding_window = js.get("sliding_window", cfg.sliding_window)
    cfg.max_window_layers = js.get("max_window_layers", cfg.max_window_layers)

    print(
        f"Detected: layers={cfg.num_hidden_layers}, hidden={cfg.hidden_size}, "
        f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads}, "
        f"head_dim={cfg.head_dim}, ffn={cfg.intermediate_size}"
    )
    return cfg


def remap_weights(hf: dict, cfg: Qwen2Config) -> dict:
    out = {}
    out["embed_tokens.weight"] = hf["model.embed_tokens.weight"]
    out["norm.weight"] = hf["model.norm.weight"]
    out["lm_head.weight"] = hf.get("lm_head.weight", hf["model.embed_tokens.weight"])
    for i in range(cfg.num_hidden_layers):
        ph, pm = f"model.layers.{i}", f"layers.{i}"
        out[f"{pm}.norm1.weight"] = hf[f"{ph}.input_layernorm.weight"]
        out[f"{pm}.norm2.weight"] = hf[f"{ph}.post_attention_layernorm.weight"]
        out[f"{pm}.attn.q_proj.weight"] = hf[f"{ph}.self_attn.q_proj.weight"]
        out[f"{pm}.attn.q_proj.bias"] = hf[f"{ph}.self_attn.q_proj.bias"]
        out[f"{pm}.attn.k_proj.weight"] = hf[f"{ph}.self_attn.k_proj.weight"]
        out[f"{pm}.attn.k_proj.bias"] = hf[f"{ph}.self_attn.k_proj.bias"]
        out[f"{pm}.attn.v_proj.weight"] = hf[f"{ph}.self_attn.v_proj.weight"]
        out[f"{pm}.attn.v_proj.bias"] = hf[f"{ph}.self_attn.v_proj.bias"]
        out[f"{pm}.attn.o_proj.weight"] = hf[f"{ph}.self_attn.o_proj.weight"]
        out[f"{pm}.ffn.gate.weight"] = hf[f"{ph}.mlp.gate_proj.weight"]
        out[f"{pm}.ffn.up.weight"] = hf[f"{ph}.mlp.up_proj.weight"]
        out[f"{pm}.ffn.down.weight"] = hf[f"{ph}.mlp.down_proj.weight"]
    return out


def load_model(
    model_path: str | None = None, model_id="Qwen/Qwen2-0.5B-Instruct", device="cpu"
):
    if model_path is not None:
        hf_state = load_file(model_path, device=device)
        local_dir = Path(model_path).parent
    else:
        print(f"Downloading {model_id} ...")
        local_dir = snapshot_download(model_id, ignore_patterns=["*.msgpack", "*.h5"])

        shards = sorted(Path(local_dir).glob("*.safetensors"))
        hf_state = {}
        for s in shards:
            hf_state.update(load_file(str(s), device=device))
        print(f"Loaded {len(hf_state)} tensors from {len(shards)} shard(s)")

    cfg = detect_config(hf_state, local_dir)
    model = Qwen2(cfg)

    remapped = remap_weights(hf_state, cfg)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"[WARN] Missing    ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"[WARN] Unexpected ({len(unexpected)}): {unexpected[:8]}")
    if not missing and not unexpected:
        print("✓ All weights loaded cleanly.")

    return model.to(device).eval(), Path(local_dir)


def format_prompt(
    user_msg: str, system: str = "You are a helpful assistant.", reasoning=False
) -> str:
    if not reasoning:
        return (
            f"/no_think\n"
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"/no_think\n"
        )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

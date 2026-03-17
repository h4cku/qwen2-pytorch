import torch
import torch.nn as nn
from config import Qwen2Config
from tqdm import tqdm
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f32 = x.float()
        normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(dtype)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    T = q.shape[2]
    c = cos[:T].unsqueeze(0).unsqueeze(0)
    s = sin[:T].unsqueeze(0).unsqueeze(0)
    return (q * c + rotate_half(q) * s, k * c + rotate_half(k) * s)


class Attention(nn.Module):
    def __init__(self, cfg: Qwen2Config, layer_idx: int):
        super().__init__()
        nq, nkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        hid = cfg.hidden_size
        self.nq, self.nkv, self.D = nq, nkv, D
        self.num_kv_groups = nq // nkv
        self.scale = D**-0.5
        self.use_window = layer_idx < cfg.max_window_layers
        self.sliding_window = cfg.sliding_window

        self.q_proj = nn.Linear(hid, nq * D, bias=True)
        self.k_proj = nn.Linear(hid, nkv * D, bias=True)
        self.v_proj = nn.Linear(hid, nkv * D, bias=True)
        self.o_proj = nn.Linear(nq * D, hid, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, _ = x.shape
        nq, nkv, D = self.nq, self.nkv, self.D

        q = self.q_proj(x).view(B, T, nq, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, nkv, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, nkv, D).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)
        S = k.shape[2]

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.full((T, S), float("-inf"), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=S - T + 1)
        if self.use_window and S > self.sliding_window:
            mask[:, : S - self.sliding_window] = float("-inf")
        attn = F.softmax(attn + mask[None, None], dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, nq * D)
        return self.o_proj(out), new_cache


class FeedForward(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Qwen2Block(nn.Module):
    def __init__(self, cfg: Qwen2Config, layer_idx: int):
        super().__init__()
        self.attn = Attention(cfg, layer_idx)
        self.ffn = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, cache=None):
        h, new_cache = self.attn(self.norm1(x), cos, sin, cache)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class Qwen2(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen2Block(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        cos, sin = build_rope_cache(
            cfg.max_position_embeddings,
            cfg.head_dim,
            cfg.rope_theta,
            torch.device("cpu"),
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids, past_caches=None):
        device = input_ids.device
        T = input_ids.shape[1]
        past_len = past_caches[0][0].shape[2] if past_caches else 0

        x = self.embed_tokens(input_ids)
        cos = self.rope_cos[past_len : past_len + T].to(device)
        sin = self.rope_sin[past_len : past_len + T].to(device)
        caches = []
        for i, layer in enumerate(self.layers):
            x, c = layer(x, cos, sin, past_caches[i] if past_caches else None)
            caches.append(c)
        return self.lm_head(self.norm(x)), caches

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=512, temperature=0.7, top_p=0.9):
        logits, caches = self(input_ids)
        generated = input_ids.clone()
        for _ in tqdm(range(max_new_tokens)):
            nl = logits[:, -1, :] / max(temperature, 1e-8)
            sl, si = torch.sort(nl, descending=True)
            p = F.softmax(sl, dim=-1)
            cp = p.cumsum(-1)
            sl[(cp - p) > top_p] = float("-inf")
            p = F.softmax(sl, dim=-1)
            nt = si.gather(-1, torch.multinomial(p, 1))
            generated = torch.cat([generated, nt], dim=1)
            if nt.item() == self.cfg.eos_token_id:
                break
            logits, caches = self(nt, caches)
        return generated

import torch
from utils import load_model, format_prompt
from tokenizers import Tokenizer

if __name__ == "__main__":

    MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}  |  Model: {MODEL_ID}")

    model, local_dir = load_model(model_path="./hf/model.safetensors", device=DEVICE)
    tokenizer = Tokenizer.from_file(str(local_dir / "tokenizer.json"))

    user_message = "What is the difference between a list and a tuple in Python?"
    prompt = format_prompt(user_message)
    print(f"\nPrompt:\n{prompt}\n")

    token_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    print(f"Prompt tokens: {len(token_ids)}")

    out = model.generate(input_ids, max_new_tokens=512, temperature=0.7, top_p=0.9)
    new_ids = out[0, len(token_ids) :].tolist()
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    print("\n" + "=" * 60 + "\nRESPONSE:\n" + "=" * 60)
    print(response)

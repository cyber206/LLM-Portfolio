import sys
sys.path.insert(0, r"D:\Tiny-Stories")

import os
import torch
from Model import GPT, GPT_config
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# ---- paths / config ----
CKPT_PATH = r"D:\Tiny-Stories\out-tinystories\ckpt.pt"
SFT_PATH  = r"D:\Tiny-Stories\sft_model.pt"
TOK_PATH  = r"D:\Tiny-Stories\tokenizer.json"
EOS       = 1
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ---- load model: architecture from base ckpt, weights from SFT ----
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model = GPT(GPT_config(**ckpt["model_args"]))
model.load_state_dict(torch.load(SFT_PATH, map_location="cpu"))
model = model.to(DEVICE).eval()

# ---- load tokenizer ----
tok = Tokenizer.from_file(TOK_PATH)

# ---- decoder fix (MUST run before any tok.decode call) ----
print("decoder before:", tok.decoder)
if tok.decoder is None:
    tok.decoder = ByteLevelDecoder()
print("decoder after :", tok.decoder)
print("pretokenizer  :", tok.pre_tokenizer)

# ---- diagnostics ----
print("=== DIAGNOSTICS ===")
print("SFT file:", SFT_PATH, "| size:", os.path.getsize(SFT_PATH))
print("vocab size:", tok.get_vocab_size())
try:
    emb_rows = model.transformer.wte.weight.shape[0]
except AttributeError:
    emb_rows = model.lm_head.weight.shape[0]
print("model vocab (embedding rows):", emb_rows)

test = "### Instruction:\nhey\n\n### Response:\n"
ids = tok.encode(test, add_special_tokens=False).ids
print("prompt ids:", ids[:20], "...")
print("roundtrip:", repr(tok.decode(ids)))
print("===================")

# ---- generation ----
@torch.no_grad()
def generate(prompt, max_new=120, temperature=0.0):
    ids = tok.encode(prompt, add_special_tokens=False).ids
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    for _ in range(max_new):
        logits, _ = model(x)
        next_logits = logits[:, -1, :]

        if temperature == 0.0:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_id], dim=1)
        if next_id.item() == EOS:
            break

    full = tok.decode(x[0].tolist())
    return full[len(prompt):].strip()

# ---- interactive loop ----
def chat():
    print("SFT TinyGPT — type an instruction. 'quit' to exit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("quit", "exit", ""):
            break

        prompt = f"### Instruction:\n{user}\n\n### Response:\n"
        reply = generate(prompt)
        print(f"Model: {reply}\n")

if __name__ == "__main__":
    chat()
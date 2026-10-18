import os, sys, torch
sys.path.insert(0, ".")
from Model import GPT, GPT_config, TinyGPTConfig, TinyGPTForCausalLM

out_dir = "out-tinystories"
hf_out_dir = os.path.join(out_dir, "hf-model")

ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
model_args = ckpt["model_args"]

raw = GPT(GPT_config(**model_args))
raw.load_state_dict(
    {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()},
    strict=True,
)

hf_config = TinyGPTConfig(**model_args)
hf_config.auto_map = {
    "AutoConfig": "Model.TinyGPTConfig",
    "AutoModelForCausalLM": "Model.TinyGPTForCausalLM",
}
hf_config.tie_word_embeddings = True
hf_model = TinyGPTForCausalLM(hf_config)
hf_model.gpt.load_state_dict(raw.state_dict(), strict=True)
hf_model.eval()
os.makedirs(hf_out_dir, exist_ok=True)
hf_model.save_pretrained(hf_out_dir, safe_serialization=True)
print("Saved to", hf_out_dir)
print(os.listdir(hf_out_dir))
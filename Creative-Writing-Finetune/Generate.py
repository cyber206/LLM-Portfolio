import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_path = "LiquidAI/LFM2-1.2B"
adapter_path = "model_output/sft_adapter"

# 1. Loading the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(base_model,adapter_path)
model.eval()


prompt = """USER:""" #when cloning please remove the user thing and type ur own prompt.
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens = 500,
        temperature = 0.6,
        top_p=0.9,
        repetition_penalty=1.1,     #let's see how curious it becomes
        do_sample = True

    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
from trl import SFTTrainer,SFTConfig
from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_from_disk
from peft import PeftModel
import torch
model_path = "LiquidAI/LFM2-1.2B"
sft_data_path = "Data/sft_dataset"
cpt_data_path = "model_output/adapter"
output_dir =  "model_output/sft_adapter"
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model,cpt_data_path,is_trainable = True)
model.config.use_cache =  False

dataset = load_from_disk(sft_data_path)

sft_config = SFTConfig(
    output_dir=output_dir,
    max_steps = 200,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate = 1e-6,
    warmup_steps = 15,
    logging_steps = 10,
    save_steps = 100,
    bf16 = True,
    max_length = 512,
    dataset_text_field="text",
    completion_only_loss=True,  #  Only train on the Assistant part
    packing=False,              # Keep False for 6GB VRAM
    report_to="none"


)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 6. Train
print("Starting SFT Training...")
trainer.train()

# 7. Save
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("SFT Done. Adapter saved to:", output_dir)


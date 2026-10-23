import os
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,get_cosine_schedule_with_warmup
import numpy as np
import time
import json
from dotenv import load_dotenv
load_dotenv()
from peft import PeftModel,get_peft_model,LoraConfig
import random
_init_from = "scratch"
out_dir = "model_output"
device = torch.device("cuda")
device_type = 'cuda'
train = "Data/train.bin"
test = "Data/test.bin"
model_path = "LiquidAI/LFM2-1.2B"
os.environ["HF_HUB_DISABLE_XET"] = "1" #idk why but downloading is taking alot lets see if this fixes it
always_checkpoint = True
log_interval = 10
batch_size = 1
graident_accumulation_steps = 8
block_size = 512
learning_rate = 1e-5
max_iter = 500
warmup_iter = 25
weight_decay = 0.01
grad_clip = 1.0
eval_interval = 50
eval_iter = 20
save_interval = 50
seed = 1337
use_lora = True
os.makedirs(out_dir, exist_ok=True)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found, please run with --cuda")
use_bf16 = torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16
print("GPU:",torch.cuda.get_device_name(0))
print("Precision:",dtype)
print("Lora:", use_bf16)


ctx = torch.amp.autocast('cuda',dtype=dtype,enabled = device_type =="cuda")
use_scaled = device_type == "cuda" and dtype == torch.float16
scaler = torch.amp.GradScaler('cuda',enabled=use_scaled)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=dtype).to(device)
model.config.use_cache = False
model.gradient_checkpointing_enable()
if use_lora:
    lora_config = LoraConfig(
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0.01,
        bias = "none",
        task_type = "CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    model = get_peft_model(model,lora_config)
    model.print_trainable_parameters()
lora_layers = [name for name, _ in model.named_modules() if "lora_A" in name]
print("LoRA attached to:", lora_layers[:10]) # Prints the first 10 to keep it clean
model.to(device)
model.train()

train_data = np.memmap(train,dtype = np.uint16,mode = "r")
valid_data = np.memmap(test,dtype = np.uint16,mode = "r")

if len(train_data)<= block_size + 1  and len(valid_data)<= block_size + 1:
    raise RuntimeError("Samples and block_size didn't matched please look through.")

print(f"Train tokens: {len(train_data):,}")
print(f"Valid tokens: {len(valid_data):,}")

def get_batch(split):
    filename = train_data if split == "train" else valid_data
    if len(filename) <= block_size:
        raise RuntimeError(f"File{filename} is too small for the chosen block size")
    ix = torch.randint(len(filename) - block_size,(batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(filename[i:i+block_size].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device,non_blocking=True)
    else:
        x = x.to(device,non_blocking=True)

    return x

checkpoint_path = os.path.join(out_dir, "ckpt.pt")
adapter_path = os.path.join(out_dir, "adapter")

iter_num = 0
best_val_loss = float("inf")
checkpoint = None

if _init_from == "scratch":
    pass
elif _init_from == "resume":
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist")

    if not os.path.isdir(adapter_path):
        raise RuntimeError(f"Adapter not found:{adapter_path}")

    del model
    base_model = AutoModelForCausalLM.from_pretrained(model_path,dtype=dtype).to(device)
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(base_model,adapter_path,is_trainable=True)
    model.to(device)
    model.train()

    checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=True)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError(f"Unrecognized initialization method: {_init_from}")

print(f"Initialization: {_init_from}")
print(f"Starting iteration: {iter_num}")

trainable_parameters = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate,betas=(0.9,0.999),weight_decay=weight_decay)
scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps = warmup_iter,num_training_steps = max_iter)

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    scheduler.load_state_dict(checkpoint["scheduler"])


@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for name in ['train','valid']:
        losses = []
        for _ in range(eval_iter):
            x = get_batch(name)
            with ctx:
                outputs = model(input_ids=x,labels = x)
            losses.append(outputs.loss.float().item())
        results[name] = sum(losses)/len(losses)
    model.train()
    return results


def save_checkpoint(valid_loss):
    global best_val_loss
    checkpoint_data = {
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": float(best_val_loss),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "valid_loss": float(valid_loss),
    }
    torch.save(checkpoint_data,checkpoint_path)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    metadata = {
        "iter_num": iter_num,
        "best_val_loss": float(best_val_loss),
        "valid_loss": float(valid_loss),
        "model_name": model_path,
        "use_lora": use_lora,
    }
    with open(os.path.join(out_dir,"training_metadata.json"),"w") as f:
        json.dump(metadata,f)
    print(f"Checkpoint saved at {iter_num} iterations")

optimizer.zero_grad(set_to_none=True)
t0 = time.time()
last_losses = None

while iter_num <max_iter:
    lr = optimizer.param_groups[0]['lr']
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        last_losses = losses
        print(f"step {iter_num}/{max_iter}:"
              f"train loss {losses['train']:.4f}"
              f"valid loss {losses['valid']:.4f}"
              f"lr {lr:.6g}")
        should_save = (losses['valid'] < best_val_loss or always_checkpoint)
        if should_save:
            best_val_loss = min(best_val_loss,losses['valid'])
            save_checkpoint(losses["valid"])
    accumulated_loss = 0.0
    for micro_step in range(graident_accumulation_steps):
        x = get_batch("train")
        with ctx:
            outputs = model(input_ids=x,labels = x)
            loss = outputs.loss/ graident_accumulation_steps
            accumulated_loss += loss.detach().float().item()
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    iter_num += 1
    if iter_num % log_interval == 0:
        elapsed = time.time() - t0

        print(
            f"step {iter_num}/{max_iter} | "
            f"loss {accumulated_loss:.4f} | "
            f"lr {lr:.6g} | "
            f"elapsed {elapsed:.1f}s"
            )

last_losses = estimate_loss()

if last_losses["valid"] < best_val_loss:
    best_val_loss = last_losses["valid"]

save_checkpoint(last_losses["valid"])

model.config.use_cache = True

print("Training finished.")
print("Final iteration:", iter_num)
print("Best validation loss:", best_val_loss)
print("Saved checkpoint:", checkpoint_path)
print("Saved LoRA adapter:", adapter_path)


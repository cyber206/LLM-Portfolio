import os
import math
import time
import json
import random
import numpy as np
import torch
from contextlib import nullcontext
from Model import GPT,GPT_config,TinyGPTConfig,TinyGPTForCausalLM
out_dir = "out-tinystories"
hf_out_dir = os.path.join(out_dir,"hf-model")

eval_interval = 100
log_interval = 10
eval_iter = 100
eval_only = False
always_checkpoint = True
init_from = "resume"

dataset = "tinystories"
gradient_accumulation_steps = 64
batch_size = 4
n_head: int = 8
n_layer: int = 16
n_embed: int = 512
dropout: float = 0.2
block_size: int = 378
vocab_size: int = 50000

learning_rate = 3e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iter = 500
lr_decay_iter = 20000
min_lr = 3e-5

device = "cuda" if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if device=='cuda' else 'cpu'

if device_type=='cuda':
    dtype = ("bfloat16"if torch.cuda.is_bf16_supported() else "float16")

else :
    dtype = "float32"

compile = False
seed = 1337
os.makedirs(out_dir,exist_ok=True)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if device_type=='cuda':
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = (
    torch.amp.autocast("cuda",dtype = ptdtype) if device =='cuda' else nullcontext()

)
use_scaler = device_type == 'cuda' and dtype == 'float16'
scaler = torch.amp.GradScaler(enabled=use_scaler)

tokens_per_iter = (gradient_accumulation_steps * batch_size * block_size)

print(f"Device: {device}")
print(f"Precision: {dtype}")
print(f"Tokens per optimizer step: {tokens_per_iter}")


def get_batch(split):
    filename = 'train.bin' if split == 'train' else 'valid.bin'
    data = np.memmap(filename,dtype = np.uint16,mode = "r")
    if len(data) <= block_size:
        raise ValueError(
            f"{filename} has too few tokens for block_size={block_size}"
        )
    ix = torch.randint(len(data)-block_size-1,(batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.uint16)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.uint16)) for i in ix])
    if device_type=='cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else :
        x = x.to(device=device, dtype=torch.long, non_blocking=True)
        y = y.to(device=device, dtype=torch.long, non_blocking=True)

    return x,y

iter_num  = 0
best_val_loss = float('inf')
model_args = {
    "n_layer":n_layer,
    "n_head":n_head,
    "n_embed":n_embed,
    "block_size":block_size,
    "vocab_size":vocab_size,
}
checkpoint = None
if init_from == "scratch":
    print("Initializing GPT from scratch")
    model = GPT(GPT_config(**model_args))

elif init_from == "resume":
    checkpoint_path = os.path.join(out_dir,"ckpt.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} does not exist")
    print(f"Resuming from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path,
                            map_location = "cpu",
                            weights_only=False)
    model_args = checkpoint["model_args"]
    model = GPT(GPT_config(**model_args))
    state_dict = checkpoint["model"]
    state_dict = {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = float(checkpoint["best_val_loss"])


else:
    raise ValueError(f"init_ from must be scratch or resume")

model.to(device)
optimizer = model.configure_optimizers(weight_decay,(beta1,beta2),learning_rate,device_type)
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint["optimizer"])

    for state in optimizer.state.values():
        for key,value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

del checkpoint
raw_model = model

if compile:
    print("Compiling model...")
    model = torch.compile(model)
@torch.no_grad()
def estimate_loss():
    results = {}
    model.eval()

    for split in ["train","valid"]:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X,Y = get_batch(split)
            with ctx:
                _,loss = model(X,Y)
            losses[k] = loss.detach().float().cpu()
        results[split] = losses.mean().item()

    model.train()
    return results

def get_lr(step):
    if step < warmup_iter:
        return learning_rate * (step +1) /(warmup_iter+1)

    if step >= lr_decay_iter:
        return min_lr

    decay_ratio = (step - warmup_iter) / (lr_decay_iter - warmup_iter)

    coeff = 0.5 *(1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def save_checkpoint(valid_loss):
    global best_val_loss

    checkpoint_path = os.path.join(out_dir,"ckpt.pt")
    checkpoint_data = {
        "model":raw_model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "iter_num":iter_num,
        "model_args":model_args,
        "best_val_loss":best_val_loss,
    }
    torch.save(checkpoint_data,checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    hf_config = TinyGPTConfig(**model_args)
    hf_model = TinyGPTForCausalLM(hf_config)

    # Transfer the learned pretrained weights
    hf_model.gpt.load_state_dict(raw_model.state_dict(), strict=True)

    hf_model.eval()
    os.makedirs(hf_out_dir, exist_ok=True)
    hf_model.save_pretrained(hf_out_dir, safe_serialization=False)

    print(f"Exported HF model: {hf_out_dir}")
    del hf_model


    with open(
        os.path.join(out_dir, "training_metadata.json",),'w',encoding='utf-8') as f:
        json.dump({"iteration":iter_num,"best_val_loss":float(valid_loss),"model_args":model_args},f,indent=2)
        print(f"Exported HF model: {hf_out_dir}")



if __name__ == "__main__":
    X,Y = get_batch("train")
    t0 = time.time()
    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {iter_num}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['valid']:.4f}, "
                f"lr {lr:.6g}"
            )

            should_save = (
                losses["valid"] < best_val_loss
                or always_checkpoint
            )
            if should_save:
                best_val_loss = min(best_val_loss, losses["valid"])
                save_checkpoint(losses["valid"])

        if iter_num == 0 and eval_only:
            break

        if iter_num >= max_iters:
            break

        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits,loss = model(X,Y)
                loss = loss/gradient_accumulation_steps

            X,Y = get_batch("train")
            scaler.scale(loss).backward()

        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        iter_num += 1
        if iter_num % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"step {iter_num}/{max_iters} | "
                f"elapsed {elapsed:.1f}s"
                f"loss {loss:.4f}"
            )

    print("Pretraining finished.")













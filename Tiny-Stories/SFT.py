from Model import GPT,GPT_config
import torch
from tokenizers import Tokenizer
from datasets import load_dataset
from torch.utils.data import Dataset,DataLoader
from torch.optim import AdamW
path = r"D:\Tiny-Stories\out-tinystories\ckpt.pt"
data_path = r"D:\Tiny-Stories\databricks-dolly-15k.jsonl"
ckpt = torch.load(path,map_location='cpu',weights_only=False)
model = GPT(GPT_config(**ckpt["model_args"]))
model = model.to(torch.device('cuda'))
x = torch.randint(0, 1000, (1, 16)).cuda()
out = model(x, targets=x)
print(type(out), len(out) if isinstance(out, tuple) else out)

tokenizer = Tokenizer.from_file("tokenizer.json")
EOS = 1
ds = load_dataset("json",data_files =data_path,split = "train")
def fmt(ex):
    if ex.get("context"):
        p = f"### Instruction:\n{ex['instruction']}\n\n### Context:\n{ex['context']}\n\n### Response:\n"
    else:
        p = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n"
    return {"prompt": p, "completion": ex["response"]}

dataset = ds.map(fmt,remove_columns = ds.column_names)

class SFTDataset(Dataset):
    def __init__(self,ds,tok,max_len = 350):
        self.ds = ds
        self.tok = tok
        self.max_len = max_len
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        ex = self.ds[idx]
        self.tok.eos_token_id = 1
        p = self.tok.encode(ex["prompt"], add_special_tokens = False).ids
        c = self.tok.encode(ex["completion"], add_special_tokens = False).ids + [self.tok.eos_token_id]
        ids = (p+c)[:self.max_len]
        labels = ([-100]*len(p)+c)[:self.max_len]
        return {"input_ids":ids, "labels":labels}

PAD = 1
def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    ids,mask,labels = [],[],[]
    for b in batch:
        pad = L-len(b["input_ids"])
        ids.append(b["input_ids"] + [PAD] * pad)
        labels.append(b["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),}

loader = DataLoader(SFTDataset(dataset, tokenizer), batch_size=4, shuffle=True, collate_fn=collate)
# batch = next(iter(loader))
# print("labels sample:", batch["labels"][0][:20])
# print("num -100 in row 0:", (batch["labels"][0] == -100).sum().item())
# print("total tokens row 0:", batch["labels"][0].numel())
optim = AdamW(model.parameters(), lr=2e-5)
model.train()
batch = next(iter(loader))
batch = {k: v.to("cuda") for k, v in batch.items()}
# with torch.no_grad():
#     _, loss_masked   = model(batch["input_ids"], targets=batch["labels"])
#     _, loss_unmasked = model(batch["input_ids"], targets=batch["input_ids"])
# print("masked:", loss_masked.item(), "unmasked:", loss_unmasked.item())
for step, batch in enumerate(loader):
    batch = {k: v.to("cuda") for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch["input_ids"], targets=batch["labels"])
        loss = out[1] if isinstance(out, tuple) else out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    optim.zero_grad()
    if step % 50 == 0:
        print(f"step {step} | loss {loss.item():.4f}")
    if step >= 2000:
        break

torch.save(model.state_dict(), "sft_model.pt")
print("saved sft_model.pt")
model.eval()
prompt = "### Instruction:\nhey\n\n### Response:\n"
ids =tokenizer.encode(prompt, add_special_tokens=False).ids
x = torch.tensor([ids], dtype=torch.long).cuda()
with torch.no_grad():
    for _ in range(60):
        logits, _ = model(x)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], 1)
        if nxt.item() == EOS:
            break
print("IN-PROCESS:", tokenizer.decode(x[0].tolist()))
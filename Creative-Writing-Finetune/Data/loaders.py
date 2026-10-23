from datasets import load_from_disk
import numpy as np
from transformers import AutoTokenizer

path = 'literotica-reddit-dirty-writing-prompts'
model_id = "LiquidAI/LFM2-1.2B"
data = load_from_disk(path)
split =  data.train_test_split(test_size=0.1, seed=42)
train_data = split['train']
val_data = split['test']
# print(len(train_data))
# print(len(val_data))

tokenizer = AutoTokenizer.from_pretrained(model_id)
train_ids = []
for text in train_data['text']:
    ids = tokenizer.encode(text,add_special_tokens=True)
    train_ids.extend(ids)
    train_ids.append(tokenizer.eos_token_id)

test_ids = []
for text in val_data["text"]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    test_ids.extend(ids)
    test_ids.append(tokenizer.eos_token_id)

# print(len(train_ids))
# print(len(test_ids))

np.array(train_ids,dtype = np.uint16).tofile('train.bin')
np.array(test_ids,dtype = np.uint16).tofile('test.bin')
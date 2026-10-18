#%%
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from datasets import load_dataset

dataset = load_dataset("parquet",
                       data_files = {
                           'train':["data/train-00000-of-00004-2d5a1467fff1081b.parquet",
                                    "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
                                    "data/train-00002-of-00004-a26307300439e943.parquet",
                                    "data/train-00003-of-00004-d243063613e5a057.parquet"],
                           'test':["data/validation-00000-of-00001-869c898b519ad725.parquet"]
                       })

train = dataset['train']
test = dataset['test']
# len(train)

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()
tokenizer.post_processor = ByteLevelProcessor(train_offsets = False )
trainer = BpeTrainer(vocab_size=50000,min_frequency=2,special_tokens=["<unk>","<eos>"],show_progress=True)
def batch_iter(batch_size=1000):
    for i in range(0,len(train),batch_size):
        yield train[i:i+batch_size]["text"]
tokenizer.train_from_iterator(batch_iter(),trainer = trainer,length=len(train))
tokenizer.save("tokenizer.json")


tokenizer = tokenizer.from_file("tokenizer.json")
print("vocab size:", tokenizer.get_vocab_size())
assert tokenizer.get_vocab_size() == 50_000, "vocab_size mismatch — check initial_alphabet / data size"
probes = [
    "### Instruction:\nhey\n\n### Response:\n",
    "a\tb\n\nc   d",           # tabs + runs of spaces
    "trailing space ",
    "café",                     # non-ASCII
    ":",                        # the token the old model collapsed miserably T_T
]
for p in probes:
    ids = tokenizer.encode(p, add_special_tokens=False).ids
    back = tokenizer.decode(ids)
    assert back == p, f"ROUND-TRIP FAILED\n  in:  {p!r}\n  out: {back!r}\n  ids: {ids}"
print("round-trip gate: PASS")
EOS = tokenizer.token_to_id("<eos>")

train_ids = []
for text in train["text"]:
    ids = tokenizer.encode(text).ids
    train_ids.extend(ids)
    train_ids.append(tokenizer.token_to_id("<eos>"))

valid_ids = []
for text in test['text']:
    ids = tokenizer.encode(text).ids
    valid_ids.extend(ids)
    valid_ids.append(tokenizer.token_to_id("<eos>"))

#%%
import numpy as np
np.array(train_ids,dtype=np.uint16).tofile("train.bin")
np.array(valid_ids,dtype=np.uint16).tofile("valid.bin")
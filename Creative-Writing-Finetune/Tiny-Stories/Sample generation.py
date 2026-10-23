import numpy as np
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
tr = np.memmap("train.bin", dtype=np.uint16, mode="r")

print("train tokens:", len(tr))
print("max id:", int(tr.max()))
assert tr.max() < 50_000, "ID out of range — wrong tokenizer"

print("\n--- first 300 tokens decoded ---")
print(repr(tok.decode(tr[:300].tolist())))
print("\n--- tokens 1,000,000–1,000,300 decoded ---")
print(repr(tok.decode(tr[1_000_000:1_000_300].tolist())))
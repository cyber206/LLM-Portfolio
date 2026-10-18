from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


SRC = r"/tokenizer.json"
DST = r"D:\Tiny-Stories\out-tinystories\hf-model"

raw = Tokenizer.from_file(SRC)
print("loaded raw tokenizer, vocab size:", raw.get_vocab_size())

tok = PreTrainedTokenizerFast(tokenizer_object=raw)
tok.add_special_tokens({"pad_token": "<eos>"})

tok.save_pretrained(DST)

print("pad_token:", tok.pad_token, "id:", tok.pad_token_id)
print("eos_token:", tok.eos_token, "id:", tok.eos_token_id)
print("Saved to", DST)
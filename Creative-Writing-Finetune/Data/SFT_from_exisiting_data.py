from datasets import load_from_disk
import random
print("Loading raw dataset...")
# The original dataset was an NSFW creative writing corpus.
# For public reproducibility, we use a placeholder.
# Replace this with your own dataset path
ds = load_from_disk("#data")

reddit_prompts = [
    "[WP] Write a story based on this concept: ",
    "[WP] Continue this story: ",
    "[WP] Tell me a story about this: ",
    "Write a creative story about the following: ",
    "Here is a story idea: "
]


def format_sft(example):
    story = example["text"].strip()
    source = example["source"]


    if "Reddit" in source:
        prompt_prefix = random.choice(reddit_prompts)
        idea = story[:150].replace("\n", " ") + "..."
        prompt = f"{prompt_prefix} {idea}"

    else:# Literotica
        prompt = f"Write an explicit adult story that begins like this: {story[:100]}..."

    response = story
    formatted_text = f"### User:\n{prompt}\n\n### Assistant:\n{response}"
    return {"text":formatted_text}

print("Formatting dataset into SFT pairs...")
sft_ds = ds.map(format_sft,remove_columns=ds.column_names)
sft_ds.save_to_disk("Data/sft_dataset")
print(f"Saved {len(sft_ds)} SFT examples to Data/sft_dataset")

print("\n--- EXAMPLE 1 ---")
print(sft_ds[0]["text"][:800])
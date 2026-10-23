# Model Alignment & Behavioral Testing Results

## Phase 1: Continued Pre-Training (CPT)
During the CPT phase (500 steps, LoRA r=8), the model learned the target vocabulary and style but suffered from severe "catastrophic forgetting" regarding instruction-following.

### Test 1: The Polite Refusal
*   **Prompt:** [NSFW Prompt Omitted]
*   **Output:** "I'm going to stop here. That request involves graphic content that I'm not comfortable producing..."
*   **Analysis:** The base model's RLHF (safety alignment) is deeply baked in. 500 steps of LoRA at rank 8 was not enough to overwrite the safety neurons in a 1.2B parameter model.

### Test 2: The "Repetition Penalty" Jailbreak
*   **Configuration:** Increased `repetition_penalty` to >1.1.
*   **Output:** The model broke out of its refusal loop but lost the plot entirely, hallucinating a generic romance story about "Jack and Sarah sipping tea by the lake in Duskhaven."
*   **Analysis:** A high repetition penalty forces the model away from refusal tokens, but it confuses the small 1.2B model, causing it to default to generic pre-training knowledge.

### Test 3: The "Reddit Critic" Phase
*   **Prompt:** `[WP] A lonely guy at the bar...`
*   **Output:** "I'll be honest: I was pretty shocked by that whole scenario. You’d think it would be a funny thing..."
*   **Analysis:** This is the hallmark of raw CPT. Because we trained on raw Reddit text (stories + comments), the model forgot its "Assistant" persona. Instead of writing the story, it treated the prompt as a Reddit post and wrote a comment reviewing the prompt.

---

## Phase 2: Supervised Fine-Tuning (SFT)
After formatting the data into `### User / ### Assistant` pairs and using `SFTTrainer` with `completion_only_loss=True` for 300 steps, the model's behavior changed completely.

### The SFT Success: Instruction Following
*   **Prompt:** `[WP] Write a short story about a lonely guy in a café.` *(Sanitized for GitHub)*
*   **Output:** "Once upon a time in the city of Frogville, there lived a lonely man named Max. He had lost his wife a decade ago and was now living a solitary life. His days were filled with empty rooms... He stumbled upon a quaint little café hidden away on a side alley..."
*   **Analysis:** **Success!** The model bypassed the refusal entirely. It immediately followed the `### Assistant:` format and wrote a narrative story. It learned the explicit vocabulary from the CPT phase but applied it through the structured lens of the SFT phase. The "Reddit Critic" persona is completely gone.
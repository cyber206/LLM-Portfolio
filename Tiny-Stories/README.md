# From-Scratch TinyStories GPT (75.9M)

This project contains a complete, hand-written pipeline for training a GPT-style language model from scratch using raw PyTorch (no HuggingFace `Trainer`, no TRL). 

##  The Engineering Story (The "Why")
Building the model was easy; making it not collapse was the real challenge. This pipeline was built through debugging three distinct failure modes:

**1. The Tokenizer Collapse (`:` loop)**
*   **Symptom:** The model generated the token `:` infinitely.
*   **Root Cause:** The initial tokenizer used a `Whitespace()` pre-tokenizer, which stripped all newlines. The prompt/response boundary (`### Response:\n`) was invisible to the model.
*   **Fix:** Rebuilt the tokenizer from scratch using a ByteLevel BPE (50k vocab). Added a "round-trip gate" (`decode(encode(x)) == x`) to guarantee no whitespace is ever lost. 

**2. The Checkpoint Overwrite**
*   **Symptom:** A 1,000-step base model was accidentally overwritten by a fresh initialization.
*   **Fix:** Implemented strict checkpoint management, including timestamped run directories and a `tokenizer_sha` hash saved inside the checkpoint to ensure weights are never loaded against the wrong tokenizer.

**3. The Undertrained-Base SFT Collapse (`\n` loop)**
*   **Symptom:** After SFT, the model outputted only newlines.
*   **Root Cause:** We diagnosed this by greedy-decoding the model and checking the logits. The model was outputting token 132 (`\n`) at 100% probability. The SFT dataset was fine, but the *base model* had only trained for 2k steps—it hadn't learned language yet. SFT forced it into a local minimum.
*   **Fix:** Establish an evaluation gate: *Never run SFT until the base model can generate coherent text on its own.* Train the base for 20k+ steps.

##  File Structure
*   `Tokenizers.py`: Trains the ByteLevel BPE and converts text to `train.bin` / `valid.bin`.
*   `Model.py`: The raw PyTorch GPT architecture (multi-head attention, MLP, LayerNorm).
*   `Pre_training.py`: The hand-written pretraining loop using memmapped data.
*   `SFT.py`: The SFT loop. Handles prompt masking (using `-100`), EOS appending, and padding.
*   `chat.py`: Inference script.
*   `Sample generation.py`: Diagnostic script to check generation at different stages.

##  Key Takeaways & Reflections

**Hardware Constraints drive Engineering Creativity:** 
My goal was to build a 75.9M parameter model entirely from scratch. Because I was constrained to a local RTX 3050 (6GB VRAM), I couldn't rely on brute force. I had to carefully manage memory, implement gradient accumulation, and avoid unnecessary overhead to train this model without crashing the GPU.

**The Learning Curve:**
Compared to my previous raw-script projects (like a 30M-parameter Harry Potter GPT where I didn't use gradient accumulation, context management, or `bfloat16`, yet still maxed out 5GB of VRAM), this project forced me to build a disciplined pipeline. I learned how crucial tokenizers, gradient masking, and memory management are to the training loop.

**What's Next?**
This project proves I can build from scratch. The logical next step is to transition to industry-standard workflows:
*   Use a pretrained base model (e.g., GPT-2) and perform **continued pretraining**.
*   Implement my own tokenizer using the HuggingFace `tokenizers` library.
*   Apply techniques like **LoRA / PEFT** for efficient fine-tuning.
*   The ultimate goal: A working instruction-following chatbot that can hold a coherent conversation.

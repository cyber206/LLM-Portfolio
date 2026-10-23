# LFM2-1.2B Fine-Tuning Pipeline (CPT + SFT) on 6GB VRAM

##  Overview
This repository demonstrates a complete, end-to-end post-training pipeline for a Small Language Model (`LiquidAI/LFM2-1.2B`). The project was engineered to run on a **single consumer GPU (NVIDIA RTX 3050 Laptop, 6GB VRAM)**, proving that model alignment and style adaptation can be achieved on budget hardware.

The pipeline consists of two phases:
1. **Continued Pre-Training (CPT):** Custom PyTorch loop for raw text style adaptation.
2. **Supervised Fine-Tuning (SFT):** Instruction alignment using Hugging Face `TRL` and `SFTTrainer`.

*⚠️ **Disclaimer:** This project is for educational and research purposes to study model alignment, instruction tuning, and catastrophic forgetting. The original training dataset contains NSFW text and is **NOT** hosted in this repository.*

## 🛠️ Technical Achievements & Debugging
*   **Custom CPT Loop:** Built a raw PyTorch training loop using `np.memmap` for token streaming, handling gradient accumulation, cosine warmup, and mixed precision (`bfloat16`).
*   **QLoRA Implementation:** Utilized 4-bit quantization (`bitsandbytes`) and targeted LFM2-specific modules (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `in_proj`, `w1`, `w2`, `w3`).
*   **SFT with TRL:** Transitioned to `SFTTrainer` to handle complex text masking (`completion_only_loss=True`), ensuring the model only trains on the Assistant's responses, not the User prompts.
*   **Windows-Specific Bug Fixes:** 
    *   Bypassed the Hugging Face `hf-xet` download deadlock by setting `HF_HUB_DISABLE_XET=1`.
    *   Resolved VRAM OOM (Out of Memory) errors by optimizing `max_length` and using gradient checkpointing.

##  Repository Structure
*   `CPT.py`: Custom PyTorch loop for raw text Continued Pre-Training.
*   `make_sft_data.py`: Script to format raw text into `### User / ### Assistant` pairs.
*   `SFT.py`: TRL-based Supervised Fine-Tuning script.
*   `Generate.py`: Inference script to test the final model.
*   `CPT_result.md`: Detailed logs and behavioral analysis of the model's shift from refusing prompts to generating narratives.

##  Results & Behavioral Observations
*   **CPT Phase:** The model successfully learned the target text style but suffered from *catastrophic forgetting* (it started reviewing prompts instead of answering them).
*   **SFT Phase:** By training for 300 steps with a low learning rate (`1e-5`), the model re-learned instruction-following. The safety alignment was successfully overwritten, and the model reliably generated narratives directly from prompts.
*   **Limitations:** Due to the 1.2B parameter size, the model occasionally suffers from "Lost in the Middle" context loss after 150+ tokens, reverting to generic story tropes.
*   **Full Analysis:** Read the detailed behavioral testing logs in [`CPT_result.md`](./CPT_result.md).

##  How to Run

**1. Clone the repo and install dependencies:**

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt (please open the requirmenet file the torch libary idk why but it keeps installing the cpu one though i have written what to do)

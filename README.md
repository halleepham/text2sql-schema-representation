# Structure Matters: Evaluating Schema Representations for Text-to-SQL LLMs
**Hallee Pham | COMP-SCI 5590 (Graduate) - Generative AI & LLMs | UMKC Spring 2026**

Full repository: https://github.com/halleepham/text2sql-schema-representation

---

## Project Overview
This project studies how database schema representation format affects SQL generation performance of LLaMA 3.1 8B on the ATIS airline query dataset. Four schema formats are compared: relational, CREATE TABLE, JSON, and natural language, using zero-shot prompting and LoRA fine-tuning. Schema format is the sole variable under study; all other factors are held constant.

---

## Environment and Setup
- Google Colab Pro with A100 GPU required
- HuggingFace account with approved access to `meta-llama/Llama-3.1-8B`
- No manual file uploads needed — all data, schemas, scripts, and results are cloned automatically from GitHub
- Install dependencies (handled automatically in Section 0.2):
```bash
pip install transformers peft bitsandbytes accelerate
```

---

## How to Run

**Step 1:** Open `code/02_text_to_sql.ipynb` in Google Colab and change runtime to A100 GPU (Runtime → Change runtime type)

**Step 2:** By default, `SKIP_GENERATION = True` and all `SKIP_TRAINING` flags are `True` — existing predictions, metrics, and adapters are loaded from the cloned repo. No generation or training will run unless you change these flags.

**Step 3:** Run Section 0.1 (HuggingFace Login) alone first and enter your token when prompted. **Do not press Run All.**

**Step 4:** Click Section 0.2 to make it the focused cell, then click the down arrow next to "Run all" and select "Run focused cells and all cells below."

> If a HuggingFace authentication error occurs, only rerun the cell with the error and all cells below it.

**To reproduce from scratch (optional):**
Set `SKIP_GENERATION = False` and/or `SKIP_TRAINING_RELATIONAL = False`, `SKIP_TRAINING_CREATE_TABLE = False` before running. Each generation or training cell takes approximately 25-30 minutes (meaning a few hours to run the whole notebook).

**Preprocessing (optional):**
```bash
python scripts/preprocessing.py
```

---

## Outputs and Artifacts

| Artifact | Location |
|---|---|
| Base model metrics | `results/base/` |
| LoRA metrics | `results/lora/` |
| Loss curve plots | `results/lora/` |
| Full predictions, adapters, data splits | https://github.com/halleepham/text2sql-schema-representation |

---

## Demo
1. Run all cells in Sections 0 and 1
2. Run Section 4.1 to load the base model
3. Edit the `question` variable in Section 4.2
4. Run Section 4.2 to generate SQL and execute against the ATIS database

---

## Hardware and Compute Limitations
- A100 GPU (40GB) required — CPU inference is not practical
- Each base model evaluation run: ~25 minutes (402 examples, batch size 8)
- Each LoRA training run: ~30 minutes (2 epochs, batch size 2)
- Full reproduction from scratch: ~3-4 hours of A100 compute
- LoRA adapters were trained for 2 epochs due to compute constraints
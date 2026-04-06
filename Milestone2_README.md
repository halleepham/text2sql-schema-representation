# Milestone 2 - Directions
Hallee Pham

## Zip File Contents
These are the files that are included in the submitted zip file. NO manual uploads in Google Colab session is required. The data files are downloaded automatically from my GitHub Repo, and helper functions are defined in the notebook.

```
Milestone2_HPham.zip
│
├── training_code/
│   ├── data/                        # Contains the data files created in preprocessing notebook
│   │   ├── split_train.jsonl        # 4055 training question-SQL pairs
│   │   ├── split_dev.jsonl          # 436 dev question-SQL pairs
│   │   └── split_test.jsonl         # 404 test question-SQL pairs (not used in this milestone)
│   ├── schemas/                     # These schema .txt files were manually created by me
│   │   ├── schema_relational.txt    # relational format (baseline)
│   │   ├── schema_create_table.txt  # CREATE TABLE format
│   │   ├── schema_json.txt          # JSON format
│   │   └── schema_nl.txt            # natural language format
│   ├── 00_preprocessing.ipynb       # preprocessing pipeline
│   └── 01_milestone2_smoke_test.ipynb          # main milestone 2 notebook
│
├── results/
│   ├── smoke_test_log.txt           # loss values at each evaluation step
│   └── loss_curve.png               # train vs val loss curve
│
├── report.pdf                       # Summarization notebooks
└── README.md                        # This file
```
> The `report.pdf` provides a summarization of explainations and outputs in the notebooks, as the notebooks go into great (and extra) detail. All required fields and explainations as stated in the assignment requirements.

 --- 

## To Run the Smoke Test
**Step 1 - Setup Colab Session**
*   Extract all files from `Milestone2_HalleePham.zip`
*   Open Google Colab
*   Upload/open `01_milestone2_smoke_test.ipynb` in the `training_code/` folder of the zip file
*   Click "Runtime" -> "Change runtime type" -> Select a GPU (I used the A100 GPU with my Student Colab Pro subscription) 
*   Data files are downloaded automatically from my GitHub repository (https://github.com/halleepham/text2sql-schema-representation) - no manual upload needed.

**Step 2 - Get HuggingFace Access**
This notebook requires access to `meta-llama/Llama-3.1-8B`.  
*   Create a free account at https://huggingface.co if you don't have an account
*   Request access at https://huggingface.co/meta-llama/Llama-3.1-8B - approval is usually instant or within a few hours
*   Click your profile picture -> "Access Tokens" -> "+ Create new token" -> Change token type to "Read" -> name your token (can be anything) -> "Create token" -> Copy the token or save it somewhere

**Step 3 - Run the notebook**
*   Only run the first cell in the notebook (the set up cell with imports). DO NOT PRESS "RUN ALL"
*   This cell will prompt you to input your HuggingFace token.
*   Input the token you copied and click "Login"
*   Wait until the cell is done executing
*   Click/select the second code cell of the notebook (Under Section 1: Load Data) so it is the focused cell
*   Click the down arrow next to the "Run all" button (in the top bar)
*   Select "Run focused cells and all cells below"
> If this process is not followed, a HuggingFace authenticaton error may occur in the tokenizer section. If an error does occur, rerun the code cell with the error and all cells below it.

---

## To Run the Preprocessing Pipeline (00_preprocessing.ipynb)
> It is not necessary to run this notebook before the milestone2 notebook. This is simply to show the preprocessing process if interested.
**Step 1 - Setup Colab Session**
*   Extract all files from `Milestone2_HalleePham.zip`
*   Open Google Colab
*   Upload/open `00_preprocessing.ipynb`
*   GPU runtime is NOT required to run this notebook
*   Data files are downloaded automatically from my GitHub repository (https://github.com/halleepham/text2sql-schema-representation) - no manual upload needed.

**Step 2 - Run the notebook**
*   Click "Run all" button. Notebook should run all the way through without errors.

---

## Key Hyperparameters
| Parameter | Value |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B` |
| Quantization | 4-bit NF4 via `bitsandbytes` |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA target modules | `q_proj`, `v_proj` |
| LoRA dropout | 0.05 |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Batch size | 2 |
| Smoke test examples | 100 |
| Steps | 50 |
| Context length | 1024 |
| Python seed | 42 |
| PyTorch seed | 42 |

---

## Outputs
The only outputs are `loss_curve.png` and `smoke_test_log.txt` saved under the `results/` folder of the Google Colab session files when you run `01_milestone2_smoke_test.ipynb`.
> I have also included the outputs from MY run in the zip folder under `results/` and the `report.pdf`
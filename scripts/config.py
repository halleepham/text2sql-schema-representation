# =============================================================================
# config.py
# Central configuration file for this project
# all important paths, hyperparameters, and setting are defined here
# other scripts imports from this file
# =============================================================================

import os

# =============================================================================
# PATHS
# =============================================================================

# Root directory of my GitHub repo inside the Colab session
REPO_ROOT = "/content/text2sql-schema-representation"
# The actual URL of my repo for documentation purposes
REPO_URL = "https://github.com/halleepham/text2sql-schema-representation"

# Data
DATA_DIR = os.path.join(REPO_ROOT, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "split_train.jsonl")
DEV_PATH = os.path.join(DATA_DIR, "split_dev.jsonl")
TEST_PATH = os.path.join(DATA_DIR, "split_test.jsonl")

# Schemas
SCHEMA_DIR = os.path.join(REPO_ROOT, "schemas")
SCHEMA_PATHS = {
    "relational": os.path.join(SCHEMA_DIR, "schema_relational.txt"),
    "create_table": os.path.join(SCHEMA_DIR, "schema_create_table.txt"),
    "json": os.path.join(SCHEMA_DIR, "schema_json.txt"),
    "nl": os.path.join(SCHEMA_DIR, "schema_nl.txt"),
}

# Database
DB_PATH = os.path.join(REPO_ROOT, "database", "atis.sqlite")

# Adapters
ADAPTER_DIR = "/content/adapters"

def get_adapter_path(schema_name, rank):
    """Return the save/load path for a specific schema + rank adapter."""
    return os.path.join(ADAPTER_DIR, f"{schema_name}_r{rank}")

# Results
RESULTS_DIR = "/content/results"

def get_predictions_path(schema_name, model_type):
    """
    Return the save path for per-example prediction results.
    model_type is either 'base' or 'lora'.
    Example: results/predictions_relational_base.json
    """
    return os.path.join(RESULTS_DIR, f"predictions_{schema_name}_{model_type}.json")

def get_metrics_path(schema_name, model_type):
    """
    Return the save path for aggregate metrics summary.
    Example: results/metrics_relational_base.json
    """
    return os.path.join(RESULTS_DIR, f"metrics_{schema_name}_{model_type}.json")

# =============================================================================
# MODEL
# =============================================================================

# HuggingFace model ID for the base model
MODEL_ID = "meta-llama/Llama-3.1-8B"

# Maximum sequence length (prompt + SQL tokens combined)
MAX_SEQ_LEN = 1024

# =============================================================================
# LORA HYPERPARAMETERS
# =============================================================================

# Default rank. Can be overridden at training time for the rank ablation
LORA_R = 8

# Alpha controls the scaling of LoRA updates (convention is to set 2x rank)
LORA_ALPHA = 16

# Attention projection matrices to attach LoRA adapters to
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Dropout applied to LoRA layers during training
LORA_DROPOUT = 0.05

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

EPOCHS = 3
BATCH_SIZE = 2
LR = 5e-5 # Learning rate (AdamW optimizer)
EVAL_EVERY = 200 # Evaluate on dev set every N steps during training

# =============================================================================
# INFERENCE HYPERPARAMETERS
# =============================================================================

# Maximum number of new tokens the model can generate at inference time.
MAX_NEW_TOKENS = 64

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42
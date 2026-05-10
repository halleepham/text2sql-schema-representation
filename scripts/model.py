# =============================================================================
# model.py
# Model loading functions for LLaMA 3.1 8B with and without LoRA adapters.

# Used by both train.py and evaluate.py to avoid code duplication.

# Three functions:
#   - load_base_model_and_tokenizer(): loads raw quantized LLaMA (no adapter)
#   - attach_lora(): attaches LoRA adapters for training
#   - load_lora_model_and_tokenizer(): loads quantized LLaMA + saved adapter
# =============================================================================

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from config import (
    MODEL_ID,
    LORA_R,
    LORA_ALPHA,
    LORA_TARGET_MODULES,
    LORA_DROPOUT,
    get_adapter_path,
)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_jsonl(path):
    """Load a JSONL file and return a list of dicts."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# =============================================================================
# SHARED QUANTIZATION CONFIG
# =============================================================================

def get_bnb_config():
    """
    Return the 4-bit NF4 quantization configuration used for all model loads.

    Returns:
        BitsAndBytesConfig for 4-bit NF4 quantization (QLoRA)
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,  # load model in 4-bit precision
        bnb_4bit_use_double_quant=True, # nested quantization for extra memory savings
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


# =============================================================================
# BASE MODEL LOADING
# =============================================================================

def load_base_model_and_tokenizer():
    """
    Load LLaMA 3.1 8B in 4-bit NF4 quantization (QLoRA) and its tokenizer.

    Returns:
        model: quantized base model, no LoRA attached
        tokenizer: LLaMA tokenizer with pad token set to EOS
    """

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # LLaMA has no dedicated pad token —> use EOS token instead
    # This is the same decision made in Milestone 2 and documented in the report.
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=get_bnb_config(),
        device_map="auto",      # automatically place layers on available GPU(s)
    )

    # Disable KV cache during training (only used for faster inference, not training)
    model.config.use_cache = False

    # Set to eval mode (disable dropout)
    model.eval()

    return model, tokenizer


# =============================================================================
# LORA ADAPTER — ATTACH FOR TRAINING
# =============================================================================

def attach_lora(model, rank=LORA_R):
    """
    Attach LoRA adapters to the base model.

    Only the adapter weights are trainable.
    The base model weights remain frozen.

    Args:
        model: quantized base model
        rank: LoRA rank r.

    Returns:
        model: base model with LoRA adapters attached
    """

    lora_config = LoraConfig(
        r=rank,                              # rank of the LoRA decomposition
        lora_alpha=LORA_ALPHA,               # scaling factor (set to 2x rank by convention)
        target_modules=LORA_TARGET_MODULES,  # attach to q_proj and v_proj in all layers
        lora_dropout=LORA_DROPOUT,           # regularization
        bias="none",                         # don't train bias terms
        task_type=TaskType.CAUSAL_LM,        # causal language modeling task
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameter summary
    model.print_trainable_parameters()

    return model


# =============================================================================
# LORA MODEL LOADING — FOR INFERENCE
# =============================================================================

def load_lora_model_and_tokenizer(schema_name, rank=LORA_R):
    """
    Load LLaMA 3.1 8B with a saved LoRA adapter for inference.

    Loads the frozen base model first, then loads the saved adapter weights
    on top of it using PeftModel.from_pretrained().

    Args:
        schema_name (str): schema the adapter was trained on. Used to find the correct adapter folder.
        rank (int): LoRA rank the adapter was trained with.

    Returns:
        model: base model + LoRA adapter in eval mode
        tokenizer: LLaMA tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the frozen quantized base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=get_bnb_config(),
        device_map="auto",
    )

    # Load the saved LoRA adapter on top of the base model
    # Reads adapter_config.json and adapter_model.safetensors from adapter_path
    adapter_path = get_adapter_path(schema_name, rank)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Set to eval mode —> disables dropout, enables KV cache for faster inference
    model.eval()

    return model, tokenizer

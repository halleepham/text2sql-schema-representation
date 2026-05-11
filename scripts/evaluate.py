# =============================================================================
# evaluate.py
# Inference and evaluation pipeline for text-to-SQL.

# Supports two modes:
#   - Base model evaluation (no LoRA adapter)
#   - LoRA-adapted model evaluation (loads saved adapter)

# Main entry point: run_evaluation()
# =============================================================================

import torch

from config import (
    MAX_NEW_TOKENS,
)

# =============================================================================
# INFERENCE
# =============================================================================

def generate_sql(model, tokenizer, prompt, device, max_new_tokens=MAX_NEW_TOKENS):
    """
    Generate SQL from a prompt using greedy decoding.

    Args:
        model: model in eval mode (base or LoRA-adapted)
        tokenizer: LLaMA tokenizer
        prompt (str): full prompt string built by build_prompt()
        device (str): 'cuda' or 'cpu'
        max_new_tokens (int): maximum number of tokens to generate

    Returns:
        pred_sql (str): the generated SQL string, extracted from model output
    """

    # Tokenize the prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                  # greedy decoding —> no randomness
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # output_ids contains the full sequence: prompt tokens + generated tokens
    # We only want the generated part (after the prompt)
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_length:]

    # Decode generated token IDs back to a string
    pred_sql = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract SQL up to and including the first semicolon
    if ";" in pred_sql:
        pred_sql = pred_sql.split(";")[0].strip() + ";"
    else:
        # fallback if base model doesn't reliably generate semicolons
        pred_sql = pred_sql.split("\n")[0].strip()

    return pred_sql


def generate_sql_batch(model, tokenizer, prompts, device, max_new_tokens=MAX_NEW_TOKENS):
    """
    Generate SQL for a batch of prompts simultaneously.

    Args:
        model: model in eval mode
        tokenizer: LLaMA tokenizer
        prompts (list): list of fully assembled prompt strings
        device (str): 'cuda' or 'cpu'
        max_new_tokens: maximum tokens to generate per example

    Returns:
        list of predicted SQL strings, one per prompt
    """
    # Left padding required for decoder-only batched generation
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=True,
    ).to(device)

    # Per-example actual prompt lengths (excluding padding tokens)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Reset padding side for training compatibility
    tokenizer.padding_side = "right"

    # Extract and post-process generated tokens per example
    results = []
    for i in range(len(prompts)):
        generated_ids = output_ids[i][input_length:]
        pred_sql = tokenizer.decode(generated_ids, skip_special_tokens=True)
        pred_sql = pred_sql.lstrip(": \t")

        if ";" in pred_sql:
            pred_sql = pred_sql.split(";")[0].strip() + ";"
        else:
            pred_sql = pred_sql.split("\n")[0].strip()

        results.append(pred_sql)

    return results
# =============================================================================
# evaluate.py
# Inference and evaluation pipeline for text-to-SQL.

# Supports two modes:
#   - Base model evaluation (no LoRA adapter)
#   - LoRA-adapted model evaluation (loads saved adapter)

# Main entry point: run_evaluation()
# =============================================================================

import os
import json
import torch

from config import (
    MODEL_ID,
    MAX_NEW_TOKENS,
    SCHEMA_PATHS,
    TEST_PATH,
    get_predictions_path,
    get_metrics_path,
    RESULTS_DIR,
    LORA_R,
    SEED,
)
from prompt_builder import load_schema, build_prompt
from metrics import aggregate_metrics
from model import load_jsonl, load_base_model_and_tokenizer, load_lora_model_and_tokenizer

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

    # Take only the first line (the SQL should be single-line)
    # discard any text after the generated SQL
    pred_sql = pred_sql.split("\n")[0].strip()

    return pred_sql


# =============================================================================
# EVALUATION LOOP
# =============================================================================

def evaluate_model(model, tokenizer, test_data, schema, device):
    """
    Run inference on the full test set and compute metrics.

    For each test example:
      - Builds the prompt
      - Generates predicted SQL
      - Stores question, gold SQL, predicted SQL, and per-example metric results

    Args:
        model: model in eval mode
        tokenizer: LLaMA tokenizer
        test_data (list[dict]): test examples from split_test.jsonl
        schema (str): schema string for this experiment
        device (str): 'cuda' or 'cpu'

    Returns:
        predictions (list[dict]): per-example results
        metrics (dict): aggregate EM, EX, ESM scores
    """
    predictions = []
    n = len(test_data)

    for i, entry in enumerate(test_data):
        # Progress indicator
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Evaluating example {i+1}/{n}...")

        question = entry["question"]
        gold_sql = entry["sql"]

        # Build prompt and generate SQL
        prompt = build_prompt(question, schema)
        pred_sql = generate_sql(model, tokenizer, prompt, device)

        # Store per-example result
        predictions.append({
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
        })

    # Compute aggregate metrics over all predictions
    metrics = aggregate_metrics(predictions)

    return predictions, metrics


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_evaluation(schema_name, model_type, rank=LORA_R):
    """
    Full evaluation pipeline for one schema format and model type.

    model_type controls whether to evaluate the base model or a LoRA adapter:
      - 'base': evaluates the raw pretrained LLaMA 3.1 8B (no LoRA)
      - 'lora': evaluates the LoRA-adapted model for this schema and rank

    Args:
        schema_name (str): one of 'relational', 'create_table', 'json', 'nl'
        model_type (str): 'base' or 'lora'
        rank (int): LoRA rank (only used when model_type='lora')

    Returns:
        metrics (dict): aggregate EM, EX, ESM scores
    """

    # 1. Set random seed
    torch.manual_seed(SEED)

    print(f"\n{'='*60}")
    print(f"Evaluation | schema: {schema_name} | model: {model_type}" + (f" | rank: {rank}" if model_type == "lora" else ""))
    print(f"{'='*60}\n")

    # 2. Load test data
    test_data = load_jsonl(TEST_PATH)
    print(f"Loaded {len(test_data)} test examples")

    # 3. Load schema
    schema = load_schema(SCHEMA_PATHS[schema_name])
    print(f"Schema '{schema_name}' loaded ({len(schema)} characters)")

    # 4. Load model (base or LoRA) from model.py
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "base":
        print("\nLoading base model (no LoRA)...")
        model, tokenizer = load_base_model_and_tokenizer()
    elif model_type == "lora":
        print(f"\nLoading LoRA model (schema={schema_name}, rank={rank})...")
        model, tokenizer = load_lora_model_and_tokenizer(schema_name, rank)
    else:
        raise ValueError(f"model_type must be 'base' or 'lora', got '{model_type}'")

    # 5. Run inference on all test examples
    print(f"\nRunning inference on {len(test_data)} test examples...\n")
    predictions, metrics = evaluate_model(model, tokenizer, test_data, schema, device)

    # 6. Save per-example predictions to results/
    os.makedirs(RESULTS_DIR, exist_ok=True)
    predictions_path = get_predictions_path(schema_name, model_type)
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to {predictions_path}")

    # 7. Save aggregate metrics to results/
    metrics_path = get_metrics_path(schema_name, model_type)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # 8. Print metrics summary
    print(f"\n--- Results: {schema_name} | {model_type} ---")
    print(f"  Exact Match: {metrics['exact_match']:.4f} ({int(metrics['exact_match'] * metrics['n_total'])}/{metrics['n_total']})")
    print(f"  Execution Acc: {metrics['execution_acc']:.4f} ({int(metrics['execution_acc'] * metrics['n_total'])}/{metrics['n_total']})")
    print(f"  Exact Set Match: {metrics['exact_set_match']:.4f} ({int(metrics['exact_set_match'] * metrics['n_total'])}/{metrics['n_total']})")
    print(f"  Exec Errors: {metrics['n_exec_error']}/{metrics['n_total']}")

    return metrics

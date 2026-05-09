# =============================================================================
# train.py
# Full LoRA fine-tuning pipeline for text-to-SQL on ATIS dataset

# Adapted from train_sft_simple() in Milestone 2 notebook.

# main entry point: run_training()
# =============================================================================

import os
import json
import torch

from config import (
    MODEL_ID,
    LORA_R,
    EPOCHS,
    LR,
    EVAL_EVERY,
    SCHEMA_PATHS,
    TRAIN_PATH,
    DEV_PATH,
    get_adapter_path,
    RESULTS_DIR,
    SEED,
)
from prompt_builder import load_schema
from dataset import make_dataloaders
from model import load_jsonl, load_base_model_and_tokenizer, attach_lora


# =============================================================================
# LOSS FUNCTIONS
# (Adapted from calc_loss_batch and eval_loss_loader in Milestone 2 and class notebook)
# =============================================================================

def calc_loss_batch(model, input_ids, target_ids):
    """
    Compute cross-entropy loss for one batch.
    Loss is only computed on SQL tokens. Prompt and padding positions
    are masked to -100 in target_ids by collate_sft_batch.

    Args:
        model: LoRA-adapted model in training mode
        input_ids: LongTensor [B, T]
        target_ids: LongTensor [B, T] with -100 at masked positions

    Returns:
        loss: scalar tensor
    """
    outputs = model(input_ids=input_ids, labels=target_ids)
    return outputs.loss  # HuggingFace models compute cross-entropy loss internally


def eval_loss_loader(model, loader, max_batches=5):
    """
    Estimate loss over a dataloader by averaging over up to max_batches batches.
    Uses torch.no_grad() to skip gradient computation during evaluation (no weight updates).

    Args:
        model: model in eval mode
        loader: DataLoader (train or dev)
        max_batches: number of batches to average over

    Returns:
        average loss (float)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_ids, target_ids, _ in loader:
            if n_batches >= max_batches:
                break
            total_loss += calc_loss_batch(model, input_ids, target_ids).item()
            n_batches += 1

    model.train()  # put model back in training mode after eval

    return total_loss / n_batches if n_batches > 0 else float("nan")


# =============================================================================
# TRAINING LOOP
# (Adapted from train_sft_simple() in Milestone 2 and class notebook)
# =============================================================================

def train_model(model, train_loader, dev_loader, epochs=EPOCHS, lr=LR, eval_every=EVAL_EVERY):
    """
    SFT training loop with per-step loss logging.

    At every eval_every steps, computes train and dev loss and logs them.
    Returns a history dict for plotting loss curves later.

    Args:
        model: LoRA-adapted model
        train_loader: DataLoader for training split
        dev_loader: DataLoader for dev split
        epochs: number of full passes over the training data (default 3)
        lr: learning rate for AdamW (default 5e-5)
        eval_every: log train/dev loss every N steps (default 200)

    Returns:
        history (dict) with keys: step, train_loss, val_loss
    """

    # AdamW is the standard optimizer for transformer fine-tuning.
    # Only LoRA adapter parameters are passed here (base model weights are frozen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"step": [], "train_loss": [], "val_loss": []}
    step = 0

    for epoch in range(epochs):
        model.train()

        for input_ids, target_ids, _ in train_loader:
            step += 1

            # Core training step
            loss = calc_loss_batch(model, input_ids, target_ids)    # forward pass + loss
            optimizer.zero_grad()   # clear gradients from previous step
            loss.backward() # compute gradients via backpropagation
            optimizer.step()    # update LoRA adapter weights

            # Periodic evaluation
            if step % eval_every == 0:
                tr = eval_loss_loader(model, train_loader, max_batches=5)
                va = eval_loss_loader(model, dev_loader,   max_batches=5)

                history["step"].append(step)
                history["train_loss"].append(tr)
                history["val_loss"].append(va)

                print(f"Epoch {epoch+1} | Step {step:5d} | train loss {tr:.4f} | val loss {va:.4f}")

    return history


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_training(schema_name, rank=LORA_R):
    """
    Full training pipeline for one schema format and LoRA rank.

    Args:
        schema_name (str): one of 'relational', 'create_table', 'json', 'nl'
        rank (int): LoRA rank r

    Returns:
        history (dict): training loss history for plotting
    """

    # 1. Set random seed for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*60}")
    print(f"Training | schema: {schema_name} | rank: {rank}")
    print(f"{'='*60}\n")

    # 2. Load train and dev data from JSONL files
    train_data = load_jsonl(TRAIN_PATH)
    dev_data = load_jsonl(DEV_PATH)
    print(f"Loaded {len(train_data)} train examples, {len(dev_data)} dev examples")

    # Load schemas
    schema = load_schema(SCHEMA_PATHS[schema_name])
    print(f"Schema '{schema_name}' loaded ({len(schema)} characters)")

    # 3. Load base model and tokenizer (QLoRA)
    print("\nLoading base model and tokenizer...")
    model, tokenizer = load_base_model_and_tokenizer()

    # 4. Attach LoRA adapters
    print(f"\nAttaching LoRA adapters (rank={rank})...")
    model = attach_lora(model, rank=rank)

    # 5. Build DataLoaders for this schema format
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, dev_loader = make_dataloaders(
        train_data, dev_data, tokenizer, schema, device
    )
    print(f"\nTrain batches: {len(train_loader)} | Dev batches: {len(dev_loader)}")

    # 6. Run training loop
    print(f"\nStarting training ({EPOCHS} epochs, eval every {EVAL_EVERY} steps)...\n")
    history = train_model(model, train_loader, dev_loader)
    print("\nTraining complete.")

    # 7. Save adapter weights
    adapter_path = get_adapter_path(schema_name, rank)
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")

    # 8. Save training log (loss history)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f"train_log_{schema_name}_r{rank}.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training log saved to {log_path}")

    return history

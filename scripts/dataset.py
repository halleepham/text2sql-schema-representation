# =============================================================================
# dataset.py
# PyTorch Dataset class and collate function for supervised fine-tuning

# TextToSQLDataset and collate_sft_batch from my milestone 2 notebook (01_milestone2_smoke_test.ipynb)
# which was iteself adapted from InstructionDataset and collate_sft_batch from the
# finetuning_for_instruction.ipynb notebook from class

# Changes from Milestone 2:
# - removed hardcoded globals (PAD_ID, ALLOWED_MAX_LEN) -> now imported from config.py
# - schema string is passed as a parameter to collate_sft_batch so the same dataset class works for all schema formats
# - tokenizer is passed explicitly so the module has no global state
# =============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial

from config import TRAIN_BATCH_SIZE, MAX_SEQ_LEN
from prompt_builder import build_prompt

# =============================================================================
# DATASET CLASS
# =============================================================================

class TextToSQLDataset(Dataset):
    """
    Stores raw question-SQL pairs as a list of dictionaries.
    Tokenization is deferred to collate_sft_batch so that dynamic padding works correctly across batches

    Each item in self.data is a dict with keys:
        question: the natural language question
        sql: the gold SQL query
        split: 'train', 'dev', or 'test'
    """

    def __init__(self, data):
        # list of dicts loaded from JSONL
        self.data = data

    def __len__(self):
        # number of examples
        return len(self.data)
    
    def __getitem__(self, idx):
        # return raw dict (no tokenization here)
        return self.data[idx]

# =============================================================================
# COLLATE FUNCTION
# =============================================================================

def collate_sft_batch(
        batch,
        tokenizer,
        schema,
        device,
        ignore_index=100,
        allowed_max_length=MAX_SEQ_LEN,
):
    """
    Tokenizes a batch of question-SQL pairs and prepares tensors for SFT.

    Args:
        batch (list[dict]): raw dicts from TextToSQLDataset
        tokenizer: HuggingFace tokenizer for LLaMA
        schema (str): schema string for this experiment
        device (str): 'cuda' or 'cpu'
        ignore_index (int): target positions to ignore in loss (-100)
        allowed_max_length (int): max sequence length cap (default 1024)

    Returns:
        input_ids : LongTensor [B, T] - tokenized input sequences
        target_ids: LongTensor [B, T] - shifted targets with -100 at masked positions
        prompt_lens: LongTensor [B] - prompt token lengths (used during inference)
    """

    pad_token_id = tokenizer.pad_token_id

    texts = [] # full strings: prompt + sql
    prompt_lens = [] # token lengths of the prompt-only portion

    for entry in batch:
        prompt = build_prompt(entry["question"], schema) # assemble prompt string
        full_text = prompt + entry["sql"] # append gold SQL as target
        texts.append(full_text)

        # Tokenize only the prompt to get its token length
        # This length is used later to know where to start the loss mask
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_lens.append(len(prompt_ids))

    # Tokenize all full sequences (prompt + SQL in one pass)
    tokenized = [tokenizer.encode(t, add_special_tokens=False) for t in texts]

    # Dynamic padding:
    # pad each batch to the longest sequence in it
    # plus 1 for the EOS token, capped at allowed_max_length
    # this avoids wasting memory on short batches
    batch_max = min(
        max(len(ids) for ids in tokenized) + 1,
        allowed_max_length
    )

    input_ids_lst = []
    target_ids_lst = []

    for ids, p_len in zip(tokenized, prompt_lens):
        # Append EOS token then truncate to batch_max
        ids = (ids + [pad_token_id])[:batch_max]

        # Pad shorter sequences to batch_max
        pad_amount = batch_max - len(ids)
        ids = ids + [pad_token_id] * pad_amount

        # Shift to create next-token prediction pairs
        input_ids = ids[:-1]    # input = tokens[0 .. T-2]
        target_ids = ids[1:]    # target = tokens[1 .. T-1]

        # Mask any remaining padding positions in targets
        # padding is not real data so the model should not be penalized for failed to predict pad tokens
        target_ids = [
            ignore_index if t == pad_token_id else t
            for t in target_ids
        ]

        input_ids_lst.append(input_ids)
        target_ids_lst.append(target_ids)

    # Convert lists to tensors and move to GPU
    input_ids_tensor = torch.tensor(input_ids_lst, dtype=torch.long).to(device)
    target_ids_tensor = torch.tensor(target_ids_lst, dtype=torch.long).to(device)
    prompt_lens_tensor = torch.tensor(prompt_lens, dtype=torch.long).to(device)

    return input_ids_tensor, target_ids_tensor, prompt_lens_tensor


# =============================================================================
# DATALOADER BUILDER
# =============================================================================

def make_dataloaders(train_data, dev_data, tokenizer, schema, device):
    """
    Wraps train and dev datasets in PyTorch DataLoaders.
 
    Uses functools.partial to prefill tokenizer, schema, and device into
    collate_sft_batch so DataLoader can call it with just the batch argument.
 
    Args:
        train_data (list[dict]): training examples
        dev_data (list[dict]): dev examples
        tokenizer: HuggingFace tokenizer
        schema (str): schema string for this experiment
        device (str): 'cuda' or 'cpu'
 
    Returns:
        train_loader, dev_loader
    """
 
    # Bind tokenizer, schema, and device into the collate function
    collate_fn = partial(
        collate_sft_batch,
        tokenizer=tokenizer,
        schema=schema,
        device=device,
    )
 
    train_loader = DataLoader(
        TextToSQLDataset(train_data),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,   # shuffle training data each epoch
        collate_fn=collate_fn,
    )
 
    dev_loader = DataLoader(
        TextToSQLDataset(dev_data),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,  # no shuffle for evaluation
        collate_fn=collate_fn,
    )
 
    return train_loader, dev_loader

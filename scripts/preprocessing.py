# =============================================================================
# preprocessing.py
# Preprocessing pipeline for the ATIS text-to-SQL dataset.

# Extracted from 00_preprocessing.ipynb

# Changes from original notebook:
#   - Filters out entries tagged with comments in the original dataset.
#     Any entry with a non-empty "comments" field is excluded, as comments
#     indicate the SQL is known to be incorrect or problematic (e.g. "Wrong SQL", "Incorrect SQL")
#   - Structured as a standalone script for reproducibility

# Output:
#   data/split_train.jsonl
#   data/split_dev.jsonl
#   data/split_test.jsonl
# =============================================================================

import json
import os
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

ATIS_URL = "https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/refs/heads/master/data/atis.json"
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "atis.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# =============================================================================
# VARIABLE SUBSTITUTION
# =============================================================================

def substitute_variables(text, variables):
    """
    Substitute placeholder variables in a text string with their real values.

    Args:
        text (str): question or SQL string with placeholders
        variables (dict): mapping from placeholder name to real value

    Returns:
        str: text with all placeholders replaced
    """
    for placeholder, value in variables.items():
        # Replace quoted placeholder first (SQL context)
        text = text.replace(f'"{placeholder}"', f'"{value}"')
        # Then unquoted (question text context)
        text = text.replace(placeholder, value)
    return text


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

def run_preprocessing():
    """
    Full preprocessing pipeline for the ATIS dataset.

    Steps:
      1. Load raw ATIS JSON
      2. Filter out entries with non-empty comments (known bad SQL)
      3. Flatten sentences — one row per (question phrasing, SQL) pair
      4. Substitute variables into questions and SQL
      5. Normalize SQL — strip whitespace
      6. Deduplicate on (question, SQL) pair
      7. Split by question-split label (train/dev/test)
      8. Write splits to JSONL files
    """
    # --- Download raw data if not present ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Downloading atis.json from {ATIS_URL}...")
        import urllib.request
        urllib.request.urlretrieve(ATIS_URL, RAW_DATA_PATH)
        print("Download complete.")
    else:
        print("atis.json already exists, skipping download.")

    # --- Load raw data ---
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    with open(RAW_DATA_PATH, "r") as f:
        raw_data = json.load(f)
    print(f"Total entries in raw data: {len(raw_data)}")

    # --- Filter out entries with comments ---
    # Any entry with a non-empty comments field is flagged as problematic
    # by the dataset authors (e.g. "Wrong SQL", "Incorrect SQL")
    # These are excluded entirely to avoid training or evaluating on bad SQL
    n_before_filter = len(raw_data)
    raw_data = [entry for entry in raw_data if not entry.get("comments", [])]
    n_filtered = n_before_filter - len(raw_data)
    print(f"Filtered out {n_filtered} entries with comments (known bad SQL)")
    print(f"Remaining entries: {len(raw_data)}")

    # --- Flatten and substitute variables ---
    # Each entry has multiple sentence phrasings
    # Create one row per phrasing, substituting variables into both the question text and the SQL string
    rows = []
    for entry in raw_data:
        sql_template = entry["sql"][0]  # take first SQL variant only

        for sentence in entry["sentences"]:
            variables = sentence["variables"]
            question = substitute_variables(sentence["text"], variables)
            sql = substitute_variables(sql_template, variables)
            split = sentence["question-split"]
            rows.append({"question": question, "sql": sql, "split": split})

    print(f"\nTotal rows after flattening: {len(rows)}")

    # --- Verify no internal semicolons ---
    multi_statement = [
        r for r in rows
        if r["sql"].strip().rstrip(";").strip().count(";") > 0
    ]
    print(f"Queries with internal semicolons: {len(multi_statement)}")
    if multi_statement:
        print("WARNING: Found queries with internal semicolons — review these manually")

    # --- Normalize and deduplicate ---
    seen = set()
    cleaned = []

    for row in rows:
        # Strip trailing whitespace and semicolon from SQL
        sql = row["sql"].strip().replace(" ;", ";")
        question = row["question"].strip()

        # Deduplicate on (question, SQL) pair
        key = (question, sql)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"question": question, "sql": sql, "split": row["split"]})

    print(f"\nRows before deduplication : {len(rows)}")
    print(f"Rows after deduplication  : {len(cleaned)}")
    print(f"Duplicates removed        : {len(rows) - len(cleaned)}")

    # --- Split by question-split label ---
    split_counts = Counter([row["split"] for row in cleaned])
    print(f"\nSplit counts: {dict(split_counts)}")

    splits = {"train": [], "dev": [], "test": []}
    for row in cleaned:
        splits[row["split"]].append(row)

    # --- Write JSONL files ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, split_rows in splits.items():
        out_path = os.path.join(OUTPUT_DIR, f"split_{split_name}.jsonl")
        with open(out_path, "w") as f:
            for row in split_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {len(split_rows)} rows to split_{split_name}.jsonl")

    total = len(cleaned)
    print(f"\nTrain: {len(splits['train'])} ({len(splits['train'])/total*100:.2f}%)")
    print(f"Dev: {len(splits['dev'])} ({len(splits['dev'])/total*100:.2f}%)")
    print(f"Test: {len(splits['test'])} ({len(splits['test'])/total*100:.2f}%)")
    print(f"Total: {total}")
    print("\nPreprocessing complete.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_preprocessing()

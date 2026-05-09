# =============================================================================
# metrics.py
# Evaluation metrics for outputs

# Three metrics are implemented:
#   1. Exact Match (EM): predicted SQL == gold SQL (string comparison)
#   2. Execution Accuracy (EX): predicted SQL produces same result set as gold SQL
#   3. Exact Set Match (ESM): clause-level set comparison, order-insensitive
#
# Each metric function takes a predicted SQL string and a gold SQL string
# and returns True (correct) or False (incorrect).
#
# aggregate_metrics() runs all three over a list of predictions and returns
# accuracy scores (0.0 to 1.0) for each metric.
# =============================================================================

import re
import sqlite3
from config import DB_PATH


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_sql(sql):
    """
    Normalize a SQL string for comparison.

    This function handles minor formatting differences that don't affect SQL semantics,
    such as extra spaces or inconsistent casing.

    Args:
        sql (str): raw SQL string

    Returns:
        normalized SQL string
    """
    sql = sql.strip()
    sql = re.sub(r"\s+", " ", sql)  # collapse all whitespace runs to a single space
    sql = sql.lower()
    return sql


# =============================================================================
# METRIC 1: EXACT MATCH
# =============================================================================

def exact_match(pred_sql, gold_sql):
    """
    Check whether predicted SQL exactly matches gold SQL after normalization.

    Args:
        pred_sql (str): model-generated SQL
        gold_sql (str): gold SQL from the dataset

    Returns:
        bool: True if they match after normalization
    """
    return normalize_sql(pred_sql) == normalize_sql(gold_sql)


# =============================================================================
# METRIC 2: EXECUTION ACCURACY
# =============================================================================

def execute_sql(sql, db_path=DB_PATH):
    """
    Execute a SQL query against the ATIS SQLite database.

    Args:
        sql (str): SQL query to execute
        db_path (str): path to the SQLite database file

    Returns:
        results (list of tuples) if execution succeeds
        None if execution fails (syntax error, invalid table/column, etc.)
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception:
        # Any SQL error (syntax error, invalid table, etc.) counts as a failure
        return None


def execution_accuracy(pred_sql, gold_sql):
    """
    Check whether predicted SQL produces the same result set as gold SQL
    when executed against the ATIS database.

    Args:
        pred_sql (str): model-generated SQL
        gold_sql (str): gold SQL from the dataset

    Returns:
        bool: True if result sets match, False otherwise
    """
    pred_results = execute_sql(pred_sql)
    gold_results = execute_sql(gold_sql)

    # If either query fails to execute, mark as incorrect
    if pred_results is None or gold_results is None:
        return False

    # Sort both result sets before comparing to handle row order differences.
    # SQL does not guarantee row ordering unless ORDER BY is specified.
    return sorted(pred_results) == sorted(gold_results)


# =============================================================================
# METRIC 3: EXACT SET MATCH
# =============================================================================

def parse_sql_clauses(sql):
    """
    Parse a SQL string into its major clauses for set-based comparison.

    Extracts: SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY
    Each clause is normalized and stored as a frozenset of tokens so that
    comparison is insensitive to token ordering within a clause.

    Args:
        sql (str): SQL string to parse

    Returns:
        dict mapping clause name (str) to frozenset of normalized tokens
    """
    sql = normalize_sql(sql)

    # Clause boundary keywords (order matters for splitting)
    clause_keywords = ["select", "from", "where", "group by", "having", "order by"]

    # Build a regex that splits on clause keywords.
    # \\b ensures we match whole words only (not substrings).
    # The alternation is ordered longest-first to avoid partial matches (e.g. "group by" before "group")
    pattern = r"\b(" + "|".join(re.escape(k) for k in clause_keywords) + r")\b"

    # Split the SQL into (keyword, content) pairs
    parts = re.split(pattern, sql)
    clauses = {}

    # parts alternates: [pre_keyword_text, keyword, content, keyword, content, ...]
    i = 1  # start at index 1 to skip any text before the first keyword
    while i < len(parts) - 1:
        keyword = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # Tokenize the clause content into a frozenset for order-insensitive comparison
        clauses[keyword] = frozenset(content.split())
        i += 2

    return clauses


def exact_set_match(pred_sql, gold_sql):
    """
    Check whether predicted SQL matches gold SQL at the clause level,
    insensitive to token ordering within each clause.

    Args:
        pred_sql (str): model-generated SQL
        gold_sql (str): gold SQL from the dataset

    Returns:
        bool: True if all clauses match as sets
    """
    pred_clauses = parse_sql_clauses(pred_sql)
    gold_clauses = parse_sql_clauses(gold_sql)
    return pred_clauses == gold_clauses


# =============================================================================
# AGGREGATE METRICS
# =============================================================================

def aggregate_metrics(predictions):
    """
    Compute EM, EX, and ESM accuracy over a list of prediction dicts.

    Args:
        predictions (list[dict]): each dict must have keys:
            pred_sql (str): model-generated SQL
            gold_sql (str): gold SQL from the dataset

    Returns:
        dict with keys:
            exact_match (float): fraction of predictions that are exact matches
            execution_acc (float): fraction of predictions with correct execution results
            exact_set_match (float): fraction of predictions that match at clause level
            n_total (int): total number of predictions evaluated
            n_exec_error (int): number of predictions that failed to execute
    """
    n_total = len(predictions)
    n_em = 0
    n_ex = 0
    n_esm = 0
    n_exec_error = 0

    for p in predictions:
        pred_sql = p["pred_sql"]
        gold_sql = p["gold_sql"]

        if exact_match(pred_sql, gold_sql):
            n_em += 1

        if execution_accuracy(pred_sql, gold_sql):
            n_ex += 1
        elif execute_sql(pred_sql) is None:
            # Track how many predictions failed to execute at all
            n_exec_error += 1

        if exact_set_match(pred_sql, gold_sql):
            n_esm += 1

    return {
        "exact_match"     : round(n_em  / n_total, 4) if n_total > 0 else 0.0,
        "execution_acc"   : round(n_ex  / n_total, 4) if n_total > 0 else 0.0,
        "exact_set_match" : round(n_esm / n_total, 4) if n_total > 0 else 0.0,
        "n_total"         : n_total,
        "n_exec_error"    : n_exec_error,
    }

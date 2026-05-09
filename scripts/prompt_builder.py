# =============================================================================
# prompt_builder.py

# Contains utility functions for loading schema files and assembling model prompts
# the prompt format is fixed across all experiments
# only the schema string changes between runs
# This ensures the schema format is the only variable

def load_schema(schema_path):
    """Load schema string from a .txt file."""
    with open(schema_path, "r") as f:
        return f.read().strip()

def build_prompt(question, schema):
    """
    Assemble a prompt from a question and schema string.

    Format:
        Translate the following question into SQL.

        Schema:
        {schema}

        Question: {question}

        SQL:

        The model is trained to predict everyting after 'SQL:'.
        During inference, generation stops at a newlline since all
        ATIS SQL queries have been confirmed to be single-line.
    """
    return(
        f"Translate the following question into SQL.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        f"SQL:"
    )

def load_schema(schema_path):
    """Load schema string from a .txt file."""
    with open(schema_path, "r") as f:
        return f.read().strip()

def build_prompt(question, schema):
    """Assemble a prompt from a question and schema string."""
    return(
        f"Translate the following question into SQL.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        f"SQL:"
    )
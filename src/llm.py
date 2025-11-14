from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def query_model(prompt, model):
    """
    Queries the specified AI model with the given prompt.
    """
    if Path(model).is_dir():
        # Load the fine-tuned model from the local directory
        tokenizer = AutoTokenizer.from_pretrained(model)
        model_obj = AutoModelForCausalLM.from_pretrained(model)

        # For now, we'll just return a placeholder response
        return f"Response from fine-tuned model at '{model}' for prompt: '{prompt}'"
    else:
        # For now, we'll just return a placeholder response
        return f"Response from {model} for prompt: '{prompt}'"

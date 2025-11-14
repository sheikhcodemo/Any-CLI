#!/usr/bin/env python3
import typer
import json
from pathlib import Path
from src.router import route_prompt
from src.llm import query_model

app = typer.Typer()
STATE_FILE = Path(".cli_state.json")

def read_state():
    if not STATE_FILE.exists():
        return {"model": None}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def write_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

@app.command()
def model(model_name: str = typer.Argument(..., help="The name of the model to use")):
    """
    Select or change the AI model.
    """
    state = read_state()
    state["model"] = model_name
    write_state(state)
    typer.echo(f"INFO: Model set to: {model_name}")

@app.command()
def run(prompt: str = typer.Argument(..., help="The user's prompt")):
    """
    Any-CLI: A tool that uses AI to help with software engineering tasks.
    """
    state = read_state()
    try:
        # 1. Route the prompt to the best model
        if state.get("model"):
            model = state["model"]
            typer.echo(f"INFO: Using selected model: {model}")
        else:
            model = route_prompt(prompt)
            typer.echo(f"INFO: Routing prompt to model: {model}")

        # 2. Query the selected model
        response = query_model(prompt, model)
        typer.echo(f"SUCCESS: Response: {response}")

    except Exception as e:
        typer.echo(f"ERROR: An unexpected error occurred: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()

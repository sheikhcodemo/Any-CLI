#!/usr/bin/env python3
import argparse
import sys
from src.router import route_prompt
from src.llm import query_model

def main():
    """
    Main function for the Any-CLI application.
    """
    parser = argparse.ArgumentParser(
        description="Any-CLI: A tool that uses AI to help with software engineering tasks."
    )
    parser.add_argument("prompt", type=str, help="The user's prompt")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    try:
        # 1. Route the prompt to the best model
        model = route_prompt(args.prompt)
        print(f"INFO: Routing prompt to model: {model}")

        # 2. Query the selected model
        response = query_model(args.prompt, model)
        print(f"SUCCESS: Response: {response}")

    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

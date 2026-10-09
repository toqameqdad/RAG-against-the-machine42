"""Entry point for the RAG command-line interface."""

import fire

from src.cli import (
    answer_command,
    answer_dataset_command,
    evaluate,
    index,
    search,
    search_dataset_command,
)


def main() -> int:
    """Run the command-line interface."""
    try:
        fire.Fire(
            {
                "index": index,
                "search": search,
                "search_dataset": search_dataset_command,
                "answer": answer_command,
                "answer_dataset": answer_dataset_command,
                "evaluate": evaluate,
            }
        )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130
    except Exception as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Utility helpers for validation and JSON loading."""

import json
from pathlib import Path
from typing import Any


def require_file(path_string: str) -> Path:
    """Return an existing file path or raise a clear error."""
    path = Path(path_string)

    if not path.exists():
        raise ValueError(f"File not found: {path_string}")

    if not path.is_file():
        raise ValueError(f"Not a file: {path_string}")

    return path


def require_directory(path_string: str) -> Path:
    """Return an existing directory path or raise a clear error."""
    path = Path(path_string)

    if not path.exists():
        raise ValueError(f"Directory not found: {path_string}")

    if not path.is_dir():
        raise ValueError(f"Not a directory: {path_string}")

    return path


def require_positive_k(k: int) -> None:
    """Validate that k is greater than zero."""
    if k <= 0:
        raise ValueError("k must be greater than 0")


def require_non_empty_text(value: str, name: str) -> None:
    """Validate that a text value is not empty."""
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def load_json_file(path_string: str) -> Any:
    """Load JSON from a file with clear validation errors."""
    path = require_file(path_string)

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON file: {path_string}"
        ) from error
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Invalid UTF-8 file: {path_string}"
        ) from error

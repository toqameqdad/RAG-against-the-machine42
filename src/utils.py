"""Utility helpers for validation and file loading."""

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

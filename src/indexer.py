"""Build searchable chunks from the raw vLLM codebase."""

from pathlib import Path
import json

from src.chunkers import chunk_python, chunk_text
from src.models import Chunk


def chunk_file(
    file_path: Path,
    project_root: Path,
    max_chunk_size: int,
) -> list[Chunk]:
    """Read one file and split it using the appropriate chunker."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    relative_path = file_path.resolve().relative_to(
        project_root.resolve()
    ).as_posix()

    if file_path.suffix == ".py":
        return chunk_python(
            text,
            relative_path,
            max_chunk_size,
        )

    if file_path.suffix in {".md", ".txt"}:
        return chunk_text(
            text,
            relative_path,
            max_chunk_size,
        )

    return []


def build_chunks(
    raw_directory: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Read supported files and build searchable chunks."""
    project_root = Path.cwd().resolve()
    raw_path = Path(raw_directory).resolve()

    chunks: list[Chunk] = []

    for file_path in raw_path.rglob("*"):
        if not file_path.is_file():
            continue

        chunks.extend(
            chunk_file(
                file_path,
                project_root,
                max_chunk_size,
            )
        )

    return chunks


def save_chunks(
    chunks: list[Chunk],
    output_path: str,
) -> None:
    """Save chunks as JSON under the processed data directory."""
    path = Path(output_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    data = [chunk.model_dump() for chunk in chunks]

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

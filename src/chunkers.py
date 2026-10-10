"""Functions for splitting source files into searchable chunks."""

from src.models import Chunk
import ast


def split_large_section(
    text: str,
    file_path: str,
    start_index: int,
    max_chunk_size: int,
) -> list[Chunk]:
    """Split a section while preserving line boundaries when possible."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than 0")

    chunks: list[Chunk] = []

    lines = text.splitlines(keepends=True)

    current_chunk = ""
    current_start = start_index

    for line in lines:
        if len(current_chunk) + len(line) <= max_chunk_size:
            current_chunk += line

        else:
            if current_chunk:
                chunks.append(
                    Chunk(
                        text=current_chunk,
                        file_path=file_path,
                        first_character_index=current_start,
                        last_character_index=(
                            current_start + len(current_chunk)
                        ),
                    )
                )

                current_start += len(current_chunk)
                current_chunk = ""

            if len(line) <= max_chunk_size:
                current_chunk = line

            else:
                for offset in range(0, len(line), max_chunk_size):
                    piece = line[offset:offset + max_chunk_size]

                    chunks.append(
                        Chunk(
                            text=piece,
                            file_path=file_path,
                            first_character_index=current_start + offset,
                            last_character_index=(
                                current_start
                                + offset
                                + len(piece)
                            ),
                        )
                    )

                current_start += len(line)

    if current_chunk:
        chunks.append(
            Chunk(
                text=current_chunk,
                file_path=file_path,
                first_character_index=current_start,
                last_character_index=(
                    current_start + len(current_chunk)
                ),
            )
        )

    return chunks


def chunk_text(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Split Markdown or plain text into searchable chunks."""
    chunks: list[Chunk] = []

    is_markdown = file_path.lower().endswith(".md")

    if not is_markdown:
        return split_large_section(
            text,
            file_path,
            0,
            max_chunk_size,
        )

    lines = text.splitlines(keepends=True)

    current_section = ""
    section_start = 0
    current_index = 0

    for line in lines:
        is_heading = (
            line.startswith("# ")
            or line.startswith("## ")
            or line.startswith("### ")
            or line.startswith("#### ")
            or line.startswith("##### ")
            or line.startswith("###### ")
        )

        if is_heading and current_section:
            chunks.extend(
                split_large_section(
                    current_section,
                    file_path,
                    section_start,
                    max_chunk_size,
                )
            )

            current_section = line
            section_start = current_index
        else:
            current_section += line

        current_index += len(line)

    if current_section:
        chunks.extend(
            split_large_section(
                current_section,
                file_path,
                section_start,
                max_chunk_size,
            )
        )

    return chunks


def get_line_start_indexes(text: str) -> list[int]:
    """Return the character index where each line starts."""
    indexes = [0]

    for index, character in enumerate(text):
        if character == "\n":
            indexes.append(index + 1)

    return indexes


def chunk_python(
    text: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Split Python source code into searchable chunks."""
    chunks: list[Chunk] = []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return split_large_section(
            text,
            file_path,
            0,
            max_chunk_size,
        )

    line_starts = get_line_start_indexes(text)

    small_block = ""
    small_block_start = 0

    for node in tree.body:
        if not hasattr(node, "lineno"):
            continue

        start_index = line_starts[node.lineno - 1]

        if (
            hasattr(node, "end_lineno")
            and node.end_lineno is not None
            and node.end_lineno < len(line_starts)
        ):
            end_index = line_starts[node.end_lineno]
        else:
            end_index = len(text)

        block_text = text[start_index:end_index]

        is_large_structure = isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        )

        if is_large_structure:
            if small_block:
                chunks.extend(
                    split_large_section(
                        small_block,
                        file_path,
                        small_block_start,
                        max_chunk_size,
                    )
                )
                small_block = ""

            chunks.extend(
                split_large_section(
                    block_text,
                    file_path,
                    start_index,
                    max_chunk_size,
                )
            )

        else:
            if not small_block:
                small_block_start = start_index

            if len(small_block) + len(block_text) <= max_chunk_size:
                small_block += block_text
            else:
                chunks.extend(
                    split_large_section(
                        small_block,
                        file_path,
                        small_block_start,
                        max_chunk_size,
                    )
                )

                small_block = block_text
                small_block_start = start_index

    if small_block:
        chunks.extend(
            split_large_section(
                small_block,
                file_path,
                small_block_start,
                max_chunk_size,
            )
        )

    return chunks

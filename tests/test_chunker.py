from src.chunkers import chunk_text


text = """# Big Section
ABCDEFGHIJ
"""

chunks = chunk_text(
    text=text,
    file_path="example.md",
    max_chunk_size=15,
)

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}")
    print(f"Start: {chunk.first_character_index}")
    print(f"End: {chunk.last_character_index}")
    print(f"Text: {repr(chunk.text)}")
    print("-" * 30)

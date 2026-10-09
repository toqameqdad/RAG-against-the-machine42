from src.chunkers import chunk_python


text = """import os

VERSION = "1.0"

def add(a, b):
    return a + b


class User:
    def login(self):
        return True
"""

chunks = chunk_python(
    text=text,
    file_path="example.py",
    max_chunk_size=2000,
)

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}")
    print(f"Start: {chunk.first_character_index}")
    print(f"End: {chunk.last_character_index}")
    print("Text:")
    print(repr(chunk.text))
    print("-" * 30)
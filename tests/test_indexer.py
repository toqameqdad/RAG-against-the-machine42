from src.indexer import build_chunks, save_chunks


chunks = build_chunks(
    raw_directory="data/raw/vllm-0.10.1",
    max_chunk_size=2000,
)

save_chunks(
    chunks,
    "data/processed/chunks.json",
)

print(f"Saved {len(chunks)} chunks.")
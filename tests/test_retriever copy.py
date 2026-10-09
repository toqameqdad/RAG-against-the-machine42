from src.retriever import search


results = search(
    query="add function",
    chunks_path="data/processed/chunks.json",
    k=3,
)

for result in results:
    print(result)

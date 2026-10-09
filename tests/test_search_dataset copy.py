from src.retriever import (
    save_search_results,
    search_dataset,
)


results = search_dataset(
    dataset_path=(
        "data/datasets/UnansweredQuestions/"
        "dataset_code_public.json"
    ),
    chunks_path="data/processed/chunks.json",
    k=10,
)

save_search_results(
    results,
    (
        "data/output/search_results/"
        "UnansweredQuestions/"
        "dataset_code_public.json"
    ),
)

print(f"Processed {len(results.search_results)} questions.")
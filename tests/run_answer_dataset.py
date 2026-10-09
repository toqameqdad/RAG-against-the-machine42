from src.generator import answer_dataset, save_answers


results = answer_dataset(
    "data/output/search_results/"
    "UnansweredQuestions/dataset_docs_public.json"
)

save_answers(
    results,
    "data/output/answers/"
    "dataset_docs_public.json",
)

print(f"Saved {len(results.search_results)} answers.")

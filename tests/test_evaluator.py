from src.evaluator import evaluate_recall


recall = evaluate_recall(
    student_results_path=(
        "data/output/search_results/"
        "UnansweredQuestions/"
        "dataset_code_public.json"
    ),
    answered_dataset_path=(
        "data/datasets/AnsweredQuestions/"
        "dataset_code_public.json"
    ),
    k=5,
)

print(f"Recall@5: {recall:.3f}")
print(f"Recall@5: {recall * 100:.1f}%")
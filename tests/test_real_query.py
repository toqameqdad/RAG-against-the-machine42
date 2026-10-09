import json

from src.retriever import search


dataset_path = (
    "data/datasets/UnansweredQuestions/"
    "dataset_docs_public.json"
)

with open(dataset_path, encoding="utf-8") as file:
    data = json.load(file)

first_question = data["rag_questions"][0]

question_id = first_question["question_id"]
question = first_question["question"]

print(f"Question ID: {question_id}")
print(f"Question: {question}")
print()

results = search(
    query=question,
    chunks_path="data/processed/chunks.json",
    k=5,
)

print("Top 5 sources:")

for i, result in enumerate(results, start=1):
    print(f"{i}. {result.file_path}")
    print(
        f"   {result.first_character_index}"
        f" -> {result.last_character_index}"
    )


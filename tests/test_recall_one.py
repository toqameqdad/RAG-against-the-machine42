import json

from src.retriever import search


unanswered_path = (
    "data/datasets/UnansweredQuestions/"
    "dataset_docs_public.json"
)

answered_path = (
    "data/datasets/AnsweredQuestions/"
    "dataset_docs_public.json"
)


with open(unanswered_path, encoding="utf-8") as file:
    unanswered_data = json.load(file)

with open(answered_path, encoding="utf-8") as file:
    answered_data = json.load(file)


question = unanswered_data["rag_questions"][0]
question_id = question["question_id"]
question_text = question["question"]


expected_question = next(
    item
    for item in answered_data["rag_questions"]
    if item["question_id"] == question_id
)


results = search(
    query=question_text,
    chunks_path="data/processed/chunks.json",
    k=5,
)


print("QUESTION")
print(question_text)

print("\nEXPECTED SOURCES")

for source in expected_question["sources"]:
    print(
        source["file_path"],
        source["first_character_index"],
        source["last_character_index"],
    )


print("\nRETRIEVED SOURCES")

for source in results:
    print(
        source.file_path,
        source.first_character_index,
        source.last_character_index,
    )

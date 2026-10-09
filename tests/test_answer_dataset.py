import json
from pathlib import Path

from src.generator import AnswerGenerator
from src.models import StudentSearchResults


path = Path(
    "data/output/search_results/"
    "UnansweredQuestions/dataset_docs_public.json"
)

with path.open("r", encoding="utf-8") as file:
    data = json.load(file)

results = StudentSearchResults(**data)

generator = AnswerGenerator()

for result in results.search_results[:3]:
    answer = generator.generate(
        question=result.question,
        sources=result.retrieved_sources,
    )

    print("\nQUESTION:")
    print(result.question)

    print("\nANSWER:")
    print(answer)

    print("=" * 60)

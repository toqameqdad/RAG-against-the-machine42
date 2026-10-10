from src.generator import AnswerGenerator
from src.retriever import BM25Retriever


question = (
    "What HTTP endpoint is used to dynamically "
    "load a LoRA adapter in vLLM?"
)

retriever = BM25Retriever(
    "data/processed/chunks.json"
)

sources = retriever.search(
    query=question,
    k=3,
)

generator = AnswerGenerator()

answer = generator.generate(
    question=question,
    sources=sources,
)

print("QUESTION:")
print(question)

print("\nANSWER:")
print(answer)

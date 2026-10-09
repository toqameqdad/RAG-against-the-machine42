from src.generator import build_context, build_prompt
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

context = build_context(sources)

prompt = build_prompt(
    question=question,
    context=context,
)

print(prompt)

from src.generator import answer


result = answer(
    question=(
        "What HTTP endpoint is used to dynamically "
        "load a LoRA adapter in vLLM?"
    ),
    chunks_path="data/processed/chunks.json",
    k=3,
)

print(result.model_dump_json(indent=2))

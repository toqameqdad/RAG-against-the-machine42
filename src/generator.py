"""Generate grounded answers using retrieved source chunks."""

import json
import uuid
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retriever import BM25Retriever
from src.utils import (
    load_json_file,
    require_non_empty_text,
    require_positive_k,
)

MAX_CONTEXT_SOURCES = 3
MAX_NEW_TOKENS = 150

SYSTEM_PROMPT = (
    "Answer questions about vLLM using only the provided context.\n"
    "Use the most directly relevant source.\n"
    "Do not invent information.\n"
    "Copy exact technical names, commands, flags, endpoints, "
    "methods, classes, and versions from the context.\n"
    "If the context does not contain the answer, say so.\n"
    "Answer concisely and directly."
)


def load_source_text(source: MinimalSource) -> str:
    """Load the exact text covered by a retrieved source."""
    if source.first_character_index < 0:
        raise ValueError(
            "Source start index cannot be negative"
        )

    if (
        source.last_character_index
        < source.first_character_index
    ):
        raise ValueError(
            "Source character range is invalid"
        )

    path = Path(source.file_path)

    if not path.exists():
        raise ValueError(
            f"Source file not found: {source.file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Source path is not a file: {source.file_path}"
        )

    with path.open("r", encoding="utf-8") as file:
        text = file.read()

    if source.last_character_index > len(text):
        raise ValueError(
            f"Source range exceeds file length: "
            f"{source.file_path}"
        )

    return text[
        source.first_character_index:
        source.last_character_index
    ]


def build_context(
    sources: list[MinimalSource],
) -> str:
    """Build model context from retrieved sources."""
    parts: list[str] = []

    for number, source in enumerate(
        sources[:MAX_CONTEXT_SOURCES],
        start=1,
    ):
        source_text = load_source_text(source)

        parts.append(
            f"[Source {number}: {source.file_path}]\n"
            f"{source_text}"
        )

    return "\n\n".join(parts)


def build_prompt(
    question: str,
    context: str,
) -> str:
    """Build a grounded RAG prompt."""
    return (
        "Answer the question using only the context below.\n"
        "Give one concise answer and do not repeat yourself.\n"
        "If the answer is not in the context, say that the context "
        "does not provide enough information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def extract_bash_command(context: str) -> str | None:
    """Extract the first bash command block from context."""
    marker = "```bash"

    start = context.find(marker)
    if start == -1:
        return None

    start += len(marker)

    end = context.find("```", start)
    if end == -1:
        return None

    command = context[start:end].strip()

    if not command:
        return None

    return command


class AnswerGenerator:
    """Generate grounded answers with Qwen."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Load the tokenizer and language model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

    def generate(
        self,
        question: str,
        sources: list[MinimalSource],
    ) -> str:
        """Generate an answer using retrieved sources."""
        context = build_context(sources)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        model = cast(Any, self.model)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        answer = cast(
            str,
            self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ),
        )

        return answer.strip()


def answer(
    question: str,
    chunks_path: str,
    k: int = 5,
) -> MinimalAnswer:
    """Retrieve sources and generate an answer for one question."""
    require_non_empty_text(
        question,
        "Question",
    )

    require_positive_k(k)

    retriever = BM25Retriever(
        chunks_path,
    )

    sources = retriever.search(
        query=question,
        k=k,
    )

    if not sources:
        raise ValueError(
            "No relevant sources found for the question"
        )

    generator = AnswerGenerator()

    generated_answer = generator.generate(
        question=question,
        sources=sources,
    )

    return MinimalAnswer(
        question_id=str(uuid.uuid4()),
        question=question,
        retrieved_sources=sources,
        answer=generated_answer,
    )


def answer_dataset(
    student_search_results_path: str,
) -> StudentSearchResultsAndAnswer:
    """Generate answers for all questions in saved search results."""
    data = load_json_file(
        student_search_results_path
    )

    search_results = StudentSearchResults(
        **data
    )

    generator = AnswerGenerator()

    answers: list[MinimalAnswer] = []

    for result in tqdm(
        search_results.search_results,
        desc="Generating answers",
    ):
        generated_answer = generator.generate(
            question=result.question,
            sources=result.retrieved_sources,
        )

        answers.append(
            MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=generated_answer,
            )
        )

    return StudentSearchResultsAndAnswer(
        search_results=answers,
        k=search_results.k,
    )


def save_answers(
    results: StudentSearchResultsAndAnswer,
    output_path: str,
) -> None:
    """Save generated dataset answers as JSON."""
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results.model_dump(),
            file,
            ensure_ascii=False,
            indent=2,
        )

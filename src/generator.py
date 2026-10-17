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

MAX_NEW_TOKENS = 150
TOKEN_SAFETY_MARGIN = 128

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
            "Source range exceeds file length: "
            f"{source.file_path}"
        )

    return text[
        source.first_character_index:
        source.last_character_index
    ]



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
        """Generate an answer within the model token budget."""
        require_non_empty_text(question, "Question")

        model_limit = self.model.config.max_position_embeddings

        max_input_tokens = (
            model_limit
            - MAX_NEW_TOKENS
            - TOKEN_SAFETY_MARGIN
        )

        if max_input_tokens <= 0:
            raise ValueError(
                "Model context window is too small"
            )

        def make_prompt(context_text: str) -> str:
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_text}\n\n"
                        f"Question: {question}"
                    ),
                },
            ]

            return cast(
                str,
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
            )

        def tokenize_prompt(prompt_text: str) -> Any:
            return self.tokenizer(
                prompt_text,
                return_tensors="pt",
            )

        empty_inputs = tokenize_prompt(make_prompt(""))

        if (
            empty_inputs["input_ids"].shape[1]
            > max_input_tokens
        ):
            raise ValueError(
                "Question and system prompt exceed "
                "the model token limit"
            )

        selected_parts: list[str] = []

        for source in sources:
            source_text = load_source_text(source)

            part = (
                f"[Source: {source.file_path}]\n"
                f"{source_text}"
            )

            candidate_parts = selected_parts + [part]
            candidate_context = "\n\n".join(
                candidate_parts
            )

            candidate_prompt = make_prompt(
                candidate_context
            )

            candidate_inputs = tokenize_prompt(
                candidate_prompt
            )

            candidate_tokens = (
                candidate_inputs["input_ids"].shape[1]
            )

            if candidate_tokens <= max_input_tokens:
                selected_parts.append(part)

        if not selected_parts:
            raise ValueError(
                "No retrieved source fits within "
                "the model token budget"
            )

        context = "\n\n".join(selected_parts)
        prompt = make_prompt(context)
        inputs = tokenize_prompt(prompt)

        input_tokens = inputs["input_ids"].shape[1]

        if input_tokens > max_input_tokens:
            raise ValueError(
                "Input exceeds the allowed token budget"
            )

        if input_tokens + MAX_NEW_TOKENS > model_limit:
            raise ValueError(
                "Input and output exceed "
                "the model context window"
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

        generated_tokens = outputs[0][input_tokens:]

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

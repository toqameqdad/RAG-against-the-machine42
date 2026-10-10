"""Generate grounded answers using retrieved source chunks."""

import uuid
import json
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retriever import BM25Retriever
from src.utils import load_json_file

MAX_CONTEXT_SOURCES = 7
MAX_NEW_TOKENS = 150

SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase and documentation "
    "using ONLY the provided context.\n"
    "Rules:\n"
    "1. Answer in one or two complete sentences.\n"
    "2. Copy exact names from the context (endpoints, flags, methods, "
    "class names, versions, commands). Never invent a name.\n"
    "3. Start with Yes or No ONLY if the question begins with Is, Are, "
    "Can, Does or Do. Otherwise answer directly.\n"
    "4. Always give the best answer you can from the context, even if "
    "it is partial. Reply 'The context does not contain this "
    "information.' ONLY if nothing in the context relates to the "
    "question.\n"
    "5. Do not repeat the question.\n"
    "6. Never write a URL unless it appears exactly in the context. "
    "To point to documentation, give the file path from the source "
    "header.\n"
    "7. For questions asking for a command, give the full command "
    "with its main arguments as shown in the context."
)


def load_source_text(source: MinimalSource) -> str:
    """Load the exact text covered by a retrieved source."""
    path = Path(source.file_path)

    with path.open("r", encoding="utf-8") as file:
        text = file.read()

    return text[
        source.first_character_index:
        source.last_character_index
    ]


def build_context(sources: list[MinimalSource]) -> str:
    """Build model context from retrieved sources."""
    parts: list[str] = []

    for number, source in enumerate(sources, start=1):
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
        context = build_context(sources[:MAX_CONTEXT_SOURCES])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()


def answer(
    question: str,
    chunks_path: str,
    k: int = 5,
) -> MinimalAnswer:
    """Retrieve sources and generate an answer for one question."""
    retriever = BM25Retriever(chunks_path)
    sources = retriever.search(
        query=question,
        k=k,
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

    data = load_json_file(student_search_results_path)

    search_results = StudentSearchResults(**data)

    generator = AnswerGenerator()
    answers: list[MinimalAnswer] = []

    for result in tqdm(
        search_results.search_results[10:20],
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

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            results.model_dump(),
            file,
            ensure_ascii=False,
            indent=2,
        )

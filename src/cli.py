"""Command-line interface for the RAG project."""

from pathlib import Path

from src.generator import answer, answer_dataset, save_answers
from src.indexer import build_chunks, save_chunks
from src.evaluator import evaluate_recall
from src.retriever import (
    BM25Retriever,
    save_search_results,
    search_dataset,
)


DEFAULT_CHUNKS_PATH = "data/processed/chunks.json"


def index(
    raw_directory: str = "data/raw/vllm-0.10.1",
    output_path: str = DEFAULT_CHUNKS_PATH,
    max_chunk_size: int = 2000,
) -> None:
    """Index the source codebase into searchable chunks."""
    chunks = build_chunks(
        raw_directory=raw_directory,
        max_chunk_size=max_chunk_size,
    )

    save_chunks(
        chunks=chunks,
        output_path=output_path,
    )

    print(f"Indexed {len(chunks)} chunks.")
    print(f"Saved to: {output_path}")


def search(
    query: str,
    k: int = 5,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
) -> None:
    """Search for the most relevant sources."""
    retriever = BM25Retriever(chunks_path)

    results = retriever.search(
        query=query,
        k=k,
    )

    for index_number, source in enumerate(results, start=1):
        print(
            f"{index_number}. "
            f"{source.file_path} "
            f"[{source.first_character_index}:"
            f"{source.last_character_index}]"
        )


def search_dataset_command(
    dataset_path: str,
    k: int = 5,
    save_directory: str = "data/output/search_results",
    chunks_path: str = DEFAULT_CHUNKS_PATH,
) -> None:
    """Search all questions from a dataset."""
    results = search_dataset(
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        k=k,
    )

    dataset_name = Path(dataset_path).name

    output_path = str(
        Path(save_directory) / dataset_name
    )

    save_search_results(
        results=results,
        output_path=output_path,
    )

    print(f"Saved {len(results.search_results)} results.")
    print(f"Saved to: {output_path}")


def answer_command(
    query: str,
    k: int = 5,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
) -> None:
    """Generate an answer for one question."""
    result = answer(
        question=query,
        chunks_path=chunks_path,
        k=k,
    )

    print(result.answer)


def answer_dataset_command(
    student_search_results_path: str,
    save_directory: str = "data/output/answers",
) -> None:
    """Generate answers for a saved search dataset."""
    results = answer_dataset(
        student_search_results_path
    )

    dataset_name = Path(
        student_search_results_path
    ).name

    output_path = str(
        Path(save_directory) / dataset_name
    )

    save_answers(
        results=results,
        output_path=output_path,
    )

    print(f"Saved {len(results.search_results)} answers.")
    print(f"Saved to: {output_path}")


def evaluate(
    student_search_results_path: str,
    dataset_path: str,
    k: int = 5,
) -> None:
    """Evaluate recall@k for saved search results."""
    recall = evaluate_recall(
        student_results_path=student_search_results_path,
        answered_dataset_path=dataset_path,
        k=k,
    )

    print(f"Recall@{k}: {recall:.3f} ({recall * 100:.1f}%)")

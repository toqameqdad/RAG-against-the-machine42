"""Retrieve the most relevant chunks using BM25."""

import json
import re
from pathlib import Path
from tqdm import tqdm

from rank_bm25 import BM25Okapi

from src.models import (
    Chunk,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)
from src.utils import load_json_file


def load_chunks(chunks_path: str) -> list[Chunk]:
    """Load indexed chunks from a JSON file."""
    path = Path(chunks_path)

    data = load_json_file(chunks_path)

    return [Chunk(**item) for item in data]


def tokenize(text: str) -> list[str]:
    """Convert text into normalized tokens for BM25."""
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


class BM25Retriever:
    """Retrieve relevant chunks using a reusable BM25 index."""

    def __init__(self, chunks_path: str) -> None:
        """Load chunks and build the BM25 index once."""
        self.chunks = load_chunks(chunks_path)

        tokenized_corpus = [
            tokenize(chunk.text)
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 5) -> list[MinimalSource]:
        """Return the top-k most relevant sources for a query."""

        if not query.strip():
            return []

        if k <= 0:
            return []

        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[MinimalSource] = []

        for index in ranked_indexes[:k]:
            chunk = self.chunks[index]

            results.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )

        return results


def search(
    query: str,
    chunks_path: str,
    k: int = 5,
) -> list[MinimalSource]:
    """Search a single query."""
    retriever = BM25Retriever(chunks_path)

    return retriever.search(
        query=query,
        k=k,
    )


def search_dataset(
    dataset_path: str,
    chunks_path: str,
    k: int,
) -> StudentSearchResults:
    """Search all questions in a dataset."""
    path = Path(dataset_path)

    data = load_json_file(dataset_path)

    retriever = BM25Retriever(chunks_path)

    results: list[MinimalSearchResults] = []

    for item in tqdm(
        data["rag_questions"],
        desc="Searching questions",
    ):
        question_id = item["question_id"]
        question = item["question"]

        retrieved_sources = retriever.search(
            query=question,
            k=k,
        )

        results.append(
            MinimalSearchResults(
                question_id=question_id,
                question=question,
                retrieved_sources=retrieved_sources,
            )
        )

    return StudentSearchResults(
        search_results=results,
        k=k,
    )


def save_search_results(
    results: StudentSearchResults,
    output_path: str,
) -> None:
    """Save dataset search results as JSON."""
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

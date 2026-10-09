"""Evaluate retrieval results using recall@k."""

import json
from pathlib import Path
from src.utils import load_json_file


def ranges_overlap(
    retrieved_start: int,
    retrieved_end: int,
    expected_start: int,
    expected_end: int,
) -> bool:
    """Return True when two character ranges overlap."""
    return (
        retrieved_start < expected_end
        and expected_start < retrieved_end
    )


def source_matches(
    retrieved: dict,
    expected: dict,
) -> bool:
    """Return True when a retrieved source matches an expected source."""
    if retrieved["file_path"] != expected["file_path"]:
        return False

    return ranges_overlap(
        retrieved["first_character_index"],
        retrieved["last_character_index"],
        expected["first_character_index"],
        expected["last_character_index"],
    )


def evaluate_recall(
    student_results_path: str,
    answered_dataset_path: str,
    k: int = 5,
) -> float:
    """Calculate average recall@k over all questions."""
    student_path = Path(student_results_path)
    answered_path = Path(answered_dataset_path)

    student_data = load_json_file(student_results_path)

    answered_data = load_json_file(answered_dataset_path)

    expected_by_id = {
        item["question_id"]: item
        for item in answered_data["rag_questions"]
    }

    recalls: list[float] = []

    for result in student_data["search_results"]:
        question_id = result["question_id"]

        if question_id not in expected_by_id:
            continue

        expected_sources = expected_by_id[question_id]["sources"]
        retrieved_sources = result["retrieved_sources"][:k]

        if not expected_sources:
            continue

        matched = 0

        for expected in expected_sources:
            found = any(
                source_matches(retrieved, expected)
                for retrieved in retrieved_sources
            )

            if found:
                matched += 1

        recall = matched / len(expected_sources)
        recalls.append(recall)

    if not recalls:
        return 0.0

    return sum(recalls) / len(recalls)
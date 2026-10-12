"""Evaluate retrieval results using recall@k."""

from typing import Any
from src.utils import (
    load_json_file,
    require_positive_k,
)


def ranges_overlap(
    retrieved_start: int,
    retrieved_end: int,
    expected_start: int,
    expected_end: int,
) -> bool:
    """Return True when the range IoU is at least 0.05."""

    intersection_start = max(
        retrieved_start,
        expected_start,
    )

    intersection_end = min(
        retrieved_end,
        expected_end,
    )

    intersection = max(
        0,
        intersection_end - intersection_start,
    )

    if intersection == 0:
        return False

    retrieved_length = retrieved_end - retrieved_start
    expected_length = expected_end - expected_start

    union = (
        retrieved_length
        + expected_length
        - intersection
    )

    if union <= 0:
        return False

    iou = intersection / union

    return iou >= 0.05


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

    require_positive_k(k)
    student_data = load_json_file(student_results_path)

    answered_data = load_json_file(answered_dataset_path)
    if not isinstance(student_data, dict):
        raise ValueError(
            "Student results JSON root must be an object"
        )

    if "search_results" not in student_data:
        raise ValueError(
            "Student results are missing 'search_results'"
        )

    if not isinstance(answered_data, dict):
        raise ValueError(
            "Dataset JSON root must be an object"
        )

    if "rag_questions" not in answered_data:
        raise ValueError(
            "Dataset is missing 'rag_questions'"
        )

    expected_by_id: dict[str, dict[str, Any]] = {
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

"""Compare generated answers with the reference answers (for your own testing).

Usage:
    python compare_answers.py \
        data/output/search_results_and_answer/UnansweredQuestions/dataset_docs_public.json \
        data/datasets/AnsweredQuestions/dataset_docs_public.json

Optional third argument: how many of the worst answers to print (default 10).

The score is a simple keyword recall: the share of the reference answer's
important words that also appear in your answer. It is only a rough guide.
Always read the worst answers yourself.
"""

import json
import re
import sys

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "on",
    "with", "by", "as", "at", "be", "it", "its", "this", "that", "from",
    "for", "you", "can", "use", "used", "using", "vllm", "which", "when",
    "how", "what", "does", "do", "your", "will", "if", "then", "into",
}


def keywords(text: str) -> set[str]:
    """Return the important lowercase words of a text."""
    found = re.findall(r"[a-z0-9_./\-]+", text.lower())
    result: set[str] = set()
    for word in found:
        word = word.strip("./-")
        if re.fullmatch(r"v\d[\w.]*", word):
            word = word[1:]
        if not word or word in STOPWORDS:
            continue
        if len(word) > 1 or word.isdigit():
            result.add(word)
    return result


def score(generated: str, reference: str) -> float:
    """Share of reference keywords found in the generated answer."""
    ref = keywords(reference)
    if not ref:
        return 0.0
    return len(ref & keywords(generated)) / len(ref)


def main() -> int:
    """Run the comparison."""
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    with open(sys.argv[1], encoding="utf-8") as file:
        generated = json.load(file)["search_results"]
    with open(sys.argv[2], encoding="utf-8") as file:
        reference = {
            q["question_id"]: q for q in json.load(file)["rag_questions"]
        }
    worst_count = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    rows = []
    for item in generated:
        ref = reference.get(item["question_id"])
        if ref is None:
            continue
        rows.append((score(item["answer"], ref["answer"]), item, ref))

    if not rows:
        print("No matching question_id between the two files.")
        return 1

    rows.sort(key=lambda row: row[0])
    average = sum(row[0] for row in rows) / len(rows)
    low = sum(1 for row in rows if row[0] < 0.3)
    print(f"Questions compared : {len(rows)}")
    print(f"Average keyword recall : {average:.2f}")
    print(f"Answers below 0.30 : {low}\n")

    print(f"=== {worst_count} worst answers ===")
    for value, item, ref in rows[:worst_count]:
        print(f"\n[{value:.2f}] Q: {item['question']}")
        print(f"  YOURS    : {item['answer'][:250]!r}")
        print(f"  REFERENCE: {ref['answer'][:250]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Pydantic data models used by the RAG pipeline."""

import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represent a source location retrieved from the indexed codebase."""

    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(BaseModel):
    """Represent a searchable chunk extracted from a source file."""

    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question that has not been answered yet."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with its reference sources and answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a dataset containing answered or unanswered RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent retrieval results for a single question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieval results together with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Represent search results for multiple questions."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent search results and generated answers."""

    search_results: list[MinimalAnswer]
    k: int

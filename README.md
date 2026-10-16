*This activity has been created as part of the 42 curriculum by tmeqdad.*

# RAG Against the Machine

## Description

RAG Against the Machine is a Retrieval-Augmented Generation system built to answer questions about the provided vLLM codebase.

Instead of relying only on the knowledge stored inside a language model, the system retrieves relevant information directly from the vLLM source code and documentation before generating an answer.

The project implements the four main stages of a RAG pipeline:

1. **Indexing**: read and split the vLLM repository into searchable chunks.
2. **Retrieval**: rank the indexed chunks according to their relevance to a question.
3. **Augmentation**: load the exact retrieved source text and place it in the model context.
4. **Generation**: use `Qwen/Qwen3-0.6B` to generate an answer grounded in the retrieved information.

The system also stores the exact file path and character positions of every retrieved source, allowing retrieval quality to be evaluated using Recall@k.

---

## System Architecture

The project is divided into independent stages:

```text
                         vLLM repository
                               |
                               v
                         File ingestion
                               |
                  +------------+------------+
                  |                         |
                  v                         v
          Python chunking          Markdown/Text chunking
             using AST              using text structure
                  |                         |
                  +------------+------------+
                               |
                               v
                       Searchable chunks
                               |
                               v
                         BM25 retrieval
                               |
                               v
                     Top-k source locations
                               |
                               v
                    Exact source text loading
                               |
                               v
                 Token-budget source selection
                               |
                               v
                       Context augmentation
                               |
                               v
                      Qwen/Qwen3-0.6B
                               |
                               v
                        Grounded answer
```

The main source files are:

```text
src/
├── __init__.py
├── __main__.py
├── cli.py
├── models.py
├── chunkers.py
├── indexer.py
├── retriever.py
├── generator.py
├── evaluator.py
└── utils.py
```

### Component responsibilities

- `models.py`: Pydantic models used to validate data exchanged between stages.
- `chunkers.py`: Python and Markdown/text chunking logic.
- `indexer.py`: reads the vLLM repository and builds the chunk index.
- `retriever.py`: BM25 indexing and top-k retrieval.
- `generator.py`: context construction and Qwen answer generation.
- `evaluator.py`: local Recall@k evaluation.
- `cli.py`: commands exposed through Python Fire.
- `utils.py`: shared validation and file-loading utilities.
- `__main__.py`: CLI entry point and top-level exception handling.

---

## Chunking Strategy

The project uses different chunking strategies for Python code and Markdown/text because source code and documentation have different structures.

The maximum chunk size is configurable through:

```bash
--max_chunk_size
```

The default is:

```text
2000 characters
```

### Python chunking

Python files are parsed using Python's built-in `ast` module.

The chunker identifies top-level structures such as:

- functions
- async functions
- classes

These structures are treated as meaningful code blocks.

Smaller top-level statements, such as imports and assignments, may be grouped together when they fit within the configured maximum chunk size.

If a Python file cannot be parsed because of a syntax error, the implementation falls back to text-based splitting instead of crashing.

### Markdown chunking

Markdown documents are primarily divided around headings:

```text
#
##
###
####
#####
######
```

This keeps related documentation sections together whenever possible.

If a section exceeds the maximum size, it is split further using line boundaries.

### Plain text chunking

Plain text files do not use Markdown heading detection.

They are split using line boundaries while respecting the configured maximum chunk size.

If a single line is larger than the limit, it is split into smaller character ranges.

### Character positions

Every chunk stores:

```text
file_path
first_character_index
last_character_index
```

These values allow the system to identify the exact original source region returned by retrieval.

---

## Retrieval Method

The retrieval system uses **BM25**, implemented with `rank-bm25`.

BM25 is a lexical ranking algorithm that scores documents according to how strongly their terms match the query.

### Tokenization

Before BM25 scoring, text is normalized and tokenized.

The tokenizer:

- converts text to lowercase
- extracts alphanumeric tokens
- preserves underscores in identifiers

For example:

```text
load_lora_adapter
```

remains a meaningful searchable identifier.

### Retrieval process

For a query:

```text
question
   |
   v
tokenization
   |
   v
BM25 scoring
   |
   v
sort scores
   |
   v
top-k chunks
   |
   v
MinimalSource objects
```

Each returned source contains the exact file path and character range.

The BM25 corpus is built once inside the retriever and reused when processing a complete dataset. This avoids rebuilding the retrieval structure for every individual question.

---

## Answer Generation

Answer generation uses:

```text
Qwen/Qwen3-0.6B
```

The retrieved source locations are converted back into exact source text using their stored character ranges.

The resulting context is passed to Qwen together with the question.

The model is used through its chat template with thinking disabled for this RAG task.

The system prompt instructs the model to:

- use only the supplied context
- avoid inventing unsupported information
- preserve exact technical identifiers
- provide direct answers
- avoid unnecessary repetition

This grounding is important because the language model is intentionally small and should rely primarily on the retrieved sources.

### Token budget and source selection

The generator uses the required default model, `Qwen/Qwen3-0.6B`, and builds the prompt with the model's chat template. It does **not** impose a fixed three-source limit. Instead, it considers the retrieved chunks in their BM25 ranking order and includes each complete source when it fits into the available token budget. A source that does not fit is skipped; later, shorter sources may still be included.

The input budget is calculated from the model context window:

```text
maximum_input_tokens = max_position_embeddings - MAX_NEW_TOKENS - TOKEN_SAFETY_MARGIN
MAX_NEW_TOKENS = 150
TOKEN_SAFETY_MARGIN = 128
```

The implementation tokenizes candidate prompts before accepting an additional source. It preserves the full system prompt and user question, and checks the final input size before calling `model.generate()`. If no retrieved source fits, or the question and system prompt alone exceed the budget, generation raises a clear error rather than proceeding with an oversized prompt.

**Trade-off:** this strategy avoids arbitrarily discarding all sources after the third result and avoids splitting a source in the middle, but it can skip a highly ranked chunk that is too large. More context can also introduce distractions, so adding more sources does not guarantee a better answer.

During development, a temporary 1,000-token context limit was used to test the guard: 680 input tokens plus a 150-token output budget totaled 830 of 1,000 tokens. The normal model limit was not permanently changed.

---

## Data Models

The project uses Pydantic models for validation.

The required models are:

```text
MinimalSource
UnansweredQuestion
AnsweredQuestion
RagDataset
MinimalSearchResults
MinimalAnswer
StudentSearchResults
StudentSearchResultsAndAnswer
```

Example retrieved source:

```json
{
  "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
  "first_character_index": 4695,
  "last_character_index": 6100
}
```

Example dataset search result:

```json
{
  "question_id": "example-question-id",
  "question": "Example question",
  "retrieved_sources": [
    {
      "file_path": "data/raw/vllm-0.10.1/example.md",
      "first_character_index": 0,
      "last_character_index": 500
    }
  ]
}
```

---

## Instructions

### Requirements

The project requires:

```text
Python >= 3.10
uv
```

The main Python dependencies are:

```text
fire
pydantic
rank-bm25
torch
tqdm
transformers
```

Development tools include:

```text
flake8
mypy
```

---

### Installation

Install all dependencies using:

```bash
make install
```

or directly:

```bash
uv sync
```

---

### 42 goinfre support

The Makefile checks whether the following directory exists:

```text
/goinfre/$(USER)
```

When available, caches and the virtual environment are stored under `goinfre`.

This includes:

```text
UV_CACHE_DIR
HF_HOME
UV_PROJECT_ENVIRONMENT
```

This is especially useful because PyTorch, Transformers, and Hugging Face model files can require significant disk space.

On systems without `/goinfre/$(USER)`, the project uses the normal local uv environment.

---

## Makefile

### Install

```bash
make install
```

Installs project dependencies with `uv`.

### Run

```bash
make run
```

Runs:

```bash
uv run python -m src
```

### Debug

```bash
make debug
```

Runs the project using Python's built-in debugger.

### Clean

```bash
make clean
```

Removes temporary files and caches such as:

```text
__pycache__
*.pyc
.mypy_cache
.pytest_cache
```

### Lint

```bash
make lint
```

Runs:

```text
flake8
mypy
```

against the project source code.

### Strict lint

```bash
make lint-strict
```

Runs stricter mypy checking for development.

---

## CLI Usage

The CLI is implemented using Python Fire.

General syntax:

```bash
uv run python -m src <command> [options]
```

Available commands:

```text
index
search
search_dataset
answer
answer_dataset
evaluate
```

---

## Example Usage

### 1. Index the vLLM repository

Place the provided vLLM repository under:

```text
data/raw/vllm-0.10.1/
```

Then run:

```bash
uv run python -m src index --max_chunk_size 2000
```

The generated index is stored under:

```text
data/processed/
```

---

### 2. Search a single question

```bash
uv run python -m src search \
"What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
--k 5
```

Example output:

```text
1. data/raw/vllm-0.10.1/docs/features/lora.md [4695:6100]
2. data/raw/vllm-0.10.1/docs/features/lora.md [6100:8082]
3. data/raw/vllm-0.10.1/examples/others/tensorize_vllm_model.py [2618:4457]
```

---

### 3. Search a complete dataset

```bash
uv run python -m src search_dataset \
--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
--k 10 \
--save_directory data/output/search_results/UnansweredQuestions
```

The output is a `StudentSearchResults` JSON file.

---

### 4. Answer a single question

```bash
uv run python -m src answer \
"What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" \
--k 5
```

Example output:

```text
The HTTP endpoint used to dynamically load a LoRA adapter in vLLM is `/v1/load_lora_adapter`.
```

---

### 5. Generate answers for a dataset

First generate the search results.

Then run:

```bash
uv run python -m src answer_dataset \
--student_search_results_path \
data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
--save_directory \
data/output/search_results_and_answer/UnansweredQuestions
```

The resulting file follows the `StudentSearchResultsAndAnswer` model.

A progress bar is displayed while answers are being generated.

---

### 6. Evaluate retrieval locally

```bash
uv run python -m src evaluate \
data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
data/datasets/AnsweredQuestions/dataset_docs_public.json \
--k 5
```

The built-in evaluator is intended for development and iteration.

The official defense score is calculated externally using the provided moulinette.

The project itself does not import or call moulinette.

---

## Expected Data Layout

```text
data/
├── raw/
│   └── vllm-0.10.1/
│
├── processed/
│
├── datasets/
│   ├── UnansweredQuestions/
│   └── AnsweredQuestions/
│
└── output/
    ├── search_results/
    │   ├── UnansweredQuestions/
    │   └── AnsweredQuestions/
    │
    └── search_results_and_answer/
        ├── UnansweredQuestions/
        └── AnsweredQuestions/
```

Input and output paths used by dataset commands are configurable through CLI arguments.

---

## Performance Analysis

The retrieval system was tested using the provided public datasets and the official moulinette. Results vary with the dataset and environment, so the earlier development measurements below are kept separate from the latest private examination.

### Latest private retrieval examination (2026-10-16)

The official `exams/scripts/exam_retrieval.sh` script completed all four checks successfully:

| Check | Recorded result | Required limit | Status |
| --- | ---: | ---: | --- |
| Indexing time | 26 seconds | at most 300 seconds | PASS |
| Warm retrieval throughput | 200 questions in 30 seconds (15 seconds for docs + 15 for code) | at most 90 seconds | PASS |
| Documentation Recall@5 | 82.0% (100 questions) | at least 80% | PASS |
| Code Recall@5 | 55.0% (100 questions) | at least 50% | PASS |

The script reported **FINAL RESULT: 4/4, STATUS: PASS**.

Additional measured recall values:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| Documentation (private) | 54.0% | 76.0% | 82.0% | 88.0% |
| Code (private) | 31.0% | 49.0% | 55.0% | 64.0% |

Run the retrieval examination from the `rag-benchmark` parent directory (not from `student`):

```bash
bash exams/scripts/exam_retrieval.sh \
  --student-path ./student \
  --moulinette-path ./moulinette
```

### Earlier development measurements

The figures below are from earlier public-dataset/development runs and should not be confused with the private examination above.

### Documentation dataset

Number of evaluated questions:

```text
100
```

Results:

```text
Recall@1  = 62.0%
Recall@3  = 80.0%
Recall@5  = 83.0%
Recall@10 = 83.0%
```

Required documentation Recall@5:

```text
80%
```

Achieved:

```text
83.0%
```

The documentation retrieval requirement is satisfied.

### Code dataset

Number of evaluated questions:

```text
99
```

Results:

```text
Recall@1  = 28.3%
Recall@3  = 43.4%
Recall@5  = 51.5%
Recall@10 = 58.6%
```

Required code Recall@5:

```text
50%
```

Achieved:

```text
51.5%
```

The code retrieval requirement is satisfied.

### Indexing performance

Indexing the full provided vLLM repository took approximately:

```text
2 minutes 7 seconds
```

on the development machine.

The required maximum indexing time is:

```text
5 minutes
```

### Retrieval performance

After changing the retriever so that the BM25 corpus is built once and reused, processing approximately 100 questions took around:

```text
13 seconds
```

on the development machine.

Performance depends on the hardware used.

### Analysis

Documentation retrieval performs better than code retrieval.

This is expected because natural-language documentation tends to share more vocabulary with natural-language questions.

Code questions can refer to:

- function behavior
- class behavior
- identifiers
- implementation details

without necessarily using the exact tokens present in the relevant source chunk.

The current BM25 implementation satisfies the required public Recall@5 thresholds, although code retrieval has a smaller margin above the required threshold.

---

## Design Decisions

### BM25 instead of TF-IDF

BM25 was selected as the required lexical retrieval method because it provides useful term-frequency normalization and works well for both source code identifiers and documentation terms.

### AST-based Python chunking

Python code is structurally different from normal prose.

Using the AST makes it possible to keep functions and classes together rather than cutting code only according to arbitrary character boundaries.

### Separate Markdown strategy

Markdown headings naturally identify documentation sections.

Using headings keeps related explanations together and improves retrieval context.

### Exact source ranges

The system stores exact character positions instead of only file names.

This makes retrieval outputs precise and compatible with Recall@k evaluation.

### Reusable BM25 retriever

The first implementation rebuilt the BM25 corpus for every question.

This was too slow for dataset retrieval.

The retriever was redesigned so the chunks and BM25 index are loaded once and reused across all questions.

This substantially reduced dataset retrieval time.

### Source count versus token budget

An earlier version used a fixed maximum of three context sources. The current approach considers all retrieved top-k source candidates while enforcing a token budget. This increases flexibility for multi-file questions, at the cost of more tokenization work and the possibility that less-relevant context affects the small generator.

### Small grounded generation model

The required `Qwen/Qwen3-0.6B` model has limited reasoning capacity compared with larger models.

For this reason, the generation strategy focuses on:

- high-quality retrieval
- concise context
- strong grounding instructions
- short direct answers

instead of asking the model to solve questions from its internal knowledge.

---

## Challenges Faced

### Chunk size limits

One challenge was preserving meaningful code and documentation structure while ensuring that no retrieved chunk exceeds the configured character limit.

The solution was to use structure-aware chunking first, followed by line-based splitting for blocks that are still too large.

### Different file types

Treating every line beginning with `#` as a Markdown heading caused problems for non-Markdown text files.

The solution was to apply Markdown heading logic only to Markdown files and use plain text splitting for other text files.

### Exact character indices

The evaluator requires exact file paths and character ranges.

The chunking system therefore tracks positions relative to the original source file rather than only storing chunk text.

### Retrieval speed

Initially, BM25 was rebuilt for every dataset question.

For 100 questions this was unnecessarily slow.

The solution was to introduce a reusable `BM25Retriever` that builds the corpus once.

### Code retrieval quality

BM25 performs lexical matching, while code questions may describe behavior without using the exact identifiers present in the implementation.

The tokenizer was improved to normalize text and preserve code identifiers such as snake_case names.

### Language model repetition

Early Qwen generation tests repeatedly produced the same answer several times.

The solution was to use Qwen's chat template, disable thinking mode for this task, provide clearer grounding instructions, and configure generation appropriately.

### Hallucination

The small language model sometimes generated plausible but unsupported details.

The system prompt was strengthened so that the model must rely on the provided source context and avoid inventing identifiers, commands, URLs, or implementation details.

### Static typing with Transformers

Some Transformers APIs expose broad or incomplete type information to mypy.

The implementation uses explicit type casts in the few places where the runtime type is known but the external library typing is not precise enough.

---

## Error Handling

The CLI is designed to handle invalid input without exposing an unhandled traceback.

Handled cases include:

- empty queries
- `k <= 0`
- invalid chunk sizes
- missing files
- malformed JSON
- invalid UTF-8 input
- keyboard interruption

Long-running dataset operations display progress using `tqdm`.

---

## Resources

The following resources were useful while developing the project:

- 42 `RAG Against the Machine` subject
- Python documentation
  - `ast`
  - `pathlib`
  - `json`
  - type hints
- Pydantic documentation
- Python Fire documentation
- `rank-bm25` documentation
- Hugging Face Transformers documentation
- Qwen3 model documentation
- vLLM source code and documentation
- mypy documentation
- flake8 documentation
- uv documentation

### Use of AI

AI tools were used as a learning and development assistant during the activity.

They were used for:

- explaining RAG concepts
- explaining BM25 and Recall@k
- discussing Python AST-based chunking
- reviewing implementation ideas
- debugging Python and CLI errors
- investigating flake8 and mypy errors
- improving prompts used with Qwen
- reviewing edge cases
- reviewing the project against the subject requirements
- helping structure project documentation

AI-generated suggestions were reviewed, tested, and modified before being integrated into the project.

The implementation was validated using:

- direct CLI testing
- retrieval tests
- Qwen generation tests
- flake8
- mypy
- the provided public datasets
- the official moulinette retrieval evaluator

---

## Limitations

The retrieval system is based on lexical BM25 matching.

As a result, retrieval can be weaker when a question describes a concept using terminology that does not appear directly in the relevant code.

The generation model is intentionally small, so generated answers may occasionally be incomplete even when the correct context is retrieved.

For this reason, the project prioritizes retrieval accuracy and grounding over free-form generation.

---

## Repository Notes

Large data files, generated output, downloaded model weights, and caches should not be committed to the Git repository.

Examples include:

```text
data/raw/
data/processed/
data/output/
.venv/
Hugging Face model cache
uv cache
```

These files are generated or provided separately during evaluation.

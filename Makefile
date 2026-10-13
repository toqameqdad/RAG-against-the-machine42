
ifneq ($(wildcard /goinfre/$(USER)),)
UV_ENV := UV_CACHE_DIR=/goinfre/$(USER)/uv-cache \
	HF_HOME=/goinfre/$(USER)/hf-cache \
	UV_PROJECT_ENVIRONMENT=/goinfre/$(USER)/rag-venv \
	UV_LINK_MODE=copy
else
UV_ENV :=
endif

.PHONY: install run debug clean lint lint-strict \
	index search run_index run_search_dataset run_my_evaluate

install:
	$(UV_ENV) uv sync

run:
	$(UV_ENV) uv run python -m src

debug:
	$(UV_ENV) uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf htmlcov .coverage

lint:
	$(UV_ENV) uv run flake8 src
	$(UV_ENV) uv run mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	$(UV_ENV) uv run flake8 src
	$(UV_ENV) uv run mypy src --strict

index:
	$(UV_ENV) uv run python -m src index --max_chunk_size 2000

run_index: index

search:
	$(UV_ENV) uv run python -m src search "What is vLLM?" --k 5

run_search_dataset:
	$(UV_ENV) uv run python -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--k 5 \
		--save_directory data/output/search_results/UnansweredQuestions
	$(UV_ENV) uv run python -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
		--k 5 \
		--save_directory data/output/search_results/UnansweredQuestions

run_my_evaluate:
	$(UV_ENV) uv run python -m src evaluate \
		--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
	$(UV_ENV) uv run python -m src evaluate \
		--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json

ifneq ($(wildcard /goinfre/$(USER)),)
UV_ENV := UV_CACHE_DIR=/goinfre/$(USER)/uv-cache \
	HF_HOME=/goinfre/$(USER)/hf-cache \
	UV_PROJECT_ENVIRONMENT=/goinfre/$(USER)/rag-venv \
	UV_LINK_MODE=copy
else
UV_ENV :=
endif

.PHONY: install run debug clean lint lint-strict index search

install:
	$(UV_ENV) uv sync

run:
	$(UV_ENV) uv run python -m src

debug:
	$(UV_ENV) uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage

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

search:
	$(UV_ENV) uv run python -m src search "What is vLLM?" --k 5
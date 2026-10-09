.PHONY: install run index search test lint typecheck clean

install:
	uv sync

run:
	uv run python -m src

index:
	uv run python -m src index

search:
	uv run python -m src search

test:
	uv run python -m pytest

lint:
	uv run flake8 src tests

typecheck:
	uv run mypy src

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
.PHONY: help install lock test format lint type-check clean

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:' Makefile | cut -d: -f1 | sort | uniq | sed 's/^/ - /'

install:
	uv pip install --system -e .[test,litellm]

lock:
	uv lock

test:
	uv run pytest -q

format:
	uv tool run ruff format .

lint:
	uv tool run ruff check .

type-check:
	uv tool run pyright

clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache __pycache__

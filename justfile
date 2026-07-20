set dotenv-load

default:
  just --list

sync:
  uv sync --frozen --group dev

sync-train:
  uv sync --frozen --group dev --extra train

lock:
  uv lock

test:
  uv run --frozen pytest

lint:
  uv run --frozen ruff check .
  uv run --frozen ruff format --check .

framework-info:
  uv run --frozen cosmos-h-surgical framework-info

release-check:
  uv run --frozen cosmos-h-surgical validate-release
  uv run --frozen pytest
  uv run --frozen ruff check .
  uv run --frozen ruff format --check .

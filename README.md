# VectorDB

Modular vector database with a FastAPI backend.

## Requirements

- Python 3.12+

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and adjust values as needed.

## Run

```bash
vectordb
# or
uvicorn vectordb.api.app:create_app --factory --reload
```

## Test

```bash
pytest
```

.PHONY: install ingest serve run test eval lint format ci docker-build docker-run clean

install:
	uv sync

ingest:
	uv run python run.py --create-db

serve:
	uv run python app.py

run:
	uv run python run.py

test:
	uv run pytest tests/ -v

eval:
	uv run python evals/run_eval.py

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

ci: lint test

docker-build:
	docker build -t rag-medical .

docker-run:
	docker run --rm -p 7860:7860 -e OPENAI_API_KEY -v ./docs:/app/docs -v ./chroma_db:/app/chroma_db rag-medical

clean:
	rm -rf chroma_db/

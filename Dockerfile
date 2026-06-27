FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and metadata
COPY README.md src/ app.py run.py ./

# Install the project itself
RUN uv sync --frozen --no-dev

EXPOSE 7860

CMD ["uv", "run", "python", "app.py"]

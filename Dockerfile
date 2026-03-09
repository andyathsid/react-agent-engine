FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

# -----------------------------
# Builder stage
# -----------------------------
FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .python-version ./
COPY uv.loc[k] ./
RUN uv sync --no-dev --no-install-project --compile-bytecode

COPY src/ ./src/
RUN uv sync --no-dev --compile-bytecode

# Pre-download roberta-base tokenizer so it's baked into the image
RUN uv run python -c "from transformers import RobertaTokenizer; RobertaTokenizer.from_pretrained('roberta-base')"

# -----------------------------
# Runtime stage
# -----------------------------
FROM base AS final

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libpq5 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.cache /app/.cache
COPY aegra.json .
COPY src/ ./src/
COPY models/ ./models/

RUN chown -R app:app /app/.cache

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

USER app

CMD ["sh", "-c", "aegra serve --host 0.0.0.0 --port ${PORT:-8080}"]

# --- Stage 1: Build ---
FROM python:3.13-slim AS builder

WORKDIR /app

# Install Poetry - Ensuring version 2.0+ support
RUN pip install "poetry>=2.0.0"

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies - we use --no-root because we are in non-package mode
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi --no-root

# --- Stage 2: Final Runtime ---
FROM python:3.13-slim AS runner

WORKDIR /app

# Copy the virtualenv from the builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy your source code and model
COPY src/ ./src/
COPY models/ ./models/

# Pre-download NLTK data for production stability
RUN python -m nltk.downloader stopwords wordnet averaged_perceptron_tagger_eng punkt_tab

EXPOSE 8000

# We use the full path to the uvicorn in our venv
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
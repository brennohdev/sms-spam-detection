# SMS Spam Detection: End-to-End MLOps Pipeline

## Overview

A production-ready asynchronous API and training orchestration system for SMS spam classification. This project implements Clean Architecture principles and follows MLOps best practices for data versioning, experiment tracking, and containerization.

## System Architecture

The project is strictly divided into two lifecycles to ensure high maintainability and prevent training-serving skew:

- **Inference Engine (src/)**: A decoupled FastAPI application following hexagonal principles. The core business logic (NLP processing) is isolated from the delivery mechanism (API).
- **Training Pipeline (pipelines/)**: A modular orchestration system (Ingest → Train → Evaluate) designed for reproducibility and experimentation.

## Technology Stack

- **Language**: Python 3.13
- **Framework**: FastAPI (Asynchronous API)
- **Machine Learning**: Scikit-Learn (Multinomial Naive Bayes)
- **NLP**: NLTK (Lemmatization, Tokenization, POS-tagging)
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC (Data Version Control)
- **Dependency Management**: Poetry 2.0
- **Infrastructure**: Docker (Multi-stage builds)

## Model Performance

The current model is a Naive Bayes classifier optimized via TF-IDF vectorization with custom lemmatization.

| Metric           | Score |
| ---------------- | ----- |
| Accuracy         | 98.2% |
| Precision (Spam) | 98.1% |
| Recall (Spam)    | 94.0% |
| F1-Score         | 96.0% |

> **Note**: The model is specifically tuned to minimize False Positives (classifying legitimate messages as spam), ensuring a high-quality user experience.

## MLOps Workflow

### Data Versioning (DVC)

We use DVC to track the `spam.csv` dataset without bloating the Git history. The data is hashed and tracked via `.dvc` files.

```bash
dvc pull  # To fetch the versioned dataset
```

### Experiment Tracking (MLflow)

Every training run is logged into an MLflow Tracking Server. We record hyperparameters (alpha, ngram_range) and performance metrics.

```bash
# To view the experiment dashboard:
poetry run mlflow ui
```

## API Reference

### Predict Spam

- **POST** `/v1/predict`

#### Request Body:

```json
{
  "text": "Congratulations! You've won a $1000 Walmart gift card. Click here to claim."
}
```

#### Response:

```json
{
  "is_spam": true,
  "confidence": 0.9982,
  "model_version": "20260221_v1",
  "processing_time_ms": 12.5
}
```

## Installation and Execution

### Local Development

Clone the repository and install dependencies:

```bash
git clone https://github.com/brennohdev/sms-spam-detection.git
cd sms-spam-detection
poetry install
```

### Run the Training Pipeline:

```bash
PYTHONPATH=. poetry run python pipelines/main.py
```

### Start the API:

```bash
poetry run uvicorn src.api.main:app --reload
```

## Docker Deployment

The project uses a multi-stage Dockerfile to minimize image size and increase security.

```bash
docker build -t sms-spam-api:latest .
docker run -p 8000:8000 sms-spam-api:latest
```

## Project Structure

```json
plaintext
├── data/               # Versioned datasets (DVC)
├── models/             # Serialized .joblib artifacts
├── pipelines/          # Training orchestration logic
│   └── steps/          # Modular pipeline components
├── src/                # Inference Engine (Production Code)
│   ├── api/            # FastAPI routes and schemas
│   ├── core/           # Domain logic and ports
│   └── adapters/       # NLP and Model implementations
├── pyproject.toml      # Poetry 2.0 configuration
└── Dockerfile          # Multi-stage production build
```

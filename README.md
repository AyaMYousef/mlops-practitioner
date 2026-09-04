# prodml

**Production-oriented machine learning service for predicting NYC Green Taxi trip duration.**

`prodml` transforms a machine learning workflow from a notebook into a structured, tested, and deployable Python application. The project uses a **Random Forest Regressor** to predict taxi trip duration in minutes based on the pickup/drop-off location pair (`PU_DO`) and trip distance. The trained model is exposed through a **FastAPI** service with single and batch prediction endpoints, health checks, metadata, request validation, structured logging, and correlation IDs. The project also includes automated tests with a coverage gate, Docker multi-stage builds, Docker Compose configuration, model serialization, and ONNX export.

---

## Quick Start

A new user can go from installation to a running prediction API in **three commands**.

Run these commands from the project root:

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Train the model

```bash
python -m prodml.train
```

This trains the Random Forest model and saves the model artifact to:

```text
models/baseline.pkl
```

### 3. Start the API

```bash
uvicorn prodml.api.main:app --host 0.0.0.0 --port 8000
```

The API is now available at:

```text
http://localhost:8000
```

---

## Make a Prediction

With the API running, open another terminal and send a prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PU_DO": "161_237",
    "trip_distance": 2.5
  }'
```

Example response:

```json
{
  "prediction": 14.32,
  "model_version": "baseline.pkl",
  "correlation_id": "7d3c5c8e-..."
}
```

`prediction` represents the estimated trip duration in **minutes**.

The exact prediction depends on the trained model and dataset.

---

## Try the Project Yourself

Clone the repository and enter the project directory:

```bash
git clone <your-repository-url>
cd mlops-practitioner
```

Then follow the **Quick Start** section above.

The project includes the required dataset under:

```text
data/raw/green_tripdata_2024-01.parquet
```

so the training workflow can be executed after installing the project.

---

## API Documentation

Once the API is running, FastAPI provides interactive Swagger documentation at:

```text
http://localhost:8000/docs
```

You can use the Swagger UI to test the endpoints directly from your browser.

### Available Endpoints

| Method | Endpoint         | Description                 |
| ------ | ---------------- | --------------------------- |
| `GET`  | `/health`        | Checks API and model health |
| `GET`  | `/metadata`      | Returns model metadata      |
| `POST` | `/predict`       | Makes a single prediction   |
| `POST` | `/predict/batch` | Makes multiple predictions  |

### Health Check

```bash
curl http://localhost:8000/health
```

Example:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## Input Format

The prediction endpoint expects:

```json
{
  "PU_DO": "161_237",
  "trip_distance": 2.5
}
```

Where:

* `PU_DO` is the pickup/drop-off location pair.
* `trip_distance` is the trip distance.
* The model predicts trip duration in minutes.

Input validation is handled by Pydantic. For example, `trip_distance` must be greater than `0` and less than `200`.

---

## Machine Learning Pipeline

The project follows a simple production-oriented ML workflow:

```text
NYC Green Taxi Data
        │
        ▼
Data Loading
        │
        ▼
Feature Engineering
        │
        ├── PU_DO
        └── trip_distance
        │
        ▼
DictVectorizer
        │
        ▼
Random Forest Regressor
        │
        ▼
Model Serialization
        │
        ▼
FastAPI Prediction Service
```

### Features

The model uses:

* **`PU_DO`** — pickup and drop-off location combination
* **`trip_distance`** — trip distance

The categorical location-pair feature is converted into numerical features using `DictVectorizer`.

### Model Artifact

The trained model and feature vectorizer are stored together in:

```text
models/baseline.pkl
```

> **Security warning:** Pickle can execute arbitrary code when loaded. Never load a `.pkl` file that you did not produce or explicitly trust.

---

## Project Structure

```text
mlops-practitioner/
│
├── data/
│   └── raw/
│       └── green_tripdata_2024-01.parquet
│
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.single
│   └── docker-compose.yml
│
├── models/
│   ├── baseline.pkl
│   └── model.onnx
│
├── notebooks/
│   └── 00-baseline.ipynb
│
├── reports/
│   ├── docker_report.txt
│   └── module-1.md
│
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── schemas.py
│       │
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       └── train.py
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_predict.py
│   └── test_serialization.py
│
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```

---

## Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest -v --cov=src/prodml --cov-report=term-missing
```

The project enforces a minimum test coverage of **70%**.

### Lint and Format

```bash
ruff check src tests
```

```bash
black --check src tests
```

### Train

```bash
python -m prodml.train
```

### Export the Model to ONNX

```bash
python -m prodml.export
```

The ONNX model is saved to:

```text
models/model.onnx
```

---

## Docker

Build the production image:

```bash
docker build -f docker/Dockerfile -t prodml-api:0.1.0 .
```

Run the API:

```bash
docker run --rm -p 8000:8000 prodml-api:0.1.0
```

Check the service:

```bash
curl http://localhost:8000/health
```

The production container runs as the non-root user:

```text
appuser
```

Docker Compose is also provided:

```bash
docker compose -f docker/docker-compose.yml up
```

---

## Technology Stack

* **Python**
* **Pandas**
* **Scikit-learn**
* **FastAPI**
* **Pydantic**
* **Pytest**
* **Docker**
* **ONNX**
* **Uvicorn**

---

## Project Goal

The goal of this project is to demonstrate the transition from a machine learning notebook to a **production-oriented ML service** that can be tested, packaged, containerized, and consumed through an API.

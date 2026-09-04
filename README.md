# prodml

Production-oriented machine learning project for predicting NYC Green Taxi trip duration using a Random Forest regression model. The project includes data preprocessing, feature engineering, model training, and a FastAPI prediction service.

## Quick Start

Run these **three commands** from the repository root:

```bash
pip install -e ".[dev]"
```

```bash
python -m prodml.train
```

```bash
uvicorn prodml.api.main:app --host 0.0.0.0 --port 8000
```

Then, in another terminal, make a prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PU_DO":"161_237","trip_distance":2.5}'
```

Example response:

```json
{
  "prediction": 14.32,
  "model_version": "baseline.pkl",
  "correlation_id": "..."
}
```

## Repository Tree

```text
mlops-practitioner/
├── data/
│   └── raw/
│       └── green_tripdata_2024-01.parquet
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.single
│   └── docker-compose.yml
├── models/
│   ├── baseline.pkl
│   └── model.onnx
├── notebooks/
│   └── 00-baseline.ipynb
├── reports/
│   ├── docker_report.txt
│   └── module-1.md
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── schemas.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       └── train.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_predict.py
│   └── test_serialization.py
├── .dockerignore
├── .gitignore
├── pyproject.toml
└── README.md
```

# prodml

Production-oriented machine learning project for predicting NYC Green Taxi trip duration.

## Setup

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
````

Verify that the package is correctly installed:

```bash
python -c "import prodml; print(prodml.__file__)"
```

## Predictor Interface

The prediction interface accepts feature dictionaries:

```python
def predict(self, features: dict[str, Any]) -> float:
    ...
```

## Development Workflow

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Lint and format check

```bash
ruff check src tests && black --check src tests
```

### 3. Run tests

```bash
pytest -v --cov=src/prodml --cov-report=term-missing
```

### 4. Train the model

```bash
python -m prodml.train
```

### 5. Serve the API

```bash
uvicorn prodml.api.main:app --reload --port 8000
```

## Complete Workflow

Run these commands in order:

```bash
pip install -e ".[dev]"
python -c "import prodml; print(prodml.__file__)"
ruff check src tests && black --check src tests
pytest -v --cov=src/prodml --cov-report=term-missing
python -m prodml.train
uvicorn prodml.api.main:app --reload --port 8000
```

````

### Important: one thing to check

Your workflow assumes these files/modules already exist:

```text
src/
└── prodml/
    ├── __init__.py
    ├── train.py
    └── api/
        └── main.py

tests/
````

In particular:

```bash
python -m prodml.train
```

and:

```bash
uvicorn prodml.api.main:app --reload --port 8000
```

will **not work yet** unless `train.py` and `api/main.py` are implemented.

Also, the earlier `pyproject.toml` we created used:

```toml
[project.scripts]
prodml-train = "prodml.train:main"
```

but your assignment specifically wants:

```bash
python -m prodml.train
```

So we should make sure `train.py` contains:

```python
if __name__ == "__main__":
    main()
```

That way both approaches can work:

```bash
prodml-train
```

and:

```bash
python -m prodml.train
```

For your **RandomForestRegressor service**, you can add this to your report/README:

### Serialization Format Comparison

| Format       | Human-readable | Cross-language         | Schema-enforced | Safe to load from an untrusted source      |
| ------------ | -------------- | ---------------------- | --------------- | ------------------------------------------ |
| **JSON**     | ✅ Yes          | ✅ Yes                  | ❌ No*           | ✅ Yes                                      |
| **Protobuf** | ❌ No (binary)  | ✅ Yes                  | ✅ Yes           | ✅ Generally yes*                           |
| **Pickle**   | ❌ No           | ❌ No (Python-specific) | ❌ No            | ❌ **No**                                   |
| **ONNX**     | ❌ No (binary)  | ✅ Yes                  | ✅ Yes           | ⚠️ **Only with validation/trust controls** |

* JSON can have a schema enforced separately, e.g. JSON Schema. Protobuf uses a defined schema, but applications should still validate untrusted input.

### Service format

> **The service serves the RandomForest model as ONNX because ONNX provides a portable, cross-language model format while avoiding the security risks and Python-specific dependency of loading Pickle models.**

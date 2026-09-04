import pickle
import time

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import get_features_and_target, prepare_features


N_ROWS = 500
N_RUNS = 100


def benchmark():
    # Load validation data
    df = load_data(settings.data_path)
    df = prepare_features(df)

    _, validation_df = split_data(
        df,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    validation_dicts, _ = get_features_and_target(validation_df)

    # Load RandomForest from pickle
    with settings.model_path.open("rb") as file:
        vectorizer, rf_model = pickle.load(file)

    # Same 500 rows for both models
    X = vectorizer.transform(validation_dicts[:N_ROWS])
    X_onnx = X.toarray().astype(np.float32)

    # Load ONNX RandomForest
    onnx_path = settings.project_root / "models" / "model.onnx"

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    # Warm-up
    for _ in range(10):
        rf_model.predict(X)
        session.run(None, {input_name: X_onnx})

    # Pickle latency
    pickle_times = []

    for _ in range(N_RUNS):
        start = time.perf_counter()

        rf_model.predict(X)

        pickle_times.append((time.perf_counter() - start) * 1000)

    # ONNX latency
    onnx_times = []

    for _ in range(N_RUNS):
        start = time.perf_counter()

        session.run(
            None,
            {input_name: X_onnx},
        )

        onnx_times.append((time.perf_counter() - start) * 1000)

    # Calculate metrics
    pickle_mean = np.mean(pickle_times)
    pickle_p95 = np.percentile(pickle_times, 95)

    onnx_mean = np.mean(onnx_times)
    onnx_p95 = np.percentile(onnx_times, 95)

    # Build report
    report = f"""
RandomForest Inference Benchmark
=================================

Rows per inference: {N_ROWS}
Benchmark runs:     {N_RUNS}

Results
---------------------------------
Model           Mean (ms)    P95 (ms)
Pickle RF       {pickle_mean:10.3f}  {pickle_p95:10.3f}
ONNX RF         {onnx_mean:10.3f}  {onnx_p95:10.3f}
"""

    # Save report
    report_path = settings.project_root / "reports" / "benchmark.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(report.strip() + "\n")

    print(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    benchmark()

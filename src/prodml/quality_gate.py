import sys

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://localhost:5001"
MODEL_NAME = "ride-duration-predictor"
MAX_REGRESSION = 0.05


def get_mae_from_run(run_id: str) -> float:
    client = MlflowClient()
    run = client.get_run(run_id)

    mae = run.data.metrics.get("mae")

    if mae is None:
        raise ValueError(f"Run {run_id} does not contain an 'mae' metric.")

    return float(mae)


def get_production_mae() -> float:
    client = MlflowClient()

    versions = client.get_latest_versions(
        MODEL_NAME,
        stages=["Production"],
    )

    if not versions:
        raise ValueError(f"No Production version found for model '{MODEL_NAME}'.")

    production_version = versions[0]
    return get_mae_from_run(production_version.run_id)


def check_quality(new_run_id: str) -> None:
    new_mae = get_mae_from_run(new_run_id)
    production_mae = get_production_mae()

    allowed_mae = production_mae * (1 + MAX_REGRESSION)

    print(f"New model MAE:        {new_mae:.4f}")
    print(f"Production model MAE: {production_mae:.4f}")
    print(f"Allowed MAE:          {allowed_mae:.4f}")

    if new_mae > allowed_mae:
        regression = ((new_mae - production_mae) / production_mae) * 100

        print(f"QUALITY GATE FAILED: MAE regressed by " f"{regression:.2f}% (> 5%).")

        sys.exit(1)

    improvement = ((production_mae - new_mae) / production_mae) * 100

    print(
        f"QUALITY GATE PASSED: MAE change is "
        f"{improvement:.2f}% compared with Production."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m prodml.quality_gate <new_run_id>")
        sys.exit(2)

    mlflow.set_tracking_uri(TRACKING_URI)

    check_quality(sys.argv[1])

import logging
import pickle
from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestRegressor

from prodml.config import settings
from prodml.logging_conf import configure_logging

logger = logging.getLogger(__name__)


def export_random_forest(
    model_path: Path,
    output_path: Path,
) -> None:
    """Export only the Random Forest model to ONNX."""

    with model_path.open("rb") as file:
        _, model = pickle.load(file)

    if not isinstance(model, RandomForestRegressor):
        raise TypeError("Expected RandomForestRegressor.")

    n_features = model.n_features_in_

    initial_types = [
        (
            "float_input",
            FloatTensorType([None, n_features]),
        )
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("wb") as file:
        file.write(onnx_model.SerializeToString())

    logger.info(
        "Random Forest exported to ONNX",
        extra={
            "output_path": str(output_path),
            "n_features": n_features,
        },
    )


def main() -> None:
    """Export the trained Random Forest model."""

    configure_logging()

    output_path = settings.project_root / "models" / "model.onnx"

    export_random_forest(
        model_path=settings.model_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

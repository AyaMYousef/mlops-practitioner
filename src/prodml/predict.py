import logging
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

logger = logging.getLogger(__name__)

T = TypeVar("T")


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Log the execution time of a function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start

        logger.info(
            "%s executed",
            func.__name__,
            extra={"latency_seconds": elapsed},
        )

        return result

    return wrapper


class DurationPredictor:
    """Predict taxi trip duration."""

    def __init__(
        self,
        vectorizer: DictVectorizer,
        model: RandomForestRegressor,
    ) -> None:
        self.vectorizer = vectorizer
        self.model = model

    @classmethod
    def load(cls, path: Path) -> "DurationPredictor":
        """Load a trained predictor from disk."""
        try:
            with path.open("rb") as file:
                vectorizer, model = pickle.load(file)

            logger.info("Model loaded successfully")
            return cls(vectorizer, model)

        except Exception:
            logger.exception("Model load failure")
            raise

    @timed
    def predict_one(
        self,
        features: dict[str, Any],
    ) -> float:
        """Predict duration for one trip."""

        logger.debug(
            "Feature vector received",
            extra={"features": features},
        )

        trip_distance = features.get("trip_distance")

        if trip_distance is not None and trip_distance > 100:
            logger.warning(
                "Input outside training range",
                extra={"trip_distance": trip_distance},
            )

        start = time.perf_counter()

        X = self.vectorizer.transform([features])
        prediction = self.model.predict(X)[0]

        latency = time.perf_counter() - start

        logger.info(
            "Prediction served",
            extra={"latency_seconds": latency},
        )

        return float(prediction)

    def predict_batch(
        self,
        features: list[dict[str, Any]],
    ) -> list[float]:
        """Predict duration for multiple trips."""

        X = self.vectorizer.transform(features)
        predictions = self.model.predict(X)

        return predictions.tolist()

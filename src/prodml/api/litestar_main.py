import logging
import time

from litestar import Litestar, post

from prodml.api.schemas import PredictionRequest, PredictionResponse
from prodml.config import settings
from prodml.predict import DurationPredictor

logger = logging.getLogger(__name__)

predictor = DurationPredictor.load(settings.model_path)


@post("/predict")
def predict(data: PredictionRequest) -> PredictionResponse:
    """Predict taxi trip duration."""

    start = time.perf_counter()

    prediction = predictor.predict_one(
        {
            "PU_DO": data.PU_DO,
            "trip_distance": data.trip_distance,
        }
    )

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Prediction completed",
        extra={
            "prediction": prediction,
            "latency_ms": latency_ms,
        },
    )

    return PredictionResponse(
        prediction=prediction,
        model_version=settings.model_path.stem,
        correlation_id="-",
        latency_ms=latency_ms,
    )


app = Litestar(
    route_handlers=[predict],
)

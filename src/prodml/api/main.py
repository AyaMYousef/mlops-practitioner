import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.config import settings
from prodml.logging_conf import configure_logging, correlation_id
from prodml.predict import DurationPredictor


configure_logging()

logger = logging.getLogger(__name__)

predictor: DurationPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the application starts."""

    global predictor

    logger.info("Application startup")

    try:
        predictor = DurationPredictor.load(settings.model_path)

        logger.info(
            "Model loaded successfully",
            extra={
                "model_path": str(settings.model_path),
            },
        )

        yield

    except Exception:
        logger.exception("Model load failure")
        raise

    finally:
        logger.info("Application shutdown")
        predictor = None


app = FastAPI(
    title="prodml API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(
    request: Request,
    call_next,
):
    """Attach a unique correlation ID to every request."""

    request_id = str(uuid.uuid4())

    token = correlation_id.set(request_id)

    logger.info(
        "Request received",
        extra={
            "method": request.method,
            "path": request.url.path,
        },
    )

    try:
        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = request_id

        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
            },
        )

        return response

    finally:
        correlation_id.reset(token)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Return a clean 422 response for invalid requests."""

    logger.warning(
        "Request validation failed",
        extra={
            "path": request.url.path,
            "errors": exc.errors(),
        },
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "message": "The request contains invalid or missing fields.",
            "details": exc.errors(),
            "correlation_id": correlation_id.get(),
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
):
    """Log unexpected errors with traceback without exposing details."""

    logger.exception(
        "Unexpected application error",
        extra={
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred.",
            "correlation_id": correlation_id.get(),
        },
    )


def get_model_version() -> str:
    """Return the model version from the artifact filename."""

    return settings.model_path.stem


def get_artifact_hash() -> str:
    """Calculate SHA-256 hash of the model artifact."""

    sha256 = hashlib.sha256()

    with settings.model_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_training_date() -> str:
    """Return the model artifact modification date."""

    timestamp = settings.model_path.stat().st_mtime

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc,
    ).isoformat()


@app.get("/health")
def health():
    """Return healthy only when the model is loaded in memory."""

    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
            },
        )

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.get("/metadata")
def metadata():
    """Return metadata describing the loaded model."""

    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model is not loaded.",
            },
        )

    return {
        "model_version": get_model_version(),
        "training_date": get_training_date(),
        "feature_names": predictor.vectorizer.feature_names_,
        "framework": "scikit-learn",
        "model_type": "RandomForestRegressor",
        "artifact_hash": get_artifact_hash(),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):
    """Predict duration for one trip."""

    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model is not loaded.",
            },
        )

    request_id = correlation_id.get()

    features = {
        "PU_DO": request.PU_DO,
        "trip_distance": request.trip_distance,
    }

    start = time.perf_counter()

    prediction = predictor.predict_one(features)

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
        model_version=get_model_version(),
        correlation_id=request_id,
        latency_ms=latency_ms,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(request: BatchPredictionRequest):
    """Predict duration for multiple trips."""

    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model is not loaded.",
            },
        )

    request_id = correlation_id.get()

    features = [
        {
            "PU_DO": item.PU_DO,
            "trip_distance": item.trip_distance,
        }
        for item in request.requests
    ]

    start = time.perf_counter()

    predictions = predictor.predict_batch(features)

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Batch prediction completed",
        extra={
            "prediction_count": len(predictions),
            "latency_ms": latency_ms,
        },
    )

    responses = [
        PredictionResponse(
            prediction=prediction,
            model_version=get_model_version(),
            correlation_id=request_id,
            latency_ms=latency_ms,
        )
        for prediction in predictions
    ]

    return BatchPredictionResponse(
        predictions=responses,
    )

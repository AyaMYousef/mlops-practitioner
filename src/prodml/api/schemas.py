from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Request schema for a single prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "PU_DO": "74_42",
                "trip_distance": 3.2,
            }
        }
    )

    PU_DO: str = Field(
        ...,
        description="Pickup and dropoff location pair.",
        examples=["74_42"],
    )

    trip_distance: float = Field(
        gt=0,
        lt=200,
        description="Trip distance in miles.",
        examples=[3.2],
    )


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "requests": [
                    {
                        "PU_DO": "74_42",
                        "trip_distance": 3.2,
                    },
                    {
                        "PU_DO": "42_74",
                        "trip_distance": 5.7,
                    },
                ]
            }
        }
    )

    requests: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[PredictionResponse]

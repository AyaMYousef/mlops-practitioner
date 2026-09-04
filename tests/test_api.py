from unittest.mock import patch

from prodml.config import settings


def test_health(client):
    """Health endpoint confirms model is loaded."""

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_predict_happy_path(
    client,
    sample_features,
):
    """Predict endpoint returns a valid prediction."""

    response = client.post(
        "/predict",
        json=sample_features,
    )

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert "model_version" in body
    assert "correlation_id" in body
    assert "latency_ms" in body

    assert isinstance(body["prediction"], float)
    assert isinstance(body["correlation_id"], str)
    assert isinstance(body["latency_ms"], float)

    assert body["model_version"] == settings.model_path.stem


def test_predict_invalid_distance(client):
    """Negative distance should return a readable 422."""

    response = client.post(
        "/predict",
        json={
            "PU_DO": "74_42",
            "trip_distance": -5,
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert body["error"] == "Validation error"
    assert "trip_distance" in str(body["details"])


def test_predict_response_has_correlation_header(
    client,
    sample_features,
):
    """Correlation ID should appear in the response header."""

    response = client.post(
        "/predict",
        json=sample_features,
    )

    assert response.status_code == 200

    body = response.json()

    assert response.headers["X-Correlation-ID"] == body["correlation_id"]
    assert response.headers["X-Request-ID"] == body["correlation_id"]


def test_predict_uses_predictor(
    client,
    sample_features,
):
    """API delegates prediction to the predictor."""

    with patch(
        "prodml.api.main.predictor.predict_one",
        return_value=10.5,
    ) as mock_predict:

        response = client.post(
            "/predict",
            json=sample_features,
        )

    assert response.status_code == 200

    mock_predict.assert_called_once_with(
        sample_features,
    )

    assert response.json()["prediction"] == 10.5


def test_batch_prediction(client):
    """Batch endpoint returns one prediction per input."""

    response = client.post(
        "/predict/batch",
        json={
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
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "predictions" in body
    assert len(body["predictions"]) == 2

    for prediction in body["predictions"]:
        assert "prediction" in prediction
        assert "model_version" in prediction
        assert "correlation_id" in prediction
        assert "latency_ms" in prediction

import pandas as pd
import pytest

from prodml.features import prepare_features


@pytest.mark.parametrize(
    "pickup,dropoff,distance",
    [
        (1, 2, 0.0),  # zero distance
        (None, 2, 3.2),  # missing pickup category
        (74, 999, 3.2),  # unseen PU_DO pair
    ],
)
def test_feature_edge_cases(
    pickup,
    dropoff,
    distance,
):
    """Feature engineering handles edge-case inputs."""

    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": pd.to_datetime(["2024-01-01 10:00:00"]),
            "lpep_dropoff_datetime": pd.to_datetime(["2024-01-01 10:10:00"]),
            "PULocationID": [pickup],
            "DOLocationID": [dropoff],
            "trip_distance": [distance],
        }
    )

    result = prepare_features(df)

    assert "duration" in result.columns
    assert "PU_DO" in result.columns


def test_duration_is_calculated():
    """Duration is calculated in minutes."""

    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": pd.to_datetime(["2024-01-01 10:00:00"]),
            "lpep_dropoff_datetime": pd.to_datetime(["2024-01-01 10:15:00"]),
            "PULocationID": [74],
            "DOLocationID": [42],
            "trip_distance": [3.2],
        }
    )

    result = prepare_features(df)

    assert result["duration"].iloc[0] == 15.0


def test_invalid_duration_is_removed():
    """Trips outside the 1-60 minute range are removed."""

    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-01 10:00:00",
                ]
            ),
            "lpep_dropoff_datetime": pd.to_datetime(
                [
                    "2024-01-01 10:00:30",
                    "2024-01-01 12:00:00",
                ]
            ),
            "PULocationID": [74, 74],
            "DOLocationID": [42, 42],
            "trip_distance": [1.0, 3.0],
        }
    )

    result = prepare_features(df)

    assert len(result) == 0

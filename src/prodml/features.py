import pandas as pd


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create target and engineered features for taxi duration prediction."""
    df = df.copy()

    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]

    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)

    return df


def get_features_and_target(
    df: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.Series]:
    """Return model features as dictionaries and the target."""
    features = ["PU_DO", "trip_distance"]

    X = df[features].to_dict(orient="records")
    y = df["duration"]

    return X, y

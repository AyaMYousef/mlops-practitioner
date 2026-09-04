from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: Path) -> pd.DataFrame:
    """Load taxi trip data from a Parquet file."""
    return pd.read_parquet(path)


def split_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into training and validation sets."""
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    return train_df, val_df

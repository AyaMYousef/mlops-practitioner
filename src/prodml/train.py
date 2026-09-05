import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import get_features_and_target, prepare_features


def train_model() -> None:
    """Train the baseline Random Forest model and persist it."""

    df = load_data(settings.data_path)

    df = prepare_features(df)

    train_df, _ = split_data(
        df,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    train_dicts, y_train = get_features_and_target(train_df)

    vectorizer = DictVectorizer()

    X_train = vectorizer.fit_transform(train_dicts)

    model = RandomForestRegressor(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        random_state=settings.random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    settings.model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with settings.model_path.open("wb") as f:
        pickle.dump(
            (vectorizer, model),
            f,
        )

    print(f"Model saved to {settings.model_path}")


def main() -> None:
    """Run the training pipeline."""
    train_model()


if __name__ == "__main__":
    main()

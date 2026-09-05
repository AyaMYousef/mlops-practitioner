def test_prediction_returns_float(
    trained_model,
    sample_features,
):
    """Prediction should return a float."""

    vectorizer, model = trained_model

    X = vectorizer.transform([sample_features])

    prediction = model.predict(X)[0]

    assert isinstance(float(prediction), float)


def test_prediction_is_sane(
    trained_model,
    sample_features,
):
    """Prediction should be within a reasonable duration range."""

    vectorizer, model = trained_model

    X = vectorizer.transform([sample_features])

    prediction = float(model.predict(X)[0])

    assert 1 <= prediction <= 60


def test_prediction_is_deterministic(
    trained_model,
    sample_features,
):
    """Same input should produce the same prediction."""

    vectorizer, model = trained_model

    X = vectorizer.transform([sample_features])

    prediction_1 = float(model.predict(X)[0])
    prediction_2 = float(model.predict(X)[0])

    assert prediction_1 == prediction_2

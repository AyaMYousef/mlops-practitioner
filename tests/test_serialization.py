import pickle

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import (
    get_features_and_target,
    prepare_features,
)


def test_pickle_onnx_parity():
    """Pickle and ONNX models should produce equivalent predictions."""

    with settings.model_path.open("rb") as file:
        vectorizer, model = pickle.load(file)

    df = load_data(settings.data_path)
    df = prepare_features(df)

    _, validation_df = split_data(
        df,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    validation_dicts, _ = get_features_and_target(validation_df)

    X_validation = vectorizer.transform(validation_dicts)

    X_500 = X_validation[:500]

    assert X_500.shape[0] == 500

    pred_pkl = model.predict(X_500)

    onnx_path = settings.project_root / "models" / "model.onnx"

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    pred_onnx = session.run(
        None,
        {input_name: X_500.toarray().astype(np.float32)},
    )[0]

    pred_onnx = np.asarray(pred_onnx).reshape(-1)

    assert np.allclose(
        pred_pkl,
        pred_onnx,
        atol=1e-4,
    )

import pickle

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import get_features_and_target, prepare_features


def test_onnx_parity():
    # Load the pickle model
    with settings.model_path.open("rb") as file:
        vectorizer, model = pickle.load(file)

    # Load and prepare the same data used for training
    df = load_data(settings.data_path)
    df = prepare_features(df)

    # Get validation/test data
    _, validation_df = split_data(
        df,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    # Prepare validation features
    validation_dicts, _ = get_features_and_target(validation_df)

    # Use the same vectorizer that was fitted during training
    X_validation = vectorizer.transform(validation_dicts)

    # Use exactly 500 validation rows
    X_500 = X_validation[:500]

    # Predictions from the pickle model
    pred_pkl = model.predict(X_500)

    # Load ONNX model
    onnx_path = settings.project_root / "models" / "model.onnx"

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    # ONNX prediction
    pred_onnx = session.run(
        None,
        {input_name: X_500.toarray().astype(np.float32)},
    )[0]

    # Flatten ONNX output
    pred_onnx = np.asarray(pred_onnx).reshape(-1)

    # Compare predictions
    assert np.allclose(
        pred_pkl,
        pred_onnx,
        atol=1e-4,
    )

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import CHURN_THRESHOLD, MODELS_DIR
from src.models.mlp import ChurnMLP, build_model

_DEFAULT_PREPROCESSOR = MODELS_DIR / "preprocessor.joblib"
_DEFAULT_MODEL = MODELS_DIR / "churn_mlp.pt"


def load_artifacts(
    preprocessor_path=None,
    model_path=None,
) -> tuple:
    preprocessor = joblib.load(preprocessor_path or _DEFAULT_PREPROCESSOR)
    model = build_model()
    model.load_state_dict(
        torch.load(model_path or _DEFAULT_MODEL, map_location="cpu", weights_only=True)
    )
    model.eval()
    return preprocessor, model


def predict_proba(
    X: pd.DataFrame,
    preprocessor,
    model: ChurnMLP,
) -> np.ndarray:
    X_t = preprocessor.transform(X)
    with torch.no_grad():
        logits = model(torch.tensor(X_t, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def predict(
    X: pd.DataFrame,
    preprocessor,
    model: ChurnMLP,
    *,
    threshold: float = CHURN_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    probs = predict_proba(X, preprocessor, model)
    labels = (probs >= threshold).astype(int)
    return probs, labels

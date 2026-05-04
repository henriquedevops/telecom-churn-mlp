import logging

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, RANDOM_SEED
from src.features.pipeline import build_preprocessor

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "telecom-churn-baselines"


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_baseline(
    name: str,
    estimator,
    X,
    y,
    *,
    extra_params: dict | None = None,
) -> dict:
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    metrics = _compute_metrics(y, y_prob)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("random_state", RANDOM_SEED)
        if extra_params:
            mlflow.log_params(extra_params)
        mlflow.log_metrics({k: round(v, 4) for k, v in metrics.items()})
        pipe.fit(X, y)
        mlflow.sklearn.log_model(pipe, "model")

    logger.info(
        "%s — AUC-ROC: %.4f | PR-AUC: %.4f | F1: %.4f",
        name,
        metrics["auc_roc"],
        metrics["pr_auc"],
        metrics["f1"],
    )
    return {"name": name, "y_prob": y_prob, **metrics}


def train_all(X, y) -> list[dict]:
    return [
        run_baseline(
            "dummy_most_frequent",
            DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
            X,
            y,
            extra_params={"strategy": "most_frequent"},
        ),
        run_baseline(
            "logistic_regression",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
            X,
            y,
            extra_params={"class_weight": "balanced", "max_iter": 1000},
        ),
    ]

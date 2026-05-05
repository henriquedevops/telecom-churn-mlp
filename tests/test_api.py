import pytest
from fastapi.testclient import TestClient

from src.config import CHURN_THRESHOLD, MODELS_DIR

_ARTIFACTS_EXIST = (MODELS_DIR / "churn_mlp.pt").exists() and (
    MODELS_DIR / "preprocessor.joblib"
).exists()

VALID_PAYLOAD = {
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0,
    "gender": "Female",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


@pytest.fixture(scope="module")
def client():
    if not _ARTIFACTS_EXIST:
        pytest.skip("Requer artefatos treinados (churn_mlp.pt e preprocessor.joblib)")
    from src.api.app import app

    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in (0, 1)
    assert data["threshold"] == CHURN_THRESHOLD


def test_predict_invalid_payload_returns_422(client):
    r = client.post("/predict", json={"SeniorCitizen": 9, "tenure": -5})
    assert r.status_code == 422

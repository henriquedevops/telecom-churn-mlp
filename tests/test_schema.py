import pandas as pd
import pytest

from src.data.loader import _SCHEMA


def _valid_row() -> dict:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
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
        "MonthlyCharges": 65.0,
        "TotalCharges": 780.0,
        "Churn": "No",
    }


def test_schema_accepts_valid_row():
    df = pd.DataFrame([_valid_row()])
    result = _SCHEMA.validate(df)
    assert len(result) == 1


def test_schema_rejects_invalid_gender():
    row = _valid_row()
    row["gender"] = "Unknown"
    with pytest.raises(Exception):
        _SCHEMA.validate(pd.DataFrame([row]))


def test_schema_rejects_negative_tenure():
    row = _valid_row()
    row["tenure"] = -1
    with pytest.raises(Exception):
        _SCHEMA.validate(pd.DataFrame([row]))

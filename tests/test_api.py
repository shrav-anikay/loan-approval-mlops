import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_api_health_endpoint():
    """Test 7: Verify GET /health returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["service"] == "loan-approval-api"
    assert json_data["model_loaded"] is True

def test_api_prediction_endpoint():
    """Test 8: Verify POST /predict accepts valid payload and returns loan approval decision."""
    payload = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 6000.0,
        "CoapplicantIncome": 2500.0,
        "LoanAmount": 180.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert json_data["prediction"] in [0, 1]
    assert json_data["approval_status"] in ["Approved", "Rejected"]
    assert "confidence" in json_data
    assert json_data["applicant_income"] == 6000.0

def test_invalid_api_input():
    """Test 9: Verify POST /predict rejects invalid payload (e.g. negative income) with 422 error."""
    invalid_payload = {
        "ApplicantIncome": -500.0,  # invalid (must be > 0)
        "LoanAmount": 150.0,
        "Credit_History": 1.0
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

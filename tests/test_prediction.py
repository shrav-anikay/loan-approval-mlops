import os
import pytest
import joblib
from src.prediction.predict import LoanPredictor

def test_model_artifact_loading():
    """Test 10: Verify saved model and preprocessor artifacts exist and load correctly."""
    model_path = "data/processed/model.joblib"
    preprocessor_path = "data/processed/preprocessor.joblib"

    assert os.path.exists(model_path)
    assert os.path.exists(preprocessor_path)

    predictor = LoanPredictor(model_path=model_path, preprocessor_path=preprocessor_path)
    assert predictor.model is not None
    assert predictor.preprocessor is not None

def test_prediction_output():
    """Test 6: Verify LoanPredictor handles sample data and outputs expected prediction structure."""
    predictor = LoanPredictor(
        model_path="data/processed/model.joblib",
        preprocessor_path="data/processed/preprocessor.joblib"
    )

    sample = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 8000,
        'CoapplicantIncome': 3000,
        'LoanAmount': 200,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Semiurban'
    }

    pred_class, status, confidence = predictor.predict(sample)
    assert pred_class in [0, 1]
    assert status in ["Approved", "Rejected"]
    assert 0.0 <= confidence <= 1.0

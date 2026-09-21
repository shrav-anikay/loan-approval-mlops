import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from src.utils.logger import get_logger

logger = get_logger("model_prediction")

class LoanPredictor:
    def __init__(
        self,
        model_path: str = "data/processed/model.joblib",
        preprocessor_path: str = "data/processed/preprocessor.joblib"
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        logger.info("Successfully loaded model and preprocessor artifacts.")

    def predict(self, applicant_data: Dict[str, Any]) -> Tuple[int, str, float]:
        """
        Accepts a dictionary of applicant features, applies preprocessing,
        returns prediction (1/0), human-readable status, and confidence score.
        """
        df = pd.DataFrame([applicant_data])
        X_trans = self.preprocessor.transform(df)

        pred_class = int(self.model.predict(X_trans)[0])
        status = "Approved" if pred_class == 1 else "Rejected"

        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_trans)[0]
            confidence = float(probs[pred_class])

        return pred_class, status, round(confidence, 4)

if __name__ == "__main__":
    predictor = LoanPredictor()
    sample_applicant = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 6000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Semiurban'
    }
    pred, status, conf = predictor.predict(sample_applicant)
    print(f"Sample Prediction: {pred} | Status: {status} | Confidence: {conf}")

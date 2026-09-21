import os
import time
import pandas as pd
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from src.prediction.predict import LoanPredictor
from src.utils.logger import get_logger

logger = get_logger("fastapi_app")

# Global Predictor Instance
predictor: Optional[LoanPredictor] = None

def init_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = LoanPredictor(
                model_path="data/processed/model.joblib",
                preprocessor_path="data/processed/preprocessor.joblib"
            )
            logger.info("Model predictor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize model predictor: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_predictor()
    yield

app = FastAPI(
    title="Loan Approval Prediction API",
    description="MLOps FastAPI Service for predicting loan approval decisions with Prometheus metrics instrumentation.",
    version="1.0.0",
    lifespan=lifespan
)

# Prometheus Metrics Definition
REQUEST_COUNT = Counter(
    "loan_requests_total",
    "Total HTTP requests to Loan API",
    ["method", "endpoint", "status"]
)

LATENCY = Histogram(
    "loan_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

PREDICTION_COUNT = Counter(
    "loan_predictions_total",
    "Total loan predictions performed",
    ["decision"]
)

# Pydantic Schemas
class LoanApplicantRequest(BaseModel):
    Gender: Optional[str] = Field("Male", description="Applicant Gender ('Male'/'Female')")
    Married: Optional[str] = Field("Yes", description="Marital Status ('Yes'/'No')")
    Dependents: Optional[str] = Field("0", description="Number of Dependents ('0','1','2','3+')")
    Education: Optional[str] = Field("Graduate", description="Education level ('Graduate'/'Not Graduate')")
    Self_Employed: Optional[str] = Field("No", description="Employment status ('Yes'/'No')")
    ApplicantIncome: float = Field(..., gt=0, description="Applicant monthly income (must be > 0)")
    CoapplicantIncome: float = Field(0.0, ge=0, description="Co-applicant monthly income (must be >= 0)")
    LoanAmount: float = Field(..., gt=0, description="Loan amount in thousands (must be > 0)")
    Loan_Amount_Term: float = Field(360.0, gt=0, description="Loan term in months")
    Credit_History: float = Field(..., description="Credit history flag (1.0 or 0.0)")
    Property_Area: str = Field("Semiurban", description="Property location ('Urban'/'Semiurban'/'Rural')")

class LoanPredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for Rejected, 1 for Approved")
    approval_status: str = Field(..., description="'Approved' or 'Rejected'")
    confidence: float = Field(..., description="Model prediction probability score")
    applicant_income: float
    loan_amount: float

# Ensure predictor is initialized on import if artifacts exist
init_predictor()

@app.get("/health")
def health_check():
    """Health check endpoint."""
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status="200").inc()
    return {
        "status": "healthy",
        "service": "loan-approval-api",
        "model_loaded": predictor is not None
    }

@app.post("/predict", response_model=LoanPredictionResponse)
def predict_loan(applicant: LoanApplicantRequest):
    """Predict loan approval decision based on applicant profile."""
    start_time = time.time()
    endpoint = "/predict"

    if predictor is None:
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail="Model predictor is not initialized.")

    if applicant.Credit_History not in [0.0, 1.0, 0, 1]:
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status="422").inc()
        raise HTTPException(status_code=422, detail="Credit_History must be 0.0 or 1.0")

    valid_areas = {'Urban', 'Semiurban', 'Rural'}
    if applicant.Property_Area not in valid_areas:
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status="422").inc()
        raise HTTPException(status_code=422, detail=f"Property_Area must be one of {valid_areas}")

    try:
        data_dict = applicant.model_dump()
        pred_class, status, confidence = predictor.predict(data_dict)

        PREDICTION_COUNT.labels(decision=status).inc()
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status="200").inc()
        LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)

        return LoanPredictionResponse(
            prediction=pred_class,
            approval_status=status,
            confidence=confidence,
            applicant_income=applicant.ApplicantIncome,
            loan_amount=applicant.LoanAmount
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status="500").inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

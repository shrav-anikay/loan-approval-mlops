import os
import pytest
from src.training.train_model import train_and_evaluate

def test_model_training():
    """Test 5: Model training evaluates metrics (accuracy > 0.70) and outputs model.joblib artifact."""
    model, metrics = train_and_evaluate(
        train_path="data/processed/train.csv",
        test_path="data/processed/test.csv",
        model_output_path="data/processed/model.joblib",
        model_type="random_forest",
        random_state=42
    )

    assert model is not None
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] > 0.70
    assert os.path.exists("data/processed/model.joblib")

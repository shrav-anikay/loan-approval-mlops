import os
import pytest
import pandas as pd
from src.transformation.transform_data import transform_and_save_data

def test_preprocessing_transformation():
    """Test 4: Transformation pipeline produces valid train/test CSVs and fitted preprocessor artifact."""
    raw_path = "data/raw/loan_data.csv"
    train_df, test_df, preprocessor = transform_and_save_data(
        raw_data_path=raw_path,
        processed_dir="data/processed",
        test_size=0.2,
        random_state=42
    )

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert "target" in train_df.columns
    assert "target" in test_df.columns
    assert not train_df.isnull().values.any()
    assert not test_df.isnull().values.any()
    assert os.path.exists("data/processed/preprocessor.joblib")

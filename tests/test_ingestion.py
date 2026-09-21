import os
import pytest
import pandas as pd
from src.ingestion.load_data import load_raw_data

def test_data_loading():
    """Test 1: Verify raw data loading returns valid DataFrame with expected columns and non-empty rows."""
    data_path = "data/raw/loan_data.csv"
    assert os.path.exists(data_path), f"Raw dataset missing at {data_path}"
    
    df = load_raw_data(data_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert df.shape[1] == 13
    assert "Loan_Status" in df.columns

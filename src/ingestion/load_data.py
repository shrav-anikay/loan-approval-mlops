import os
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("data_ingestion")

def load_raw_data(data_path: str = "data/raw/loan_data.csv") -> pd.DataFrame:
    """
    Ingests the raw loan approval dataset and performs initial data inspection checks.
    """
    if not os.path.exists(data_path):
        err_msg = f"Raw dataset not found at path: {data_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path)

    logger.info(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns present: {list(df.columns)}")

    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    logger.info(f"Total missing values across dataset: {total_missing}")
    if total_missing > 0:
        missing_summary = missing_counts[missing_counts > 0].to_dict()
        logger.info(f"Missing values breakdown: {missing_summary}")

    duplicates = df.duplicated().sum()
    logger.info(f"Duplicate rows identified: {duplicates}")

    target_col = "Loan_Status"
    if target_col in df.columns:
        logger.info(f"Target column '{target_col}' distribution:\n{df[target_col].value_counts(dropna=False)}")
    else:
        logger.warning(f"Target column '{target_col}' not found in dataset!")

    return df

if __name__ == "__main__":
    df = load_raw_data("data/raw/loan_data.csv")
    print(f"\nIngestion check complete. Head:\n{df.head(3)}")

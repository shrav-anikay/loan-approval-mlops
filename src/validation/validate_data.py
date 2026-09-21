import os
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("data_validation")

REQUIRED_COLUMNS = [
    'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
    'Property_Area', 'Loan_Status'
]

NUMERICAL_COLUMNS = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
CATEGORICAL_COLUMNS = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

def validate_schema(df: pd.DataFrame) -> bool:
    """Validates that all required columns are present in the dataset."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        err = f"Schema validation failed! Missing required columns: {missing_cols}"
        logger.error(err)
        raise ValueError(err)
    logger.info("Schema validation passed: All required columns present.")
    return True

def validate_target_column(df: pd.DataFrame, target_col: str = "Loan_Status") -> bool:
    """Validates the target column presence and distinct values."""
    if target_col not in df.columns:
        err = f"Target column '{target_col}' not found!"
        logger.error(err)
        raise ValueError(err)
    
    unique_vals = set(df[target_col].dropna().unique())
    valid_targets = {'Y', 'N', 1, 0}
    if not unique_vals.issubset(valid_targets):
        err = f"Target column validation failed! Invalid values found: {unique_vals}"
        logger.error(err)
        raise ValueError(err)
    
    logger.info(f"Target column validation passed. Unique values: {unique_vals}")
    return True

def validate_missing_values(df: pd.DataFrame, max_missing_pct: float = 0.25) -> bool:
    """Validates that no column exceeds the maximum allowed missing value percentage."""
    missing_pcts = df.isnull().mean()
    high_missing = missing_pcts[missing_pcts > max_missing_pct]
    if not high_missing.empty:
        err = f"Missing value validation failed! Columns exceeding {max_missing_pct*100}% missing: {high_missing.to_dict()}"
        logger.error(err)
        raise ValueError(err)
    logger.info("Missing value ratio validation passed.")
    return True

def validate_duplicates(df: pd.DataFrame) -> bool:
    """Validates that there are no exact duplicate rows."""
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        err = f"Duplicate validation failed! Found {duplicate_count} duplicate rows."
        logger.error(err)
        raise ValueError(err)
    logger.info("Duplicate row validation passed: 0 duplicate rows found.")
    return True

def validate_numerical_ranges(df: pd.DataFrame) -> bool:
    """Validates that numerical values (incomes, loan amounts) are non-negative."""
    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            min_val = df[col].dropna().min()
            if min_val < 0:
                err = f"Numerical bounds validation failed! Column '{col}' contains negative value: {min_val}"
                logger.error(err)
                raise ValueError(err)
    logger.info("Numerical bounds validation passed.")
    return True

def validate_dataset(df: pd.DataFrame) -> bool:
    """Runs all validation checks on the dataset."""
    logger.info("Starting complete data validation suite...")
    validate_schema(df)
    validate_target_column(df)
    validate_missing_values(df)
    validate_duplicates(df)
    validate_numerical_ranges(df)
    logger.info("All data validation checks passed successfully!")
    return True

if __name__ == "__main__":
    raw_path = "data/raw/loan_data.csv"
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        validate_dataset(df)
    else:
        logger.error(f"Cannot run validation. File not found: {raw_path}")

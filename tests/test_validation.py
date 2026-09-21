import pytest
import pandas as pd
from src.validation.validate_data import (
    validate_schema,
    validate_target_column,
    validate_missing_values,
    validate_duplicates,
    validate_numerical_ranges
)

def test_schema_validation():
    """Test 2: Schema validation passes for correct columns and raises ValueError for missing columns."""
    valid_data = pd.DataFrame({
        'Loan_ID': ['LP1'], 'Gender': ['Male'], 'Married': ['Yes'],
        'Dependents': ['0'], 'Education': ['Graduate'], 'Self_Employed': ['No'],
        'ApplicantIncome': [5000], 'CoapplicantIncome': [2000], 'LoanAmount': [150],
        'Loan_Amount_Term': [360], 'Credit_History': [1.0], 'Property_Area': ['Urban'],
        'Loan_Status': ['Y']
    })
    assert validate_schema(valid_data) is True

    invalid_data = valid_data.drop(columns=['ApplicantIncome'])
    with pytest.raises(ValueError):
        validate_schema(invalid_data)

def test_missing_value_validation():
    """Test 3: Missing value validation succeeds under threshold and fails when exceeded."""
    valid_df = pd.DataFrame({'col1': [1, 2, 3, None], 'col2': [10, 20, 30, 40]})
    assert validate_missing_values(valid_df, max_missing_pct=0.30) is True

    invalid_df = pd.DataFrame({'col1': [None, None, None, 4]})
    with pytest.raises(ValueError):
        validate_missing_values(invalid_df, max_missing_pct=0.50)

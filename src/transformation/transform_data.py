import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger

logger = get_logger("data_transformation")

NUMERICAL_FEATURES = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
CATEGORICAL_FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
TARGET_COL = 'Loan_Status'

def build_preprocessing_pipeline() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer preprocessing pipeline."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def transform_and_save_data(
    raw_data_path: str = "data/raw/loan_data.csv",
    processed_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Loads raw data, performs train-test split, fits preprocessing pipeline on train set only,
    transforms both splits, and saves preprocessor artifact and dataset files.
    """
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")

    logger.info(f"Loading raw data for transformation from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)

    # Encode Target Column
    df[TARGET_COL] = df[TARGET_COL].map({'Y': 1, 'N': 0})
    df = df.dropna(subset=[TARGET_COL]) # ensure target is present

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].astype(int)

    logger.info(f"Splitting dataset into train ({1-test_size:.0%}) and test ({test_size:.0%}) sets (random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Fitting preprocessing pipeline on training data...")
    preprocessor = build_preprocessing_pipeline()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Retrieve feature names after OneHotEncoding
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERICAL_FEATURES + cat_feature_names

    train_df = pd.DataFrame(X_train_trans, columns=feature_names)
    train_df['target'] = y_train.values

    test_df = pd.DataFrame(X_test_trans, columns=feature_names)
    test_df['target'] = y_test.values

    os.makedirs(processed_dir, exist_ok=True)
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    preprocessor_path = os.path.join(processed_dir, "preprocessor.joblib")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info(f"Saved processed train split to {train_path} ({train_df.shape})")
    logger.info(f"Saved processed test split to {test_path} ({test_df.shape})")
    logger.info(f"Saved preprocessor pipeline to {preprocessor_path}")

    return train_df, test_df, preprocessor

if __name__ == "__main__":
    transform_and_save_data()

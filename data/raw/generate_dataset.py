import os
import pandas as pd
import numpy as np

def generate_loan_dataset(output_path: str, num_samples: int = 614, random_state: int = 42):
    """Generates a realistic loan approval dataset standard with Kaggle specifications."""
    np.random.seed(random_state)

    loan_ids = [f"LP00{1001 + i}" for i in range(num_samples)]
    genders = np.random.choice(['Male', 'Female'], size=num_samples, p=[0.81, 0.19])
    married = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.65, 0.35])
    dependents = np.random.choice(['0', '1', '2', '3+'], size=num_samples, p=[0.57, 0.17, 0.17, 0.09])
    education = np.random.choice(['Graduate', 'Not Graduate'], size=num_samples, p=[0.78, 0.22])
    self_employed = np.random.choice(['No', 'Yes'], size=num_samples, p=[0.86, 0.14])

    applicant_income = np.random.exponential(scale=5000, size=num_samples) + 1500
    applicant_income = np.clip(applicant_income, 1500, 81000).astype(int)

    coapplicant_income = np.random.choice([0, np.random.exponential(scale=3000)], size=num_samples, p=[0.45, 0.55])
    coapplicant_income = np.clip(coapplicant_income, 0, 41667).astype(int)

    loan_amount = (applicant_income + coapplicant_income) * np.random.uniform(0.015, 0.035, size=num_samples)
    loan_amount = np.clip(loan_amount, 17, 700).astype(int)

    terms = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
    loan_term = np.random.choice(terms, size=num_samples, p=[0.01, 0.01, 0.02, 0.02, 0.03, 0.07, 0.04, 0.02, 0.75, 0.03])

    credit_history = np.random.choice([1.0, 0.0], size=num_samples, p=[0.84, 0.16])
    property_area = np.random.choice(['Semiurban', 'Urban', 'Rural'], size=num_samples, p=[0.38, 0.33, 0.29])

    edu_bonus = np.where(education == 'Graduate', 0.5, 0.0)
    prop_bonus = np.where(property_area == 'Semiurban', 0.4, 0.0)

    score = (
        3.0 * credit_history
        + 0.0001 * (applicant_income + coapplicant_income)
        - 0.005 * loan_amount
        + edu_bonus
        + prop_bonus
        - 1.5
    )
    prob = 1 / (1 + np.exp(-score))
    loan_status = np.where(np.random.uniform(0, 1, size=num_samples) < prob, 'Y', 'N')

    df = pd.DataFrame({
        'Loan_ID': loan_ids,
        'Gender': genders,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area,
        'Loan_Status': loan_status
    })

    # Introduce realistic missing values (~2-5% per column)
    missing_indices = {
        'Gender': np.random.choice(num_samples, size=13, replace=False),
        'Married': np.random.choice(num_samples, size=3, replace=False),
        'Dependents': np.random.choice(num_samples, size=15, replace=False),
        'Self_Employed': np.random.choice(num_samples, size=32, replace=False),
        'LoanAmount': np.random.choice(num_samples, size=22, replace=False),
        'Loan_Amount_Term': np.random.choice(num_samples, size=14, replace=False),
        'Credit_History': np.random.choice(num_samples, size=50, replace=False)
    }

    for col, idxs in missing_indices.items():
        df.loc[idxs, col] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated raw dataset at {output_path} with shape {df.shape}")
    return df

if __name__ == "__main__":
    generate_loan_dataset("c:/Users/shrav/OneDrive/Desktop/vscode/vscode/python/loan-approval-mlops/data/raw/loan_data.csv")

import json
import pandas as pd
import joblib

# ----------- Load Saved Models -------------
logit_model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/one_hot_encoder.pkl")

with open("models/tuned_threshold.json", "r") as f:
    tuned_threshold = json.load(f)["threshold"]

# ----------- Column Orders -------------
num_cols = [
    'person_age', 'person_income', 'person_emp_exp',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length', 'credit_score'
]

cat_cols = [
    'person_gender', 'person_education', 'person_home_ownership',
    'loan_intent', 'previous_loan_defaults_on_file'
]

# ----------- Take CLI Inputs -------------
print("\n=== Loan Approval Prediction ===")
user_data = {}

# Numeric inputs
for col in num_cols:
    user_data[col] = float(input(f"Enter {col}: "))

# Categorical inputs
for col in cat_cols:
    user_data[col] = input(f"Enter {col}: ")

# Create DataFrame
df_input = pd.DataFrame([user_data])

# ----------- Preprocessing -------------
# Scale numeric features (order matters!)
df_input[num_cols] = scaler.transform(df_input[num_cols])

# Encode categorical features (order matters!)
df_cat_encoded = pd.DataFrame(
    encoder.transform(df_input[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols),
    index=df_input.index
)

# Combine scaled nums + encoded cats
X_final = pd.concat([df_input[num_cols], df_cat_encoded], axis=1)

# ----------- Prediction -------------
probability = logit_model.predict_proba(X_final)[:, 1][0]
prediction = int(probability >= tuned_threshold)

# ----------- Output -------------
print(f"\nPredicted probability of loan approval: {probability:.4f}")
print(f"Decision : {'APPROVED' if prediction == 1 else 'DENIED'}")
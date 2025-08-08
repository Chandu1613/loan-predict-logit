# Loan Approval Classification (Logistic Regression)

## 📊 Exploratory Data Analysis (EDA)

We analyzed the dataset to understand feature distributions, relationships, and detect outliers.

### 1. Loan Status Distribution
![Loan Status](artifacts/loan_status_distribution.png)  
The dataset is slightly imbalanced, with more approved loans than rejected ones.

### 2. Categorical Features
Example: **Loan Intent**
![Loan Intent Distribution](artifacts/loan_intent_distribution.png)  
Most loans are for **Education** and **Medical** purposes, with varying approval rates.

### 3. Numerical Features
![Histograms](artifacts/numerical_histograms.png)  
Income, credit score, and loan amount show skewed distributions.

### 4. Correlation Matrix
![Correlation Matrix](artifacts/correlation_matrix.png)  
Credit score and interest rate have a strong negative correlation.

### 5. Outliers

Outliers were detected using the **Z-score method** (threshold = 3).  
Below are boxplots highlighting outliers for each numerical feature.

#### 1. Person Age
![Person Age Outliers](artifacts/person_age_outliers.png)

#### 2. Person Income
![Person Income Outliers](artifacts/person_income_outliers.png)

#### 3. Employment Experience
![Employment Experience Outliers](artifacts/person_emp_exp_outliers.png)

#### 4. Loan Amount
![Loan Amount Outliers](artifacts/loan_amnt_outliers.png)

#### 5. Loan Interest Rate
![Loan Interest Rate Outliers](artifacts/loan_int_rate_outliers.png)

#### 6. Loan Percent Income
![Loan Percent Income Outliers](artifacts/loan_percent_income_outliers.png)

#### 7. Credit History Length
![Credit History Length Outliers](artifacts/cb_person_cred_hist_length_outliers.png)

#### 8. Credit Score
![Credit Score Outliers](artifacts/credit_score_outliers.png)

---

**Summary of Outliers (Count per Feature)**  
See [`artifacts/outlier_report.txt`](artifacts/outlier_report.txt) for the number of detected outliers per numerical column.
---

## 📈 Skewness Handling
We measured skewness for each numerical feature and applied **Yeo-Johnson transformation** to normalize distributions.  
This helps Logistic Regression perform better.

**Before Transformation:**  
Highly skewed distributions were present in `person_income`, `loan_amnt`, and `credit_score`.

**After Transformation:**  
![Histograms After Transformation](artifacts/histograms_after_transformation.png)  
All numerical features now show a more symmetric distribution.

---

## 🚨 Outlier Treatment
We used the **IQR method** to detect and cap outliers for each numerical feature.  
This prevents extreme values from overly influencing the model.

---
## Train-Test Split
   - 80% training, 20% testing.
   - Stratified split to maintain class balance in both sets.

---
## Feature Scaling & Encoding*
   - **StandardScaler** for numerical columns.
   - **OneHotEncoder** for categorical columns (drop first to avoid dummy trap).
   - Applied **only after splitting** to avoid data leakage.

## 🗂 Artifacts
All generated plots and reports are stored in the [`artifacts/`](artifacts/) folder for reference.

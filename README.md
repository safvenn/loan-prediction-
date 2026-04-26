# Loan Prediction Model — Analysis Report

---

## 1. Project Overview

This project builds a binary classification model to predict whether a loan application will be **approved (1)** or **rejected (0)**. Two machine learning algorithms were trained and compared — Logistic Regression and Random Forest Classifier — using the same dataset and preprocessing pipeline.

**Target Variable:** `loan_status` (1 = Approved, 0 = Rejected)

---

## 2. Dataset Summary

| Property | Value |
|---|---|
| Total Records | 5,000 |
| Total Features | 12 |
| Target Classes | 2 (Approved / Rejected) |
| Approved (Class 1) | 3,570 — 71.4% |
| Rejected (Class 0) | 1,430 — 28.6% |
| Train / Test Split | 80% / 20% (4,000 train · 1,000 test) |
| Random State | 42 |

> **Note:** The dataset is moderately imbalanced — approved loans make up ~71% of records. This means a naive classifier that predicts "Approved" every time would already reach ~71% accuracy, so accuracy alone must be interpreted alongside precision, recall, and AUC-ROC.

### Features Used

| Feature | Type | Description |
|---|---|---|
| `person_age` | Numerical | Age of the applicant |
| `person_income` | Numerical | Annual income |
| `person_emp_length` | Numerical | Years of employment |
| `loan_amnt` | Numerical | Requested loan amount |
| `loan_int_rate` | Numerical | Loan interest rate (%) |
| `loan_percent_income` | Numerical | Loan amount as % of income |
| `cb_person_cred_hist_length` | Numerical | Credit history length (years) |
| `person_gender` | Categorical (encoded) | Gender of applicant |
| `person_home_ownership` | Categorical (encoded) | RENT / OWN / MORTGAGE |
| `loan_intent` | Categorical (encoded) | Purpose of loan |
| `person_education` | Categorical (encoded) | Highest education level |
| `previous_loan_defaults_on_file` | Categorical (encoded) | Prior default history |

### Preprocessing

All categorical columns were encoded using **scikit-learn's LabelEncoder** before model training.

---

## 3. Model 1 — Logistic Regression

### Configuration
- Algorithm: Logistic Regression
- Max Iterations: 1,000
- Solver: lbfgs (default)

### Performance

| Metric | Value |
|---|---|
| **Accuracy** | **86.40%** |
| **AUC-ROC** | **0.9340** |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Rejected (0) | 0.76 | 0.72 | 0.74 | 265 |
| Approved (1) | 0.90 | 0.92 | 0.91 | 735 |
| **Weighted Avg** | **0.86** | **0.86** | **0.86** | **1,000** |

### Confusion Matrix

|  | Predicted Rejected | Predicted Approved |
|---|---|---|
| **Actual Rejected** | 190 ✅ | 75 ❌ |
| **Actual Approved** | 61 ❌ | 674 ✅ |

**Interpretation:**
- 190 rejected loans were correctly identified
- 674 approved loans were correctly identified
- 75 rejected loans were wrongly predicted as approved (False Positives — risky for lenders)
- 61 approved loans were wrongly predicted as rejected (False Negatives — lost business)

---

## 4. Model 2 — Random Forest Classifier

### Configuration
- Algorithm: Random Forest (Ensemble of Decision Trees)
- Number of Trees: 100 (default)
- Random State: 42

### Performance

| Metric | Value |
|---|---|
| **Accuracy** | **94.10%** |
| **AUC-ROC** | **0.9821** |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Rejected (0) | 0.93 | 0.84 | 0.88 | 265 |
| Approved (1) | 0.94 | 0.98 | 0.96 | 735 |
| **Weighted Avg** | **0.94** | **0.94** | **0.94** | **1,000** |

### Confusion Matrix

|  | Predicted Rejected | Predicted Approved |
|---|---|---|
| **Actual Rejected** | 222 ✅ | 43 ❌ |
| **Actual Approved** | 16 ❌ | 719 ✅ |

**Interpretation:**
- 222 rejected loans correctly identified (vs 190 by Logistic Regression)
- 719 approved loans correctly identified (vs 674 by Logistic Regression)
- Only 43 false positives — significantly fewer risky misclassifications
- Only 16 false negatives — nearly all approved applicants reached correctly

---

## 5. Model Comparison

| Metric | Logistic Regression | Random Forest | Winner |
|---|---|---|---|
| Accuracy | 86.40% | **94.10%** | 🏆 Random Forest |
| AUC-ROC | 0.9340 | **0.9821** | 🏆 Random Forest |
| Precision (Rejected) | 0.76 | **0.93** | 🏆 Random Forest |
| Recall (Rejected) | 0.72 | **0.84** | 🏆 Random Forest |
| F1-Score (Rejected) | 0.74 | **0.88** | 🏆 Random Forest |
| Precision (Approved) | 0.90 | **0.94** | 🏆 Random Forest |
| Recall (Approved) | 0.92 | **0.98** | 🏆 Random Forest |
| False Positives | 75 | **43** | 🏆 Random Forest |
| False Negatives | 61 | **16** | 🏆 Random Forest |

**Random Forest outperforms Logistic Regression across every metric.**

---

## 6. Feature Importance (Random Forest)

Random Forest ranks features by how much each one contributes to prediction decisions.

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | `person_income` | 38.15% |
| 2 | `previous_loan_defaults_on_file` | 22.71% |
| 3 | `loan_int_rate` | 12.13% |
| 4 | `loan_percent_income` | 10.06% |
| 5 | `loan_amnt` | 3.91% |
| 6 | `person_age` | 3.02% |
| 7 | `cb_person_cred_hist_length` | 2.87% |
| 8 | `person_emp_length` | 2.56% |

**Key Insight:** The top 2 features — income and prior default history — together account for over **60%** of the model's prediction power. Interest rate and loan-to-income ratio are also strong predictors. Demographic features (gender, education, home ownership) contribute minimally compared to financial indicators.

---

## 7. Key Findings

1. **Random Forest is the stronger model** — 7.7 percentage points higher accuracy and an AUC-ROC of 0.98, indicating near-excellent discrimination between approved and rejected loans.

2. **AUC-ROC of 0.98** for Random Forest means it is extremely capable of ranking applicants by their probability of approval, making it reliable for risk scoring.

3. **Financial features dominate** — income, default history, interest rate, and loan-to-income ratio drive the prediction. Demographics contribute very little.

4. **Class imbalance is mild** (71/29) — both models handle it reasonably, but Random Forest is more robust, achieving 84% recall on the minority class (Rejected) versus Logistic Regression's 72%.

5. **Logistic Regression** still achieves solid results (86.4% accuracy, AUC 0.93) and is more interpretable — suitable if explainability is a priority over raw performance.

---

## 8. Conclusion & Recommendation

**Recommended Model: Random Forest Classifier**

The Random Forest model achieves **94.1% accuracy** and an **AUC-ROC of 0.9821**, making it highly reliable for loan approval prediction. It drastically reduces both false positives (risky loan approvals) and false negatives (missed valid applicants) compared to Logistic Regression.

### Next Steps to Improve Further
- **Hyperparameter tuning** — Use GridSearchCV to optimize `n_estimators`, `max_depth`, and `min_samples_split`
- **Handle class imbalance** — Apply SMOTE or `class_weight='balanced'` to improve minority class recall
- **Feature scaling** — Apply StandardScaler to numerical features; this is required to help Logistic Regression converge properly
- **Cross-validation** — Use k-fold CV for more reliable performance estimates
- **Save the model** — Export the trained model using `joblib` for use in the Streamlit app

---

*Report generated from: `loan_prediction.ipynb`*

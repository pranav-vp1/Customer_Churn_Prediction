# 📡 Customer Churn Prediction

A machine learning web application that predicts whether a telecom customer is likely to cancel their subscription, built with **Scikit-learn** and deployed using **Streamlit**.

> Internship Project — Week 12 Deployment | Built by **Pranav V P**

---

## 🧠 Problem Statement

Customer churn is a major concern for businesses in telecom, banking, and SaaS industries. Losing customers leads to revenue loss and increased acquisition costs. This project builds a binary classification model that predicts whether a customer will leave (**Churn: Yes/No**) based on their account, demographic, and service details.

---

## 🎯 Project Objectives

- Explore and understand the telecom customer dataset
- Perform data preprocessing — missing values, duplicates, and outliers
- Conduct Exploratory Data Analysis (EDA)
- Apply encoding techniques for categorical variables and feature scaling
- Train and compare multiple classification algorithms
- Handle class imbalance using SMOTE
- Tune the best model using RandomizedSearchCV
- Deploy the final model as an interactive web app using Streamlit

---

## 📁 Project Structure

```
├── app.py                          # Streamlit web application
├── save_model.py                   # Script to train and save model artifacts
├── model_artifacts.pkl             # Saved model, scaler, encoders (generated)
├── telecom_churn_data.csv          # Dataset (place in project root)
├── customer_churn_prediction.ipynb # Full ML notebook with EDA & training
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 📊 Dataset Features

| Feature | Type | Description |
|---|---|---|
| `SeniorCitizen` | Numeric | Whether the customer is a senior citizen (0/1) |
| `gender` | Categorical | Male / Female |
| `tenure` | Numeric | Number of months with the company |
| `Partner` | Categorical | Has a partner (Yes/No) |
| `Dependents` | Categorical | Has dependents (Yes/No) |
| `PhoneService` | Categorical | Has phone service (Yes/No) |
| `MultipleLines` | Categorical | Has multiple phone lines |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Categorical | Has online security add-on |
| `OnlineBackup` | Categorical | Has online backup add-on |
| `DeviceProtection` | Categorical | Has device protection add-on |
| `TechSupport` | Categorical | Has tech support add-on |
| `StreamingTV` | Categorical | Has streaming TV |
| `StreamingMovies` | Categorical | Has streaming movies |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categorical | Uses paperless billing (Yes/No) |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | Numeric | Monthly amount charged to the customer |
| `TotalCharges` | Numeric | Total amount charged over the customer's tenure |
| **`Churn`** | **Target** | **Yes (1) = Left, No (0) = Stayed** |

---

## ⚙️ ML Pipeline

```
Raw Data
   │
   ▼
Data Cleaning         → Drop customerID, fix TotalCharges, impute nulls, remove duplicates
   │
   ▼
EDA                   → Univariate, bivariate analysis, correlation heatmap
   │
   ▼
Outlier Treatment     → IQR clipping on tenure, MonthlyCharges, TotalCharges
   │
   ▼
Encoding              → Label Encoding (binary cols) + One-Hot Encoding (multi-class cols)
   │
   ▼
Feature Scaling       → StandardScaler (zero mean, unit variance)
   │
   ▼
Train-Test Split      → 80/20 stratified split
   │
   ▼
SMOTE                 → Balance class ratio on training data only (no data leakage)
   │
   ▼
Model Training        → 7 classifiers compared on Accuracy, Precision, Recall, F1, ROC-AUC
   │
   ▼
Hyperparameter Tuning → RandomizedSearchCV (15 iterations, 3-fold CV, scoring='f1')
   │
   ▼
Custom Threshold      → 0.40 (maximises Recall — catching more churners is business-critical)
   │
   ▼
Best Model            → Tuned Gradient Boosting Classifier
```

---

## 🏆 Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| K-Nearest Neighbors | Distance-based |
| Decision Tree | Rule-based |
| Random Forest | Ensemble — bagging |
| Support Vector Classifier (SVC) | Margin-based |
| Gradient Boosting ✅ | Ensemble — boosting **(selected)** |
| XGBoost | Optimised boosting |

**Final Model:** Tuned Gradient Boosting Classifier with a decision threshold of **0.40**, selected for its superior ROC-AUC and Recall balance.

---

## 🔑 Top Churn Drivers

| Feature | Insight |
|---|---|
| Tenure | Short-tenure customers churn the most |
| Monthly Charges | Higher charges correlate with higher churn risk |
| Contract Type | Month-to-month contracts have the highest churn rate |
| Tech Support | Customers without tech support churn significantly more |
| Total Charges | Low total charges indicate newer, higher-risk customers |

---

## 💼 Business Recommendations

- Offer **long-term contract discounts** to new customers
- **Bundle tech support** in base plans to reduce churn
- Flag customers with **tenure < 12 months** for proactive retention calls
- Introduce **loyalty pricing** for customers spending above $70/month

---

## 🚀 Setup & Deployment

### Prerequisites
- Python 3.10+
- `telecom_churn_data.csv` placed in the project root

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the model artifacts (run once)

```bash
python save_model.py
```

This trains the Gradient Boosting model and saves `model_artifacts.pkl`.

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy to Streamlit Cloud (Free Public URL)

1. Push all project files to a **GitHub repository** (include `model_artifacts.pkl`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo and set `app.py` as the main file
4. Click **Deploy** — your app gets a public shareable URL instantly

---

## 📦 Dependencies

| Package | Version |
|---|---|
| streamlit | 1.45.0 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| scikit-learn | 1.6.1 |
| imbalanced-learn | 0.13.0 |
| xgboost | 2.0.3 |
| joblib | 1.5.3 |

> ⚠️ Always regenerate `model_artifacts.pkl` using `save_model.py` inside the same virtual environment. Pickle files are not portable across different library versions.

---

## 📸 App Preview

The app provides:
- Input form for all 19 customer features
- Real-time churn probability score with a visual progress bar
- Clear High / Low risk verdict
- Personalised risk factor breakdown per customer

---

## 👨‍💻 Author

**Pranav V P**  
Internship Project — Machine Learning | Week 12 Deployment  
Built with Scikit-learn, imbalanced-learn & Streamlit
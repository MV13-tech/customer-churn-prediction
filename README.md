# Customer Churn Prediction 📊🔮

An end-to-end Machine Learning project to analyze telecom customer demographic, service usage, and account data to predict customer churn. The project implements data processing, exploratory data analysis (EDA), feature engineering, and evaluation of **Logistic Regression** and **Random Forest Classifier** models.

---

## 📌 Project Overview

Customer churn occurs when customers stop doing business with a company. Predicting customer churn is vital for telecom companies to proactively retain at-risk customers, design targeted retention campaigns, and maximize Customer Lifetime Value (CLTV).

### Key Highlights
- **Dataset**: Telco Customer Churn (`Telco_customer_churn.xlsx` with 7,043 rows & 33 features)
- **Models Used**: Logistic Regression & Random Forest Classifier
- **Achieved Accuracy**:
  - 🌲 **Random Forest**: **93.46%**
  - 📈 **Logistic Regression**: **91.40%**

---

## 📁 Repository Structure

```
customer-churn-prediction/
├── data/
│   └── Telco_customer_churn.xlsx     # Telecom Customer Churn Dataset
├── customer_churn_prediction.ipynb   # Executed Jupyter Notebook with visualizations & outputs
├── train.py                          # Modular Python pipeline script for training & evaluation
├── README.md                         # Project documentation
└── .gitignore                        # Git ignore file
```

---

## 📊 Exploratory Data Analysis (EDA)

The notebook includes visual analysis of key factors influencing churn:
1. **Churn Distribution**: Analysis of churned vs. retained customers.
2. **Contract Type vs Churn**: Highlighting how month-to-month contracts exhibit significantly higher churn rates compared to two-year contracts.
3. **CLTV & Total Charges**: Distribution of Customer Lifetime Value across demographic segments.
4. **Correlation Matrix**: Identification of key feature interactions and correlations.

---

## 🛠️ Machine Learning Pipeline

1. **Data Cleaning**: Handling missing values, removing non-informative columns (`CustomerID`, `Lat Long`, `Churn Reason`).
2. **Feature Encoding**: Categorical variables encoded using `LabelEncoder`.
3. **Feature Scaling**: Numerical feature standardization via `StandardScaler`.
4. **Model Training**:
   - **Logistic Regression**: Baseline linear model trained on scaled data (`max_iter=1000`).
   - **Random Forest**: Ensemble decision tree model (`n_estimators=100`).
5. **Model Evaluation**: Metrics include Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## 📈 Model Performance Summary

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **91.40%** | 0.83 | 0.85 | 0.84 |
| **Random Forest** | **93.46%** | **0.88** | **0.87** | **0.88** |

---

## 🚀 Getting Started

### 1. Prerequisites & Dependencies
Ensure Python 3.8+ is installed. Install required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### 2. Run the Notebook
Launch Jupyter Notebook to view pre-executed cells, plots, and metrics:

```bash
jupyter notebook customer_churn_prediction.ipynb
```

### 3. Run Standalone Script
Execute the standalone pipeline script directly from terminal:

```bash
python train.py
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

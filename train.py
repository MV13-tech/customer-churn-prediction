"""
Customer Churn Prediction - Machine Learning Pipeline
Author: MV13-tech
Description: End-to-end churn prediction using Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

def load_data(filepath):
    print(f"[*] Loading data from {filepath}...")
    df = pd.read_excel(filepath)
    print(f"[*] Dataset shape: {df.shape}")
    return df

def preprocess_data(df):
    print("[*] Preprocessing data...")
    df_model = df.copy()

    # Remove duplicates
    dups = df_model.duplicated().sum()
    if dups > 0:
        df_model.drop_duplicates(inplace=True)
        print(f"    - Removed {dups} duplicate rows.")

    # Drop non-predictive or unique identifier columns
    cols_to_drop = ['CustomerID', 'Lat Long', 'Churn Reason', 'Count', 'State', 'Country', 'Zip Code', 'City']
    existing_drops = [c for c in cols_to_drop if c in df_model.columns]
    df_model.drop(columns=existing_drops, inplace=True)

    # Convert numeric fields
    if 'Total Charges' in df_model.columns:
        df_model['Total Charges'] = pd.to_numeric(df_model['Total Charges'], errors='coerce')

    # Drop missing values
    df_model.dropna(inplace=True)

    # Encode categorical features
    label_encoders = {}
    for col in df_model.select_dtypes(include=['object']).columns:
        if col != 'Churn Label':
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le

    # Features and Target
    X = df_model.drop(columns=['Churn Value', 'Churn Label'], errors='ignore')
    y = df_model['Churn Value']

    return X, y

def train_and_evaluate(X, y):
    print("[*] Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Logistic Regression
    print("\n" + "="*50)
    print(" 1. LOGISTIC REGRESSION MODEL ")
    print("="*50)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"Accuracy Score: {lr_acc * 100:.2f}%\n")
    print("Classification Report:\n", classification_report(y_test, lr_pred))

    # 2. Random Forest Classifier
    print("="*50)
    print(" 2. RANDOM FOREST CLASSIFIER ")
    print("="*50)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Accuracy Score: {rf_acc * 100:.2f}%\n")
    print("Classification Report:\n", classification_report(y_test, rf_pred))

if __name__ == "__main__":
    df = load_data("data/Telco_customer_churn.xlsx")
    X, y = preprocess_data(df)
    train_and_evaluate(X, y)

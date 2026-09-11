# FraudShield — Credit Card Fraud Detection

A machine learning project for real-time credit card transaction fraud detection using **XGBoost, FastAPI, and a web frontend**.

## Features

- XGBoost fraud classification
- Real-time fraud probability
- FastAPI REST API
- Banking-style web interface
- High/Low risk detection

## Project Structure

```text
fraud-detection-project/
├── app_premium.py
├── index_bank.html
├── fraud_detection_model.pkl
├── fraud_threshold.pkl
└── requirements.txt
Dataset
25,000 transactions
37 features
1,250 fraudulent transactions
23,750 legitimate transactions
5% fraud rate
Tech Stack

Python · Pandas · Scikit-learn · XGBoost · FastAPI · HTML · CSS · JavaScript

Run Locally

Install dependencies:
pip install -r requirements.txt

Start the API:

uvicorn app_premium:app --reload

Open:

http://12......
Workflow
Transaction
    ↓
Frontend
    ↓
FastAPI
    ↓
XGBoost Model
    ↓
Fraud Probability
    ↓
FRAUD / LEGITIMATE

This project is intended for demonstration and fraud-screening purposes.

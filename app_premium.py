from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime

app = FastAPI(title="FraudShield Transaction Review", version="2.1")

model = joblib.load("fraud_detection_model.pkl")
threshold = float(joblib.load("fraud_threshold.pkl"))


class UserTransaction(BaseModel):
    transaction_amount: float
    merchant_category: str
    payment_method: str
    billing_country: str
    transaction_country: str
    customer_age: int
    device_type: str
    authentication_method: str
    distance_from_home_km: float
    transaction_hour: int


def build_model_input(t: UserTransaction) -> dict:
    international = int(t.billing_country != t.transaction_country)
    location_mismatch = international

    # These values are neutral defaults for features not collected
    # by the compact user-facing form.
    return {
        "transaction_amount": t.transaction_amount,
        "merchant_category": t.merchant_category,
        "payment_method": t.payment_method,
        "card_present": int(t.payment_method != "digital_wallet"),
        "customer_age": t.customer_age,
        "account_age_days": 500,
        "average_transaction_amount": 80.0,
        "transaction_amount_deviation": max(t.transaction_amount - 80.0, 0.0),
        "billing_country": t.billing_country,
        "transaction_country": t.transaction_country,
        "location_mismatch": location_mismatch,
        "distance_from_home_km": t.distance_from_home_km,
        "international_transaction": international,
        "device_type": t.device_type,
        "device_trust_score": 0.8,
        "new_device": 0,
        "ip_risk_score": 0.2,
        "vpn_or_proxy": 0,
        "failed_authentication_attempts": 0,
        "transactions_last_1h": 1,
        "transactions_last_24h": 5,
        "transactions_last_7d": 15,
        "amount_spent_last_24h": 300.0,
        "unique_merchants_last_24h": 3,
        "merchant_risk_score": 0.2,
        "is_new_merchant": 0,
        "transaction_hour": t.transaction_hour,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_night": int(t.transaction_hour < 6 or t.transaction_hour >= 22),
        "authentication_method": t.authentication_method,
        "authentication_success": 1,
        "cvv_match": 1,
        "previous_fraud_count": 0,
    }


@app.get("/")
def home():
    return FileResponse("index_bank.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "XGBoost",
        "threshold": threshold
    }


@app.post("/predict")
def predict(transaction: UserTransaction):
    try:
        payload = build_model_input(transaction)
        data = pd.DataFrame([payload])

        probability = float(model.predict_proba(data)[0, 1])
        is_fraud = int(probability >= threshold)

        return {
            "prediction": "FRAUD" if is_fraud else "LEGITIMATE",
            "fraud_probability": round(probability * 100, 2),
            "risk_level": "HIGH" if is_fraud else "LOW",
            "analyzed_at": datetime.now().strftime("%d %b %Y, %I:%M:%S %p"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

from api.model_loader import load_model_and_metadata


# ============================================================
# Initialize FastAPI
# ============================================================

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Churn prediction using the best model automatically selected by the pipeline",
    version="1.1.0"
)


# ============================================================
# Load model and metadata
# ============================================================

metadata = load_model_and_metadata()
model = metadata["model"]
explainer = metadata["explainer"]
scaler = metadata["scaler"]
requires_scaling = metadata["requires_scaling"]
feature_columns = metadata["feature_columns"]


# ============================================================
# Input schema (Pydantic)
# ============================================================

class CustomerInput(BaseModel):
    TenureinMonths: float
    MonthlyCharge: float
    TotalCharges: float
    TotalRevenue: float
    TotalServices: float
    EngagementScore: float
    BillingRiskScore: float
    CLTV_Normalized: float
    Cluster: int

# ============================================================
# Fallback for linear models: coef * value as a contribution proxy
# ============================================================

def _prepare_input(customer: CustomerInput) -> np.ndarray:
    X = np.array([getattr(customer, col) for col in feature_columns]).reshape(1, -1)
    return scaler.transform(X) if requires_scaling else X

def _top_drivers(X: np.ndarray) -> list:
    if explainer is not None:
        raw = explainer.shap_values(X)
        # TreeExplainer can return a list (one entry per class) depending on the
        # model/shap version; we keep the positive class (churn=1).
        shap_row = raw[1][0] if isinstance(raw, list) else raw[0]
        idx = np.argsort(np.abs(shap_row))[::-1][:5]
        return [{"feature": feature_columns[i], "impact": float(shap_row[i])} for i in idx]
    contributions = model.coef_[0] * X[0]
    idx = np.argsort(np.abs(contributions))[::-1][:5]
    return [{"feature": feature_columns[i], "impact": float(contributions[i])} for i in idx]

# ============================================================
# Endpoint: health check
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "model_used": metadata["model_name"]}

# ============================================================
# Endpoint: single prediction
# ============================================================

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    X = _prepare_input(customer)
    prob = model.predict_proba(X)[0][1]
    return {
        "model_used": metadata["model_name"],
        "churn_probability": float(prob),
        "top_drivers": _top_drivers(X)
    }


# ============================================================
# Endpoint: batch prediction
# ============================================================

@app.post("/predict_batch")
def predict_batch(customers: List[CustomerInput]):
    results = []
    for customer in customers:
        X = _prepare_input(customer)
        prob = model.predict_proba(X)[0][1]
        results.append({"churn_probability": float(prob), "top_drivers": _top_drivers(X)})
    return {"model_used": metadata["model_name"], "results": results}

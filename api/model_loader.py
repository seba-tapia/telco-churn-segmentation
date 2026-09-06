import joblib
import shap

from src.config import load_config

TREE_BASED_MODELS = {"rf", "xgb"}


def load_model_and_metadata():
    cfg = load_config()
    artifact = joblib.load(cfg["paths"]["model_artifact"])

    model = artifact["model"]
    model_name = artifact["model_name"]

    # TreeExplainer only applies to tree-based models; if log_reg wins,
    # coefficients are used as an importance proxy instead (see main.py/dashboard.py).
    explainer = shap.TreeExplainer(model) if model_name in TREE_BASED_MODELS else None

    return {
        "model": model,
        "model_name": model_name,
        "explainer": explainer,
        "scaler": artifact["scaler"],
        "requires_scaling": artifact["requires_scaling"],
        "feature_columns": artifact["feature_columns"],
    }

import argparse
import os
import json
import joblib

from src.config import load_config
from src.data.load import load_raw_data
from src.data.clean import clean_data
from src.features.engineering import add_features
from src.segmentation.clustering import run_clustering
from src.modeling.train import prepare_data, train_models
from src.modeling.evaluate import evaluate_model
from src.modeling.select import select_best_model


def step_load(cfg):
    print("📥 Loading raw data...")
    df = load_raw_data(cfg)
    print(f"✔ Data loaded: {df.shape[0]} rows")
    return df


def step_clean(df, cfg):
    print("🧹 Cleaning data...")
    df = clean_data(df, cfg)
    if cfg["export"]["save_clean"]:
        df.to_csv(cfg["paths"]["clean_data"], index=False)
        print(f"✔ Clean data saved to {cfg['paths']['clean_data']}")
    return df


def step_features(df, cfg):
    print("🧩 Generating features...")
    df = add_features(df, cfg)
    if cfg["export"]["save_features"]:
        df.to_csv(cfg["paths"]["features_data"], index=False)
        print(f"✔ Features saved to {cfg['paths']['features_data']}")
    return df


def step_segmentation(df, cfg):
    print("🔍 Running clustering...")
    df = run_clustering(df, cfg)
    if cfg["export"]["save_segments"]:
        df.to_csv(cfg["paths"]["segments_data"], index=False)
        print(f"✔ Segments saved to {cfg['paths']['segments_data']}")
    return df


def step_modeling(df, cfg):
    print("🤖 Training models...")
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df, cfg)
    models = train_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, cfg)

    print("\n📊 Model evaluation:")
    for name, (model, X_t) in models.items():
        metrics = evaluate_model(model, X_t, y_test)
        print(f"\nModel: {name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return models, y_test, scaler

def step_export_model(models, y_test, scaler, cfg):
    print("\n💾 Selecting the best model by ROC-AUC...")
    best_name, best_metrics, all_metrics = select_best_model(models, y_test, metric="roc_auc")
    best_model, _ = models[best_name]

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "requires_scaling": best_name == "log_reg",
        "scaler": scaler if best_name == "log_reg" else None,
        "feature_columns": cfg["modeling"]["feature_columns"],
        "decision_threshold": cfg["modeling"]["decision_threshold"],
    }

    os.makedirs(os.path.dirname(cfg["paths"]["model_artifact"]), exist_ok=True)
    joblib.dump(artifact, cfg["paths"]["model_artifact"])

    with open(cfg["paths"]["metrics_report"], "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"✔ Winning model: {best_name} (ROC-AUC={best_metrics['roc_auc']:.4f})")
    print(f"✔ Artifact saved to {cfg['paths']['model_artifact']}")
    print(f"✔ Full metrics for all 3 models saved to {cfg['paths']['metrics_report']}")



def run_all(cfg):
    df = step_load(cfg)
    df = step_clean(df, cfg)
    df = step_features(df, cfg)
    df = step_segmentation(df, cfg)
    models, y_test, scaler = step_modeling(df, cfg)
    step_export_model(models, y_test, scaler, cfg)
    print("\n🎉 Full pipeline executed successfully!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Telco Churn pipeline")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["load", "clean", "features", "segmentation", "modeling", "all"],
        help="Pipeline step to run"
    )

    args = parser.parse_args()
    cfg = load_config()

    if args.step == "load":
        step_load(cfg)

    elif args.step == "clean":
        df = step_load(cfg)
        step_clean(df, cfg)

    elif args.step == "features":
        df = step_load(cfg)
        df = step_clean(df, cfg)
        step_features(df, cfg)

    elif args.step == "segmentation":
        df = step_load(cfg)
        df = step_clean(df, cfg)
        df = step_features(df, cfg)
        step_segmentation(df, cfg)

    elif args.step == "modeling":
        df = step_load(cfg)
        df = step_clean(df, cfg)
        df = step_features(df, cfg)
        df = step_segmentation(df, cfg)
        models, y_test, scaler = step_modeling(df, cfg)
        step_export_model(models, y_test, scaler, cfg)

    elif args.step == "all":
        run_all(cfg)

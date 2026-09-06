from src.modeling.evaluate import evaluate_model


def select_best_model(models, y_test, metric="roc_auc"):
    all_metrics = {
        name: evaluate_model(model, X_t, y_test)
        for name, (model, X_t) in models.items()
    }
    best_name = max(all_metrics, key=lambda n: all_metrics[n][metric])
    return best_name, all_metrics[best_name], all_metrics

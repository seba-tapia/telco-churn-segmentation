from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def prepare_data(df, cfg):
    df["ChurnLabel"] = df["ChurnLabel"].map({"Yes": 1, "No": 0})
    feature_cols = cfg["modeling"]["feature_columns"]
    test_size = cfg["modeling"]["test_size"]
    random_state = cfg["modeling"]["random_state"]

    X = df[feature_cols]
    y = df["ChurnLabel"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, cfg):
    models = {}

    # Logistic Regression
    lr_cfg = cfg["modeling"]["algorithms"]["logistic_regression"]
    lr = LogisticRegression(max_iter=lr_cfg["max_iter"])
    lr.fit(X_train_scaled, y_train)
    models["log_reg"] = (lr, X_test_scaled)

    # Random Forest
    rf_cfg = cfg["modeling"]["algorithms"]["random_forest"]
    rf = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        random_state=rf_cfg["random_state"]
    )
    rf.fit(X_train, y_train)
    models["rf"] = (rf, X_test)

    # XGBoost
    xgb_cfg = cfg["modeling"]["algorithms"]["xgboost"]
    xgb_model = xgb.XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        learning_rate=xgb_cfg["learning_rate"],
        max_depth=xgb_cfg["max_depth"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        random_state=xgb_cfg["random_state"]
    )
    xgb_model.fit(X_train, y_train)
    models["xgb"] = (xgb_model, X_test)

    return models

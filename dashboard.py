import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config
from api.model_loader import load_model_and_metadata


# ============================================================
# Load configuration and data
# ============================================================

@st.cache_resource
def get_model_metadata():
    return load_model_and_metadata()

cfg = load_config()
metadata = get_model_metadata()
model = metadata["model"]
explainer = metadata["explainer"]
feature_columns = metadata["feature_columns"]

df_segments = pd.read_csv(cfg["paths"]["segments_data"])


# ============================================================
# Title
# ============================================================

st.title("📊 Telco Churn Dashboard")
st.caption(f"Model in production: **{metadata['model_name']}**")


# ============================================================
# Section 1: Segments
# ============================================================

st.header("🔍 Segments (Clusters)")

cluster_counts = df_segments["Cluster"].value_counts().sort_index()

st.bar_chart(cluster_counts)

st.write("Customer distribution by segment.")


# ============================================================
# Section 2: PCA
# ============================================================

st.header("🌀 PCA Visualization of Segments")

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    df_segments["PCA1"],
    df_segments["PCA2"],
    c=df_segments["Cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("Customer Segments in PCA Space")

st.pyplot(fig)


# ============================================================
# Section 3: Funnel
# ============================================================

st.header("📥 CustomerStatus Funnel")

funnel_counts = df_segments["CustomerStatus"].value_counts()

st.bar_chart(funnel_counts)

st.write("Funnel stages: Joined → Stayed → Churned")


# ============================================================
# Section 4: Churn drivers (SHAP)
# ============================================================

st.header("🧠 Churn Drivers")
X_input = df_segments[feature_columns].values
if metadata["requires_scaling"]:
    X_input = metadata["scaler"].transform(X_input)

if explainer is not None:
    import shap
    raw = explainer.shap_values(X_input)
    shap_values_for_plot = raw[1] if isinstance(raw, list) else raw
    fig2 = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_for_plot, X_input, feature_names=feature_columns, show=False)
    st.pyplot(fig2)
else:
    st.info(f"The production model ({metadata['model_name']}) is linear — showing coefficients instead of SHAP.")
    st.bar_chart(pd.Series(model.coef_[0], index=feature_columns).sort_values())


# ============================================================
# Section 5: Individual prediction
# ============================================================

st.header("🔮 Individual Churn Prediction")
st.write("Select a customer to see their churn probability:")

customer_index = st.slider("Customer ID (row)", 0, len(df_segments) - 1, 0)
customer = df_segments.iloc[customer_index]

st.write("### Customer data")
st.json(customer.to_dict())

X_single = customer[feature_columns].values.reshape(1, -1)
if metadata["requires_scaling"]:
    X_single = metadata["scaler"].transform(X_single)

prob = model.predict_proba(X_single)[0][1]
st.write(f"### Churn probability: **{prob:.2f}**")

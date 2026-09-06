# Telco Churn — Customer Segmentation & Churn Prediction

End-to-end project that identifies **which telecom customers are most at risk of leaving (churn)**, groups them into **actionable segments**, and exposes that information through an **API** and an **interactive dashboard**.

**Author:** Sebastian Tapia

*(This README is also available in Spanish: [README.es.md](README.es.md))*

---

> **Note:** my professional background is in banks and insurance companies, environments that handle highly confidential information that cannot be shared or used outside those contexts. For that reason, this portfolio project is built entirely on a public dataset (Kaggle), with no data or information coming from my professional activity.

---

## 📌 Business Summary (no technical background required)

This section is written for marketing, customer retention, or anyone on the business side who wants to understand **what this project does and what decisions can be made with it**, without going into technical detail.

### What problem does it solve?

Out of every 10 customers, **about 3 end up leaving (churn)**. This project analyzes customer history (how much they pay, how long they've been customers, which services they use, how they interact with support, etc.) to answer two questions:

1. **What types of customers do we have?** (segmentation)
2. **Which ones are most likely to leave, and why?** (prediction + explanation)

With this, the retention team can **prioritize who to contact first** instead of treating every customer the same way.

### The 4 customer segments we found

| Segment | Who they are | Churn rate | What to do |
|---|---|---|---|
| 🟢 **Loyal low-cost** | Long-tenured customers (4+ years), pay little per month, barely use add-on services | **3%** — almost nobody leaves | No urgent attention needed. They're the stable base of the business. |
| 🔵 **Loyal premium** | Long-tenured customers, pay a lot per month, use several add-on services (security, tech support, streaming) | **13%** | Nurture with exclusive perks — they're the highest-value, already-committed customers. |
| 🟠 **At risk, high spend** | Mid-tenure customers, pay a fair amount, but use few add-on services | **33%** | Retention focus: these are valuable customers who are leaving. Offering services that increase engagement (support, security) may prevent churn. |
| 🔴 **New, low engagement** | Very recent customers (under 1 year), pay a mid-range amount, barely use add-on services | **37%** — the riskiest group | Onboarding focus: strengthen the experience in the first few months (the first 90 days appear to be critical). |

**The most important finding for the business**: it's not how much a customer pays that drives churn the most, but **the combination of short tenure + low use of add-on services**. A new customer who isn't using the extra services (online security, premium tech support, device protection) is the earliest warning sign we have, even before their contract renewal date approaches.

### How reliable is the churn prediction model?

In plain terms: if we take 10 customers who **really are** going to leave, the model correctly catches **between 4 and 7 of them**, depending on how "sensitive" we configure the alert:

- **Conservative setting** (fewer alerts, but more accurate): catches 4 out of 10 who leave, and almost every alert it raises is correct.
- **Sensitive setting** (more alerts, some of which are false alarms): catches 6-7 out of 10 who leave, at the cost of raising more alerts about customers who weren't actually going to leave.

**The decision of which setting to use is a business decision, not a technical one**: it depends on how much it costs to reach out to a customer who wasn't going to leave (a call, a discount) versus how much it costs to lose a customer who was (their entire future value). That decision should be made by the retention team together with the data team — it's documented in detail in `06_Churn_Model.ipynb` for whenever the team wants to move forward on it.

### Honest limitations of this analysis

- The model is trained on historical data from a specific period; if the business changes (new competitors, new plans, etc.), the model should be retrained.
- The model predicts churn probability, it doesn't guarantee anything — it's a prioritization tool, not a crystal ball.
- The original data included a variable ("satisfaction score") that seemed to predict churn almost perfectly, but on closer inspection it turned out to be a consequence of churn rather than a cause (it's measured after the customer has already decided to leave). It was therefore excluded from the model — otherwise, the model would have looked far more accurate than it really is in practice.

---

## 🛠️ Technical Documentation

### Project architecture

```
Raw data (CSV)
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│   Cleaning   │ ──▶ │   Feature    │ ──▶ │  Segmentation  │ ──▶ │   Modeling    │
│  (clean.py)  │     │ Engineering  │     │ (clustering.py)│     │  (train.py)   │
└─────────────┘     └──────────────┘     └───────────────┘     └──────┬───────┘
                                                                        │
                                                              Automatic selection
                                                              of the best model by
                                                              ROC-AUC (select.py)
                                                                        │
                                                                        ▼
                                                          models/churn_model.joblib
                                                                        │
                                              ┌─────────────────────────┴───────────────────────┐
                                              ▼                                                   ▼
                                    REST API (api/main.py)                          Dashboard (dashboard.py)
                                    /predict, /predict_batch                        Segments, PCA, funnel,
                                                                                     drivers, individual prediction
```

The whole pipeline is **config-driven**: paths, columns, hyperparameters, and thresholds live in `config/config.yaml`, not hardcoded in the code.

### Repository structure

```
telco-churn-segmentation/
├── config/
│   └── config.yaml              # Central project configuration
├── data/
│   ├── raw/                     # Original dataset (Telco-Customer-Churn.csv)
│   └── processed/               # Clean, feature-engineered, and segmented data
├── models/
│   ├── churn_model.joblib       # Winning model artifact (+ scaler if applicable)
│   └── metrics_report.json      # Comparative metrics for the 3 evaluated models
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Cleaning.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Segmentation.ipynb
│   ├── 05_Funnel_Analysis.ipynb
│   └── 06_Churn_Model.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── load.py
│   │   └── clean.py
│   ├── features/
│   │   └── engineering.py
│   ├── segmentation/
│   │   └── clustering.py
│   └── modeling/
│       ├── train.py
│       ├── evaluate.py
│       └── select.py
├── api/
│   ├── main.py                  # FastAPI (prediction + drivers)
│   └── model_loader.py          # Loads the trained artifact (does not retrain)
├── dashboard.py                 # Streamlit dashboard
├── run_pipeline.py              # Orchestrator: runs the full pipeline or individual steps
└── requirements.txt
```

### How to run the project

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Run the full pipeline** (cleaning → features → segmentation → training → best-model selection → export)

```bash
python run_pipeline.py --step all
```

It can also be run step by step for debugging:

```bash
python run_pipeline.py --step load           # load only
python run_pipeline.py --step clean          # load + cleaning
python run_pipeline.py --step features       # + feature engineering
python run_pipeline.py --step segmentation   # + clustering
python run_pipeline.py --step modeling       # + training, evaluation, and model export
```

Once it finishes, you'll have `models/churn_model.joblib` (the winning model, automatically selected by ROC-AUC) and `models/metrics_report.json` (metrics for all 3 evaluated models).

**3. Start the API**

```bash
uvicorn api.main:app --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`. Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — confirms the API is alive and which model is loaded |
| `/predict` | POST | Churn prediction + drivers for a single customer |
| `/predict_batch` | POST | Prediction for a list of customers |

**4. Start the dashboard**

```bash
streamlit run dashboard.py
```

Includes: segment distribution, PCA visualization, customer status funnel, model drivers (SHAP or coefficients depending on the winning model), and interactive individual prediction.

### Current technical results

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** (winner) | 0.789 | 0.435 | 0.522 | **0.831** |
| Random Forest | 0.786 | 0.437 | 0.520 | 0.827 |
| XGBoost | 0.781 | 0.463 | 0.529 | 0.820 |

The winning model is automatically selected by ROC-AUC in `src/modeling/select.py`. All three models end up very close to each other because, after removing variables with data leakage, the relationship between features and churn is mostly linear.

**Documented methodological decisions:**
- `SatisfactionScore` was excluded from the model and from clustering because it's a variable with severe data leakage (see the conclusions of `06_Churn_Model.ipynb` and `04_Segmentation.ipynb`).
- `k=4` in clustering does not maximize the silhouette score (which peaks at k=2), but was chosen because it produces segments with more differentiated and business-actionable churn rates (see the conclusions of `04_Segmentation.ipynb`).
- The recommended decision threshold (0.35 instead of the default 0.5) is documented in `06_Churn_Model.ipynb` and is saved in the artifact (`churn_model.joblib`) when the model is exported, but it is not yet read or applied in the API/dashboard responses — this remains a future improvement pending validation with the retention team.

### Possible future improvements

- Read `decision_threshold` from the artifact in `model_loader.py` and apply it in `main.py`/`dashboard.py` (for example, by adding an `at_risk: true/false` field to the API response, computed with that threshold instead of assuming the implicit 0.5 from `predict_proba`). The value is already persisted in `churn_model.joblib`; the last step of reading and using it is still missing.
- Pin exact library versions in `requirements.txt` (`pip freeze`) for guaranteed reproducibility.
- Add automated tests for the pipeline (loading, cleaning, features) and for the API endpoints.
- Periodically retrain with new data and track whether the winning model changes over time (model drift).
- The notebooks (`01`–`06`) duplicate part of the logic in `src/*.py` instead of importing it, in order to keep exploration separate from production code. This already caused one real inconsistency (a fix applied to `clean.py` was not propagated to `02_Cleaning.ipynb`), so it's worth periodically checking that both sides stay in sync, or migrating the notebooks to import directly from `src/`.

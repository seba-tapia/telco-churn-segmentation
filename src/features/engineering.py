from sklearn.preprocessing import MinMaxScaler

def tenure_bucket(months):
    if months < 6:
        return "0-6 months"
    elif months < 12:
        return "6-12 months"
    elif months < 24:
        return "1-2 years"
    elif months < 48:
        return "2-4 years"
    else:
        return "4+ years"

def add_features(df, cfg):

    binary_cols = cfg["features"]["binary_columns"]
    service_cols = cfg["features"]["service_columns"]
    engagement_cols = cfg["features"]["engagement_columns"]

    # Tenure bucket
    df["TenureBucket"] = df["TenureinMonths"].apply(tenure_bucket)

    # TotalServices
    df["TotalServices"] = df[service_cols].apply(lambda r: sum(r == "Yes"), axis=1)

    # EngagementScore
    df["EngagementScore"] = df[engagement_cols].apply(lambda r: sum(r == "Yes"), axis=1)

    # BillingRiskScore
    df["BillingRiskScore"] = (
        df["MonthlyCharge"] * 0.6 +
        df["TotalExtraDataCharges"] * 0.2 +
        df["TotalLongDistanceCharges"] * 0.2
    )

    # CLTV normalized
    scaler = MinMaxScaler()
    df["CLTV_Normalized"] = scaler.fit_transform(df[["CLTV"]])

    # Binary columns
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    return df

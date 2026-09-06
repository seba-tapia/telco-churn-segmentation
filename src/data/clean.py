import pandas as pd

def clean_data(df, cfg):
    numeric_cols = cfg["cleaning"]["numeric_columns"]
    drop_cols = cfg["cleaning"]["drop_columns"]

    # Convert numeric columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute missing values
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Strip categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    # Drop irrelevant columns
    df = df.drop(columns=drop_cols, errors="ignore")

    # Drop duplicates
    df = df.drop_duplicates()

    return df

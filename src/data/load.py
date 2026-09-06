import pandas as pd

def load_raw_data(cfg):
    path = cfg["paths"]["raw_data"]
    return pd.read_csv(path)

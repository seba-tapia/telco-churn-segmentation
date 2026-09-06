import yaml

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

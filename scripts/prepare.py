import yaml
from src.dataio import load_csv

cfg = yaml.safe_load(open("config/data.yaml"))

df = load_csv(cfg["paths"]["raw_csv"])

print(df.head())
print(df.info())
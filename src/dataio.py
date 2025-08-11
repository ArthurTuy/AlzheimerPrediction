from pathlib import Path
import pandas as pd
import yaml

import kagglehub
from kagglehub import KaggleDatasetAdapter


OPCAO_CARREGAMENTO = 2  #1 para csv e 2 para usar a API

# Carrega configs
CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "data.yaml"
cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))

def load_csv(path_csv: str) -> pd.DataFrame:
    if OPCAO_CARREGAMENTO == 1:
        print("[INFO] Carregando do CSV local…")
        p = Path(path_csv)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo CSV não encontrado: {p.resolve()}")
        return pd.read_csv(p)

    elif OPCAO_CARREGAMENTO == 2:
        print("[INFO] Carregando via KaggleHub API…")
        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            cfg["kaggle"]["dataset"],
            cfg["kaggle"]["file_path"]
        )

    else:
        raise ValueError("Opção inválida! Escolha 1 (CSV local) ou 2 (KaggleHub API).")

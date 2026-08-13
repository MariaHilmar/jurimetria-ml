"""Gera um dataset sintético com sinal aprendível.

Não usa dados reais de processos. O rótulo `procedente` é amostrado de um
processo generativo conhecido, só para o ciclo de treino/validação ser
demonstrável em público.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import AREAS_DIREITO, LABEL_COLUMN, TRIBUNAIS

_TRIBUNAL_EFFECT = {
    "TJSP": 1.1,
    "TJRJ": 0.4,
    "TJMG": 0.0,
    "TJRS": -0.35,
    "TJBA": -0.9,
}
_AREA_EFFECT = {
    "consumidor": 1.2,
    "trabalhista": 0.7,
    "civel": 0.0,
    "tributario": -1.1,
}


def _logit(row: dict[str, float | str | int]) -> float:
    valor = float(row["valor_causa"])
    return float(
        -0.2
        + _TRIBUNAL_EFFECT[str(row["tribunal"])]
        + _AREA_EFFECT[str(row["area_direito"])]
        + 0.9 * ((np.log1p(valor) - 9.5) / 2.0)
        + (0.85 if int(row["tem_liminar"]) == 1 else -0.7)
        + (0.55 if int(row["dias_tramitacao"]) > 365 else -0.15)
        + (-0.6 if int(row["grau"]) == 2 else 0.35)
        + 0.25 * (int(row["qtd_movimentacoes"]) / 20.0)
    )


def generate_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    if n_samples < 50:
        raise ValueError("n_samples deve ser pelo menos 50 para o split estratificado.")

    rng = np.random.default_rng(seed)
    tribunais = rng.choice(TRIBUNAIS, size=n_samples)
    areas = rng.choice(AREAS_DIREITO, size=n_samples)
    grau = rng.integers(1, 3, size=n_samples)
    valor_causa = np.clip(
        rng.lognormal(mean=9.2, sigma=1.1, size=n_samples), 800, 2_000_000
    )
    qtd_movimentacoes = rng.poisson(lam=12, size=n_samples).clip(1, 80)
    dias_tramitacao = rng.integers(15, 1800, size=n_samples)
    tem_liminar = rng.integers(0, 2, size=n_samples)

    rows: list[dict[str, float | str | int]] = []
    labels: list[int] = []
    for i in range(n_samples):
        row: dict[str, float | str | int] = {
            "tribunal": str(tribunais[i]),
            "area_direito": str(areas[i]),
            "grau": int(grau[i]),
            "valor_causa": float(valor_causa[i]),
            "qtd_movimentacoes": int(qtd_movimentacoes[i]),
            "dias_tramitacao": int(dias_tramitacao[i]),
            "tem_liminar": int(tem_liminar[i]),
        }
        noise = float(rng.normal(0.0, 0.25))
        probability = 1.0 / (1.0 + np.exp(-(_logit(row) + noise)))
        labels.append(int(rng.random() < probability))
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame[LABEL_COLUMN] = labels
    return frame


def save_dataset(path: Path, n_samples: int = 2000, seed: int = 42) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_dataset(n_samples=n_samples, seed=seed)
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera CSV sintético de processos.")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=Path, default=Path("data/processos_sinteticos.csv")
    )
    args = parser.parse_args()
    destination = save_dataset(args.out, n_samples=args.n_samples, seed=args.seed)
    print(f"dataset gravado em {destination}")


if __name__ == "__main__":
    main()

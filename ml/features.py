"""Contrato de features compartilhado entre treino e predição."""

from __future__ import annotations

CATEGORICAL_COLUMNS = ["tribunal", "area_direito"]
NUMERIC_COLUMNS = [
    "grau",
    "valor_causa",
    "qtd_movimentacoes",
    "dias_tramitacao",
    "tem_liminar",
]
FEATURE_COLUMNS = [*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS]
LABEL_COLUMN = "procedente"

TRIBUNAIS = ("TJSP", "TJRJ", "TJMG", "TJRS", "TJBA")
AREAS_DIREITO = ("civel", "trabalhista", "tributario", "consumidor")

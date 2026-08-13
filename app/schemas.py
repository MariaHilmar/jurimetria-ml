"""Contratos Pydantic da API."""

from typing import Literal

from pydantic import BaseModel, Field

Tribunal = Literal["TJSP", "TJRJ", "TJMG", "TJRS", "TJBA"]
AreaDireito = Literal["civel", "trabalhista", "tributario", "consumidor"]


class ProcessoFeatures(BaseModel):
    tribunal: Tribunal
    area_direito: AreaDireito
    grau: int = Field(ge=1, le=2)
    valor_causa: float = Field(gt=0)
    qtd_movimentacoes: int = Field(ge=0, le=200)
    dias_tramitacao: int = Field(ge=0, le=5000)
    tem_liminar: bool

    def to_row(self) -> dict[str, str | int | float]:
        return {
            "tribunal": self.tribunal,
            "area_direito": self.area_direito,
            "grau": self.grau,
            "valor_causa": self.valor_causa,
            "qtd_movimentacoes": self.qtd_movimentacoes,
            "dias_tramitacao": self.dias_tramitacao,
            "tem_liminar": int(self.tem_liminar),
        }


class PredictResponse(BaseModel):
    probabilidade_procedente: float = Field(ge=0, le=1)
    classe: Literal["procedente", "improcedente"]
    limiar: float
    versao_modelo: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    modelo_carregado: bool
    versao_modelo: str | None = None
    dataset: str = "sintetico"


class MetricsResponse(BaseModel):
    model_id: str
    trained_at: str
    n_samples: int
    n_train: int
    n_test: int
    calibrated: bool
    threshold: float
    positive_rate: float
    feature_columns: list[str]
    metrics: dict[str, float]
    dataset: str
    notes: str

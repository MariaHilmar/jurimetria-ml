"""Carrega o artefato treinado e executa inferência."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import Settings
from app.schemas import ProcessoFeatures
from ml.features import FEATURE_COLUMNS
from ml.train import METRICS_FILENAME, MODEL_FILENAME


class ModelNotLoadedError(RuntimeError):
    """Artefato de modelo ausente ou ilegível."""


class ModelService:
    def __init__(self, artifacts_dir: Path, threshold: float) -> None:
        self._artifacts_dir = artifacts_dir
        self._threshold = threshold
        self.pipeline: Any | None = None
        self.metrics: dict[str, Any] | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> ModelService:
        return cls(settings.artifacts_dir, settings.predict_threshold)

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None and self.metrics is not None

    def load(self) -> None:
        model_path = self._artifacts_dir / MODEL_FILENAME
        metrics_path = self._artifacts_dir / METRICS_FILENAME
        if not model_path.exists() or not metrics_path.exists():
            self.pipeline = None
            self.metrics = None
            return
        self.pipeline = joblib.load(model_path)
        self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    def predict(self, features: ProcessoFeatures) -> dict[str, str | float]:
        if self.pipeline is None or self.metrics is None:
            raise ModelNotLoadedError("Modelo não carregado. Rode python -m ml.train.")

        frame = pd.DataFrame([features.to_row()], columns=FEATURE_COLUMNS)
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        threshold = float(self.metrics.get("threshold", self._threshold))
        label = "procedente" if probability >= threshold else "improcedente"
        return {
            "probabilidade_procedente": probability,
            "classe": label,
            "limiar": threshold,
            "versao_modelo": str(self.metrics["model_id"]),
        }

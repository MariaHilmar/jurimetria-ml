"""API Flask equivalente: mesmos contratos /health, /metrics e /predict.

Evidencia o framework Flask pedido em vagas Python, reusando o ModelService
do recorte sklearn/XGBoost. A API principal do repo continua em FastAPI.
"""

from __future__ import annotations

from flask import Flask, request
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import ProcessoFeatures
from app.service import ModelNotLoadedError, ModelService


def create_flask_app(service: ModelService | None = None) -> Flask:
    settings = get_settings()
    app = Flask("jurimetria-ml-flask")
    model = service or ModelService.from_settings(settings)
    model.load()
    app.config["MODEL_SERVICE"] = model

    @app.get("/health")
    def health() -> tuple[dict[str, object], int]:
        current: ModelService = app.config["MODEL_SERVICE"]
        version = None
        if current.metrics is not None:
            version = str(current.metrics.get("model_id"))
        return {
            "status": "ok" if current.is_ready else "degraded",
            "modelo_carregado": current.is_ready,
            "versao_modelo": version,
            "dataset": "sintetico",
            "runtime": "flask",
        }, 200

    @app.get("/metrics")
    def metrics() -> tuple[dict[str, object], int]:
        current: ModelService = app.config["MODEL_SERVICE"]
        if current.metrics is None:
            return {"detail": "Métricas indisponíveis. Rode python -m ml.train."}, 503
        return current.metrics, 200

    @app.post("/predict")
    def predict() -> tuple[dict[str, object], int]:
        current: ModelService = app.config["MODEL_SERVICE"]
        try:
            payload = ProcessoFeatures.model_validate(
                request.get_json(silent=True) or {}
            )
        except ValidationError as exc:
            return {"detail": exc.errors()}, 422
        try:
            result: dict[str, object] = dict(current.predict(payload))
            return result, 200
        except ModelNotLoadedError as exc:
            return {"detail": str(exc)}, 503

    return app

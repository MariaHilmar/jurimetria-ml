"""API HTTP: health, métricas do último treino e inferência."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas import (
    HealthResponse,
    MetricsResponse,
    PredictResponse,
    ProcessoFeatures,
)
from app.service import ModelNotLoadedError, ModelService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    service: ModelService = app.state.service
    service.load()
    yield


def create_app(service: ModelService | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Jurimetria ML",
        description=(
            "Recorte de portfólio do ciclo clássico de ML: dataset sintético, "
            "treino com XGBoost calibrado e API de predição. "
            "Não é um modelo de desfecho judicial real."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.service = service or ModelService.from_settings(settings)

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(
        _request: Request, exc: ModelNotLoadedError
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current: ModelService = app.state.service
        version = None
        if current.metrics is not None:
            version = str(current.metrics.get("model_id"))
        return HealthResponse(
            status="ok" if current.is_ready else "degraded",
            modelo_carregado=current.is_ready,
            versao_modelo=version,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        current: ModelService = app.state.service
        if current.metrics is None:
            raise HTTPException(
                status_code=503,
                detail="Métricas indisponíveis. Rode python -m ml.train.",
            )
        return MetricsResponse.model_validate(current.metrics)

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: ProcessoFeatures) -> PredictResponse:
        current: ModelService = app.state.service
        result = current.predict(payload)
        return PredictResponse.model_validate(result)

    return app


app = create_app()

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.service import ModelService

VALID_PAYLOAD = {
    "tribunal": "TJSP",
    "area_direito": "consumidor",
    "grau": 1,
    "valor_causa": 25000.0,
    "qtd_movimentacoes": 14,
    "dias_tramitacao": 420,
    "tem_liminar": True,
}


def test_health_and_metrics(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["modelo_carregado"] is True
    assert body["dataset"] == "sintetico"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    payload = metrics.json()
    assert payload["dataset"] == "sintetico"
    assert "roc_auc" in payload["metrics"]
    assert payload["metrics"]["roc_auc"] > 0.65


def test_predict_returns_probability_in_unit_interval(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    probability = body["probabilidade_procedente"]
    assert 0.0 <= probability <= 1.0
    assert body["classe"] in {"procedente", "improcedente"}
    assert body["versao_modelo"]


def test_predict_rejects_invalid_tribunal(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "tribunal": "STF"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_without_model_returns_503(tmp_path: Path) -> None:
    service = ModelService(artifacts_dir=tmp_path, threshold=0.5)
    app = create_app(service=service)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        predict = client.post("/predict", json=VALID_PAYLOAD)
        assert predict.status_code == 503
        metrics = client.get("/metrics")
        assert metrics.status_code == 503

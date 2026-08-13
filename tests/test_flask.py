from __future__ import annotations

from pathlib import Path

from app.flask_app import create_flask_app
from app.service import ModelService
from tests.test_api import VALID_PAYLOAD


def test_flask_health_and_predict(artifacts_dir: Path) -> None:
    service = ModelService(artifacts_dir=artifacts_dir, threshold=0.5)
    app = create_flask_app(service=service)
    client = app.test_client()

    health = client.get("/health")
    assert health.status_code == 200
    body = health.get_json()
    assert body["status"] == "ok"
    assert body["runtime"] == "flask"

    predict = client.post("/predict", json=VALID_PAYLOAD)
    assert predict.status_code == 200
    payload = predict.get_json()
    assert 0.0 <= payload["probabilidade_procedente"] <= 1.0


def test_flask_rejects_invalid_payload(artifacts_dir: Path) -> None:
    service = ModelService(artifacts_dir=artifacts_dir, threshold=0.5)
    app = create_flask_app(service=service)
    client = app.test_client()
    response = client.post("/predict", json={**VALID_PAYLOAD, "tribunal": "STF"})
    assert response.status_code == 422

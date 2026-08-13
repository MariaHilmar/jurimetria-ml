from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.service import ModelService
from ml.train import train


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("artifacts")
    metrics = train(
        n_samples=600,
        artifacts_dir=path,
        n_estimators=40,
        calibrate=True,
        random_state=42,
    )
    assert metrics["metrics"]["roc_auc"] > 0.7
    return path


@pytest.fixture()
def client(artifacts_dir: Path) -> Iterator[TestClient]:
    service = ModelService(artifacts_dir=artifacts_dir, threshold=0.5)
    app = create_app(service=service)
    with TestClient(app) as test_client:
        yield test_client

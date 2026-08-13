from __future__ import annotations

from pathlib import Path

import pytest

from ml.train import METRICS_FILENAME, MODEL_FILENAME, train
from ml.train import main as train_cli


def test_train_writes_artifacts_and_reports_learnable_signal(tmp_path: Path) -> None:
    metrics = train(
        n_samples=600,
        artifacts_dir=tmp_path,
        n_estimators=40,
        random_state=42,
    )
    assert (tmp_path / MODEL_FILENAME).exists()
    assert (tmp_path / METRICS_FILENAME).exists()
    assert metrics["dataset"] == "sintetico"
    assert metrics["metrics"]["roc_auc"] > 0.7
    assert metrics["metrics"]["f1"] > 0.55
    assert 0.0 <= metrics["metrics"]["brier"] < 0.3


def test_train_rejects_invalid_test_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="test_size"):
        train(n_samples=80, test_size=1.5, artifacts_dir=tmp_path)


def test_train_cli_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "ml.train",
            "--n-samples",
            "200",
            "--n-estimators",
            "20",
            "--artifacts-dir",
            str(tmp_path),
        ],
    )
    train_cli()
    assert (tmp_path / MODEL_FILENAME).exists()
    assert (tmp_path / METRICS_FILENAME).exists()

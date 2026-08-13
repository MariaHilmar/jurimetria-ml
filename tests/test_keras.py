from __future__ import annotations

from pathlib import Path

import pytest

keras = pytest.importorskip("keras")


def test_keras_mlp_trains_and_writes_metrics(tmp_path: Path) -> None:
    from ml.train_keras import KERAS_METRICS_FILENAME, KERAS_MODEL_FILENAME, train_keras

    metrics = train_keras(
        n_samples=200,
        epochs=1,
        artifacts_dir=tmp_path,
        random_state=42,
    )
    assert metrics["framework"] == "keras"
    assert metrics["dataset"] == "sintetico"
    assert 0.0 <= metrics["metrics"]["f1"] <= 1.0
    assert (tmp_path / KERAS_MODEL_FILENAME).exists()
    assert (tmp_path / KERAS_METRICS_FILENAME).exists()

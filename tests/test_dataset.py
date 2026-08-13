from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas import AreaDireito, Tribunal
from ml.dataset import generate_dataset, save_dataset
from ml.dataset import main as dataset_cli
from ml.features import AREAS_DIREITO, FEATURE_COLUMNS, LABEL_COLUMN, TRIBUNAIS


def test_dataset_is_reproducible() -> None:
    first = generate_dataset(n_samples=80, seed=7)
    second = generate_dataset(n_samples=80, seed=7)
    assert first.equals(second)


def test_dataset_has_expected_columns_and_both_classes() -> None:
    frame = generate_dataset(n_samples=200, seed=42)
    assert list(frame.columns) == [*FEATURE_COLUMNS, LABEL_COLUMN]
    assert set(frame[LABEL_COLUMN].unique()) == {0, 1}
    positive_rate = float(frame[LABEL_COLUMN].mean())
    assert 0.2 < positive_rate < 0.8


def test_dataset_rejects_tiny_samples() -> None:
    with pytest.raises(ValueError, match="50"):
        generate_dataset(n_samples=10, seed=1)


def test_api_literals_match_feature_vocabularies() -> None:
    assert set(TRIBUNAIS) == set(Tribunal.__args__)
    assert set(AREAS_DIREITO) == set(AreaDireito.__args__)


def test_save_dataset_writes_csv(tmp_path: Path) -> None:
    destination = save_dataset(tmp_path / "processos.csv", n_samples=60, seed=1)
    assert destination.exists()
    assert (
        destination.read_text(encoding="utf-8").splitlines()[0].startswith("tribunal")
    )


def test_dataset_cli_writes_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "cli.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["ml.dataset", "--n-samples", "60", "--seed", "1", "--out", str(out)],
    )
    dataset_cli()
    assert out.exists()

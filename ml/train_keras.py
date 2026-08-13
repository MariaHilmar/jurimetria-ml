"""MLP Keras (backend TensorFlow) no mesmo dataset sintético.

Não substitui o XGBoost calibrado. Serve para evidenciar o vocabulário
TensorFlow/Keras da vaga, com treino curto em CPU (1 época).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.dataset import generate_dataset
from ml.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    NUMERIC_COLUMNS,
)

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
KERAS_MODEL_FILENAME = "keras_mlp.keras"
KERAS_METRICS_FILENAME = "keras_metrics.json"


def _preprocess() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )


def train_keras(
    n_samples: int = 400,
    test_size: float = 0.25,
    random_state: int = 42,
    artifacts_dir: Path | None = None,
    epochs: int = 1,
) -> dict[str, Any]:
    import keras
    from keras import layers

    output_dir = artifacts_dir or DEFAULT_ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = generate_dataset(n_samples=n_samples, seed=random_state)
    features = frame[FEATURE_COLUMNS]
    labels = frame[LABEL_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    encoder = _preprocess()
    x_train_np = encoder.fit_transform(x_train).astype(np.float32)
    x_test_np = encoder.transform(x_test).astype(np.float32)
    y_train_np = y_train.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    keras.utils.set_random_seed(random_state)
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train_np.shape[1],)),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x_train_np, y_train_np, epochs=epochs, batch_size=32, verbose=0)

    y_prob = model.predict(x_test_np, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    scores = {
        "f1": float(f1_score(y_test_np, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_np, y_prob)),
    }
    metrics: dict[str, Any] = {
        "model_id": f"keras-mlp-{random_state}-{n_samples}",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "framework": "keras",
        "backend": "tensorflow",
        "n_samples": n_samples,
        "epochs": epochs,
        "dataset": "sintetico",
        "metrics": scores,
        "notes": (
            "MLP didático em CPU (1 época). Baseline de produção deste repo é XGBoost."
        ),
    }
    model.save(output_dir / KERAS_MODEL_FILENAME)
    (output_dir / KERAS_METRICS_FILENAME).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina MLP Keras no dataset sintético."
    )
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()
    result = train_keras(
        n_samples=args.n_samples,
        epochs=args.epochs,
        artifacts_dir=args.artifacts_dir,
    )
    scores = result["metrics"]
    print(f"keras ok | f1={scores['f1']:.3f} auc={scores['roc_auc']:.3f}")


if __name__ == "__main__":
    main()

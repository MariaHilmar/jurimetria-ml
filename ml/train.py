"""Treina um classificador XGBoost calibrado e grava artefatos.

Ciclo: dataset sintético -> split estratificado -> pipeline sklearn ->
métricas (F1, AUC, Brier) -> model.joblib + metrics.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ml.dataset import generate_dataset
from ml.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    NUMERIC_COLUMNS,
)

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
MODEL_FILENAME = "model.joblib"
METRICS_FILENAME = "metrics.json"


def _build_pipeline(
    random_state: int,
    n_estimators: int,
    calibrate: bool,
) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )
    booster = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        random_state=random_state,
    )
    estimator: Any = booster
    if calibrate:
        estimator = CalibratedClassifierCV(booster, method="sigmoid", cv=3)
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _classification_metrics(
    y_true: pd.Series, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def train(
    n_samples: int = 2000,
    test_size: float = 0.2,
    random_state: int = 42,
    artifacts_dir: Path | None = None,
    n_estimators: int = 80,
    calibrate: bool = True,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size deve estar entre 0 e 1 (exclusive).")

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

    pipeline = _build_pipeline(
        random_state=random_state,
        n_estimators=n_estimators,
        calibrate=calibrate,
    )
    pipeline.fit(x_train, y_train)

    y_prob = pipeline.predict_proba(x_test)[:, 1]
    scores = _classification_metrics(y_test, y_prob, threshold)

    positive_rate = float(labels.mean())
    metrics: dict[str, Any] = {
        "model_id": f"xgb-calibrated-{random_state}-{n_samples}",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_samples,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "test_size": test_size,
        "random_state": random_state,
        "calibrated": calibrate,
        "threshold": threshold,
        "positive_rate": positive_rate,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": scores,
        "dataset": "sintetico",
        "notes": (
            "Dataset sintético com processo generativo conhecido. "
            "Não representa desfecho judicial real."
        ),
    }

    joblib.dump(pipeline, output_dir / MODEL_FILENAME)
    (output_dir / METRICS_FILENAME).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo e grava artefatos.")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    result = train(
        n_samples=args.n_samples,
        test_size=args.test_size,
        random_state=args.random_state,
        artifacts_dir=args.artifacts_dir,
        n_estimators=args.n_estimators,
        calibrate=not args.no_calibrate,
        threshold=args.threshold,
    )
    scores = result["metrics"]
    print(
        "treino ok | "
        f"f1={scores['f1']:.3f} auc={scores['roc_auc']:.3f} "
        f"brier={scores['brier']:.3f}"
    )
    print(f"artefatos em {args.artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()

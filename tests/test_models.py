"""Unit tests for model training and persistence."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from src.config import RAW_DATA_PATH, TARGET_COL
from src.ingestion.loader import CreditDataLoader
from src.models.persistence import ModelPersistenceManager
from src.preprocessing.ml_pipeline import build_ml_pipeline


@pytest.fixture(scope="module")
def small_dataset():
    loader = CreditDataLoader(RAW_DATA_PATH)
    df = loader.load_raw_data().drop_duplicates().head(400)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def test_persistence_pipeline_roundtrip(small_dataset, tmp_path):
    X, y = small_dataset
    pipe = build_ml_pipeline(LogisticRegression(max_iter=100), scale_numeric=True)
    pipe.fit(X, y)

    manager = ModelPersistenceManager(models_dir=tmp_path)
    save_path = tmp_path / "test_pipeline.joblib"
    manager.save_pipeline(pipe, filepath=save_path)

    assert save_path.exists()

    loaded_pipe = manager.load_pipeline(filepath=save_path)
    preds_orig = pipe.predict_proba(X)
    preds_loaded = loaded_pipe.predict_proba(X)

    np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)


def test_persistence_threshold_roundtrip(tmp_path):
    manager = ModelPersistenceManager(models_dir=tmp_path)
    th_path = tmp_path / "test_th.json"
    manager.save_threshold_config(0.35, {"note": "test"}, filepath=th_path)

    assert th_path.exists()
    loaded = manager.load_threshold_config(filepath=th_path)
    assert loaded["selected_threshold"] == 0.35
    assert loaded["metadata"]["note"] == "test"


"""Joblib and JSON persistence manager for models, pipelines, and evaluation artifacts."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    MODEL_METRICS_PATH,
    MODELS_DIR,
    OPTIMAL_THRESHOLD_PATH,
)


class ModelPersistenceManager:
    """Manages serialization and deserialization of production ML pipelines and metadata."""

    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_pipeline(
        self, pipeline: Any, filepath: Optional[Union[str, Path]] = None
    ) -> Path:
        """Persist full scikit-learn pipeline using joblib."""
        target_path = Path(filepath or BEST_MODEL_PATH)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, target_path)
        return target_path

    def load_pipeline(
        self, filepath: Optional[Union[str, Path]] = None
    ) -> Any:
        """Load persisted scikit-learn pipeline."""
        target_path = Path(filepath or BEST_MODEL_PATH)
        if not target_path.exists():
            raise FileNotFoundError(f"Pipeline artifact not found at {target_path}")
        return joblib.load(target_path)

    def save_threshold_config(
        self,
        threshold: float,
        metadata: Optional[Dict[str, Any]] = None,
        filepath: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Save selected decision threshold and rationale to JSON."""
        target_path = Path(filepath or OPTIMAL_THRESHOLD_PATH)
        payload = {
            "selected_threshold": threshold,
            "metadata": metadata or {},
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return target_path

    def load_threshold_config(
        self, filepath: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Load threshold configuration from JSON."""
        target_path = Path(filepath or OPTIMAL_THRESHOLD_PATH)
        if not target_path.exists():
            raise FileNotFoundError(f"Threshold file not found at {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_comparison_metrics(
        self,
        metrics_dict: Dict[str, Any],
        filepath: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Save model comparison table to JSON and CSV."""
        target_path = Path(filepath or MODEL_METRICS_PATH)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)

        # Also write tabular CSV for quick review
        csv_path = target_path.with_suffix(".csv")
        if isinstance(metrics_dict, list):
            df = pd.DataFrame(metrics_dict)
            df.to_csv(csv_path, index=False)
        elif isinstance(metrics_dict, dict) and "comparison" in metrics_dict:
            df = pd.DataFrame(metrics_dict["comparison"])
            df.to_csv(csv_path, index=False)

        return target_path


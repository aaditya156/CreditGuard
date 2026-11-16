"""Unit tests for ModelEvaluator and threshold optimization."""

import pytest
import numpy as np
from src.models.evaluator import ModelEvaluator, EvaluationMetrics, ThresholdOptimizationResult


@pytest.fixture
def dummy_predictions():
    np.random.seed(42)
    y_true = np.array([0] * 80 + [1] * 20)
    # Generate realistic probabilities
    y_prob = np.concatenate([
        np.random.beta(1, 5, size=80),
        np.random.beta(5, 2, size=20)
    ])
    return y_true, y_prob


def test_evaluator_compute_metrics(dummy_predictions):
    y_true, y_prob = dummy_predictions
    evaluator = ModelEvaluator(cost_fn=5000.0, cost_fp=1000.0)
    metrics = evaluator.compute_metrics(y_true, y_prob, threshold=0.50)

    assert isinstance(metrics, EvaluationMetrics)
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.pr_auc <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0
    assert metrics.true_negatives + metrics.false_positives == 80
    assert metrics.false_negatives + metrics.true_positives == 20
    assert metrics.total_cost >= 0.0


def test_evaluator_threshold_optimization(dummy_predictions):
    y_true, y_prob = dummy_predictions
    evaluator = ModelEvaluator(cost_fn=5000.0, cost_fp=1000.0)
    opt_res = evaluator.optimize_threshold_for_cost(y_true, y_prob)

    assert isinstance(opt_res, ThresholdOptimizationResult)
    assert 0.01 <= opt_res.optimal_threshold <= 0.99
    assert opt_res.min_total_cost <= opt_res.cost_at_default_threshold
    assert opt_res.cost_savings >= 0.0


"""
Evaluation metrics for coordinated account detection.

Supports:
- Account-level classification metrics
- Community detection metrics
- Edge/link prediction metrics
- 95% Bootstrap Confidence Interval computation
- Recall at specific FPR thresholds (1% and 5% FPR)
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    adjusted_rand_score, normalized_mutual_info_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


def compute_bootstrap_ci(
    data_or_y_true: np.ndarray,
    y_target: Optional[np.ndarray] = None,
    metric_fn: Optional[Callable] = None,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute empirical bootstrap confidence interval for a metric function or 1D array.

    Supports two modes:
    1. 1D array of scalars (e.g. metrics across random seeds): computes bootstrap CI of the mean.
    2. (y_true, y_target, metric_fn): resamples pairs and computes metric distribution.
    """
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci) / 2.0

    if y_target is None or metric_fn is None:
        # Mode 1: 1D array of sample values
        arr = np.asarray(data_or_y_true, dtype=float)
        n = len(arr)
        if n == 0:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n_samples": 0}
        boot_means = []
        for _ in range(n_bootstraps):
            sample = rng.choice(arr, size=n, replace=True)
            boot_means.append(np.mean(sample))
        lower = float(np.percentile(boot_means, alpha * 100))
        upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
        return {
            "mean": float(np.mean(arr)),
            "ci_lower": lower,
            "ci_upper": upper,
            "std": float(np.std(arr)),
            "n_samples": n,
        }

    # Mode 2: Pair evaluation metric
    y_true = np.asarray(data_or_y_true)
    y_target = np.asarray(y_target)
    n = len(y_true)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n_samples": 0}

    scores = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        try:
            val = float(metric_fn(y_true[idx], y_target[idx]))
            if not np.isnan(val) and not np.isinf(val):
                scores.append(val)
        except Exception:
            continue

    if not scores:
        base_val = float(metric_fn(y_true, y_target)) if len(np.unique(y_true)) > 1 else 0.0
        return {"mean": base_val, "ci_lower": base_val, "ci_upper": base_val, "std": 0.0, "n_samples": n}

    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1.0 - alpha) * 100))
    return {
        "mean": float(np.mean(scores)),
        "ci_lower": lower,
        "ci_upper": upper,
        "std": float(np.std(scores)),
        "n_samples": n,
    }


def find_threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Find the decision threshold that achieves <= target_fpr on a calibration/val set.

    Args:
        y_true: Ground truth labels (0=negative, 1=positive).
        scores: Continuous prediction scores.
        target_fpr: Target false positive rate (e.g. 0.01 for 1% FPR).

    Returns:
        float threshold to apply to unseen test scores.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # Find the largest threshold where FPR <= target_fpr
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[-1]  # Highest FPR that is still <= target_fpr
        return float(thresholds[best_idx])
    return float(np.max(scores))


def compute_recall_at_fixed_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Compute recall on test pairs using a fixed threshold chosen on a held-out split."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    positives = (y_true == 1)
    if not np.any(positives):
        return 0.0
    preds = (scores >= threshold).astype(int)
    return float(recall_score(y_true, preds, zero_division=0))


class EvaluationMetrics:
    """Compute evaluation metrics for different task formulations."""

    def __init__(self, config: dict):
        self.config = config
        self.eval_config = config.get("evaluation", {})

    def compute_classification_metrics(self, y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_scores: np.ndarray = None
                                       ) -> Dict[str, float]:
        """
        Compute classification metrics for account-level IO detection.

        Metrics:
        - Accuracy
        - Precision (weighted)
        - Recall (weighted)
        - F1 (weighted)
        - ROC-AUC
        - PR-AUC
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # Binary metrics
        if len(np.unique(y_true)) == 2:
            metrics["precision_binary"] = precision_score(y_true, y_pred, zero_division=0)
            metrics["recall_binary"] = recall_score(y_true, y_pred, zero_division=0)
            metrics["f1_binary"] = f1_score(y_true, y_pred, zero_division=0)

        if y_scores is not None and len(np.unique(y_true)) > 1:
            try:
                if y_scores.ndim > 1:
                    scores = y_scores[:, 1]
                else:
                    scores = y_scores
                metrics["roc_auc"] = roc_auc_score(y_true, scores)
                metrics["pr_auc"] = average_precision_score(y_true, scores)
            except ValueError:
                metrics["roc_auc"] = 0.0
                metrics["pr_auc"] = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        return metrics

    def compute_community_metrics(self, true_labels: np.ndarray,
                                  predicted_labels: np.ndarray
                                  ) -> Dict[str, float]:
        """
        Compute community detection quality metrics.

        Metrics:
        - Adjusted Rand Index (ARI)
        - Normalized Mutual Information (NMI)
        """
        metrics = {}

        if len(np.unique(true_labels)) > 1 and len(np.unique(predicted_labels)) > 1:
            metrics["adjusted_rand_index"] = adjusted_rand_score(
                true_labels, predicted_labels
            )
            metrics["normalized_mutual_info"] = normalized_mutual_info_score(
                true_labels, predicted_labels
            )
        else:
            metrics["adjusted_rand_index"] = 0.0
            metrics["normalized_mutual_info"] = 0.0

        return metrics

    def compute_link_prediction_metrics(self, y_true: np.ndarray,
                                        y_scores: np.ndarray
                                        ) -> Dict[str, float]:
        """
        Compute link prediction metrics.

        Metrics:
        - ROC-AUC
        - Average Precision
        """
        metrics = {}
        if len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
                metrics["average_precision"] = average_precision_score(y_true, y_scores)
            except ValueError:
                metrics["roc_auc"] = 0.0
                metrics["average_precision"] = 0.0
        return metrics

    def compute_group_purity(self, communities: Dict[int, List[str]],
                             true_labels: Dict[str, bool]) -> float:
        """
        Compute community purity.

        Purity = fraction of accounts in each community that share
        the same ground-truth label.
        """
        total_correct = 0
        total_accounts = 0

        for comm_id, members in communities.items():
            if len(members) == 0:
                continue

            labels = [true_labels.get(m, False) for m in members]
            majority_label = max(set(labels), key=labels.count)
            correct = sum(1 for l in labels if l == majority_label)

            total_correct += correct
            total_accounts += len(members)

        return total_correct / total_accounts if total_accounts > 0 else 0.0

    def generate_report(self, metrics: Dict[str, float],
                        model_name: str = "Model") -> str:
        """Generate a human-readable evaluation report."""
        lines = [f"=== {model_name} Evaluation Report ===\n"]

        for key, value in metrics.items():
            if key == "confusion_matrix":
                lines.append(f"\nConfusion Matrix:")
                cm = np.array(value)
                lines.append(f"  {cm}")
            elif isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create comparison table of multiple models."""
        rows = []
        for model_name, metrics in results.items():
            row = {"model": model_name}
            row.update({k: v for k, v in metrics.items()
                       if isinstance(v, (int, float)) and k != "confusion_matrix"})
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def format_results_table(self, results_df: pd.DataFrame) -> str:
        """Format results as a readable table."""
        key_cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        available_cols = [c for c in key_cols if c in results_df.columns]

        return results_df[available_cols].to_string(index=False, float_format="%.4f")

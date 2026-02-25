"""
Evaluation metrics for multimodal SER.

Computes:
  - Weighted F1 (WF1)   — standard SER benchmark metric
  - Unweighted Accuracy (UA) — average per-class recall
  - Per-class F1 scores
  - Confusion matrix
"""

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import numpy as np


EMOTION_NAMES = ["ang", "hap", "neu", "sad", "fru", "exc"]


def compute_metrics(labels: list, preds: list, num_classes: int = 6) -> dict:
    """Compute all evaluation metrics.

    Args:
        labels: Ground-truth class indices
        preds:  Predicted class indices
        num_classes: Number of emotion classes

    Returns:
        dict with keys: weighted_f1, ua, per_class_f1, accuracy, confusion_matrix
    """
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    # UA = Unweighted Accuracy = mean per-class recall
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    per_class_recall = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    ua = float(per_class_recall.mean())

    per_class_f1 = f1_score(
        labels, preds,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    ).tolist()

    accuracy = accuracy_score(labels, preds)

    return {
        "weighted_f1": float(weighted_f1),
        "ua": float(ua),
        "accuracy": float(accuracy),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(metrics: dict, emotion_names: list = None) -> None:
    """Pretty-print evaluation metrics."""
    names = emotion_names or EMOTION_NAMES
    print(f"  Weighted F1 : {metrics['weighted_f1']:.4f}")
    print(f"  Unweighted Acc: {metrics['ua']:.4f}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print("  Per-class F1:")
    for name, f1 in zip(names, metrics["per_class_f1"]):
        print(f"    {name:4s}: {f1:.4f}")

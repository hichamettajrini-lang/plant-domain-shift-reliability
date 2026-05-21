"""Template for computing bootstrap confidence intervals.

This script is a template because predictions are generated locally from trained
models and public datasets. Use fixed labels to keep macro metrics comparable
across bootstrap samples.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def bootstrap_metrics_fixed_labels(y_true, y_pred, labels, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    results = {"accuracy": [], "macro_f1": [], "weighted_f1": [], "macro_precision": [], "macro_recall": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        results["accuracy"].append(accuracy_score(yt, yp))
        results["macro_f1"].append(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
        results["weighted_f1"].append(f1_score(yt, yp, labels=labels, average="weighted", zero_division=0))
        results["macro_precision"].append(precision_score(yt, yp, labels=labels, average="macro", zero_division=0))
        results["macro_recall"].append(recall_score(yt, yp, labels=labels, average="macro", zero_division=0))
    return {f"{k}_{s}": v for k, vals in results.items() for s, v in {
        "mean": np.mean(vals),
        "ci95_low": np.percentile(vals, 2.5),
        "ci95_high": np.percentile(vals, 97.5)
    }.items()}

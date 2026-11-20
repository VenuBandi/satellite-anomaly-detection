"""Dynamic threshold selection for anomaly detection."""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def threshold_sweep(
    errors: np.ndarray,
    labels: np.ndarray,
    percentiles: list[float] | None = None,
) -> dict:
    if percentiles is None:
        percentiles = list(range(90, 100))

    best = {"f1": 0.0, "threshold": None, "precision": 0.0, "recall": 0.0}
    results = []

    for p in percentiles:
        t = np.percentile(errors, p)
        preds = (errors >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        pr = precision_score(labels, preds, zero_division=0)
        rc = recall_score(labels, preds, zero_division=0)
        results.append({"percentile": p, "threshold": t, "f1": f1,
                         "precision": pr, "recall": rc})
        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": t, "precision": pr, "recall": rc,
                    "percentile": p}

    return {"best": best, "sweep": results}


def mad_threshold(errors: np.ndarray, k: float = 3.5) -> float:
    """Median Absolute Deviation based threshold."""
    median = np.median(errors)
    mad = np.median(np.abs(errors - median))
    return median + k * mad * 1.4826

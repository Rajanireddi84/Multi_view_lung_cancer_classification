import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             f1_score, accuracy_score, roc_curve)


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def binary_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(thr), "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "ap": float(average_precision_score(y, p)) if both else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_auc_ci(y, p, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def subtype_metrics(y_sub, pred_sub):
    y_sub = np.asarray(y_sub); pred_sub = np.asarray(pred_sub)
    m = y_sub >= 0
    if m.sum() == 0:
        return {"n": 0}
    return {"n": int(m.sum()),
            "accuracy": float(accuracy_score(y_sub[m], pred_sub[m])),
            "macro_f1": float(f1_score(y_sub[m], pred_sub[m], average="macro", zero_division=0))}

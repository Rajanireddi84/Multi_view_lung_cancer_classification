import json, os, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def seed_everything(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_roc(curves, path):
    """curves: {name: (y, p)}"""
    plt.figure(figsize=(5, 5))
    for n, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=n)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity"); plt.legend(); plt.title("ROC"); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def plot_cm(cm, labels, path, title="Confusion matrix"):
    cm = np.asarray(cm)
    plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.xticks(range(len(labels)), labels); plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def plot_history(hist, path):
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1); plt.plot(hist["train_loss"], label="train"); plt.plot(hist["val_loss"], label="val"); plt.legend(); plt.title("Loss")
    plt.subplot(1, 2, 2); plt.plot(hist["val_auc"]); plt.title("Val AUC"); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()

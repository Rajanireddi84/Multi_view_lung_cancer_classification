import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold

from .augment import augment_patch3d, simclr_augment
from .preprocess import extract_views

VIEWS = ("axial", "coronal", "sagittal")


def load_manifest(path):
    df = pd.read_csv(path)
    need = {"path", "pid", "label"}
    assert need <= set(df.columns), f"manifest needs columns {need}"
    if "subtype" not in df.columns:
        df["subtype"] = -1
    df["subtype"] = df["subtype"].fillna(-1).astype(int)
    return df


def patient_folds(df, n_folds=5, seed=42):
    """Patient-wise stratified folds -> list of (train_idx, val_idx, test_idx).
    Fold k: test = fold k, val = fold k+1, train = rest. No patient leaks across splits."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [te for _, te in sgkf.split(df, df["label"], df["pid"])]
    out = []
    for k in range(n_folds):
        te, va = folds[k], folds[(k + 1) % n_folds]
        tr = np.concatenate([folds[i] for i in range(n_folds) if i not in (k, (k + 1) % n_folds)])
        out.append((tr, va, te))
    return out


class NoduleDataset(Dataset):
    """Returns dict(views=[3 x (3,H,W)], label, subtype). Patches stored as npz['patch'] (Z,Y,X) in [0,1]."""

    def __init__(self, df, root, views=VIEWS, offsets=(-4, 0, 4), train=False):
        self.df, self.root, self.views, self.offsets, self.train = df.reset_index(drop=True), root, views, offsets, train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        patch = torch.from_numpy(np.load(os.path.join(self.root, r["path"]))["patch"].astype(np.float32))
        if self.train:
            patch = augment_patch3d(patch)
        v = extract_views(patch.numpy(), self.offsets)
        return {"views": [torch.from_numpy(np.ascontiguousarray(v[n])) for n in self.views],
                "label": int(r["label"]), "subtype": int(r["subtype"])}


class ContrastiveDataset(Dataset):
    """Unlabelled SimCLR data: a random view of a nodule patch -> two augmented copies."""

    def __init__(self, df, root, offsets=(-4, 0, 4)):
        self.df, self.root, self.offsets = df.reset_index(drop=True), root, offsets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        patch = np.load(os.path.join(self.root, self.df.iloc[i]["path"]))["patch"].astype(np.float32)
        v = extract_views(patch, self.offsets)
        x = torch.from_numpy(np.ascontiguousarray(v[VIEWS[np.random.randint(3)]]))
        return simclr_augment(x), simclr_augment(x)


def collate(batch):
    n_views = len(batch[0]["views"])
    return {"views": [torch.stack([b["views"][k] for b in batch]) for k in range(n_views)],
            "label": torch.tensor([b["label"] for b in batch]),
            "subtype": torch.tensor([b["subtype"] for b in batch])}

"""Synthetic nodule patches for SMOKE-TESTING the pipeline only.
Results obtained on this data say nothing about real-world performance."""
import os
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def make_nodule(rng, malignant, subtype, size=64):
    zz, yy, xx = np.mgrid[:size, :size, :size].astype(np.float32) - size / 2
    r = np.sqrt(zz**2 + yy**2 + xx**2) + 1e-6
    vol = ndi.gaussian_filter(rng.normal(0.12, 0.02, (size, size, size)).astype(np.float32), 1.0)
    rad = rng.uniform(9, 14) if malignant else rng.uniform(4, 8)
    shape = rad
    if malignant:  # lobulated / spiculated boundary
        theta = np.arccos(np.clip(zz / r, -1, 1)); phi = np.arctan2(yy, xx)
        shape = rad * (1 + 0.25 * np.sin(rng.integers(3, 7) * theta) * np.cos(rng.integers(3, 7) * phi))
    blob = (r < shape).astype(np.float32)
    inten = 0.75
    if malignant and subtype == 0:      # adenocarcinoma ~ part-solid / ground-glass
        inten = 0.45
    elif malignant and subtype == 1:    # squamous ~ dense with a central cavity
        inten = 0.9
        blob[r < shape * 0.35] = 0.3 / inten
    vol = np.maximum(vol, ndi.gaussian_filter(blob, 0.8) * inten)
    vol += rng.normal(0, 0.015, vol.shape)
    return np.clip(vol, 0, 1).astype(np.float16)


def build_synthetic_dataset(out_dir, n=120, seed=0, size=64, name="synthetic"):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i in range(n):
        mal = int(rng.random() < 0.5)
        sub = int(rng.integers(0, 3)) if mal else -1
        fn = f"{name}_{i:04d}.npz"
        np.savez_compressed(os.path.join(out_dir, fn), patch=make_nodule(rng, mal, sub, size))
        rows.append(dict(path=fn, pid=f"{name}_P{i // 2:03d}", label=mal, subtype=sub,
                         malignancy=float(rng.uniform(3.5, 5) if mal else rng.uniform(1, 2.5)), source=name))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    return df

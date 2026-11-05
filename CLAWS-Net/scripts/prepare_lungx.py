"""Build 64^3 patches + manifest from the SPIE-AAPM LUNGx collection (TCIA, DOI 10.7937/K9/TCIA.2015.UZLSU3FL).

Inputs (download from the collection page):
  * DICOM images via TCIA Data Retriever ("classic directory structure": <root>/<PatientID>/<study>/<series>/*.dcm)
      calibration patients are named CT-Training-*, test patients LUNGx-*
  * CalibrationSet_NoduleData.xlsx and TestSet_NoduleData_PublicRelease_wTruth.xlsx

The Excel column names are auto-detected (case-insensitive keyword match) and printed so you can verify them;
override with --col_* if needed. COORDINATE CONVENTIONS ARE NOT GUARANTEED: always inspect results/lungx_preview.png
(written by --preview) before trusting the patches, and use --z_mode / --one_based to fix any offset.

Example:
  python scripts/prepare_lungx.py --dicom_root /data/SPIE-AAPM --xlsx CalibrationSet_NoduleData.xlsx TestSet_NoduleData_PublicRelease_wTruth.xlsx --out data/lungx --preview
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
import pydicom
from scipy import ndimage as ndi
from claws_net.preprocess import crop_center, preprocess_patch

KEYS = {  # column role -> keyword regexes tried in order
    "pid": [r"scan", r"patient", r"subject", r"case", r"\bid\b"],
    "x": [r"\bx\b", r"x[\s_]*(coord|pos|center)", r"column"],
    "y": [r"\by\b", r"y[\s_]*(coord|pos|center)", r"\brow\b"],
    "z": [r"\bz\b", r"slice", r"z[\s_]*(coord|pos|center)", r"image"],
    "diagnosis": [r"diagnos", r"malig", r"benign", r"truth", r"label"],
    "subtype": [r"histolog", r"subtype", r"cell[\s_]*type", r"pathology"],
}
SUB = [("adeno", 0), ("squam", 1)]


def detect_columns(df, overrides):
    cols = {}
    for role, pats in KEYS.items():
        if overrides.get(role):
            cols[role] = overrides[role]; continue
        for p in pats:
            hit = [c for c in df.columns if re.search(p, str(c).lower())]
            hit = [c for c in hit if c not in cols.values()]
            if hit:
                cols[role] = hit[0]; break
    return cols


def read_nodule_table(path, overrides):
    xl = pd.read_excel(path, sheet_name=None)
    frames = []
    for sheet, df in xl.items():
        df = df.dropna(how="all")
        if df.empty: continue
        cols = detect_columns(df, overrides)
        print(f"[{os.path.basename(path)} / {sheet}] columns: {list(df.columns)}\n   detected: {cols}")
        missing = [r for r in ("pid", "x", "y", "z", "diagnosis") if r not in cols]
        if missing:
            raise SystemExit(f"Could not detect columns {missing}. Pass --col_<name> explicitly (see printed column list).")
        out = pd.DataFrame({r: df[c] for r, c in cols.items()}).dropna(subset=["pid", "x", "y", "z"])
        out["src"] = os.path.basename(path); frames.append(out)
    return pd.concat(frames, ignore_index=True)


def find_series_dir(root, pid):
    pid = str(pid).strip()
    cands = [d for d in glob.glob(os.path.join(root, "**", pid + "*"), recursive=True) if os.path.isdir(d)]
    cands = [d for d in cands if glob.glob(os.path.join(d, "**", "*.dcm"), recursive=True)]
    if not cands:
        raise FileNotFoundError(f"No DICOM folder for patient '{pid}' under {root}")
    return sorted(cands, key=len)[0]


def load_series(folder):
    files = [pydicom.dcmread(f) for f in glob.glob(os.path.join(folder, "**", "*.dcm"), recursive=True)]
    files = [s for s in files if hasattr(s, "ImagePositionPatient")]
    files.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    vol = np.stack([s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1)) + float(getattr(s, "RescaleIntercept", 0))
                    for s in files])
    ps = [float(v) for v in files[0].PixelSpacing]
    dz = float(np.median(np.abs(np.diff([float(s.ImagePositionPatient[2]) for s in files]))))
    inst = np.array([int(getattr(s, "InstanceNumber", i)) for i, s in enumerate(files)])
    return vol, (dz, ps[0], ps[1]), inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_root", required=True); ap.add_argument("--xlsx", nargs="+", required=True)
    ap.add_argument("--out", default="data/lungx"); ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--z_mode", choices=["index", "instance"], default="index",
                    help="index: z = position in z-sorted stack; instance: z = DICOM InstanceNumber")
    ap.add_argument("--one_based", action="store_true", help="coordinates in the xlsx start at 1")
    ap.add_argument("--preview", action="store_true", help="write <out>/preview.png montage for visual QC")
    for r in KEYS: ap.add_argument(f"--col_{r}", default=None)
    a = ap.parse_args(); ov = {r: getattr(a, f"col_{r}") for r in KEYS}
    os.makedirs(a.out, exist_ok=True)
    tab = pd.concat([read_nodule_table(p, ov) for p in a.xlsx], ignore_index=True)
    print(f"{len(tab)} nodule rows from {tab.pid.nunique()} patients")
    rows, prev, cache = [], [], {}
    for i, r in tab.iterrows():
        pid = str(r["pid"]).strip()
        d = str(r["diagnosis"]).lower()
        if "malig" in d or d in ("m", "1"): label = 1
        elif "benign" in d or d in ("b", "0"): label = 0
        else:
            print(f"skip {pid}: unknown diagnosis '{r['diagnosis']}'"); continue
        if pid not in cache:
            cache = {pid: load_series(find_series_dir(a.dicom_root, pid))}
        vol, sp, inst = cache[pid]
        off = 1 if a.one_based else 0
        x, y, z = float(r["x"]) - off, float(r["y"]) - off, float(r["z"]) - off
        if a.z_mode == "instance":
            z = float(np.argmin(np.abs(inst - float(r["z"]))))
        rs = ndi.zoom(vol, sp, order=1)                       # resample to ~1 mm isotropic
        hu = crop_center(rs, (z * sp[0], y * sp[1], x * sp[2]), a.size)
        fn = f"lungx_{pid}_{i}.npz"
        np.savez_compressed(os.path.join(a.out, fn), patch=preprocess_patch(hu).astype(np.float16))
        sub = -1
        if "subtype" in r and isinstance(r["subtype"], str):
            sub = next((v for k, v in SUB if k in r["subtype"].lower()), 2 if label == 1 else -1)
        rows.append(dict(path=fn, pid=f"lungx_{pid}", label=label, subtype=sub, source="LUNGx", split=str(r["src"])))
        if a.preview: prev.append((hu[a.size // 2], label, pid))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "manifest.csv"), index=False)
    print(f"saved {len(df)} nodules ({int(df.label.sum())} malignant / {int((1 - df.label).sum())} benign) -> {a.out}")
    print("subtype labels available for", int((df.subtype >= 0).sum()), "nodules")
    if a.preview and prev:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n = min(len(prev), 48); cols = 8; rws = int(np.ceil(n / cols))
        fig, ax = plt.subplots(rws, cols, figsize=(2 * cols, 2 * rws)); ax = np.atleast_2d(ax)
        for k, a_ in enumerate(ax.ravel()):
            a_.axis("off")
            if k < n:
                a_.imshow(np.clip(prev[k][0], -1000, 400), cmap="gray"); a_.set_title(f"{prev[k][2][-6:]} {'M' if prev[k][1] else 'B'}", fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(a.out, "preview.png"), dpi=120); print("preview ->", a.out + "/preview.png")


if __name__ == "__main__":
    main()

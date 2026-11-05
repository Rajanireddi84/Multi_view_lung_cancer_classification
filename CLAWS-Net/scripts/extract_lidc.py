"""Build 64^3 nodule patches + manifest from LIDC-IDRI using pylidc.

Selection (as in the paper): nodule diameter >= 3 mm and >= 3 radiologist annotations
(inter-reader agreement). Label = mean malignancy: <3 benign(0), >3 malignant(1), ==3 excluded.
Prerequisite: `pip install pylidc` and ~/.pylidcrc pointing to the LIDC-IDRI DICOM folder.
  [dicom]
  path = /path/to/LIDC-IDRI
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from tqdm import tqdm
from claws_net.preprocess import crop_center, preprocess_patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lidc")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--min_diam", type=float, default=3.0)
    ap.add_argument("--min_readers", type=int, default=3)
    ap.add_argument("--max_scans", type=int, default=0, help="0 = all 1018 scans")
    a = ap.parse_args()
    import pylidc as pl
    os.makedirs(a.out, exist_ok=True)
    rows = []
    scans = pl.query(pl.Scan).all()
    scans = scans[:a.max_scans] if a.max_scans else scans
    for scan in tqdm(scans):
        groups = scan.cluster_annotations()
        groups = [g for g in groups if len(g) >= a.min_readers]
        if not groups:
            continue
        vol = scan.to_volume(verbose=False).astype(np.float32).transpose(2, 0, 1)   # -> [Z,Y,X]
        zoom = (scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing)        # -> 1 mm isotropic
        vol = ndi.zoom(vol, zoom, order=1)
        for gi, g in enumerate(groups):
            diam = float(np.mean([x.diameter for x in g]))
            mal = float(np.mean([x.malignancy for x in g]))
            if diam < a.min_diam or round(mal, 2) == 3.0:
                continue
            cen = np.mean([x.centroid for x in g], axis=0)                           # (i=row, j=col, k=slice)
            cz, cy, cx = cen[2] * zoom[0], cen[0] * zoom[1], cen[1] * zoom[2]
            hu = crop_center(vol, (cz, cy, cx), a.size)
            patch = preprocess_patch(hu).astype(np.float16)
            fn = f"{scan.patient_id}_n{gi}.npz"
            np.savez_compressed(os.path.join(a.out, fn), patch=patch)
            rows.append(dict(path=fn, pid=scan.patient_id, label=int(mal > 3), subtype=-1,
                             malignancy=mal, diameter_mm=diam, source="LIDC-IDRI"))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "manifest.csv"), index=False)
    print(f"saved {len(rows)} nodules ->", a.out)


if __name__ == "__main__":
    main()

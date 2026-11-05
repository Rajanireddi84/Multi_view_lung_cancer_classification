"""Aggregate results/*/metrics.json (grouped by experiment prefix before '_fold') into results/summary.md/csv."""
import glob, json, os, re, sys
import numpy as np, pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for f in sorted(glob.glob(f"{root}/*/metrics.json")):
    run = os.path.basename(os.path.dirname(f)); exp = re.sub(r"_fold\d+$", "", run)
    m = json.load(open(f))
    for split in ("test", "external"):
        if split in m:
            d = m[split]; rows.append(dict(experiment=exp, split=split, run=run, **{k: d[k] for k in
                  ("accuracy", "sensitivity", "specificity", "precision", "f1", "auc")},
                  subtype_acc=d.get("subtype", {}).get("accuracy", np.nan), subtype_f1=d.get("subtype", {}).get("macro_f1", np.nan)))
if not rows: sys.exit("no metrics.json found")
df = pd.DataFrame(rows); df.to_csv(f"{root}/all_runs.csv", index=False)
cols = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc", "subtype_acc", "subtype_f1"]
g = df.groupby(["experiment", "split"])[cols]
mean, std, n = g.mean(), g.std(ddof=0).fillna(0), g.size()
lines = ["| Experiment | Split | Folds | " + " | ".join(cols) + " |", "|" + "---|" * (3 + len(cols))]
for k in mean.index:
    lines.append(f"| {k[0]} | {k[1]} | {n[k]} | " + " | ".join(
        "n/a" if np.isnan(mean.loc[k, c]) else f"{mean.loc[k, c]:.4f} ± {std.loc[k, c]:.4f}" for c in cols) + " |")
open(f"{root}/summary.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

"""Generate a SYNTHETIC dataset to smoke-test the pipeline (not for reporting results)."""
import argparse
from claws_net.synthetic import build_synthetic_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic"); ap.add_argument("--n", type=int, default=160)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--name", default="synthetic")
a = ap.parse_args()
df = build_synthetic_dataset(a.out, a.n, a.seed, name=a.name)
print(df.label.value_counts().to_dict(), "->", a.out)

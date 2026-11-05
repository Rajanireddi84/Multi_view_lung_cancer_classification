"""Train / evaluate CLAWS-Net on one patient-wise CV fold (+ optional external test set)."""
import argparse, json, os
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from claws_net.data import NoduleDataset, collate, load_manifest, patient_folds, VIEWS
from claws_net.losses import CompositeLoss
from claws_net.metrics import binary_metrics, subtype_metrics, youden_threshold, bootstrap_auc_ci
from claws_net.models import CLAWSNet
from claws_net.utils import seed_everything, save_json, plot_roc, plot_cm, plot_history
from sklearn.metrics import confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--external", default=None, help="manifest of external test set (e.g. LUNGx)")
ap.add_argument("--out", default="results/run"); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n_folds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--patience", type=int, default=15); ap.add_argument("--lambda_sub", type=float, default=0.5)
ap.add_argument("--pretrained", default=None, help="SimCLR encoder .pt"); ap.add_argument("--views", nargs="+", default=list(VIEWS))
ap.add_argument("--no_ctf", action="store_true"); ap.add_argument("--no_awdf", action="store_true")
ap.add_argument("--depth", type=int, default=2); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); seed_everything(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)

df = load_manifest(a.manifest); root = os.path.dirname(a.manifest)
tr, va, te = patient_folds(df, a.n_folds, a.seed)[a.fold]
D = lambda idx, train: NoduleDataset(df.iloc[idx], root, tuple(a.views), train=train)
mk = lambda ds, sh: DataLoader(ds, a.bs, shuffle=sh, collate_fn=collate, num_workers=a.workers, pin_memory=True)
dl_tr, dl_va, dl_te = mk(D(tr, True), True), mk(D(va, False), False), mk(D(te, False), False)

model = CLAWSNet(n_views=len(a.views), depth=a.depth, use_ctf=not a.no_ctf, use_awdf=not a.no_awdf).to(dev)
if a.pretrained:
    model.load_pretrained_encoder(torch.load(a.pretrained, map_location=dev)); print("loaded SimCLR encoder")
cnt = np.bincount(df.iloc[tr]["label"], minlength=2).astype(np.float32)
crit = CompositeLoss(a.lambda_sub, torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), device=dev)).to(dev)
opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl_tr))
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")


@torch.no_grad()
def predict(dl):
    model.eval(); P, Y, S, SP, L, VW = [], [], [], [], 0.0, []
    for b in dl:
        v = [x.to(dev) for x in b["views"]]; ym, ys = b["label"].to(dev), b["subtype"].to(dev)
        out = model(v); L += crit(out, ym, ys)[0].item() * len(ym)
        P += torch.softmax(out["mal"], 1)[:, 1].cpu().tolist(); Y += b["label"].tolist()
        S += b["subtype"].tolist(); SP += out["sub"].argmax(1).cpu().tolist(); VW += out["view_w"].cpu().tolist()
    return np.array(Y), np.array(P), np.array(S), np.array(SP), L / max(1, len(Y)), np.array(VW)


hist = {"train_loss": [], "val_loss": [], "val_auc": []}; best, bad = -1, 0
for ep in range(a.epochs):
    model.train(); tot = 0
    for b in tqdm(dl_tr, desc=f"ep {ep + 1}/{a.epochs}", leave=False):
        v = [x.to(dev) for x in b["views"]]
        with torch.autocast(dev, enabled=dev == "cuda"):
            loss, _ = crit(model(v), b["label"].to(dev), b["subtype"].to(dev))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
    y, p, *_, vl, _ = predict(dl_va); auc = binary_metrics(y, p)["auc"]
    hist["train_loss"].append(tot / len(dl_tr)); hist["val_loss"].append(vl); hist["val_auc"].append(auc)
    print(f"ep {ep + 1}: train {hist['train_loss'][-1]:.4f} val {vl:.4f} AUC {auc:.4f}")
    if auc > best: best, bad = auc, 0; torch.save(model.state_dict(), f"{a.out}/best.pt")
    else:
        bad += 1
        if bad >= a.patience: print("early stop"); break

model.load_state_dict(torch.load(f"{a.out}/best.pt", map_location=dev))
yv, pv, *_ = predict(dl_va); thr = youden_threshold(yv, pv)          # threshold chosen on validation only
res = {"args": vars(a), "val_threshold": thr}; curves = {}
sets = {"test": dl_te}
if a.external:
    edf = load_manifest(a.external)
    sets["external"] = mk(NoduleDataset(edf, os.path.dirname(a.external), tuple(a.views)), False)
for name, dl in sets.items():
    y, p, s, sp, _, vw = predict(dl); m = binary_metrics(y, p, thr)
    m["auc_ci95"] = bootstrap_auc_ci(y, p); m["subtype"] = subtype_metrics(s, sp)
    m["mean_view_weights(axial,coronal,sagittal)"] = vw.mean(0).tolist() if len(a.views) == 3 else None
    res[name] = m; curves[name] = (y, p)
    pd.DataFrame({"label": y, "prob": p, "subtype": s, "subtype_pred": sp}).to_csv(f"{a.out}/pred_{name}.csv", index=False)
    plot_cm([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], ["Benign", "Malignant"], f"{a.out}/cm_{name}.png", f"{name} (thr={thr:.2f})")
    print(name, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    if (s >= 0).sum() > 0:
        mm = s >= 0
        plot_cm(confusion_matrix(s[mm], sp[mm], labels=[0, 1, 2]), ["Adeno", "Squam", "Other"], f"{a.out}/cm_subtype_{name}.png", f"subtype {name}")
plot_roc(curves, f"{a.out}/roc.png"); plot_history(hist, f"{a.out}/history.png")
pd.DataFrame(hist).to_csv(f"{a.out}/history.csv", index=False); save_json(res, f"{a.out}/metrics.json")

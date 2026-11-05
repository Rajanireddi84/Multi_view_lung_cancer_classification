"""SimCLR contrastive pre-training of the Res2Net encoder on unlabelled nodule views."""
import argparse, os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from claws_net.data import ContrastiveDataset, load_manifest
from claws_net.losses import nt_xent
from claws_net.models import SimCLRNet
from claws_net.utils import seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", nargs="+", required=True, help="one or more manifest.csv (labels are NOT used)")
ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--temp", type=float, default=0.2)
ap.add_argument("--out", default="checkpoints/simclr_encoder.pt"); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); seed_everything(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
sets = []
for m in a.manifest:
    sets.append(ContrastiveDataset(load_manifest(m), os.path.dirname(m)))
ds = torch.utils.data.ConcatDataset(sets)
dl = DataLoader(ds, a.bs, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=True)
net = SimCLRNet().to(dev)
opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
log = []
for ep in range(a.epochs):
    net.train(); tot = 0
    for x1, x2 in tqdm(dl, desc=f"SimCLR {ep + 1}/{a.epochs}", leave=False):
        loss = nt_xent(net(x1.to(dev)), net(x2.to(dev)), a.temp)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    sched.step(); log.append(tot / len(dl)); print(f"epoch {ep + 1} NT-Xent {log[-1]:.4f}")
    torch.save(net.encoder.state_dict(), a.out)

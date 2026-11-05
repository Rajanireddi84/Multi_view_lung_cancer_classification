"""Train the U-Net lung-field refiner on pseudo-labels from the classical watershed-morphology
segmentation (self-training). Slices are taken from the extracted patches' parent CT via --slices_npz
(an .npz with array `hu` of shape [N,H,W] in HU). Saves checkpoints/unet.pt."""
import argparse, os
import numpy as np, torch, torch.nn.functional as F
from claws_net.models import UNet
from claws_net.preprocess import lung_mask_slice, window_normalize
from claws_net.utils import seed_everything

ap = argparse.ArgumentParser()
ap.add_argument("--slices_npz", required=True); ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--out", default="checkpoints/unet.pt")
a = ap.parse_args(); seed_everything()
hu = np.load(a.slices_npz)["hu"]
prior = np.stack([lung_mask_slice(s) for s in hu]).astype(np.float32)
x = torch.from_numpy(np.stack([window_normalize(hu), prior], 1)); y = torch.from_numpy(prior)[:, None]
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = UNet().to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3)
for ep in range(a.epochs):
    perm = torch.randperm(len(x)); tot = 0
    for i in range(0, len(x), a.bs):
        idx = perm[i:i + a.bs]; xb, yb = x[idx].to(dev), y[idx].to(dev)
        lg = net(xb); p = torch.sigmoid(lg)
        dice = 1 - (2 * (p * yb).sum() + 1) / (p.sum() + yb.sum() + 1)
        loss = F.binary_cross_entropy_with_logits(lg, yb) + dice
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print(f"epoch {ep + 1}/{a.epochs} loss {tot / max(1, len(x) // a.bs):.4f}")
os.makedirs(os.path.dirname(a.out), exist_ok=True); torch.save(net.state_dict(), a.out)

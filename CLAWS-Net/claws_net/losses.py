import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent(z1, z2, temperature=0.2):
    """SimCLR NT-Xent loss. z1, z2: (B,D) embeddings of two views of the same samples."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    B = z1.size(0)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, target)


class CompositeLoss(nn.Module):
    """L = CE_malignancy + lambda_sub * CE_subtype (subtype term only on samples with a subtype label)."""

    def __init__(self, lambda_sub=0.5, class_weights=None, label_smoothing=0.05):
        super().__init__()
        self.lambda_sub, self.ls = lambda_sub, label_smoothing
        self.register_buffer("w", class_weights if class_weights is not None else None)

    def forward(self, out, y_mal, y_sub):
        l_mal = F.cross_entropy(out["mal"], y_mal, weight=self.w, label_smoothing=self.ls)
        m = y_sub >= 0
        l_sub = F.cross_entropy(out["sub"][m], y_sub[m]) if m.any() else out["sub"].sum() * 0.0
        return l_mal + self.lambda_sub * l_sub, {"mal": l_mal.item(), "sub": float(l_sub.item())}

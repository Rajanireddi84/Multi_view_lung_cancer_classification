"""Torch-only augmentations: SimCLR view augmentation (2D) and 3D patch augmentation."""
import random
import torch
import torch.nn.functional as F


def _gauss_kernel(sigma, ch, device):
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=device, dtype=torch.float32) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(ch, 1, k, k).contiguous(), k // 2


def simclr_augment(x):
    """x: (C,H,W) in [0,1]. Random flip/rot90, resized crop, brightness/contrast, blur, noise."""
    C, H, W = x.shape
    if random.random() < 0.5: x = x.flip(-1)
    if random.random() < 0.5: x = x.flip(-2)
    x = torch.rot90(x, random.randint(0, 3), (-2, -1))
    s = random.uniform(0.55, 1.0)
    h, w = int(H * s), int(W * s)
    i, j = random.randint(0, H - h), random.randint(0, W - w)
    x = F.interpolate(x[None, :, i:i + h, j:j + w], size=(H, W), mode="bilinear", align_corners=False)[0]
    x = x * random.uniform(0.75, 1.25) + random.uniform(-0.1, 0.1)
    if random.random() < 0.4:
        k, p = _gauss_kernel(random.uniform(0.5, 1.5), C, x.device)
        x = F.conv2d(F.pad(x[None], (p,) * 4, mode="reflect"), k, groups=C)[0]
    x = x + torch.randn_like(x) * random.uniform(0.0, 0.04)
    return x.clamp(0, 1)


def augment_patch3d(p):
    """p: (Z,Y,X) tensor. Random axis flips, in-plane rot90, mild intensity jitter."""
    for ax in range(3):
        if random.random() < 0.5: p = p.flip(ax)
    p = torch.rot90(p, random.randint(0, 3), (1, 2))
    p = p * random.uniform(0.9, 1.1) + random.uniform(-0.03, 0.03)
    return p.clamp(0, 1)

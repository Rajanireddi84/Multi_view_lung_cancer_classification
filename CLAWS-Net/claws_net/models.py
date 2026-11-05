"""CLAWS-Net architecture: Res2Net triple-view encoder, Cross-View Transformer,
Adaptive Weighted Discriminative Fusion (AWDF), Dual-Head classifier, SimCLR wrapper, U-Net."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- Res2Net encoder
class Res2Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, scale=4, base_width=26, downsample=None):
        super().__init__()
        self.width = max(1, int(math.floor(planes * base_width / 64)))
        self.scale, self.nums = scale, scale - 1
        self.stage = downsample is not None
        self.conv1 = nn.Conv2d(inplanes, self.width * scale, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width * scale)
        self.convs = nn.ModuleList([nn.Conv2d(self.width, self.width, 3, stride, 1, bias=False) for _ in range(self.nums)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in range(self.nums)])
        self.pool = nn.AvgPool2d(3, stride, 1)
        self.conv3 = nn.Conv2d(self.width * scale, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if (i == 0 or self.stage) else sp + spx[i]
            sp = F.relu(self.bns[i](self.convs[i](sp)), inplace=True)
            out = sp if i == 0 else torch.cat((out, sp), 1)
        last = self.pool(spx[self.nums]) if self.stage else spx[self.nums]
        out = torch.cat((out, last), 1)
        out = self.bn3(self.conv3(out))
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res, inplace=True)


class Res2NetEncoder(nn.Module):
    """2D Res2Net for 64x64 inputs. Output: (B, widths[-1]*4, H/8, W/8)."""

    def __init__(self, in_ch=3, widths=(16, 32, 64, 128), layers=(2, 2, 2, 2), scale=4, base_width=26):
        super().__init__()
        self.inplanes = 32
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        strides = (1, 2, 2, 2)
        self.layers = nn.Sequential(*[self._make(w, n, s, scale, base_width) for w, n, s in zip(widths, layers, strides)])
        self.out_channels = widths[-1] * Res2Bottleneck.expansion

    def _make(self, planes, blocks, stride, scale, bw):
        ds = None
        if stride != 1 or self.inplanes != planes * 4:
            ds = nn.Sequential(nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False) if stride > 1 else nn.Identity(),
                               nn.Conv2d(self.inplanes, planes * 4, 1, bias=False), nn.BatchNorm2d(planes * 4))
        mods = [Res2Bottleneck(self.inplanes, planes, stride, scale, bw, ds)]
        self.inplanes = planes * 4
        mods += [Res2Bottleneck(self.inplanes, planes, 1, scale, bw) for _ in range(1, blocks)]
        return nn.Sequential(*mods)

    def forward(self, x):
        return self.layers(self.stem(x))


# ----------------------------------------------------------------- Cross-View Transformer
class CrossViewTransformer(nn.Module):
    """Projects each view's feature map to tokens, adds learned (view, position) embeddings and
    runs joint self-attention over ALL views' tokens -> cross-view spatial consistency and
    long-range dependencies. depth=0 disables attention (ablation)."""

    def __init__(self, in_ch, d=256, heads=8, depth=2, n_views=3, n_tokens=64, drop=0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.ModuleList([nn.Conv2d(in_ch, d, 1) for _ in range(n_views)])
        self.pos = nn.Parameter(torch.zeros(1, n_views * n_tokens, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.encoder = None
        if depth > 0:
            layer = nn.TransformerEncoderLayer(d, heads, d * 4, drop, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, feats):
        x = torch.cat([p(f).flatten(2).transpose(1, 2) for p, f in zip(self.proj, feats)], 1) + self.pos
        if self.encoder is not None:
            x = self.encoder(x)
        return list(torch.split(self.norm(x), self.n_tokens, dim=1))


# ----------------------------------------------------------------- AWDF
class AWDF(nn.Module):
    """Adaptive Weighted Discriminative Fusion.
    (1) token-level attention pooling per view highlights diagnostically significant regions;
    (2) a learned softmax gate weighs each view per sample;
    (3) a residual refinement MLP sharpens the fused descriptor."""

    def __init__(self, d, n_views=3, drop=0.2):
        super().__init__()
        self.tok_attn = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_views)])
        self.gate = nn.Sequential(nn.Linear(n_views * d, d), nn.GELU(), nn.Linear(d, n_views))
        self.refine = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop))

    def forward(self, toks):
        pooled = [(torch.softmax(a(t), 1) * t).sum(1) for t, a in zip(toks, self.tok_attn)]
        P = torch.stack(pooled, 1)                                   # (B,V,d)
        g = torch.softmax(self.gate(torch.cat(pooled, -1)), -1)      # (B,V)
        fused = (g.unsqueeze(-1) * P).sum(1)
        return fused + self.refine(fused), g


class MeanFusion(nn.Module):
    """Ablation replacement for AWDF: plain mean over tokens and views."""

    def forward(self, toks):
        P = torch.stack([t.mean(1) for t in toks], 1)
        return P.mean(1), torch.full(P.shape[:2], 1.0 / P.size(1), device=P.device)


# ----------------------------------------------------------------- Dual-head + full model
class DualHead(nn.Module):
    def __init__(self, d, n_subtypes=3, drop=0.3):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(drop))
        self.mal = nn.Linear(d // 2, 2)
        self.sub = nn.Linear(d // 2, n_subtypes)

    def forward(self, f):
        h = self.shared(f)
        return self.mal(h), self.sub(h)


class CLAWSNet(nn.Module):
    def __init__(self, n_views=3, in_ch=3, d=256, heads=8, depth=2, img_size=64, n_subtypes=3,
                 use_ctf=True, use_awdf=True, widths=(16, 32, 64, 128)):
        super().__init__()
        self.encoders = nn.ModuleList([Res2NetEncoder(in_ch, widths) for _ in range(n_views)])
        n_tokens = (img_size // 8) ** 2
        self.ctf = CrossViewTransformer(self.encoders[0].out_channels, d, heads, depth if use_ctf else 0, n_views, n_tokens)
        self.fusion = AWDF(d, n_views) if use_awdf else MeanFusion()
        self.head = DualHead(d, n_subtypes)

    def load_pretrained_encoder(self, state):
        for e in self.encoders:
            e.load_state_dict(state)

    def forward(self, views):
        toks = self.ctf([e(v) for e, v in zip(self.encoders, views)])
        f, g = self.fusion(toks)
        mal, sub = self.head(f)
        return {"mal": mal, "sub": sub, "view_w": g, "feat": f}


class SimCLRNet(nn.Module):
    def __init__(self, in_ch=3, proj_dim=128, widths=(16, 32, 64, 128)):
        super().__init__()
        self.encoder = Res2NetEncoder(in_ch, widths)
        c = self.encoder.out_channels
        self.proj = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True), nn.Linear(c, proj_dim))

    def forward(self, x):
        return self.proj(self.encoder(x).mean((2, 3)))


# ----------------------------------------------------------------- U-Net (lung-field refinement)
def _dc(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    """Small U-Net. Input = [CT slice, watershed-morphology prior] (2ch) -> refined lung mask logits.
    This is the 'Hybrid Watershed-Morphology U-Net': the classical mask is a prior that the network refines."""

    def __init__(self, in_ch=2, base=16):
        super().__init__()
        b = base
        self.d1, self.d2, self.d3, self.bott = _dc(in_ch, b), _dc(b, 2 * b), _dc(2 * b, 4 * b), _dc(4 * b, 8 * b)
        self.u3, self.c3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), _dc(8 * b, 4 * b)
        self.u2, self.c2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), _dc(4 * b, 2 * b)
        self.u1, self.c1 = nn.ConvTranspose2d(2 * b, b, 2, 2), _dc(2 * b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        x1 = self.d1(x); x2 = self.d2(F.max_pool2d(x1, 2)); x3 = self.d3(F.max_pool2d(x2, 2))
        y = self.bott(F.max_pool2d(x3, 2))
        y = self.c3(torch.cat([self.u3(y), x3], 1)); y = self.c2(torch.cat([self.u2(y), x2], 1))
        y = self.c1(torch.cat([self.u1(y), x1], 1))
        return self.out(y)

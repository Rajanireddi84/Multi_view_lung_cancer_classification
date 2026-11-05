import numpy as np
from claws_net import preprocess as pp, metrics as M
from claws_net.synthetic import make_nodule


def fake_ct(n=64):
    """body ellipse (soft tissue) containing two air-filled lungs + a nodule."""
    yy, xx = np.mgrid[:256, :256]
    body = ((yy - 128) / 110.0) ** 2 + ((xx - 128) / 120.0) ** 2 < 1
    left = ((yy - 128) / 60.0) ** 2 + ((xx - 85) / 40.0) ** 2 < 1
    right = ((yy - 128) / 60.0) ** 2 + ((xx - 171) / 40.0) ** 2 < 1
    s = np.full((256, 256), -1000.0); s[body] = 40.0; s[left | right] = -850.0
    s[120:130, 160:170] = 30.0   # nodule inside right lung
    return np.stack([s] * n)


def test_window_range():
    x = pp.window_normalize(np.array([-2000, -1000, 400, 3000.]))
    assert x.min() == 0 and x.max() == 1


def test_lung_mask_finds_both_lungs():
    m = pp.lung_mask_slice(fake_ct(1)[0])
    assert m[128, 85] and m[128, 171] and not m[5, 5]
    assert m[125, 165], "juxta/intra-lung nodule must be included in the mask (hole filling)"


def test_patch_pipeline_shapes_and_range():
    hu = pp.crop_center(fake_ct(70), (35, 125, 165), 64)
    p = pp.preprocess_patch(hu)
    assert p.shape == (64, 64, 64) and 0 <= p.min() and p.max() <= 1
    v = pp.extract_views(p)
    assert all(a.shape == (3, 64, 64) for a in v.values())


def test_crop_padding_at_border():
    assert pp.crop_center(np.zeros((10, 10, 10)), (0, 0, 0), 64).shape == (64, 64, 64)


def test_edge_enhance_increases_edge_contrast():
    img = np.zeros((64, 64), np.float32); img[:, 32:] = 0.5
    out = pp.gaussian_edge_enhance(img)
    assert out[:, 34].max() >= 0.5 - 1e-3 and out.max() <= 1


def test_metrics():
    y = np.array([0, 0, 1, 1]); p = np.array([.1, .4, .6, .9])
    m = M.binary_metrics(y, p)
    assert m["auc"] == 1.0 and m["accuracy"] == 1.0 and m["sensitivity"] == 1.0
    assert 0.4 <= M.youden_threshold(y, p) <= 0.9
    s = M.subtype_metrics([0, 1, -1, 2], [0, 1, 0, 1]); assert s["n"] == 3


def test_synthetic_separable():
    rng = np.random.default_rng(0)
    b = make_nodule(rng, 0, -1).astype(float); m = make_nodule(rng, 1, 0).astype(float)
    assert (m > 0.3).sum() > (b > 0.3).sum()

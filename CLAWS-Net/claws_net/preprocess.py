"""CT preprocessing: HU windowing, hybrid Watershed-Morphology lung segmentation,
and Gaussian-based nodule-edge enhancement."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation, filters

HU_MIN, HU_MAX = -1000.0, 400.0


def window_normalize(hu, lo=HU_MIN, hi=HU_MAX):
    """Clip to the lung window and scale to [0, 1]."""
    return ((np.clip(hu, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def lung_mask_slice(hu_slice, air_thr=-320, min_area=2000):
    """Classical Watershed-Morphology lung field segmentation for one axial slice.

    1. Threshold air-like voxels and drop the border-connected background.
    2. Keep the (up to) two largest components = lung candidates.
    3. Marker-controlled watershed on the Sobel gradient (internal marker = eroded
       lungs, external marker = dilated ring) refines the lung boundary.
    4. Morphological closing + hole filling re-includes juxta-pleural nodules.
    """
    binary = hu_slice < air_thr
    binary = segmentation.clear_border(binary)
    lab = measure.label(binary)
    if lab.max() == 0:
        return np.zeros(binary.shape, dtype=bool)
    areas = np.bincount(lab.ravel())
    areas[0] = 0
    keep = [i for i in np.argsort(areas)[::-1][:2] if areas[i] >= min_area]
    if not keep:
        return np.zeros(binary.shape, dtype=bool)
    lungs = np.isin(lab, keep)

    internal = ndi.binary_erosion(lungs, morphology.disk(5))
    ext_outer = ndi.binary_dilation(lungs, morphology.disk(10))
    ext_inner = ndi.binary_dilation(lungs, morphology.disk(3))
    markers = np.zeros(lungs.shape, dtype=np.int32)
    markers[internal] = 1
    markers[ext_outer ^ ext_inner] = 2
    grad = filters.sobel(np.clip(hu_slice, HU_MIN, HU_MAX))
    ws = segmentation.watershed(grad, markers) == 1
    ws = ws | internal

    mask = ndi.binary_closing(ws, morphology.disk(10))
    mask = ndi.binary_fill_holes(mask)
    lab2 = measure.label(mask)
    if lab2.max() > 0:
        sizes = np.bincount(lab2.ravel()); sizes[0] = 0
        mask = np.isin(lab2, np.where(sizes >= min_area)[0])
    return mask.astype(bool)


def lung_mask_volume(hu_vol):
    """Apply slice-wise segmentation to a [Z, Y, X] HU volume."""
    return np.stack([lung_mask_slice(s) for s in hu_vol], axis=0)


def gaussian_edge_enhance(vol, sigma_denoise=0.6, sigma_lo=1.0, sigma_hi=2.0, alpha=1.5):
    """Gaussian-based filter: light Gaussian denoising followed by a
    Difference-of-Gaussians band-pass added back to emphasise nodule edges while
    suppressing background noise. Works on 2D or 3D arrays in [0, 1]."""
    den = ndi.gaussian_filter(vol, sigma_denoise)
    dog = ndi.gaussian_filter(den, sigma_lo) - ndi.gaussian_filter(den, sigma_hi)
    return np.clip(den + alpha * dog, 0.0, 1.0).astype(np.float32)


def preprocess_patch(hu_patch, apply_mask=True, enhance=True, mask_dilate=4):
    """HU patch [Z,Y,X] -> normalised, lung-masked, edge-enhanced patch in [0,1]."""
    norm = window_normalize(hu_patch)
    if apply_mask:
        m = lung_mask_volume(hu_patch)
        if m.any():
            # dilate in-plane so pleural-attached nodules are never cut off
            m = np.stack([ndi.binary_dilation(s, morphology.disk(mask_dilate)) for s in m])
            norm = np.where(m, norm, 0.0).astype(np.float32)
    if enhance:
        norm = gaussian_edge_enhance(norm)
    return norm


def crop_center(vol, center, size=64, fill=-1000.0):
    """Crop a cube of side `size` around `center` (z,y,x), padding with `fill`."""
    half = size // 2
    padded = np.pad(vol, [(half, half)] * 3, mode="constant", constant_values=fill)
    z, y, x = [int(round(c)) + half for c in center]
    return padded[z - half:z + half, y - half:y + half, x - half:x + half]


def extract_views(patch, offsets=(-4, 0, 4)):
    """3D patch [Z,Y,X] -> dict of 2.5D views, each [len(offsets), H, W].
    axial = fixed z, coronal = fixed y, sagittal = fixed x."""
    c = patch.shape[0] // 2
    axial = np.stack([patch[c + o, :, :] for o in offsets])
    coronal = np.stack([patch[:, c + o, :] for o in offsets])
    sagittal = np.stack([patch[:, :, c + o] for o in offsets])
    return {"axial": axial, "coronal": coronal, "sagittal": sagittal}

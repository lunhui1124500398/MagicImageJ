"""
Auto-suggest ROI candidates using temporal + spatial analysis.

Temporal method: per-frame normalization + Gaussian blur + median background subtraction
Spatial method: Difference of Gaussians with compactness filter
Both masked to liquid cell region (auto-detected or user-provided).

Returns bboxes compatible with MagicImageJ's Batch_ROI system.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.ndimage import label, gaussian_filter


def auto_detect_roi_mask(stack: np.ndarray, *, percentile: float = 30) -> np.ndarray:
    """Detect liquid cell boundary from temporal std deviation."""
    n = stack.shape[0]
    mean_acc = np.zeros(stack.shape[1:], dtype=np.float64)
    sq_acc = np.zeros(stack.shape[1:], dtype=np.float64)
    for i in range(n):
        f = stack[i].astype(np.float64)
        mean_acc += f
        sq_acc += f * f
    mean_acc /= n
    std_frame = np.sqrt(np.maximum(sq_acc / n - mean_acc * mean_acc, 0)).astype(np.float32)
    blurred = gaussian_filter(std_frame, sigma=15)
    threshold = np.percentile(blurred, percentile)
    mask = blurred > threshold
    mask = ndimage.binary_dilation(mask, iterations=10)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask, iterations=15)
    labeled_mask, n = label(mask)
    if n > 1:
        sizes = [(labeled_mask == i).sum() for i in range(1, n + 1)]
        mask = labeled_mask == (np.argmax(sizes) + 1)
    return ndimage.binary_dilation(mask, iterations=5).astype(np.uint8)


def _temporal_candidates(stack: np.ndarray, roi_mask: np.ndarray, *,
                         sigma: float = 4.0, thresh_pct: float = 92,
                         min_area: int = 100, max_area: int = 4000,
                         max_dim: int = 100, pad: int = 10) -> list[dict]:
    n, h, w = stack.shape
    blurred = np.empty((n, h, w), dtype=np.float32)
    for i in range(n):
        frame_f = stack[i].astype(np.float32)
        mean_v = frame_f.mean()
        std_v = frame_f.std()
        if std_v < 1e-6:
            std_v = 1.0
        blurred[i] = gaussian_filter((frame_f - mean_v) / std_v, sigma=sigma)

    background = np.median(blurred, axis=0)
    motion = np.zeros((h, w), dtype=np.float32)
    for i in range(n):
        motion += np.abs(blurred[i] - background)
    motion /= n
    motion_masked = motion * roi_mask

    vals = motion_masked[roi_mask > 0]
    if vals.size == 0:
        return []
    threshold = np.percentile(vals, thresh_pct)
    binary = (motion_masked > threshold) & (roi_mask > 0)
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_erosion(binary, iterations=1)
    binary = ndimage.binary_dilation(binary, iterations=1)

    labeled_arr, n = label(binary)
    h, w = motion.shape
    bboxes = []
    for i in range(1, n + 1):
        comp = labeled_arr == i
        area = comp.sum()
        if area < min_area or area > max_area:
            continue
        rows = np.where(comp.any(axis=1))[0]
        cols = np.where(comp.any(axis=0))[0]
        y1, y2 = max(0, rows[0] - pad), min(h, rows[-1] + pad)
        x1, x2 = max(0, cols[0] - pad), min(w, cols[-1] + pad)
        if (y2 - y1) > max_dim or (x2 - x1) > max_dim:
            continue
        bboxes.append({"bbox": [x1, y1, x2, y2], "method": "temporal"})
    return bboxes


def _spatial_candidates(frame: np.ndarray, roi_mask: np.ndarray, *,
                        sigma_small: float = 5.0, sigma_large: float = 25.0,
                        thresh_pct: float = 97, min_area: int = 150,
                        max_area: int = 3500, max_dim: int = 80, pad: int = 10,
                        min_compactness: float = 0.25) -> list[dict]:
    normed = frame.astype(np.float32)
    normed = (normed - normed.min()) / (normed.max() - normed.min() + 1e-8)

    dog = gaussian_filter(normed, sigma=sigma_large) - gaussian_filter(normed, sigma=sigma_small)
    dog_masked = dog * roi_mask

    vals = dog_masked[roi_mask > 0]
    if vals.size == 0:
        return []
    threshold = np.percentile(vals, thresh_pct)
    binary = (dog_masked > threshold) & (roi_mask > 0)
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_erosion(binary, iterations=2)
    binary = ndimage.binary_dilation(binary, iterations=2)

    labeled_arr, n = label(binary)
    h, w = frame.shape
    bboxes = []
    for i in range(1, n + 1):
        comp = labeled_arr == i
        area = comp.sum()
        if area < min_area or area > max_area:
            continue
        rows = np.where(comp.any(axis=1))[0]
        cols = np.where(comp.any(axis=0))[0]
        bbox_h = rows[-1] - rows[0] + 1
        bbox_w = cols[-1] - cols[0] + 1
        if bbox_h * bbox_w > 0 and area / (bbox_h * bbox_w) < min_compactness:
            continue
        y1, y2 = max(0, rows[0] - pad), min(h, rows[-1] + pad)
        x1, x2 = max(0, cols[0] - pad), min(w, cols[-1] + pad)
        if (y2 - y1) > max_dim or (x2 - x1) > max_dim:
            continue
        bboxes.append({"bbox": [x1, y1, x2, y2], "method": "spatial"})
    return bboxes


def _merge_candidates(temporal: list[dict], spatial: list[dict],
                      iou_threshold: float = 0.3) -> list[dict]:
    all_cands = list(temporal)
    for sb in spatial:
        is_dup = False
        for existing in all_cands:
            ax1, ay1, ax2, ay2 = existing["bbox"]
            bx1, by1, bx2, by2 = sb["bbox"]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            if union > 0 and inter / union > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            all_cands.append(sb)
    return all_cands


def suggest_rois(stack: np.ndarray, *,
                 roi_mask: np.ndarray | None = None,
                 skip_frames: int = 15,
                 ref_frame_idx: int | None = None,
                 progress_callback=None) -> list[list[int]]:
    """
    Main entry point. Returns list of [x1, y1, x2, y2] bboxes.

    Parameters
    ----------
    stack : (T, H, W) numpy array, the loaded frame sequence
    roi_mask : optional binary mask (H, W), 1=search area. Auto-detected if None.
    skip_frames : skip this many early frames (rolling average warmup)
    ref_frame_idx : which frame to use for spatial detection. Auto-selected if None.
    progress_callback : optional callable(message: str) for status updates
    """
    def _status(msg):
        if progress_callback:
            progress_callback(msg)

    if stack.ndim != 3:
        return []

    usable = stack[skip_frames:] if skip_frames < stack.shape[0] else stack
    if usable.shape[0] < 5:
        usable = stack

    _status("Detecting ROI mask...")
    if roi_mask is None:
        roi_mask = auto_detect_roi_mask(usable)

    _status("Running temporal analysis...")
    t_bboxes = _temporal_candidates(usable, roi_mask)

    if ref_frame_idx is not None and skip_frames <= ref_frame_idx < stack.shape[0]:
        ref = stack[ref_frame_idx].astype(np.float32)
    else:
        ref = usable[usable.shape[0] // 2].astype(np.float32)

    _status("Running spatial analysis...")
    s_bboxes = _spatial_candidates(ref, roi_mask)

    _status("Merging candidates...")
    merged = _merge_candidates(t_bboxes, s_bboxes)

    _status(f"Found {len(merged)} candidates")
    return [c["bbox"] for c in merged]

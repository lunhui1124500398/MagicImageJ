"""
Generate high-recall local candidate boxes for semi-automatic auto-crop review.

This prototype is intentionally heuristic-first. It does not try to solve the
full low-SNR nanoparticle problem in one step. Instead, it produces a ranked
proposal list plus preview images that help the user decide whether the current
interaction style is saving time.

The script supports three practical knobs:

1. Focus / ignore regions
   Restrict search to user-relevant areas and suppress clearly irrelevant zones.
2. Seed boxes
   Optional example boxes that mildly bias the ranking toward similar patches.
3. Oversized candidate boxes
   Boxes are slightly enlarged on purpose so moving particles are less likely to
   be clipped during the proposal stage.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from export_auto_crop_training_data import load_session_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate semi-automatic auto-crop candidate boxes."
    )
    parser.add_argument(
        "--session-root",
        required=True,
        help="Session root that contains processing_log.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where proposal previews and JSON outputs are written.",
    )
    parser.add_argument(
        "--review-json",
        help=(
            "Optional review template JSON. If present, focus/ignore regions are "
            "loaded from it."
        ),
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index to review. Defaults to overview_frame from processing_log.",
    )
    parser.add_argument(
        "--sequence-dir",
        help=(
            "Optional explicit enhanced-frame sequence directory. Use this when "
            "the current enhanced stack exists in MagicImageJ but has not been "
            "exported into the session archive yet."
        ),
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=0,
        help=(
            "Candidate box size in pixels. Use 0 to auto-infer from seed boxes or "
            "existing manual ROIs when available."
        ),
    )
    parser.add_argument(
        "--box-scale",
        type=float,
        default=1.15,
        help="Scale factor applied to the base box size for proposal export.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=120,
        help="Number of final ranked candidates to export.",
    )
    parser.add_argument(
        "--label-top-k",
        type=int,
        default=40,
        help="Number of candidates to label in the preview image.",
    )
    parser.add_argument(
        "--focus-rect",
        action="append",
        default=[],
        help="Additional focus rectangle as x1,y1,x2,y2. Can be repeated.",
    )
    parser.add_argument(
        "--ignore-rect",
        action="append",
        default=[],
        help="Additional ignore rectangle as x1,y1,x2,y2. Can be repeated.",
    )
    parser.add_argument(
        "--seed-rect",
        action="append",
        default=[],
        help="Seed rectangle as x1,y1,x2,y2. Can be repeated.",
    )
    parser.add_argument(
        "--seed-id",
        action="append",
        type=int,
        default=[],
        help=(
            "Existing manual ROI id from processing_log used only as a local seed "
            "for ranking experiments."
        ),
    )
    parser.add_argument(
        "--seed-weight",
        type=float,
        default=0.18,
        help="How much seed similarity contributes to the final ranking.",
    )
    parser.add_argument(
        "--copy-review-template",
        action="store_true",
        help="Copy the input review template into the output directory for convenience.",
    )
    parser.add_argument(
        "--temporal-rerank",
        action="store_true",
        help="Use the full Enh sequence to re-rank candidates after the single-frame proposal stage.",
    )
    parser.add_argument(
        "--temporal-frame-step",
        type=int,
        default=1,
        help="Use every Nth frame during temporal re-ranking. 1 uses all frames.",
    )
    parser.add_argument(
        "--temporal-search-radius",
        type=int,
        default=0,
        help="Local search radius in pixels for temporal verification. 0 auto-infers from box size.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def odd(value: int | float) -> int:
    value = int(round(value))
    if value < 3:
        value = 3
    return value if value % 2 == 1 else value + 1


def clamp_bbox(bbox_xyxy: list[int | float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    return [x1, y1, x2, y2]


def parse_rect(spec: str) -> dict[str, Any]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid rectangle spec: {spec}")
    values = [int(float(p)) for p in parts]
    return {"type": "rect", "bbox_xyxy": values}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_processing_log_manual_boxes(bundle) -> list[dict[str, Any]]:
    rois = bundle.processing_data.get("batch_crop", {}).get("rois", [])
    manual_boxes = []
    for roi in rois:
        bbox_xyxy = [int(v) for v in roi["bbox"]]
        manual_boxes.append(
            {
                "id": int(roi["id"]),
                "bbox_xyxy": bbox_xyxy,
                "folder": roi.get("folder"),
                "frame_range_used": roi.get("frame_range_used"),
            }
        )
    return manual_boxes


def shape_to_polygon(shape: dict[str, Any]) -> np.ndarray:
    if "bbox_xyxy" in shape:
        x1, y1, x2, y2 = [int(v) for v in shape["bbox_xyxy"]]
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

    if "points" in shape:
        return np.array([[int(p[0]), int(p[1])] for p in shape["points"]], dtype=np.int32)

    raise ValueError(f"Unsupported region shape: {shape}")


def build_mask(
    height: int,
    width: int,
    focus_regions: list[dict[str, Any]],
    ignore_regions: list[dict[str, Any]],
) -> np.ndarray:
    if focus_regions:
        mask = np.zeros((height, width), dtype=np.uint8)
        for region in focus_regions:
            polygon = shape_to_polygon(region)
            cv2.fillPoly(mask, [polygon], 255)
    else:
        mask = np.full((height, width), 255, dtype=np.uint8)

    for region in ignore_regions:
        polygon = shape_to_polygon(region)
        cv2.fillPoly(mask, [polygon], 0)

    return mask


def normalize_map(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = values.astype(np.float32)
    if mask is not None and np.any(mask > 0):
        sample = values[mask > 0]
    else:
        sample = values.reshape(-1)

    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return np.zeros_like(values, dtype=np.float32)

    lo = float(np.percentile(sample, 1.0))
    hi = float(np.percentile(sample, 99.5))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)

    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return normalized.astype(np.float32)


def compute_blobness(image: np.ndarray, sigmas: list[float]) -> np.ndarray:
    responses = []
    for sigma in sigmas:
        sigma = max(0.8, float(sigma))
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        response = np.maximum(-lap * (sigma * sigma), 0.0)
        responses.append(response)

    if not responses:
        return np.zeros_like(image, dtype=np.float32)
    return np.max(np.stack(responses, axis=0), axis=0).astype(np.float32)


def compute_structure_coherence(image: np.ndarray, sigma: float) -> np.ndarray:
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(grad_x * grad_x, (0, 0), sigma)
    jyy = cv2.GaussianBlur(grad_y * grad_y, (0, 0), sigma)
    jxy = cv2.GaussianBlur(grad_x * grad_y, (0, 0), sigma)

    trace = jxx + jyy
    delta = np.sqrt(np.maximum((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy, 0.0))
    lambda1 = 0.5 * (trace + delta)
    lambda2 = 0.5 * (trace - delta)
    coherence = (lambda1 - lambda2) / np.maximum(lambda1 + lambda2, 1e-6)
    return np.clip(coherence, 0.0, 1.0).astype(np.float32)


def compute_summary_images(stack: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": stack.mean(axis=0).astype(np.float32),
        "min": stack.min(axis=0).astype(np.float32),
        "std": stack.std(axis=0).astype(np.float32),
    }


def box_filter_contrast(image: np.ndarray, inner: int, outer: int) -> np.ndarray:
    inner_mean = cv2.blur(image, (inner, inner))
    outer_mean = cv2.blur(image, (outer, outer))
    return outer_mean - inner_mean


def compute_maps(
    frame: np.ndarray,
    summaries: dict[str, np.ndarray],
    box_size: int,
    search_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    mean_img = summaries["mean"]
    min_img = summaries["min"]
    std_img = summaries["std"]

    inner = odd(box_size)
    outer = odd(box_size * 1.8)
    blackhat_kernel = odd(box_size * 0.35)

    temporal_delta = mean_img - min_img
    patch_mean = box_filter_contrast(mean_img, inner, outer)
    patch_frame = box_filter_contrast(frame, inner, outer)
    spot_mean = cv2.morphologyEx(
        mean_img.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        np.ones((blackhat_kernel, blackhat_kernel), dtype=np.uint8),
    ).astype(np.float32)
    spot_frame = cv2.morphologyEx(
        frame.astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        np.ones((blackhat_kernel, blackhat_kernel), dtype=np.uint8),
    ).astype(np.float32)
    inv_mean = 255.0 - mean_img
    inv_frame = 255.0 - frame
    dog_mean = cv2.GaussianBlur(inv_mean, (0, 0), max(1.0, box_size / 18.0)) - cv2.GaussianBlur(
        inv_mean, (0, 0), max(2.0, box_size / 7.0)
    )
    temporal = cv2.blur(temporal_delta, (inner, inner))
    volatility = cv2.blur(std_img, (inner, inner))
    blob_sigmas = [max(1.4, box_size / 18.0), max(2.2, box_size / 12.0), max(3.0, box_size / 8.0)]
    blob_mean = compute_blobness(inv_mean, blob_sigmas)
    blob_frame = compute_blobness(inv_frame, blob_sigmas)
    coherence = compute_structure_coherence(mean_img, sigma=max(1.2, box_size / 10.0))

    grad_x = cv2.Sobel(mean_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mean_img, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.GaussianBlur(np.sqrt(grad_x * grad_x + grad_y * grad_y), (0, 0), box_size / 6.0)

    patch_mean_n = normalize_map(patch_mean, search_mask)
    patch_frame_n = normalize_map(patch_frame, search_mask)
    spot_mean_n = normalize_map(spot_mean, search_mask)
    spot_frame_n = normalize_map(spot_frame, search_mask)
    dog_mean_n = normalize_map(dog_mean, search_mask)
    temporal_n = normalize_map(temporal, search_mask)
    volatility_n = normalize_map(volatility, search_mask)
    gradient_n = normalize_map(gradient, search_mask)
    blob_mean_n = normalize_map(blob_mean, search_mask)
    blob_frame_n = normalize_map(blob_frame, search_mask)
    coherence_n = normalize_map(coherence, search_mask)

    combined = (
        0.24 * patch_mean_n
        + 0.13 * patch_frame_n
        + 0.12 * spot_mean_n
        + 0.06 * spot_frame_n
        + 0.10 * dog_mean_n
        + 0.16 * blob_mean_n
        + 0.07 * blob_frame_n
        + 0.07 * temporal_n
        + 0.05 * volatility_n
    )
    combined = np.clip(combined - 0.10 * gradient_n - 0.11 * coherence_n, 0.0, 1.0)

    margin = outer // 2 + 2
    combined[:margin, :] = 0.0
    combined[-margin:, :] = 0.0
    combined[:, :margin] = 0.0
    combined[:, -margin:] = 0.0

    maps = {
        "combined": combined,
        "patch_mean": patch_mean_n,
        "patch_frame": patch_frame_n,
        "spot_mean": spot_mean_n,
        "spot_frame": spot_frame_n,
        "dog_mean": dog_mean_n,
        "temporal": temporal_n,
        "volatility": volatility_n,
        "gradient": gradient_n,
        "blob_mean": blob_mean_n,
        "blob_frame": blob_frame_n,
        "coherence": coherence_n,
    }

    for key, value in maps.items():
        masked = value.copy()
        masked[search_mask == 0] = 0.0
        maps[key] = masked

    return maps


def collect_local_maxima(
    score_map: np.ndarray,
    search_mask: np.ndarray,
    threshold_percentile: float,
    suppression_radius: float,
    local_window: int,
    max_points: int,
) -> list[tuple[int, int, float]]:
    masked = score_map.copy()
    masked[search_mask == 0] = 0.0
    sample = masked[search_mask > 0]
    sample = sample[sample > 0]
    if sample.size == 0:
        return []

    threshold = float(np.percentile(sample, threshold_percentile))
    local_max = cv2.dilate(masked, np.ones((local_window, local_window), dtype=np.uint8))
    ys, xs = np.where((masked == local_max) & (masked >= threshold))
    if ys.size == 0:
        return []

    order = np.argsort(masked[ys, xs])[::-1]
    radius_sq = suppression_radius * suppression_radius
    points: list[tuple[int, int, float]] = []
    for idx in order:
        y = int(ys[idx])
        x = int(xs[idx])
        value = float(masked[y, x])
        if any((py - y) * (py - y) + (px - x) * (px - x) < radius_sq for py, px, _ in points):
            continue
        points.append((y, x, value))
        if len(points) >= max_points:
            break
    return points


def build_seed_templates(frame: np.ndarray, seed_boxes: list[list[int]]) -> list[np.ndarray]:
    templates = []
    for bbox in seed_boxes:
        x1, y1, x2, y2 = bbox
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        patch = cv2.resize(patch, (31, 31), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)
        templates.append(patch)
    return templates


def compute_seed_similarity(
    frame: np.ndarray,
    bbox_xyxy: list[int],
    templates: list[np.ndarray],
) -> float:
    if not templates:
        return 0.0

    x1, y1, x2, y2 = bbox_xyxy
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
        return 0.0

    patch = cv2.resize(patch, (31, 31), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    patch = (patch - patch.mean()) / (patch.std() + 1e-6)

    best = -1.0
    for template in templates:
        value = float(np.mean(patch * template))
        if value > best:
            best = value
    return float(np.clip((best + 1.0) / 2.0, 0.0, 1.0))


def ring_contrast_feature(
    image: np.ndarray,
    bbox_xyxy: list[int],
    ring_scale: float,
    global_scale: float,
) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half_w = max(4, (x2 - x1) // 2)
    half_h = max(4, (y2 - y1) // 2)
    ring_half_w = int(round(half_w * ring_scale))
    ring_half_h = int(round(half_h * ring_scale))

    rx1 = max(0, cx - ring_half_w)
    ry1 = max(0, cy - ring_half_h)
    rx2 = min(image.shape[1], cx + ring_half_w)
    ry2 = min(image.shape[0], cy + ring_half_h)

    outer = image[ry1:ry2, rx1:rx2]
    inner = image[y1:y2, x1:x2]
    if outer.size == 0 or inner.size == 0:
        return 0.0

    mask = np.ones_like(outer, dtype=bool)
    iy1 = y1 - ry1
    iy2 = iy1 + inner.shape[0]
    ix1 = x1 - rx1
    ix2 = ix1 + inner.shape[1]
    mask[iy1:iy2, ix1:ix2] = False
    ring = outer[mask]
    if ring.size == 0:
        return 0.0

    contrast = (float(np.mean(ring)) - float(np.mean(inner))) / max(global_scale, 1e-6)
    return float(np.clip((contrast + 1.0) / 2.0, 0.0, 1.0))


def iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float((ax2 - ax1) * (ay2 - ay1))
    area_b = float((bx2 - bx1) * (by2 - by1))
    return inter / max(area_a + area_b - inter, 1e-6)


def nms_candidates(
    candidates: list[dict[str, Any]],
    iou_threshold: float,
    top_k: int,
    cell_size: int,
    crowd_radius: float,
    cell_penalty: float,
    crowd_penalty: float,
    scan_limit: int = 320,
) -> list[dict[str, Any]]:
    def grid_key(candidate: dict[str, Any]) -> tuple[int, int]:
        cx, cy = candidate["center_xy"]
        return int(cx // cell_size), int(cy // cell_size)

    kept: list[dict[str, Any]] = []
    remaining = sorted(candidates, key=lambda item: item["score"], reverse=True)
    cell_counts: dict[tuple[int, int], int] = defaultdict(int)
    crowd_radius_sq = crowd_radius * crowd_radius

    while remaining and len(kept) < top_k:
        best_index = None
        best_effective_score = -1e9
        scan_count = min(scan_limit, len(remaining))

        for index in range(scan_count):
            candidate = remaining[index]
            if any(iou(candidate["bbox_xyxy"], existing["bbox_xyxy"]) > iou_threshold for existing in kept):
                continue

            key = grid_key(candidate)
            cx, cy = candidate["center_xy"]
            local_density = 0.0
            for existing in kept:
                ex, ey = existing["center_xy"]
                dist_sq = float((cx - ex) * (cx - ex) + (cy - ey) * (cy - ey))
                if dist_sq >= crowd_radius_sq:
                    continue
                dist = np.sqrt(dist_sq)
                local_density += max(0.0, 1.0 - dist / max(crowd_radius, 1e-6))

            effective_score = (
                float(candidate["score"])
                - cell_penalty * cell_counts[key]
                - crowd_penalty * local_density
            )
            if effective_score > best_effective_score:
                best_effective_score = effective_score
                best_index = index

        if best_index is None:
            break

        chosen = remaining.pop(best_index)
        chosen = dict(chosen)
        chosen["selection_score"] = round(float(best_effective_score), 6)
        chosen["grid_key"] = list(grid_key(chosen))
        cell_counts[tuple(chosen["grid_key"])] += 1
        kept.append(chosen)

    return kept


def longest_true_run(flags: np.ndarray) -> int:
    best = 0
    current = 0
    for value in flags.astype(bool).tolist():
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def compute_temporal_candidate_features(
    stack: np.ndarray,
    candidates: list[dict[str, Any]],
    box_size: int,
    search_mask: np.ndarray,
    frame_step: int,
    search_radius: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    frame_step = max(1, int(frame_step))
    sampled_indices = list(range(0, stack.shape[0], frame_step))
    if not sampled_indices:
        sampled_indices = [0]

    height, width = stack.shape[1], stack.shape[2]
    search_radius = max(8, int(search_radius))
    inner = odd(box_size)
    outer = odd(box_size * 1.8)
    blackhat_kernel = odd(box_size * 0.35)
    blob_sigmas = [max(1.4, box_size / 18.0), max(2.2, box_size / 12.0), max(3.0, box_size / 8.0)]

    origins = np.array([candidate["center_xy"] for candidate in candidates], dtype=np.int32)
    scores_by_candidate = [[] for _ in candidates]
    positions_by_candidate = [[] for _ in candidates]

    for frame_index in sampled_indices:
        frame_u8 = stack[frame_index]
        frame = frame_u8.astype(np.float32)
        patch_map = normalize_map(box_filter_contrast(frame, inner, outer), search_mask)
        spot_map = normalize_map(
            cv2.morphologyEx(
                frame_u8,
                cv2.MORPH_BLACKHAT,
                np.ones((blackhat_kernel, blackhat_kernel), dtype=np.uint8),
            ).astype(np.float32),
            search_mask,
        )
        blob_map = normalize_map(compute_blobness(255.0 - frame, blob_sigmas), search_mask)
        response_map = (0.42 * patch_map + 0.23 * spot_map + 0.35 * blob_map).astype(np.float32)
        response_map[search_mask == 0] = 0.0

        for idx, (origin_x, origin_y) in enumerate(origins.tolist()):
            x1 = max(0, origin_x - search_radius)
            y1 = max(0, origin_y - search_radius)
            x2 = min(width, origin_x + search_radius + 1)
            y2 = min(height, origin_y + search_radius + 1)
            local = response_map[y1:y2, x1:x2]
            if local.size == 0:
                scores_by_candidate[idx].append(0.0)
                positions_by_candidate[idx].append([origin_x, origin_y])
                continue

            local_max_index = int(np.argmax(local))
            local_y, local_x = np.unravel_index(local_max_index, local.shape)
            best_x = int(x1 + local_x)
            best_y = int(y1 + local_y)
            local_score = float(local[local_y, local_x])
            scores_by_candidate[idx].append(local_score)
            positions_by_candidate[idx].append([best_x, best_y])

    features = []
    support_target = max(8.0, len(sampled_indices) * 0.18)
    run_target = max(5.0, len(sampled_indices) * 0.10)
    compact_scale = max(10.0, float(box_size) * 0.55)
    search_scale = max(8.0, float(search_radius))

    for idx, candidate in enumerate(candidates):
        score_values = np.array(scores_by_candidate[idx], dtype=np.float32)
        if score_values.size == 0:
            features.append(
                {
                    "sampled_frame_count": 0,
                    "best8_mean": 0.0,
                    "tail_mean": 0.0,
                    "support_count": 0,
                    "longest_run": 0,
                    "compactness": 0.0,
                    "motion_span_norm": 0.0,
                    "temporal_peakiness": 0.0,
                    "temporal_score": 0.0,
                    "suggested_visible_frame_range": None,
                }
            )
            continue

        top_count = min(8, score_values.size)
        top_scores = np.sort(score_values)[-top_count:]
        best8_mean = float(np.mean(top_scores))
        tail_mean = float(np.percentile(score_values, 45.0))
        peak_gap = max(0.0, best8_mean - tail_mean)
        temporal_peakiness = float(np.clip(peak_gap / max(best8_mean, 1e-6), 0.0, 1.0))
        support_threshold = max(0.24, tail_mean + 0.45 * peak_gap)
        support_flags = score_values >= support_threshold
        support_count = int(np.count_nonzero(support_flags))
        longest_run = longest_true_run(support_flags)

        supported_positions = np.array(positions_by_candidate[idx], dtype=np.float32)[support_flags]
        if supported_positions.shape[0] >= 2:
            median_pos = np.median(supported_positions, axis=0)
            distances = np.sqrt(np.sum((supported_positions - median_pos) ** 2, axis=1))
            compactness = float(np.clip(1.0 - float(np.mean(distances)) / compact_scale, 0.0, 1.0))
            motion_span = float(np.percentile(distances, 90.0))
        elif supported_positions.shape[0] == 1:
            compactness = 1.0
            motion_span = 0.0
        else:
            compactness = 0.0
            motion_span = float(search_radius)

        support_norm = float(np.clip(support_count / support_target, 0.0, 1.0))
        run_norm = float(np.clip(longest_run / run_target, 0.0, 1.0))
        best_norm = float(np.clip((best8_mean - 0.18) / 0.50, 0.0, 1.0))
        peakiness_norm = temporal_peakiness
        motion_span_norm = float(np.clip(motion_span / search_scale, 0.0, 1.0))

        temporal_score = (
            0.34 * best_norm
            + 0.24 * peakiness_norm
            + 0.18 * support_norm
            + 0.14 * run_norm
            + 0.10 * compactness
        )

        supported_sample_indices = [sampled_indices[i] for i, flag in enumerate(support_flags.tolist()) if flag]
        if supported_sample_indices:
            visible_range = [int(supported_sample_indices[0]), int(supported_sample_indices[-1])]
        else:
            visible_range = None

        features.append(
            {
                "sampled_frame_count": int(len(sampled_indices)),
                "best8_mean": round(best8_mean, 6),
                "tail_mean": round(tail_mean, 6),
                "support_count": support_count,
                "longest_run": longest_run,
                "compactness": round(compactness, 6),
                "motion_span_norm": round(motion_span_norm, 6),
                "temporal_peakiness": round(peakiness_norm, 6),
                "temporal_score": round(float(np.clip(temporal_score, 0.0, 1.0)), 6),
                "suggested_visible_frame_range": visible_range,
            }
        )

    return features


def render_overlay(
    frame: np.ndarray,
    focus_regions: list[dict[str, Any]],
    ignore_regions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    label_top_k: int,
    output_path: Path,
    manual_boxes: list[dict[str, Any]] | None = None,
) -> None:
    rgb = np.repeat(frame.astype(np.uint8)[..., None], 3, axis=2)
    image = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for region in focus_regions:
        polygon = [tuple(point) for point in shape_to_polygon(region).tolist()]
        draw.polygon(polygon, fill=(0, 180, 80, 36), outline=(0, 180, 80, 140))

    for region in ignore_regions:
        polygon = [tuple(point) for point in shape_to_polygon(region).tolist()]
        draw.polygon(polygon, fill=(220, 60, 60, 52), outline=(220, 60, 60, 180))

    if manual_boxes:
        for manual in manual_boxes:
            x1, y1, x2, y2 = manual["bbox_xyxy"]
            draw.rectangle([x1, y1, x2, y2], outline=(80, 220, 120, 180), width=2)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for index, candidate in enumerate(candidates, start=1):
        x1, y1, x2, y2 = candidate["bbox_xyxy"]
        color = (255, 191, 0, 255) if index <= label_top_k else (255, 191, 0, 180)
        width = 3 if index <= label_top_k else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if index <= label_top_k:
            label = f"C{index}"
            draw.text((x1 + 2, max(0, y1 - 18)), label, fill=(230, 235, 255, 255), font=font)

    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)


def render_score_map(score_map: np.ndarray, output_path: Path) -> None:
    normalized = (np.clip(score_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)).save(output_path)


def evaluate_candidates(
    candidates: list[dict[str, Any]],
    manual_boxes: list[dict[str, Any]],
    top_checks: list[int],
) -> dict[str, Any]:
    if not manual_boxes:
        return {}

    metrics: dict[str, Any] = {
        "manual_roi_count": len(manual_boxes),
        "top_k_checks": [],
    }

    manual_only = [item["bbox_xyxy"] for item in manual_boxes]
    for top_k in top_checks:
        subset = candidates[:top_k]
        center_hits = set()
        overlap_hits = set()
        false_positives = 0
        for candidate in subset:
            x1, y1, x2, y2 = candidate["bbox_xyxy"]
            cx, cy = candidate["center_xy"]
            center_hit = False
            overlap_hit = False
            for idx, manual in enumerate(manual_only):
                mx1, my1, mx2, my2 = manual
                if mx1 <= cx <= mx2 and my1 <= cy <= my2:
                    center_hits.add(idx)
                    center_hit = True
                if iou(candidate["bbox_xyxy"], manual) > 0.01:
                    overlap_hits.add(idx)
                    overlap_hit = True
            if not overlap_hit:
                false_positives += 1

        metrics["top_k_checks"].append(
            {
                "top_k": top_k,
                "center_hit_count": len(center_hits),
                "overlap_hit_count": len(overlap_hits),
                "false_positive_count": false_positives,
            }
        )

    return metrics


def collect_seed_boxes(
    args: argparse.Namespace,
    review_data: dict[str, Any] | None,
    manual_boxes: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[list[int]]:
    seed_boxes: list[list[int]] = []

    for spec in args.seed_rect:
        seed_boxes.append(clamp_bbox(parse_rect(spec)["bbox_xyxy"], width, height))

    manual_by_id = {item["id"]: item["bbox_xyxy"] for item in manual_boxes}
    for seed_id in args.seed_id:
        if seed_id in manual_by_id:
            seed_boxes.append(clamp_bbox(manual_by_id[seed_id], width, height))

    if review_data:
        for candidate in review_data.get("accepted_candidates", []):
            if "bbox_xyxy" in candidate:
                seed_boxes.append(clamp_bbox(candidate["bbox_xyxy"], width, height))

    unique = []
    seen = set()
    for bbox in seed_boxes:
        key = tuple(bbox)
        if key in seen:
            continue
        seen.add(key)
        unique.append(bbox)
    return unique


def infer_box_size(
    requested_box_size: int,
    seed_boxes: list[list[int]],
    manual_boxes: list[dict[str, Any]],
) -> int:
    if requested_box_size > 0:
        return requested_box_size

    size_samples = []
    for bbox in seed_boxes:
        x1, y1, x2, y2 = bbox
        size_samples.append(max(x2 - x1, y2 - y1))

    for item in manual_boxes:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        size_samples.append(max(x2 - x1, y2 - y1))

    if not size_samples:
        return 68

    return int(np.median(np.array(size_samples)))


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir).resolve())

    sequence_dir = Path(args.sequence_dir) if args.sequence_dir else None
    bundle = load_session_bundle(
        Path(args.session_root),
        sequence_dir_override=sequence_dir,
    )
    frame_index = args.frame_index if args.frame_index is not None else bundle.overview_frame

    frame_path = bundle.sequence_dir / f"frame_{frame_index:04d}.png"
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    frame = np.array(Image.open(frame_path), dtype=np.uint8)
    height, width = frame.shape

    review_data = load_json(Path(args.review_json)) if args.review_json else None
    focus_regions = list(review_data.get("focus_regions", [])) if review_data else []
    ignore_regions = list(review_data.get("ignore_regions", [])) if review_data else []
    focus_regions.extend(parse_rect(spec) for spec in args.focus_rect)
    ignore_regions.extend(parse_rect(spec) for spec in args.ignore_rect)

    search_mask = build_mask(height, width, focus_regions, ignore_regions)
    manual_boxes = get_processing_log_manual_boxes(bundle)
    seed_boxes = collect_seed_boxes(args, review_data, manual_boxes, width, height)
    base_box_size = infer_box_size(args.box_size, seed_boxes, manual_boxes)
    export_box_size = max(20, int(round(base_box_size * args.box_scale)))

    files = sorted(bundle.sequence_dir.glob("frame_*.png"))
    stack = np.stack([np.array(Image.open(path), dtype=np.uint8) for path in files], axis=0)
    summaries = compute_summary_images(stack)
    maps = compute_maps(frame.astype(np.float32), summaries, base_box_size, search_mask)

    local_window = odd(max(5, base_box_size * 0.22))
    suppression_radius = max(10.0, base_box_size * 0.42)
    peak_specs = [
        ("combined", maps["combined"], 92.0, 800),
        ("patch_mean", maps["patch_mean"], 93.5, 500),
        ("patch_frame", maps["patch_frame"], 94.0, 400),
        ("blob_mean", maps["blob_mean"], 94.0, 420),
        ("spot_mean", maps["spot_mean"], 95.5, 400),
        ("temporal", maps["temporal"], 96.0, 300),
    ]

    raw_points: list[dict[str, Any]] = []
    for source_name, score_map, percentile, max_points in peak_specs:
        points = collect_local_maxima(
            score_map=score_map,
            search_mask=search_mask,
            threshold_percentile=percentile,
            suppression_radius=suppression_radius,
            local_window=local_window,
            max_points=max_points,
        )
        for y, x, value in points:
            raw_points.append({"source": source_name, "y": y, "x": x, "source_score": value})

    deduped_points: list[dict[str, Any]] = []
    merge_radius_sq = (base_box_size * 0.35) ** 2
    for item in sorted(raw_points, key=lambda row: row["source_score"], reverse=True):
        if any(
            (existing["y"] - item["y"]) ** 2 + (existing["x"] - item["x"]) ** 2 < merge_radius_sq
            for existing in deduped_points
        ):
            continue
        deduped_points.append(item)

    templates = build_seed_templates(frame.astype(np.float32), seed_boxes)
    frame_scale = max(8.0, float(np.std(frame.astype(np.float32))))
    temporal_scale = max(4.0, float(np.std((summaries["mean"] - summaries["min"]).astype(np.float32))))

    candidates: list[dict[str, Any]] = []
    half_box = export_box_size / 2.0
    temporal_map = maps["temporal"]
    volatility_map = maps["volatility"]
    combined_map = maps["combined"]
    patch_map = maps["patch_mean"]
    spot_map = np.maximum(maps["spot_mean"], maps["spot_frame"])
    blob_map = np.maximum(maps["blob_mean"], 0.7 * maps["blob_frame"])
    coherence_map = maps["coherence"]
    source_bias = {
        "combined": 0.018,
        "patch_mean": 0.016,
        "patch_frame": 0.004,
        "blob_mean": 0.014,
        "spot_mean": -0.012,
        "temporal": -0.004,
    }

    for point in deduped_points:
        x = int(point["x"])
        y = int(point["y"])
        bbox = clamp_bbox(
            [x - half_box, y - half_box, x + half_box, y + half_box],
            width=width,
            height=height,
        )

        if np.mean(search_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]) < 10:
            continue

        mean_contrast = ring_contrast_feature(summaries["mean"], bbox, ring_scale=1.55, global_scale=frame_scale)
        frame_contrast = ring_contrast_feature(frame.astype(np.float32), bbox, ring_scale=1.55, global_scale=frame_scale)
        delta_crop = summaries["mean"][bbox[1] : bbox[3], bbox[0] : bbox[2]] - summaries["min"][bbox[1] : bbox[3], bbox[0] : bbox[2]]
        temporal_value = float(np.clip(np.mean(delta_crop) / (2.0 * temporal_scale), 0.0, 1.0))
        seed_similarity = compute_seed_similarity(frame.astype(np.float32), bbox, templates)
        border_penalty = 0.0
        if bbox[0] <= 4 or bbox[1] <= 4 or bbox[2] >= width - 4 or bbox[3] >= height - 4:
            border_penalty = 0.08
        coherence_penalty = float(coherence_map[y, x])
        blob_value = float(blob_map[y, x])
        source_prior = source_bias.get(point["source"], 0.0)

        score = (
            0.24 * float(combined_map[y, x])
            + 0.15 * float(patch_map[y, x])
            + 0.13 * blob_value
            + 0.12 * mean_contrast
            + 0.09 * frame_contrast
            + 0.08 * float(spot_map[y, x])
            + 0.06 * temporal_value
            + 0.04 * float(temporal_map[y, x])
            + 0.03 * float(volatility_map[y, x])
            + float(args.seed_weight) * seed_similarity
            + source_prior
            - 0.10 * coherence_penalty
            - border_penalty
        )

        candidates.append(
            {
                "source": point["source"],
                "bbox_xyxy": bbox,
                "center_xy": [x, y],
                "score": round(float(score), 6),
                "base_score": round(float(score), 6),
                "features": {
                    "combined_center": round(float(combined_map[y, x]), 6),
                    "patch_center": round(float(patch_map[y, x]), 6),
                    "spot_center": round(float(spot_map[y, x]), 6),
                    "blob_center": round(blob_value, 6),
                    "temporal_center": round(float(temporal_map[y, x]), 6),
                    "volatility_center": round(float(volatility_map[y, x]), 6),
                    "coherence_center": round(coherence_penalty, 6),
                    "mean_ring_contrast": round(float(mean_contrast), 6),
                    "frame_ring_contrast": round(float(frame_contrast), 6),
                    "temporal_box_mean": round(float(temporal_value), 6),
                    "seed_similarity": round(float(seed_similarity), 6),
                    "source_prior": round(float(source_prior), 6),
                    "border_penalty": round(float(border_penalty), 6),
                },
            }
        )

    temporal_search_radius = (
        int(args.temporal_search_radius)
        if int(args.temporal_search_radius) > 0
        else max(14, int(round(export_box_size * 0.80)))
    )
    if args.temporal_rerank:
        temporal_features = compute_temporal_candidate_features(
            stack=stack,
            candidates=candidates,
            box_size=base_box_size,
            search_mask=search_mask,
            frame_step=args.temporal_frame_step,
            search_radius=temporal_search_radius,
        )
        for candidate, temporal in zip(candidates, temporal_features):
            candidate["temporal"] = temporal
            adjusted_score = (
                float(candidate["base_score"])
                + 0.16 * float(temporal["temporal_score"])
                + 0.05 * float(temporal["temporal_peakiness"])
                + 0.03 * float(temporal["compactness"])
            )
            candidate["score"] = round(float(adjusted_score), 6)
    else:
        for candidate in candidates:
            candidate["temporal"] = {
                "sampled_frame_count": 0,
                "best8_mean": 0.0,
                "tail_mean": 0.0,
                "support_count": 0,
                "longest_run": 0,
                "compactness": 0.0,
                "motion_span_norm": 0.0,
                "temporal_peakiness": 0.0,
                "temporal_score": 0.0,
                "suggested_visible_frame_range": None,
            }

    diversity_cell_size = max(32, int(round(base_box_size * 1.6)))
    crowd_radius = max(24.0, float(base_box_size * 1.25))
    ranked_candidates = nms_candidates(
        candidates,
        iou_threshold=0.32,
        top_k=args.top_k,
        cell_size=diversity_cell_size,
        crowd_radius=crowd_radius,
        cell_penalty=0.026,
        crowd_penalty=0.075,
    )
    for index, candidate in enumerate(ranked_candidates, start=1):
        candidate["candidate_id"] = f"C{index:03d}"

    overlay_all = output_dir / f"frame_{frame_index:04d}_candidates_top{len(ranked_candidates)}.png"
    overlay_labeled = output_dir / f"frame_{frame_index:04d}_candidates_top{args.label_top_k}_labeled.png"
    heatmap_path = output_dir / f"frame_{frame_index:04d}_score_heatmap.png"
    mask_path = output_dir / f"frame_{frame_index:04d}_search_mask.png"
    proposal_json_path = output_dir / f"frame_{frame_index:04d}_candidates.json"
    summary_json_path = output_dir / "proposal_summary.json"

    render_overlay(
        frame=frame.astype(np.float32),
        focus_regions=focus_regions,
        ignore_regions=ignore_regions,
        candidates=ranked_candidates,
        label_top_k=0,
        output_path=overlay_all,
        manual_boxes=None,
    )
    render_overlay(
        frame=frame.astype(np.float32),
        focus_regions=focus_regions,
        ignore_regions=ignore_regions,
        candidates=ranked_candidates,
        label_top_k=args.label_top_k,
        output_path=overlay_labeled,
        manual_boxes=manual_boxes if manual_boxes else None,
    )
    render_score_map(maps["combined"], heatmap_path)
    Image.fromarray(search_mask).save(mask_path)

    evaluation = evaluate_candidates(
        candidates=ranked_candidates,
        manual_boxes=manual_boxes,
        top_checks=[10, 20, 40, 80, min(args.top_k, 120)],
    )

    proposal_bundle = {
        "session_root": str(bundle.session_root),
        "frame_index": frame_index,
        "source_image": str(frame_path),
        "sequence_dir": str(bundle.sequence_dir),
        "review_json": str(Path(args.review_json).resolve()) if args.review_json else None,
        "focus_regions": focus_regions,
        "ignore_regions": ignore_regions,
        "seed_boxes": seed_boxes,
        "parameters": {
            "base_box_size": base_box_size,
            "export_box_size": export_box_size,
            "top_k": args.top_k,
            "label_top_k": args.label_top_k,
            "seed_weight": args.seed_weight,
            "suppression_radius": suppression_radius,
            "local_window": local_window,
            "diversity_cell_size": diversity_cell_size,
            "crowd_radius": crowd_radius,
            "temporal_rerank": bool(args.temporal_rerank),
            "temporal_frame_step": int(args.temporal_frame_step),
            "temporal_search_radius": int(temporal_search_radius),
        },
        "artifacts": {
            "overlay_all": str(overlay_all),
            "overlay_labeled": str(overlay_labeled),
            "score_heatmap": str(heatmap_path),
            "search_mask": str(mask_path),
        },
        "candidates": ranked_candidates,
        "debug_evaluation": evaluation,
    }
    proposal_json_path.write_text(
        json.dumps(proposal_bundle, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "session_root": str(bundle.session_root),
        "frame_index": frame_index,
        "candidate_count": len(ranked_candidates),
        "seed_count": len(seed_boxes),
        "manual_roi_count": len(manual_boxes),
        "focus_region_count": len(focus_regions),
        "ignore_region_count": len(ignore_regions),
        "proposal_json": str(proposal_json_path),
        "overlay_labeled": str(overlay_labeled),
        "overlay_all": str(overlay_all),
        "score_heatmap": str(heatmap_path),
        "debug_evaluation": evaluation,
        "notes": [
            "This is a heuristic proposal generator, not a trained detector.",
            "Focus or ignore regions are strongly recommended for useful ranking.",
            "Temporal disappearance is intentionally not filtered at this stage.",
        ]
        + (
            [
                "Temporal reranking is enabled and uses the Enh sequence to re-score candidates.",
            ]
            if args.temporal_rerank
            else []
        ),
    }
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.copy_review_template and args.review_json:
        review_path = Path(args.review_json)
        target = output_dir / review_path.name
        target.write_text(review_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

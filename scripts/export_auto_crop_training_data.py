"""
Export semi-automatic auto-crop training data from a MagicImageJ session.

This script creates two dataset views:

1. A canonical JSONL/JSON export that preserves rich metadata for later
   iteration, including placeholders for focus masks, primary target points,
   rejected candidates, and temporal visibility annotations.
2. A derived YOLO detection dataset that uses the manually reviewed overview
   frame(s) only. This avoids introducing false positives on frames where a
   particle later disappears.

The current design intentionally decouples:
  - "where should a box go on the overview frame?"
  - "for which frames is the particle still visible?"

That means users do not need to manually remove vanished particles before the
first export. Temporal presence can be annotated later using the canonical
review templates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_CLASS_NAME = "nanoparticle"


@dataclass
class SessionBundle:
    session_root: Path
    processing_log: Path
    sequence_dir: Path
    overview_frame: int
    processing_data: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MagicImageJ auto-crop training data."
    )
    parser.add_argument(
        "--session-root",
        action="append",
        required=True,
        help=(
            "Session root that contains processing_log.json. Repeat this flag "
            "to export multiple sessions into one dataset."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where the canonical and YOLO exports will be written.",
    )
    parser.add_argument(
        "--class-name",
        default=DEFAULT_CLASS_NAME,
        help="Class name used in YOLO export.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for the derived YOLO dataset.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to place overview images into the exported YOLO dataset.",
    )
    return parser.parse_args()


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_bbox_xyxy(
    bbox_xyxy: list[int | float], width: int, height: int
) -> list[int]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    return [x1, y1, x2, y2]


def find_processing_log(session_root: Path) -> Path:
    processing_log = session_root / "processing_log.json"
    if not processing_log.exists():
        raise FileNotFoundError(f"Missing processing_log.json under {session_root}")
    return processing_log


def validate_sequence_dir(sequence_dir: Path) -> Path:
    sequence_dir = sequence_dir.resolve()
    if not sequence_dir.exists() or not sequence_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")
    if not any(sequence_dir.glob("frame_*.png")):
        raise FileNotFoundError(
            f"Sequence directory does not contain frame_*.png files: {sequence_dir}"
        )
    return sequence_dir


def infer_sequence_dir(session_root: Path) -> Path:
    exported_root = session_root / "Exported_Sequences"
    candidates: list[Path] = []

    if exported_root.exists():
        for path in exported_root.iterdir():
            if path.is_dir() and path.name.startswith("Enh_"):
                if any(path.glob("frame_*.png")):
                    candidates.append(path)

    if not candidates:
        for path in session_root.iterdir():
            if path.is_dir() and path.name.startswith("Enh_"):
                if any(path.glob("frame_*.png")):
                    candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not infer enhanced frame sequence under {session_root}"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_session_bundle(
    session_root: Path,
    sequence_dir_override: Path | None = None,
) -> SessionBundle:
    session_root = session_root.resolve()
    processing_log = find_processing_log(session_root)
    with processing_log.open("r", encoding="utf-8") as handle:
        processing_data = json.load(handle)

    batch_crop = processing_data.get("batch_crop", {})
    overview_frame = int(batch_crop.get("overview_frame", 0))
    if sequence_dir_override is not None:
        sequence_dir = validate_sequence_dir(Path(sequence_dir_override))
    else:
        sequence_dir = infer_sequence_dir(session_root)

    return SessionBundle(
        session_root=session_root,
        processing_log=processing_log,
        sequence_dir=sequence_dir,
        overview_frame=overview_frame,
        processing_data=processing_data,
    )


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return

    if mode == "hardlink":
        dst.hardlink_to(src)
        return

    shutil.copy2(src, dst)


def derive_contrasted_folder_name(origin_folder_name: str) -> str:
    if origin_folder_name.endswith("_origin"):
        return origin_folder_name[: -len("_origin")] + "_contrasted"
    return origin_folder_name + "_contrasted"


def count_pngs(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.png"))


def to_yolo_line(bbox_xyxy: list[int], width: int, height: int, class_id: int = 0) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def build_review_template(image_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": image_record["image_id"],
        "source_image": image_record["source_image"],
        "frame_index": image_record["frame_index"],
        "focus_regions": [],
        "ignore_regions": [],
        "accepted_candidates": [],
        "rejected_candidates": [],
        "notes": "",
        "annotation_guidance": {
            "focus_regions": (
                "Optional polygons or rectangles that describe where the user "
                "wants the auto-crop algorithm to search."
            ),
            "ignore_regions": (
                "Optional polygons or rectangles that should not contribute to "
                "training, such as regions that the user intentionally ignores."
            ),
            "accepted_candidates": (
                "Candidates proposed by automation and accepted by the user."
            ),
            "rejected_candidates": (
                "Candidates proposed by automation but deleted by the user. "
                "These are valuable hard negatives."
            ),
            "primary_point": (
                "When a box contains two particles, add a primary point under "
                "the corresponding accepted candidate or ground-truth item."
            ),
            "visible_frame_range": (
                "Optional temporal visibility for later track refinement. "
                "Leave null during the first export."
            ),
        },
    }


def split_name(image_id: str, total_images: int, val_ratio: float) -> str:
    if total_images < 5:
        return "train"

    bucket = stable_hash(image_id) % 10_000
    threshold = int(val_ratio * 10_000)
    return "val" if bucket < threshold else "train"


def export_dataset(
    bundles: list[SessionBundle],
    output_root: Path,
    class_name: str,
    val_ratio: float,
    copy_mode: str,
) -> dict[str, Any]:
    canonical_root = ensure_dir(output_root / "canonical")
    review_root = ensure_dir(canonical_root / "review_templates")
    yolo_root = ensure_dir(output_root / "yolo_detect")
    ensure_dir(yolo_root / "images" / "train")
    ensure_dir(yolo_root / "images" / "val")
    ensure_dir(yolo_root / "labels" / "train")
    ensure_dir(yolo_root / "labels" / "val")

    image_records: list[dict[str, Any]] = []
    track_records: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []

    for bundle in bundles:
        batch_crop = bundle.processing_data.get("batch_crop", {})
        rois = batch_crop.get("rois", [])
        frame_path = bundle.sequence_dir / f"frame_{bundle.overview_frame:04d}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Overview frame missing: {frame_path}")

        with Image.open(frame_path) as img:
            width, height = img.size

        session_id = bundle.session_root.name
        image_id = f"{session_id}__frame_{bundle.overview_frame:04d}"
        image_record = {
            "image_id": image_id,
            "session_id": session_id,
            "source_session_root": str(bundle.session_root),
            "source_processing_log": str(bundle.processing_log),
            "source_sequence_dir": str(bundle.sequence_dir),
            "source_image": str(frame_path),
            "frame_index": bundle.overview_frame,
            "width": width,
            "height": height,
            "annotation_source": "manual_batch_crop_overview",
            "task_stage": "overview_detection",
            "focus_mask_path": None,
            "ignore_regions": [],
            "annotations": [],
            "preprocessing": {
                "planned_settings": bundle.processing_data.get("planned_settings", {}),
                "import_meta": bundle.processing_data.get("import_meta", {}),
                "data_layer": batch_crop.get("data_layer"),
            },
            "notes": [
                (
                    "Boxes are exported only on the reviewed overview frame. "
                    "Temporal disappearance is intentionally left undecided."
                )
            ],
        }

        for roi in rois:
            bbox_xyxy_raw = roi["bbox"]
            bbox_xyxy = normalize_bbox_xyxy(bbox_xyxy_raw, width, height)
            track_id = int(roi["id"])
            origin_folder_name = roi["folder"]
            origin_folder = bundle.session_root / origin_folder_name
            contrasted_folder = bundle.session_root / derive_contrasted_folder_name(
                origin_folder_name
            )

            annotation = {
                "instance_id": f"{session_id}__np_{track_id:03d}",
                "track_id": track_id,
                "category": class_name,
                "bbox_xyxy": bbox_xyxy,
                "bbox_xyxy_from_processing_log": [int(v) for v in bbox_xyxy_raw],
                "primary_point": None,
                "main_target_status": "unknown",
                "temporal_visibility": {
                    "overview_frame_state": "visible",
                    "visible_frame_range": None,
                    "full_sequence_state": "unknown",
                },
                "source_crop_origin_dir": str(origin_folder),
                "source_crop_contrasted_dir": str(contrasted_folder),
                "frame_range_used": roi.get("frame_range_used"),
            }
            image_record["annotations"].append(annotation)

            track_records.append(
                {
                    "session_id": session_id,
                    "track_id": track_id,
                    "instance_id": annotation["instance_id"],
                    "bbox_xyxy": bbox_xyxy,
                    "bbox_xyxy_from_processing_log": [int(v) for v in bbox_xyxy_raw],
                    "source_crop_origin_dir": str(origin_folder),
                    "source_crop_contrasted_dir": str(contrasted_folder),
                    "crop_origin_frame_count": count_pngs(origin_folder),
                    "crop_contrasted_frame_count": count_pngs(contrasted_folder),
                    "primary_point": None,
                    "main_target_status": "unknown",
                    "visible_frame_range": None,
                    "full_sequence_state": "unknown",
                }
            )

        image_records.append(image_record)
        session_summaries.append(
            {
                "session_id": session_id,
                "session_root": str(bundle.session_root),
                "processing_log": str(bundle.processing_log),
                "sequence_dir": str(bundle.sequence_dir),
                "overview_frame": bundle.overview_frame,
                "roi_count": len(rois),
            }
        )

    total_images = len(image_records)

    for image_record in image_records:
        split = split_name(image_record["image_id"], total_images, val_ratio)
        image_record["split"] = split

        if total_images < 5:
            image_record["split_note"] = (
                "Dataset is currently too small for a meaningful validation split. "
                "The derived YOLO export keeps everything in train."
            )
            split = "train"

        src = Path(image_record["source_image"])
        dst_image_name = f"{image_record['image_id']}.png"
        dst_image = yolo_root / "images" / split / dst_image_name
        dst_label = yolo_root / "labels" / split / f"{image_record['image_id']}.txt"

        copy_or_link(src, dst_image, copy_mode)

        yolo_lines = [
            to_yolo_line(
                ann["bbox_xyxy"],
                image_record["width"],
                image_record["height"],
                class_id=0,
            )
            for ann in image_record["annotations"]
        ]
        dst_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        review_template = build_review_template(image_record)
        review_path = review_root / f"{image_record['image_id']}.json"
        review_path.write_text(
            json.dumps(review_template, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    manifest_path = canonical_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in image_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    tracks_path = canonical_root / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        for record in track_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    sessions_path = canonical_root / "sessions.json"
    sessions_path.write_text(
        json.dumps(session_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "class_name": class_name,
        "total_sessions": len(session_summaries),
        "total_images": len(image_records),
        "total_tracks": len(track_records),
        "canonical_manifest": str(manifest_path),
        "canonical_tracks": str(tracks_path),
        "canonical_sessions": str(sessions_path),
        "review_templates_dir": str(review_root),
        "yolo_root": str(yolo_root),
        "notes": [
            (
                "Only overview frames are exported as positive full-frame "
                "detections in the first pass."
            ),
            (
                "Track disappearance is intentionally left as unknown and can "
                "be annotated later through the review templates."
            ),
            (
                "For very small datasets, the derived YOLO export keeps "
                "everything in train to avoid misleading split behavior."
            ),
        ],
    }
    summary_path = output_root / "export_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    data_yaml = "\n".join(
        [
            f"path: {yolo_root.as_posix()}",
            "train: images/train",
            "val: images/val" if total_images >= 5 else "val: images/train",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    (yolo_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

    return summary


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root).resolve())
    bundles = [load_session_bundle(Path(path)) for path in args.session_root]
    summary = export_dataset(
        bundles=bundles,
        output_root=output_root,
        class_name=args.class_name,
        val_ratio=args.val_ratio,
        copy_mode=args.copy_mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

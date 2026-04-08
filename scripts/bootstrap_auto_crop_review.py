"""
Bootstrap an auto-crop review template directly from a MagicImageJ session.

Use this when the user wants proposal assistance before any manual batch-crop
ROIs have been finalized. The script only needs:

1. A session root that contains processing_log.json.
2. A frame index chosen by the user as the overview / review frame.

It writes a review JSON compatible with annotate_auto_crop_regions.py and
propose_auto_crop_candidates.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_auto_crop_training_data import build_review_template, ensure_dir, load_session_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an auto-crop review template from a session and frame index."
    )
    parser.add_argument(
        "--session-root",
        required=True,
        help="Session root that contains processing_log.json.",
    )
    parser.add_argument(
        "--frame-index",
        required=True,
        type=int,
        help="Frame index the user wants to use as the review / overview frame.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the review JSON will be written.",
    )
    parser.add_argument(
        "--sequence-dir",
        help=(
            "Optional explicit enhanced-frame sequence directory. Use this when "
            "the current enhanced stack exists in MagicImageJ but has not been "
            "exported into the session archive yet."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir) if args.sequence_dir else None
    bundle = load_session_bundle(
        Path(args.session_root),
        sequence_dir_override=sequence_dir,
    )
    frame_index = int(args.frame_index)
    frame_path = bundle.sequence_dir / f"frame_{frame_index:04d}.png"
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    session_id = bundle.session_root.name
    image_id = f"{session_id}__frame_{frame_index:04d}"
    image_record = {
        "image_id": image_id,
        "source_image": str(frame_path),
        "frame_index": frame_index,
    }
    review = build_review_template(image_record)
    review["source_processing_log"] = str(bundle.processing_log)
    review["sequence_dir"] = str(bundle.sequence_dir)
    review["session_root"] = str(bundle.session_root)

    output_dir = ensure_dir(Path(args.output_dir).resolve())
    output_path = output_dir / f"{image_id}.json"
    output_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()

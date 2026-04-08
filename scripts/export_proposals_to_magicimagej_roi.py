"""
Convert auto-crop candidate proposals into a MagicImageJ ROI JSON file.

This creates a JSON file compatible with GeometryWidget -> Load ROIs, so users
can bring externally proposed candidate boxes back into MagicImageJ for manual
review, deletion, and refinement.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert proposal candidates into a MagicImageJ ROI JSON."
    )
    parser.add_argument(
        "--proposal-json",
        required=True,
        help="Path to frame_XXXX_candidates.json produced by propose_auto_crop_candidates.py.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path for the MagicImageJ ROI JSON output.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="How many top-ranked proposal boxes to export into the ROI JSON.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("candidate_id", "rank", "score"),
        default="candidate_id",
        help="How labels should be written into MagicImageJ.",
    )
    return parser.parse_args()


def bbox_to_rectangle(bbox_xyxy: list[int | float]) -> list[list[float]]:
    x1, y1, x2, y2 = bbox_xyxy
    return [
        [float(y1), float(x1)],
        [float(y1), float(x2)],
        [float(y2), float(x2)],
        [float(y2), float(x1)],
    ]


def make_label(candidate: dict[str, Any], rank: int, mode: str) -> str:
    if mode == "rank":
        return str(rank)
    if mode == "score":
        return f"{candidate['score']:.3f}"
    return str(candidate.get("candidate_id", f"C{rank:03d}"))


def main() -> None:
    args = parse_args()
    proposal_json = Path(args.proposal_json).resolve()
    output_json = Path(args.output_json).resolve()
    proposal = json.loads(proposal_json.read_text(encoding="utf-8"))

    candidates = proposal.get("candidates", [])[: max(0, int(args.top_k))]
    rois = []
    for rank, candidate in enumerate(candidates, start=1):
        label = make_label(candidate, rank, args.label_mode)
        frame_info = (
            f"[score={candidate['score']:.3f}]"
            if isinstance(candidate.get("score"), (int, float))
            else ""
        )
        rois.append(
            {
                "id": rank - 1,
                "coordinates": bbox_to_rectangle(candidate["bbox_xyxy"]),
                "label": label,
                "frame_range": "",
                "frame_info": frame_info,
            }
        )

    data_dump = {
        "version": "1.2",
        "type": "MagicImageJ_ROI",
        "timestamp": str(datetime.datetime.now()),
        "environment": {
            "source_layer_name": Path(proposal.get("source_image", "")).name,
            "data_layer_name": Path(proposal.get("source_image", "")).name,
            "rotation_applied": "Rotated" in str(proposal.get("sequence_dir", "")),
            "proposal_json": str(proposal_json),
            "top_k_exported": len(rois),
        },
        "rois": rois,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data_dump, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(output_json))


if __name__ == "__main__":
    main()

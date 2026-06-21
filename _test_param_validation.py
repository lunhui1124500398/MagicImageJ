"""
Regression test for param-edit validation (2026-05-29).
See discussions/2026-05-29_param_edit_validation_design.md

Covers the two pure (Qt-free) helpers added to widgets/recovery_widget.py:
  - _param_readonly_reason(key, value): Plan A read-only classifier
  - _validate_param_changes(changed, total_frames): Plan C range/sanity check

Run from the MagicImageJ env:
    python _test_param_validation.py
"""
import os
import sys
import math

# Importing recovery_widget pulls in qtpy.QtWidgets at module import time; force
# offscreen so it works headless. No QApplication / widget is constructed here —
# we only call the two module-level pure functions.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from widgets.recovery_widget import _param_readonly_reason, _validate_param_changes  # noqa: E402

_passed = 0
_failed = 0


def check(label, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")


# ---------------------------------------------------------------------------
# Plan A: read-only classifier
# ---------------------------------------------------------------------------
print("\n[Plan A] _param_readonly_reason — LOCKED keys (reason is not None):")
locked = [
    ("roi_count", 2),               # derived
    ("proposal_count", 5),          # derived
    ("frame_count", 139),           # derived
    ("total_frames", 139),          # derived
    ("source_total_frames", 139),   # derived
    ("foo_count", 1),               # *_count suffix
    ("max_shift_x", 3.2),           # measured output
    ("max_shift_y", -1.0),          # measured output
    ("source_layer", "Imported"),   # layer ref
    ("view_layer", "View"),         # layer ref
    ("data_layer", "Data"),         # layer ref
    ("rois", [[[0, 0], [1, 1]]]),   # structured (named)
    ("frame_info", [{"i": 0}]),     # structured (named)
    ("frame_indices", [0, 1, 2]),   # structured (named)
    ("unknown_list", [1, 2, 3]),    # structured catch-all (list)
    ("unknown_dict", {"a": 1}),     # structured catch-all (dict)
]
for k, v in locked:
    check(f"{k!r} locked", _param_readonly_reason(k, v) is not None)

print("\n[Plan A] _param_readonly_reason — EDITABLE keys (reason is None):")
editable = [
    ("template_frame", 70),
    ("angle", -40.0),
    ("expand", True),
    ("kernel_size", 11),
    ("clip_limit", 2.0),
    ("max_shift", 5),               # NOTE: 'max_shift' is in measured set -> see below
    ("bbox", [0, 0, 10, 10]),       # small structured whitelist
    ("roi_bbox", [1, 2, 3, 4]),     # small structured whitelist
    ("max_workers", 8),
]
for k, v in editable:
    if k == "max_shift":
        # 'max_shift' is intentionally a measured key -> should be LOCKED, not editable.
        check(f"{k!r} locked (measured)", _param_readonly_reason(k, v) is not None)
    else:
        check(f"{k!r} editable", _param_readonly_reason(k, v) is None)

# bbox/roi_bbox must stay editable EVEN THOUGH they are lists (whitelist beats list rule)
check("bbox editable despite being a list",
      _param_readonly_reason("bbox", [0, 0, 5, 5]) is None)
check("roi_bbox editable despite being a list",
      _param_readonly_reason("roi_bbox", [0, 0, 5, 5]) is None)

# ---------------------------------------------------------------------------
# Plan C: range / sanity validation
# ---------------------------------------------------------------------------
print("\n[Plan C] _validate_param_changes — should PASS (empty error list):")
ok_cases = [
    ("template_frame in range", {"template_frame": 10}, 50),
    ("template_frame top edge", {"template_frame": 49}, 50),
    ("template_frame float-int", {"template_frame": 10.0}, 50),
    ("template_frame no total (skip upper)", {"template_frame": 9999}, None),
    ("bbox valid", {"bbox": [0, 0, 10, 10]}, None),
    ("roi_bbox valid", {"roi_bbox": [3, 4, 30, 40]}, None),
    ("clip_limit positive", {"clip_limit": 2.0}, None),
    ("kernel_size positive", {"kernel_size": 11}, None),
    ("kernel_size float-int", {"kernel_size": 11.0}, None),
    ("angle finite", {"angle": -40.0}, None),
    ("unknown key ignored", {"some_other": "whatever"}, 50),
    ("empty changes", {}, 50),
]
for label, changed, total in ok_cases:
    errs = _validate_param_changes(changed, total)
    check(f"{label} -> no error", errs == [])

print("\n[Plan C] _validate_param_changes — should FAIL (>=1 error, names the key):")
bad_cases = [
    ("template_frame out of range", {"template_frame": 50}, 50, "template_frame"),
    ("template_frame negative", {"template_frame": -1}, 50, "template_frame"),
    ("template_frame non-int", {"template_frame": 1.5}, 50, "template_frame"),
    ("template_frame bool", {"template_frame": True}, 50, "template_frame"),
    ("bbox wrong length", {"bbox": [0, 0, 10]}, None, "bbox"),
    ("bbox x2<=x1", {"bbox": [10, 0, 5, 10]}, None, "bbox"),
    ("bbox y2<=y1", {"bbox": [0, 10, 5, 5]}, None, "bbox"),
    ("roi_bbox non-number", {"roi_bbox": [0, 0, "a", 10]}, None, "roi_bbox"),
    ("clip_limit zero", {"clip_limit": 0}, None, "clip_limit"),
    ("clip_limit negative", {"clip_limit": -1.0}, None, "clip_limit"),
    ("kernel_size zero", {"kernel_size": 0}, None, "kernel_size"),
    ("kernel_size non-int", {"kernel_size": 3.5}, None, "kernel_size"),
    ("angle nan", {"angle": float("nan")}, None, "angle"),
    ("angle inf", {"angle": float("inf")}, None, "angle"),
]
for label, changed, total, key in bad_cases:
    errs = _validate_param_changes(changed, total)
    check(f"{label} -> error mentions {key!r}",
          len(errs) >= 1 and any(key in e for e in errs))

# combined: multiple violations counted separately
errs = _validate_param_changes({"template_frame": 999, "bbox": [0, 0, 1]}, 50)
check("combined two violations -> 2 errors", len(errs) == 2)

# ---------------------------------------------------------------------------
print(f"\n==== _test_param_validation: {_passed} passed, {_failed} failed ====")
sys.exit(1 if _failed else 0)

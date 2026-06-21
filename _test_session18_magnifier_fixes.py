"""
Session 18 regression tests — ROI Magnifier fixes (E2E test feedback).

Covers:
1. Issue 1 — Apply persists AND restores Advanced (CLAHE + gamma), not just contrast.
2. Issue 2 — CLAHE clip limit is tunable (default 3.0) and actually changes the burn.
3. Issue 3 — magnifier frame slider clamps to the active frame range
   (per-ROI Frame Filter, else global Slider Range Lock).
4. _parse_frame_indices parity with geometry_widget.parse_indices_helper.

Runs offscreen; no real napari viewer required.
"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ----------------------------------------------------------------------------
# Minimal fakes (mirror the _ViewerLike pattern in _test_phase4_magnifier.py)
# ----------------------------------------------------------------------------
class _Sig:
    def connect(self, *a, **k): pass
    def disconnect(self, *a, **k): pass


class _EventGroup:
    def __init__(self, names):
        for n in names:
            setattr(self, n, _Sig())


def _rect(y0, x0, y1, x1):
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


class _FakeShapes:
    """Stand-in for the napari 'Batch_ROI' Shapes layer."""
    def __init__(self, n):
        # Real rectangle polygons so the magnifier's _refresh() can compute bboxes.
        self.data = [_rect(0, 0, 10 + i, 10 + i) for i in range(n)]
        self.features = pd.DataFrame({
            'label': [f'NP{i + 1}' for i in range(n)],
            'frame_range': [''] * n,
        })
        self.selected_data = set()

    def refresh(self):
        pass


class _FakeLayers:
    def __init__(self):
        self._d = {}
        self._active = None

    def add(self, name, layer):
        self._d[name] = layer

    def __contains__(self, k): return k in self._d
    def __getitem__(self, k): return self._d[k]
    def __iter__(self): return iter(self._d.values())
    def __len__(self): return len(self._d)

    @property
    def selection(self):
        outer = self
        class S:
            active = outer._active
        return S()

    @property
    def events(self):
        return _EventGroup(['inserted', 'removed'])


class _Dims:
    def __init__(self):
        self.current_step = [0]

    @property
    def events(self):
        return _EventGroup(['current_step'])


class _Viewer:
    def __init__(self):
        self.layers = _FakeLayers()
        self.dims = _Dims()


class _GeomLike:
    """Stand-in for the geometry widget holding the global Slider Range Lock."""
    def __init__(self, enabled, lo, hi):
        self._slider_lock_enabled = enabled
        self._slider_lock_min = lo
        self._slider_lock_max = hi


_APP = None  # module-level so the QApplication is never GC'd (which would delete widgets)


def _make_magnifier(n_rois=2):
    global _APP
    from qtpy.QtWidgets import QApplication
    from widgets.roi_magnifier import ROIMagnifierWindow
    _APP = QApplication.instance() or QApplication(sys.argv)
    viewer = _Viewer()
    viewer.layers.add("Batch_ROI", _FakeShapes(n_rois))
    win = ROIMagnifierWindow(viewer, roi_idx=None)
    return win


# ----------------------------------------------------------------------------
# Issue 1 + 2 — Advanced (CLAHE + clip + gamma) save -> restore roundtrip
# ----------------------------------------------------------------------------
def test_advanced_save_restore_roundtrip():
    win = _make_magnifier(n_rois=2)
    win._last_idx = 0

    # User tunes Advanced on ROI 0
    win.adv_group.setChecked(True)
    win.chk_clahe.setChecked(True)
    win.gamma_slider.setValue(200)        # gamma = 2.0
    win.clahe_clip_spin.setValue(5.0)     # clip = 5.0
    assert win._clahe_enabled is True
    assert abs(win._gamma - 2.0) < 1e-6
    assert abs(win._clahe_clip - 5.0) < 1e-6

    # Apply -> persists to Batch_ROI.features
    win._write_preview_to_features(0)
    feats = win.viewer.layers["Batch_ROI"].features
    assert 'preview_clahe_clip' in feats.columns, "clip column must be persisted"
    assert bool(feats['preview_clahe'].iloc[0]) is True
    assert abs(float(feats['preview_gamma'].iloc[0]) - 2.0) < 1e-6
    assert abs(float(feats['preview_clahe_clip'].iloc[0]) - 5.0) < 1e-6

    # Simulate "close + reopen": reset UI to defaults, then restore from features
    win.adv_group.setChecked(False)
    win.chk_clahe.setChecked(False)
    win.gamma_slider.setValue(100)
    win.clahe_clip_spin.setValue(3.0)
    win._gamma = 1.0
    win._clahe_enabled = False
    win._clahe_clip = 3.0

    win._restore_advanced_from_features(feats, 0)

    # Issue 1: gamma + CLAHE restored (previously silently dropped on reopen)
    assert win._clahe_enabled is True, "CLAHE state must restore"
    assert abs(win._gamma - 2.0) < 1e-6, "gamma must restore"
    assert abs(win._clahe_clip - 5.0) < 1e-6, "clip must restore"
    # UI widgets reflect it
    assert win.chk_clahe.isChecked() is True
    assert win.gamma_slider.value() == 200
    assert abs(win.clahe_clip_spin.value() - 5.0) < 1e-6
    # adv panel auto-expands so _apply_post_burn actually applies the restored values
    assert win.adv_group.isChecked() is True, "adv group must re-check so preview applies"
    win.close()
    print("[OK] Advanced (CLAHE+gamma+clip) persists AND restores on reopen")


def test_restore_neutralizes_for_plain_roi():
    """Switching to an ROI with no saved Advanced must reset (no state leak)."""
    win = _make_magnifier(n_rois=2)
    win._last_idx = 0
    # Save advanced on ROI 0
    win.adv_group.setChecked(True)
    win.chk_clahe.setChecked(True)
    win.gamma_slider.setValue(250)
    win.clahe_clip_spin.setValue(7.5)
    win._write_preview_to_features(0)
    feats = win.viewer.layers["Batch_ROI"].features

    # ROI 1 was never tuned -> columns carry the neutral defaults
    win._restore_advanced_from_features(feats, 1)
    assert win._clahe_enabled is False
    assert abs(win._gamma - 1.0) < 1e-6
    assert abs(win._clahe_clip - 3.0) < 1e-6
    assert win.adv_group.isChecked() is False, "plain ROI must collapse advanced"
    win.close()
    print("[OK] restore neutralizes Advanced for ROIs without saved values (no leak)")


# ----------------------------------------------------------------------------
# Issue 2 — clip limit actually changes the CLAHE burn
# ----------------------------------------------------------------------------
def test_clip_limit_changes_burn():
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[SKIP] cv2 unavailable — clip-burn test skipped")
        return
    win = _make_magnifier()
    win.adv_group.setChecked(True)
    win.chk_clahe.setChecked(True)

    rng = np.random.default_rng(0)
    u8 = rng.integers(100, 150, size=(128, 128), dtype=np.uint8)

    win._clahe_clip = 1.0
    out_low = win._apply_post_burn(u8.copy())
    win._clahe_clip = 12.0
    out_high = win._apply_post_burn(u8.copy())

    assert out_low.shape == u8.shape and out_low.dtype == np.uint8
    assert not np.array_equal(out_low, out_high), "different clip limits must yield different burns"
    win.close()
    print("[OK] CLAHE clip limit is tunable and changes the burn")


# ----------------------------------------------------------------------------
# Issue 3 — frame slider clamps to per-ROI range / global lock
# ----------------------------------------------------------------------------
def test_effective_frame_bounds():
    win = _make_magnifier(n_rois=2)
    T = 50

    # (a) no range set anywhere -> full stack
    win._last_idx = 0
    assert win._effective_frame_bounds(T) == (0, T - 1)

    # (b) per-ROI Frame Filter takes precedence
    feats = win.viewer.layers["Batch_ROI"].features.copy()
    feats.loc[0, 'frame_range'] = '5-20'
    win.viewer.layers["Batch_ROI"].features = feats
    assert win._effective_frame_bounds(T) == (5, 20), "per-ROI range must clamp"

    # (c) 'all'/empty range is ignored (treated as full)
    feats.loc[0, 'frame_range'] = ''
    win.viewer.layers["Batch_ROI"].features = feats
    assert win._effective_frame_bounds(T) == (0, T - 1)

    # (d) global Slider Range Lock when no per-ROI range
    win._geom = _GeomLike(True, 10, 30)
    assert win._effective_frame_bounds(T) == (10, 30), "global lock must clamp"

    # (e) per-ROI overrides global lock
    feats.loc[0, 'frame_range'] = '12-18'
    win.viewer.layers["Batch_ROI"].features = feats
    assert win._effective_frame_bounds(T) == (12, 18)

    # (f) disabled global lock -> ignored
    win._geom = _GeomLike(False, 10, 30)
    feats.loc[0, 'frame_range'] = ''
    win.viewer.layers["Batch_ROI"].features = feats
    assert win._effective_frame_bounds(T) == (0, T - 1)
    win.close()
    print("[OK] _effective_frame_bounds: per-ROI > global lock > full, all clamp correctly")


# ----------------------------------------------------------------------------
# Parser parity with geometry_widget.parse_indices_helper
# ----------------------------------------------------------------------------
def _ref_parse(text, total_frames):
    """Verbatim copy of geometry_widget.parse_indices_helper for parity checking."""
    if not text.strip() or text.strip().lower() == "all":
        return list(range(total_frames))
    indices = set()
    try:
        parts = [p.strip() for p in text.split(',')]
        for p in parts:
            if not p:
                continue
            if '-' in p:
                start, end = map(int, p.split('-'))
                start = max(0, start); end = min(total_frames - 1, end)
                if start <= end:
                    indices.update(range(start, end + 1))
            else:
                idx = int(p)
                if 0 <= idx < total_frames:
                    indices.add(idx)
        return sorted(list(indices))
    except Exception:
        return list(range(total_frames))


def test_parser_parity():
    from widgets.roi_magnifier import _parse_frame_indices
    T = 30
    cases = ["", "all", "All", "global", "0-10", "0-10, 15, 20-25",
             "5", "100-200", "abc", "  ", "3,3,3", "29", "30"]
    for c in cases:
        mine = _parse_frame_indices(c, T)
        ref = _ref_parse(c, T)
        assert mine == ref, f"parser mismatch on {c!r}: mine={mine[:5]}.. ref={ref[:5]}.."
    print(f"[OK] _parse_frame_indices matches parse_indices_helper on {len(cases)} cases")


if __name__ == "__main__":
    print("=== Session 18 Magnifier fix tests ===")
    test_advanced_save_restore_roundtrip()
    test_restore_neutralizes_for_plain_roi()
    test_clip_limit_changes_burn()
    test_effective_frame_bounds()
    test_parser_parity()
    print("\nAll Session 18 Magnifier fix tests passed.")

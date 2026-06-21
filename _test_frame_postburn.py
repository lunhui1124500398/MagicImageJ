# -*- coding: utf-8 -*-
"""Test #1 frame-selection parser + #3 export post-burn + override pass-through.
Run: conda run -n MagicImageJ python _test_frame_postburn.py"""
import os, sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

from widgets.recovery_widget import RecoveryWidget as RW
from widgets.geometry_widget import _apply_export_post_burn as pb

p = f = 0
def chk(n, c):
    global p, f
    if c: p += 1; print("  PASS", n)
    else: f += 1; print("  FAIL", n)

print("#1 frame-selection parser:")
chk("'0-20' -> [0..20] (21)", RW._parse_frame_selection('0-20') == list(range(21)))
chk("'all' -> None", RW._parse_frame_selection('all') is None)
chk("'' -> None", RW._parse_frame_selection('') is None)
chk("'0-30,50' -> 0..30 + 50", RW._parse_frame_selection('0-30,50') == list(range(31)) + [50])
chk("'5' -> [5]", RW._parse_frame_selection('5') == [5])
chk("reversed '20-0' -> [0..20]", RW._parse_frame_selection('20-0') == list(range(21)))

print("#3 export post-burn (CLAHE+gamma):")
img = np.arange(256, dtype=np.uint8).reshape(16, 16)
chk("gamma=1, clahe=False -> unchanged", np.array_equal(pb(img, {'gamma': 1.0, 'clahe': False}), img))
chk("None override -> unchanged", np.array_equal(pb(img, None), img))
out = pb(img, {'gamma': 0.5, 'clahe': False})
chk("gamma=0.5 brightens midtones", out.mean() > img.mean())
out2 = pb(img, {'gamma': 2.0, 'clahe': False})
chk("gamma=2.0 darkens midtones", out2.mean() < img.mean())

print("#3 dialog carries gamma/clahe into get_preview_overrides:")
class FakeDims: current_step = [0]
class FakeLayer:
    def __init__(s, d): s.data = d
class FakeViewer:
    def __init__(s): s.dims = FakeDims(); s.layers = {}
from widgets.batch_export_confirm_dialog import BatchExportConfirmDialog
stack = (np.random.rand(10, 50, 50) * 255).astype(np.uint8)
v = FakeViewer(); v.layers['data'] = FakeLayer(stack)
rois = [np.array([[5, 5], [5, 30], [30, 30], [30, 5]])]
feats = pd.DataFrame({'label': ['NP1'], 'frame_range': [''],
                      'preview_min': [10.0], 'preview_max': [200.0], 'preview_lut': ['Viridis'],
                      'preview_gamma': [0.7], 'preview_clahe': [True]})
dlg = BatchExportConfirmDialog(None, v, 'data', rois, feats, '', is_tiff=False, has_view_layer=True)
ov = dlg.get_preview_overrides()[0]
chk("override has gamma 0.7", abs(ov.get('gamma', 1.0) - 0.7) < 1e-6)
chk("override has clahe True", ov.get('clahe') is True)
chk("override lut Viridis", ov.get('lut') == 'Viridis')

print(f"\n=== {p} passed, {f} failed ===")
sys.exit(1 if f else 0)

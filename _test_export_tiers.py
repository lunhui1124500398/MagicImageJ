# -*- coding: utf-8 -*-
"""Headless test: BatchExportConfirmDialog tier defaults + no construction crash.
Run: conda run -n MagicImageJ python _test_export_tiers.py"""
import os, sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

class FakeDims:    current_step = [0]
class FakeLayer:
    def __init__(self, data): self.data = data
class FakeViewer:
    def __init__(self):
        self.dims = FakeDims()
        self.layers = {}

from widgets.batch_export_confirm_dialog import BatchExportConfirmDialog

stack = (np.random.rand(10, 60, 60) * 255).astype(np.uint8)
viewer = FakeViewer(); viewer.layers['data'] = FakeLayer(stack)
rois = [np.array([[10,10],[10,40],[40,40],[40,10]]),
        np.array([[5,5],[5,25],[25,25],[25,5]])]

passed = failed = 0
def chk(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else: failed += 1; print(f"  FAIL  {name}")

# Case A: a view layer exists AND NP1 has a saved Magnifier preview → default all 3 tiers on
featsA = pd.DataFrame({'label':['NP1','NP2'], 'frame_range':['',''],
                       'preview_min':[10.0, float('nan')], 'preview_max':[200.0, float('nan')],
                       'preview_lut':['Viridis','']})
dlgA = BatchExportConfirmDialog(None, viewer, 'data', rois, featsA, '', is_tiff=False, has_view_layer=True)
print("Case A (view layer + magnifier preview):")
chk("origin default on", dlgA.export_origin is True)
chk("contrasted default on", dlgA.export_contrasted is True)
chk("preview default on (NP1 has it)", dlgA.export_preview is True)
chk("apply_preview_contrast alias == export_preview", dlgA.apply_preview_contrast == dlgA.export_preview)

# Case B: NO view layer, NO magnifier preview → only origin (the "no magnifier" case)
featsB = pd.DataFrame({'label':['NP1','NP2'], 'frame_range':['','']})
dlgB = BatchExportConfirmDialog(None, viewer, 'data', rois, featsB, '', is_tiff=False, has_view_layer=False)
print("Case B (no view layer, no preview):")
chk("origin on", dlgB.export_origin is True)
chk("contrasted off (no view layer)", dlgB.export_contrasted is False)
chk("preview off (no magnifier)", dlgB.export_preview is False)
chk("contrasted checkbox disabled", not dlgB.chk_contrasted.isEnabled())
chk("preview checkbox disabled", not dlgB.chk_preview.isEnabled())

# Case C: view layer but no preview → 2 tiers (origin + contrasted) = the user's "2个" default
dlgC = BatchExportConfirmDialog(None, viewer, 'data', rois, featsB, '', is_tiff=False, has_view_layer=True)
print("Case C (view layer, no preview = 2-tier default):")
chk("origin on", dlgC.export_origin is True)
chk("contrasted on", dlgC.export_contrasted is True)
chk("preview off", dlgC.export_preview is False)

# Toggle guard: unchecking all tiers disables Start
dlgC.chk_origin.setChecked(False); dlgC.chk_contrasted.setChecked(False)
chk("Start disabled when no tier selected", not dlgC.btn_start.isEnabled())
dlgC.chk_origin.setChecked(True)
chk("Start re-enabled after picking a tier", dlgC.btn_start.isEnabled())

print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)

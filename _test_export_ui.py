# -*- coding: utf-8 -*-
"""Headless test + screenshot for the export dialog enhancements:
- draggable frame slider (controls preview + overview frame)
- editable per-NP Frames column
Run: conda run -n MagicImageJ python _test_export_ui.py"""
import os, sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

class FakeDims:    current_step = [3]
class FakeLayer:
    def __init__(self, data): self.data = data
class FakeViewer:
    def __init__(self):
        self.dims = FakeDims()
        self.layers = {}

from widgets.batch_export_confirm_dialog import BatchExportConfirmDialog

# 40-frame stack so the slider has a real range
stack = (np.random.rand(40, 120, 160) * 255).astype(np.uint8)
viewer = FakeViewer(); viewer.layers['data'] = FakeLayer(stack)
rois = [np.array([[20,20],[20,70],[70,70],[70,20]]),
        np.array([[10,90],[10,140],[55,140],[55,90]])]
feats = pd.DataFrame({'label':['NP1','NP2'], 'frame_range':['',''],
                      'preview_min':[12.0, float('nan')], 'preview_max':[210.0, float('nan')],
                      'preview_lut':['Viridis','']})

dlg = BatchExportConfirmDialog(None, viewer, 'data', rois, feats, '', is_tiff=False, has_view_layer=True)

passed = failed = 0
def chk(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else: failed += 1; print(f"  FAIL  {name}")

print("Slider + preview frame:")
chk("slider exists", hasattr(dlg, 'frame_slider'))
chk("slider max == total_frames-1 (39)", dlg.frame_slider.maximum() == 39)
chk("default preview frame follows viewer (3)", dlg._preview_frame == 3)
dlg.frame_slider.setValue(17)
chk("dragging slider -> _preview_frame=17", dlg._preview_frame == 17)
chk("get_overview_frame()==17", dlg.get_overview_frame() == 17)
chk("slider label shows 18/40", dlg.frame_slider_label.text() == "18/40")

print("Editable per-NP Frames column:")
# default effective ranges empty (= all)
chk("default ranges all empty", dlg.get_effective_ranges() == ["", ""])
# edit NP1 frame range cell -> fires itemChanged -> _on_range_edited
dlg.table.item(0, 3).setText("0-50")
chk("NP1 effective_range == '0-50'", dlg.get_effective_ranges()[0] == "0-50")
chk("NP2 still all", dlg.get_effective_ranges()[1] == "")
# typing 'all' resets to empty
dlg.table.item(0, 3).setText("all")
chk("NP1 reset to all -> ''", dlg.get_effective_ranges()[0] == "")
# label/bbox/preview columns are read-only
chk("Label col not editable", not bool(dlg.table.item(0,1).flags() & 0x2))  # Qt.ItemIsEditable == 2
chk("Frames col editable", bool(dlg.table.item(0,3).flags() & 0x2))

# ---- screenshot ----
dlg.resize(1120, 660)
dlg.show()
app.processEvents()
dlg.frame_slider.setValue(22)
dlg.table.item(1, 3).setText("0-30,35")
app.processEvents()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_export_dialog_shot.png")
pix = dlg.grab()
ok = pix.save(out)
print(f"\nscreenshot saved={ok} -> {out}  ({pix.width()}x{pix.height()})")

print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)

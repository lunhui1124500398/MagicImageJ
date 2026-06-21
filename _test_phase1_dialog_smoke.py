"""
Phase 1 dialog smoke test (offscreen).

Construct BatchExportConfirmDialog with a mock viewer + ROI features, verify:
- No crash on __init__
- get_selected_indices() returns initial 0..N-1
- apply_preview_contrast respects has_preview detection
- Toggling a row checkbox updates internal state
- Painting preview doesn't crash (uses napari current_step)
"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from qtpy.QtWidgets import QApplication
import pandas as pd


class _DimsLike:
    def __init__(self):
        self.current_step = [10, 0, 0]


class _LayerLike:
    def __init__(self, data, features):
        self.data = data
        self.features = features


class _LayersLike:
    def __init__(self, layers_dict):
        self._d = layers_dict

    def __contains__(self, k):
        return k in self._d

    def __getitem__(self, k):
        return self._d[k]


class _ViewerLike:
    def __init__(self, stack, features):
        layer = _LayerLike(stack, features)
        self.layers = _LayersLike({"data": layer, "Batch_ROI": layer})
        self.dims = _DimsLike()


def run():
    app = QApplication.instance() or QApplication(sys.argv)

    # Synthetic stack (T=20, H=200, W=300, uint16)
    np.random.seed(0)
    stack = (np.random.rand(20, 200, 300) * 60000).astype(np.uint16)

    # 3 ROIs:
    #  NP1 - has preview (Viridis)
    #  NP2 - has preview (Gray)
    #  NP3 - no preview (NaN values)
    rois = [
        np.array([[10, 20], [10, 80], [60, 80], [60, 20]]),
        np.array([[100, 100], [100, 200], [180, 200], [180, 100]]),
        np.array([[5, 5], [5, 90], [80, 90], [80, 5]]),
    ]
    features = pd.DataFrame({
        'label': ['NP1', 'NP2', 'NP3'],
        'frame_range': ['0-10', '', 'all'],
        'preview_min': [100.0, 500.0, np.nan],
        'preview_max': [50000.0, 40000.0, np.nan],
        'preview_lut': ['Viridis', 'Gray', ''],
    })

    viewer = _ViewerLike(stack, features)

    from widgets.batch_export_confirm_dialog import BatchExportConfirmDialog
    dlg = BatchExportConfirmDialog(
        parent=None,
        viewer=viewer,
        data_layer_name='data',
        rois=rois,
        roi_features=features,
        global_range_text='',
        is_tiff=False,
    )

    # 1. All 3 should be initially selected
    sel = dlg.get_selected_indices()
    assert sel == [0, 1, 2], f"initial selection wrong: {sel}"
    print(f"[OK] initial selection = {sel}")

    # 2. apply_preview_contrast should be True (2 ROIs have preview)
    assert dlg.apply_preview_contrast is True, "preview should be ON"
    print(f"[OK] apply_preview_contrast = {dlg.apply_preview_contrast}")

    # 3. include_overview default True
    assert dlg.include_overview is True
    print(f"[OK] include_overview = {dlg.include_overview}")

    # 4. preview_overrides aligned with rois
    pov = dlg.get_preview_overrides()
    assert len(pov) == 3
    assert pov[0] is not None and pov[0]['lut'] == 'Viridis'
    assert pov[1] is not None and pov[1]['lut'] == 'Gray'
    assert pov[2] is None, "NP3 has NaN preview, should be None"
    print(f"[OK] preview_overrides[0]={pov[0]}, [1]={pov[1]}, [2]={pov[2]}")

    # 5. Toggle NP2 off
    dlg._on_enable_toggled(1, False)
    sel = dlg.get_selected_indices()
    assert sel == [0, 2], f"after toggle off NP2: {sel}"
    print(f"[OK] after un-select NP2 = {sel}")

    # 6. set_all False -> empty
    dlg._set_all(False)
    assert dlg.get_selected_indices() == []
    print("[OK] set_all(False) -> empty")

    # 7. invert -> all back
    dlg._invert_selection()
    assert dlg.get_selected_indices() == [0, 1, 2]
    print("[OK] invert from empty -> all")

    # 8. Test no-preview scenario: dialog with all-NaN preview values
    features2 = pd.DataFrame({
        'label': ['A', 'B'],
        'frame_range': ['', ''],
    })  # No preview cols at all
    rois2 = rois[:2]
    dlg2 = BatchExportConfirmDialog(
        parent=None, viewer=viewer, data_layer_name='data',
        rois=rois2, roi_features=features2, global_range_text='', is_tiff=False,
    )
    assert dlg2.apply_preview_contrast is False, "no preview cols -> apply should be False"
    assert dlg2.chk_preview.isEnabled() is False, "chk_preview should be disabled"
    print("[OK] no-preview-features dialog: checkbox disabled, apply_preview=False")

    # 9. TIFF mode: preview still applies but LUT message in tooltip
    dlg3 = BatchExportConfirmDialog(
        parent=None, viewer=viewer, data_layer_name='data',
        rois=rois, roi_features=features, global_range_text='', is_tiff=True,
    )
    tt = dlg3.chk_preview.toolTip()
    assert "TIFF" in tt
    print(f"[OK] TIFF mode tooltip mentions TIFF: '{tt[:50]}...'")

    print("\nAll Phase 1 dialog smoke tests passed.")


if __name__ == "__main__":
    run()

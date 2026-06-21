"""
Standalone test for ROIVideoExportDialog using a real napari Viewer
so we get napari's dark theme (matches what user sees in MagicImageJ).

Usage:
    python _test_roi_dialog.py                # show dialog, save screenshots, exit
    python _test_roi_dialog.py --interactive  # keep window open

Outputs PNG to D:/Revolution_Sample_Claude/_test_roi_dialog_{state}.png
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import napari
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QTimer

# Force stdout to UTF-8 so emoji in widget text don't crash print()
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def force_cn_locale():
    """Write zh_CN into the global config so tr() returns Chinese."""
    try:
        from widgets.settings_widget import GlobalConfig as _GC
        cfg_path = _GC._config_path
        cfg = {}
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["language"] = "zh_CN"
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[setup] forced language=zh_CN in {cfg_path}")
    except Exception as e:
        print(f"[setup] could not force CN locale: {e}")


def main():
    interactive = "--interactive" in sys.argv
    force_cn_locale()

    # Build a real napari Viewer (dark theme by default)
    viewer = napari.Viewer(show=False)
    print(f"[setup] napari theme: {napari.settings.get_settings().appearance.theme}")

    # Mock 100-frame stack
    rng = np.random.default_rng(42)
    stack = (rng.random((100, 512, 512)) * 200 + 30).astype(np.uint8)
    viewer.add_image(stack, name="test_data")

    polygons = [
        np.array([[100, 100], [100, 200], [200, 200], [200, 100]], dtype=float),
        np.array([[250, 250], [250, 380], [380, 380], [380, 250]], dtype=float),
        np.array([[50, 300], [50, 450], [180, 450], [180, 300]], dtype=float),
    ]
    features = pd.DataFrame({
        "label": ["NP1", "NP2", "NP3"],
        "frame_range": ["all", "0-50", "20-80"],
    })
    viewer.add_shapes(polygons, name="Batch_ROI", shape_type="polygon", features=features)

    # Import the dialog AFTER napari Viewer exists so theme is applied
    from widgets.roi_video_export_dialog import ROIVideoExportDialog

    dlg = ROIVideoExportDialog(
        parent=None,
        viewer=viewer,
        data_layer_name="test_data",
    )
    # Mimic main.py's tab_widget stylesheet so this test reproduces what the
    # user actually sees (without it, buttons render with default Qt padding
    # and the truncation symptom disappears).
    MAIN_PY_QSS = """
        QWidget {
            font-family: "Segoe UI", "Microsoft YaHei", "San Francisco", "Helvetica Neue", sans-serif;
            font-size: 10pt;
            color: #E0E0E0;
        }
        QPushButton {
            border: 1px solid #555;
            border-radius: 4px;
            padding: 6px 12px;
            background-color: #333;
            margin: 2px;
        }
        QPushButton:hover { background-color: #444; border-color: #777; }
        QPushButton:pressed { background-color: #222; }
        QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
            padding: 4px;
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #222;
            min-height: 22px;
            selection-background-color: #2196F3;
            margin: 2px;
        }
    """
    dlg.setStyleSheet(MAIN_PY_QSS)
    dlg.show()
    dlg.resize(1280, 760)

    app = QApplication.instance()

    out_dir = "D:/Revolution_Sample_Claude"

    def _selected_rows():
        return [r.row() for r in dlg.table.selectionModel().selectedRows()]

    def _save(path, crop_table=False):
        if crop_table:
            # Grab only the table region (left side, full height of table)
            rect = dlg.table.geometry()
            # Map to dlg coords
            pos = dlg.table.mapTo(dlg, rect.topLeft())
            from qtpy.QtCore import QRect
            qr = QRect(pos, rect.size())
            pix = dlg.grab(qr)
        else:
            pix = dlg.grab()
        ok = pix.save(path)
        print(f"[save] {path}: {ok}  (selected_rows={_selected_rows()})")

    def capture_unselected():
        dlg.table.clearSelection()
        QApplication.processEvents()
        QTimer.singleShot(400, lambda: (
            _save(f"{out_dir}/_test_roi_dialog_unselected.png"),
            _save(f"{out_dir}/_test_roi_dialog_unselected_table.png", crop_table=True),
            QTimer.singleShot(400, capture_selected),
        ))

    def capture_selected():
        dlg.table.selectRow(1)
        QApplication.processEvents()
        # Print actual button dimensions to diagnose truncation
        widget = dlg.table.cellWidget(1, 5)
        if widget:
            children = widget.findChildren(__import__("qtpy.QtWidgets", fromlist=["QPushButton"]).QPushButton)
            for b in children:
                print(f"[debug] button '{b.text()}' actual={b.width()}px sizeHint={b.sizeHint().width()}px minWidth={b.minimumWidth()}px")
            print(f"[debug] col 5 actual width: {dlg.table.columnWidth(5)}px")
            print(f"[debug] action widget actual size: {widget.size().width()}x{widget.size().height()}px")
        QTimer.singleShot(400, lambda: (
            _save(f"{out_dir}/_test_roi_dialog_selected.png"),
            _save(f"{out_dir}/_test_roi_dialog_selected_table.png", crop_table=True),
            (None if interactive else QTimer.singleShot(200, app.quit)),
        ))

    QTimer.singleShot(1000, capture_unselected)
    app.exec_()


if __name__ == "__main__":
    main()

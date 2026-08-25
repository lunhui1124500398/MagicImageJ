"""
共享的"整层导出"交互驱动 (供 Enhance / Geometry 两个 tab 复用)。

各 widget 解析好 data 层 / view 层的名字后调 run_full_layer_export(...):
弹确认框 → 选输出目录 → 取整层数组 → 调 utils.layer_export.export_layer_pair_to_disk
→ 记 session log → 返回 summary。非 Qt 的写盘/manifest 逻辑在 utils/layer_export.py。
"""
import datetime
from pathlib import Path

from qtpy.QtWidgets import QMessageBox, QFileDialog
from qtpy.QtCore import QSettings

from utils.layer_export import export_layer_pair_to_disk
from utils.session_logger import get_logger
from widgets.settings_widget import tr


def _as_stack(arr):
    """2D 单帧 → (1,H,W); 3D 原样。让 write_full_stack 的 stack[fi] 恒为 2D 帧。"""
    if getattr(arr, "ndim", 0) == 2:
        return arr[None]
    return arr


def _resolve_output_dir(parent):
    archive = QSettings("NapariUser", "Global").value("archive_path", "")
    if archive and Path(archive).exists():
        return Path(archive)
    d = QFileDialog.getExistingDirectory(parent, tr("Select Output Folder"))
    return Path(d) if d else None


def run_full_layer_export(parent, viewer, data_name, view_name, is_tiff=False):
    """导出 data 层(原始→origin) + view 层(对比度→contrasted) 整层。

    data_name 必填且必须是原始层; view_name 可选(为空或等于 data 则只导 data)。
    返回 summary dict 或 None(用户取消 / 无原始层)。
    """
    if not data_name or data_name not in viewer.layers:
        QMessageBox.warning(
            parent, tr("Export Full Layer (Data+View) as PNG"),
            tr("No raw data layer found — import origin first."))
        return None

    data_arr = _as_stack(viewer.layers[data_name].data)
    view_arr = None
    if view_name and view_name in viewer.layers and view_name != data_name:
        view_arr = _as_stack(viewer.layers[view_name].data)
    else:
        view_name = None

    total = int(data_arr.shape[0]) if getattr(data_arr, "ndim", 0) >= 3 else 1
    frame_indices = list(range(total))
    fmt = "TIFF" if is_tiff else "PNG"

    text = (f"{tr('Data Layer')}: {data_name}\n"
            f"{tr('View Layer')}: {view_name or '—'}\n"
            f"{tr('Format')}: {fmt}\n"
            f"{tr('Frames')}: {total}")
    confirm = QMessageBox.question(
        parent, tr("Export Full Layer (Data+View) as PNG"), text,
        QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
    if confirm != QMessageBox.Ok:
        return None

    out = _resolve_output_dir(parent)
    if out is None:
        return None

    now = datetime.datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    summary = export_layer_pair_to_disk(
        out, data_arr, data_name, view_arr, view_name, frame_indices,
        is_tiff=is_tiff, pad=4, total_source_frames=total,
        timestamp=ts, created_at=now.isoformat())

    try:
        get_logger().log_action("export", "full_layer_png", {
            "data_layer": data_name, "view_layer": view_name or "",
            "format": fmt, "output_dir": str(out),
            "data_folder": summary.get("data_folder"),
            "view_folder": summary.get("view_folder"),
            "frame_count": summary.get("frames"),
        })
    except Exception:
        pass

    return summary

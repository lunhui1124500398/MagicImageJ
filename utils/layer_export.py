"""
共享的"整层导出"工具 (Qt-free, 供 Enhance / Geometry 两个 tab 复用)。

把一个完整图层 (整帧, 不裁切) 写成 PNG 序列或 TIFF 栈, 并附一个 manifest,
让用户调完对比度后能一键导出 data 层(原始→_origin) + view 层(对比度→_contrasted),
下次直接 Import PNG 而不必再导入巨大的 dm4。

写盘逻辑 (write_full_stack) 从 geometry_widget 原样搬来 —— 已 memmap-safe (逐帧读),
不会因 5000+ 帧整段进 RAM 而 OOM。本模块刻意不 import napari/Qt, 仿 core/geometry_helpers。
"""
import json
from pathlib import Path

import numpy as np
import cv2

from utils.video_export import export_to_tiff_stack
from utils.memory_utils import create_huge_array, release_memmap_pages


def _safe_folder_name(name):
    """图层名 → 文件系统安全文件夹名。"""
    bad = '<>:"/\\|?*\n\r\t'
    s = "".join(("_" if c in bad else c) for c in str(name)).strip().rstrip(". ")
    return s or "layer"


def write_full_stack(stack, frame_indices, folder, base_name, is_tiff, pad):
    """把完整层(整帧, 不裁切)写到 folder: PNG 序列或 TIFF 栈, 与裁切导出同款灰度归一化。

    逐帧读取 (memmap-safe)。一次性 `np.asarray(stack[frame_indices])` 会把整段选中帧
    拉进 RAM (5000 帧整帧 ~24GB), 是完整液池导出的 OOM 源, 故逐帧处理。
    """
    folder = Path(folder)
    if not frame_indices:
        return
    folder.mkdir(parents=True, exist_ok=True)

    if is_tiff:
        # 逐帧拷进"预分配(大数据自动 memmap)"子栈再落盘, 避免整段进 RAM。
        first = np.asarray(stack[frame_indices[0]])
        sub, _ = create_huge_array((len(frame_indices),) + first.shape, first.dtype)
        sub[0] = first
        for k in range(1, len(frame_indices)):
            sub[k] = stack[frame_indices[k]]
        release_memmap_pages(sub)
        export_to_tiff_stack(sub, str(folder / f"{base_name}.tiff"))
        return

    for fi in frame_indices:
        img = np.asarray(stack[fi])  # single 2D frame — a few MB even on a memmap
        if img.dtype != np.uint8:
            mn, mx = float(img.min()), float(img.max())
            img = ((img - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else img.astype(np.uint8)
        try:
            ok, buf = cv2.imencode(".png", img)
            if ok:
                buf.tofile(str(folder / f"{fi:0{pad}d}.png"))
        except Exception as e:
            print(f"Full-layer save error: {e}")


def full_layer_export_folder_names(data_name, view_name, timestamp):
    """(data_base, view_base): `{safe(name)}__{timestamp}`, 与批量全液池路径命名一致。
    name 为空时对应项返回 None。"""
    ts = timestamp or ""
    data_base = _safe_folder_name(f"{data_name}__{ts}") if data_name else None
    view_base = _safe_folder_name(f"{view_name}__{ts}") if view_name else None
    return data_base, view_base


def build_layer_export_manifest(data_entry, view_entry, fmt, total_source_frames, created_at):
    """纯函数: 生成整层导出 manifest dict。schema 仿现有 MagicImageJ_Export,
    新 type=MagicImageJ_LayerExport, 顶层 layers={data, view} (缺项为 None)。"""
    return {
        "version": 1,
        "type": "MagicImageJ_LayerExport",
        "created_at": created_at or "",
        "export": {
            "format": fmt,
            "total_source_frames": int(total_source_frames),
        },
        "layers": {
            "data": data_entry,
            "view": view_entry,
        },
    }


def export_layer_pair_to_disk(output_dir, data_arr, data_name, view_arr, view_name,
                              frame_indices, is_tiff=False, pad=4,
                              total_source_frames=None, timestamp=None, created_at=None):
    """把 data 层(原始, origin) + view 层(对比度, contrasted)整层各写一个文件夹, 附 manifest。

    data_arr/view_arr 任一为 None 则跳过该层。返回 summary dict:
    {data_folder, view_folder, manifest, frames}。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_base, view_base = full_layer_export_folder_names(data_name, view_name, timestamp)

    data_entry = None
    view_entry = None
    frames = len(frame_indices)

    if data_arr is not None and data_base:
        write_full_stack(data_arr, frame_indices, output_dir / data_base, data_base, is_tiff, pad)
        data_entry = {"role": "origin", "layer_name": data_name, "folder": data_base, "frames": frames}

    if view_arr is not None and view_base:
        write_full_stack(view_arr, frame_indices, output_dir / view_base, view_base, is_tiff, pad)
        view_entry = {"role": "contrasted", "layer_name": view_name,
                      "folder": view_base, "source_layer": data_name, "frames": frames}

    if total_source_frames is None:
        total_source_frames = frames
    fmt = "TIFF" if is_tiff else "PNG"
    manifest = build_layer_export_manifest(data_entry, view_entry, fmt, total_source_frames, created_at)
    manifest_name = f"manifest_layerexport_{timestamp}.json" if timestamp else "manifest_layerexport.json"
    manifest_path = output_dir / manifest_name
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return {
        "data_folder": data_base if data_entry else None,
        "view_folder": view_base if view_entry else None,
        "manifest": str(manifest_path),
        "frames": frames,
    }

"""
BatchExportConfirmDialog
========================
在 geometry_widget._export_batch_crops 启动 BatchExportThread 之前弹出, 让用户:
- 勾选/反勾选具体导出哪些 ROI
- 决定是否应用 Magnifier preview 对比度 (preview_min/preview_max/preview_lut)
- 决定是否生成概览图 (overview.png)

设计要点:
- 预览面板复用 napari 当前帧, 上面画各 ROI bbox + label, 仅作 "这些 ROI 会被导出" 的视觉确认。
- ROI 表格每行有 [☑][标签][bbox][帧范围][Preview 状态] 5 列。
- TIFF 模式 + LUT 不冲突: TIFF 导出走 grayscale burn, LUT 自动 strip (tooltip 说明)。
- 与 roi_video_export_dialog 不共享代码 (那个 dialog 关心视频编码, 这里关心 PNG/TIFF 批量), 复用浪费, 各自轻量。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QFont
from qtpy.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFrame, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPushButton, QSizePolicy, QSlider, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from widgets.settings_widget import tr, GlobalConfig


_LUT_NAMES = ('Gray', 'Inverted', 'Viridis', 'Inferno', 'Hot', 'Cool')


def _polygon_to_bbox(roi_polygon, stack_shape) -> tuple:
    """polygon (N,2) -> (y1, x1, y2, x2) clipped to stack shape."""
    ys = roi_polygon[:, 0]
    xs = roi_polygon[:, 1]
    y1, y2 = int(min(ys)), int(max(ys))
    x1, x2 = int(min(xs)), int(max(xs))
    if len(stack_shape) == 3:
        H, W = stack_shape[1], stack_shape[2]
    else:
        H, W = stack_shape[0], stack_shape[1]
    return max(0, y1), max(0, x1), min(H, y2), min(W, x2)


def _estimate_frame_count(range_text: str, total_frames: int) -> int:
    """粗略估算: 'all' / '' -> total; '0-100' -> 101; '0-30,50' -> 32"""
    if not range_text or range_text.strip().lower() in ('all', ''):
        return total_frames
    n = 0
    try:
        for part in range_text.split(','):
            p = part.strip()
            if not p:
                continue
            if '-' in p:
                a, b = p.split('-', 1)
                start = max(0, int(a.strip()))
                end = min(total_frames - 1, int(b.strip()))
                if start <= end:
                    n += end - start + 1
            else:
                v = int(p)
                if 0 <= v < total_frames:
                    n += 1
        return n if n > 0 else total_frames
    except Exception:
        return total_frames


def _qimage_from_array(arr) -> QImage:
    """uint8 (H,W) or (H,W,3) -> QImage. Caller MUST keep arr alive while QImage is used."""
    if arr.ndim == 2:
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()


def _normalize_to_uint8(frame: np.ndarray) -> np.ndarray:
    """Simple min-max norm for preview only."""
    if frame.dtype == np.uint8:
        return frame
    f = frame.astype(np.float32)
    mn, mx = float(np.min(f)), float(np.max(f))
    if mx <= mn:
        return np.zeros_like(f, dtype=np.uint8)
    return np.clip((f - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)


class BatchExportConfirmDialog(QDialog):
    """批量导出确认对话框。"""

    def __init__(
        self,
        parent,
        viewer,
        data_layer_name: str,
        rois: list,                   # napari shapes data: list[ (N,2) polygons ]
        roi_features,                 # pandas DataFrame from layer.features
        global_range_text: str,
        is_tiff: bool,
        has_view_layer: bool = False,  # 是否有 View 图层可导出 _contrasted 这一级
    ):
        super().__init__(parent)
        self.viewer = viewer
        self.data_layer_name = data_layer_name
        self.rois = list(rois)
        self.global_range_text = global_range_text or ""
        self.is_tiff = is_tiff
        self._has_view_layer = bool(has_view_layer)

        self.setWindowTitle(tr("Confirm Batch Export"))
        self.setWindowFlags(
            (self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint
        )
        self.resize(1100, 640)

        stack = viewer.layers[data_layer_name].data
        self.stack_shape = stack.shape
        self.total_frames = stack.shape[0] if stack.ndim == 3 else 1

        # Per-ROI metadata (label/frame_range/preview_*)
        labels = list(roi_features.get('label', []))
        franges = list(roi_features.get('frame_range', []))
        pmins = roi_features['preview_min'].tolist() if 'preview_min' in roi_features.columns else []
        pmaxs = roi_features['preview_max'].tolist() if 'preview_max' in roi_features.columns else []
        pluts = roi_features['preview_lut'].tolist() if 'preview_lut' in roi_features.columns else []
        pgammas = roi_features['preview_gamma'].tolist() if 'preview_gamma' in roi_features.columns else []
        pclahes = roi_features['preview_clahe'].tolist() if 'preview_clahe' in roi_features.columns else []
        pclips = roi_features['preview_clahe_clip'].tolist() if 'preview_clahe_clip' in roi_features.columns else []

        self._roi_rows = []   # list of dict per ROI
        for i, poly in enumerate(self.rois):
            label = labels[i] if i < len(labels) else f"NP{i + 1}"
            frange = str(franges[i]) if i < len(franges) and franges[i] else ""
            bbox = _polygon_to_bbox(np.asarray(poly), self.stack_shape)
            pm = pmins[i] if i < len(pmins) else float('nan')
            px = pmaxs[i] if i < len(pmaxs) else float('nan')
            plut = pluts[i] if i < len(pluts) else ''
            try:
                pm_ok = (pm == pm) and (px == px) and (px > pm)
            except Exception:
                pm_ok = False
            # gamma / CLAHE (Phase 4 Advanced); 缺失或 NaN → 中性值
            pg = pgammas[i] if i < len(pgammas) else 1.0
            pc = pclahes[i] if i < len(pclahes) else False
            pcl = pclips[i] if i < len(pclips) else 3.0
            try:
                pg = float(pg) if pg == pg else 1.0
            except Exception:
                pg = 1.0
            try:
                pc = bool(pc) if pc == pc else False
            except Exception:
                pc = False
            try:
                pcl = float(pcl) if (pcl == pcl and float(pcl) > 0) else 3.0
            except Exception:
                pcl = 3.0
            self._roi_rows.append({
                'label': str(label),
                'bbox': bbox,
                'frame_range': frange,
                'effective_range': frange if frange else self.global_range_text,
                'has_preview': pm_ok,
                'preview_min': float(pm) if pm_ok else None,
                'preview_max': float(px) if pm_ok else None,
                'preview_lut': plut if isinstance(plut, str) and plut in _LUT_NAMES else 'Gray',
                'preview_gamma': pg,
                'preview_clahe': pc,
                'preview_clahe_clip': pcl,
                'enabled': True,
            })

        self._any_has_preview = any(r['has_preview'] for r in self._roi_rows)

        # 预览帧 (可拖动 slider 控制; 默认取 viewer 当前帧)。同时作为概览图导出帧。
        try:
            cur = int(self.viewer.dims.current_step[0]) if self.total_frames > 1 else 0
        except Exception:
            cur = 0
        self._preview_frame = max(0, min(cur, self.total_frames - 1))
        self._populating = False   # 填表期间忽略 itemChanged

        self._build_ui()
        self._refresh_preview()
        self._update_summary()

    # ---------- UI ----------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 8)

        # [Fix 3b] 缺 view 提醒条: 没选 View 层 → contrasted 导不了。可勾「不再提醒」永久关闭。
        if (not self._has_view_layer) and self._warn_missing_view_enabled():
            warn_row = QHBoxLayout()
            warn_lbl = QLabel(
                "⚠ " + tr("No View (contrasted) layer selected — '_contrasted' will NOT be exported. "
                          "Large liquid cells usually need both origin + contrasted.")
            )
            warn_lbl.setWordWrap(True)
            warn_lbl.setStyleSheet(
                "color: #FFB74D; font-size: 12px; padding: 6px; "
                "border: 1px solid #FFB74D; border-radius: 4px; background-color: #3a2e1a;"
            )
            warn_row.addWidget(warn_lbl, stretch=1)
            self.chk_dont_warn_view = QCheckBox(tr("Don't warn again"))
            self.chk_dont_warn_view.toggled.connect(self._on_dont_warn_view_toggled)
            warn_row.addWidget(self.chk_dont_warn_view)
            root.addLayout(warn_row)

        # Top splitter: left table | right preview
        splitter = QSplitter(Qt.Horizontal)

        # ---- Left ----
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        title = QLabel(f"<b>{tr('Select ROIs to export')}</b>")
        ll.addWidget(title)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            tr("Export"), tr("Label"), tr("BBox (y1,x1,y2,x2)"),
            tr("Frames ✎ (double-click to edit)"), tr("Magnifier Preview"),
        ])
        # 只有第 3 列(Frames)可编辑；编辑触发用双击/回车
        self.table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(28)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnWidth(3, 150)
        self.table.setColumnWidth(4, 200)
        self.table.itemSelectionChanged.connect(self._refresh_preview)
        self.table.itemChanged.connect(self._on_range_edited)

        self._populate_table()
        ll.addWidget(self.table, stretch=1)

        # Select buttons
        sel_row = QHBoxLayout()
        b_all = QPushButton(tr("Select All"))
        b_all.clicked.connect(lambda: self._set_all(True))
        b_none = QPushButton(tr("Select None"))
        b_none.clicked.connect(lambda: self._set_all(False))
        b_inv = QPushButton(tr("Invert"))
        b_inv.clicked.connect(self._invert_selection)
        sel_row.addWidget(b_all)
        sel_row.addWidget(b_none)
        sel_row.addWidget(b_inv)
        sel_row.addStretch()
        ll.addLayout(sel_row)

        splitter.addWidget(left)

        # ---- Right: preview ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        ph = QLabel(f"<b>{tr('Preview (drag the slider to scrub frames)')}</b>")
        rl.addWidget(ph)

        self.preview_label = QLabel(tr("Loading preview..."))
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(360, 360)
        self.preview_label.setStyleSheet(
            "background-color: #1e1e1e; color: #888; border: 1px solid #444;"
        )
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rl.addWidget(self.preview_label, stretch=1)

        # 帧滑条: 拖动浏览不同帧, 同时决定概览图(overview)导出用哪一帧
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel(tr("Frame:")))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self._preview_frame)
        self.frame_slider.setEnabled(self.total_frames > 1)
        self.frame_slider.setToolTip(
            tr("Scrub frames for preview. This frame is also used for the overview map.")
        )
        self.frame_slider.valueChanged.connect(self._on_preview_frame_changed)
        frame_row.addWidget(self.frame_slider, stretch=1)
        self.frame_slider_label = QLabel(f"{self._preview_frame + 1}/{self.total_frames}")
        self.frame_slider_label.setMinimumWidth(60)
        self.frame_slider_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        frame_row.addWidget(self.frame_slider_label)
        rl.addLayout(frame_row)

        self.preview_info = QLabel("")
        self.preview_info.setStyleSheet("color: #aaa; font-size: 11px;")
        self.preview_info.setWordWrap(True)
        rl.addWidget(self.preview_info)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([700, 400])
        root.addWidget(splitter, stretch=1)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        root.addWidget(line)

        # Export tiers (分级导出): origin / contrasted / preview，默认导出全部可用项
        opt_box = QGroupBox(tr("Export tiers (choose what to write)"))
        ov = QVBoxLayout(opt_box)

        # Tier 1: origin (raw) — always available
        self.chk_origin = QCheckBox(tr("Origin (raw crop) → _origin"))
        self.chk_origin.setChecked(True)
        self.chk_origin.setToolTip(tr("Raw cropped frames, grayscale, no contrast/LUT applied."))
        self.chk_origin.toggled.connect(self._update_summary)
        ov.addWidget(self.chk_origin)

        # Tier 2: contrasted (View layer's current state) — only if a View layer exists
        self.chk_contrasted = QCheckBox(tr("Contrasted (View layer state) → _contrasted"))
        self.chk_contrasted.setChecked(self._has_view_layer)
        self.chk_contrasted.setEnabled(self._has_view_layer)
        self.chk_contrasted.setToolTip(
            tr("Exports the View layer's display (e.g. an enhanced/contrast layer).")
            if self._has_view_layer else
            tr("No View layer selected in the Batch panel — this tier has nothing to export.")
        )
        self.chk_contrasted.toggled.connect(self._update_summary)
        ov.addWidget(self.chk_contrasted)

        # Tier 3: preview (Magnifier contrast + LUT) — only if a ROI has saved preview
        self.chk_preview = QCheckBox(tr("Preview (Magnifier contrast + LUT) → _preview"))
        self.chk_preview.setChecked(self._any_has_preview)
        self.chk_preview.setEnabled(self._any_has_preview)
        if not self._any_has_preview:
            self.chk_preview.setToolTip(
                tr("No ROI has Magnifier-saved preview. Open Magnifier and tune contrast/LUT first "
                   "(auto-saves; no Apply needed).")
            )
        elif self.is_tiff:
            self.chk_preview.setToolTip(
                tr("Contrast burned in (grayscale; LUT dropped for TIFF). _origin keeps the raw data.")
            )
        else:
            self.chk_preview.setToolTip(
                tr("Contrast burned in; a non-Gray LUT produces RGB PNG. _origin keeps the raw data.")
            )
        self.chk_preview.toggled.connect(self._update_summary)
        ov.addWidget(self.chk_preview)

        self.chk_overview = QCheckBox(tr("Also generate overview map (overview.png)"))
        self.chk_overview.setChecked(True)
        self.chk_overview.setToolTip(
            tr("Overview map uses napari's current display contrast (not Magnifier preview).")
        )
        ov.addWidget(self.chk_overview)

        # [Fix 3a] 完整液池层: 整帧 data + view 各导到一个 {图层名}__{时间戳} 文件夹
        full_row = QHBoxLayout()
        self.chk_full_liquid = QCheckBox(
            tr("Also export full liquid-cell layers (full data + view, all frames)")
        )
        self.chk_full_liquid.setChecked(self._full_liquid_default())
        self.chk_full_liquid.setToolTip(
            tr("Writes the WHOLE data + view layers (not cropped) to new '{layer}__{timestamp}' "
               "folders — the full liquid-cell movie for archive/viewing.")
        )
        self.chk_full_liquid.toggled.connect(self._on_full_liquid_toggled)
        full_row.addWidget(self.chk_full_liquid)
        full_row.addWidget(QLabel(tr("Frames:")))
        self.full_range_edit = QLineEdit()
        self.full_range_edit.setPlaceholderText(tr("All (empty) or 0-100, 120"))
        self.full_range_edit.setMaximumWidth(160)
        full_row.addWidget(self.full_range_edit)
        full_row.addStretch()
        ov.addLayout(full_row)

        root.addWidget(opt_box)

        # Summary
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #ccc; font-size: 12px; padding: 4px;")
        root.addWidget(self.summary_label)

        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.btn_start = QPushButton(f"💾 {tr('Start Export')}")
        self.btn_start.setStyleSheet(
            "background-color: #2E7D32; color: white; font-weight: bold; padding: 6px 18px;"
        )
        self.btn_start.clicked.connect(self.accept)
        btn_box.addButton(self.btn_start, QDialogButtonBox.AcceptRole)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

        # Default: select first row to populate preview
        if len(self._roi_rows):
            self.table.selectRow(0)

    def _populate_table(self):
        self._populating = True   # 填表期间屏蔽 itemChanged
        try:
            self.table.setRowCount(len(self._roi_rows))
            for i, r in enumerate(self._roi_rows):
                # Col 0: checkbox in a centered wrap
                chk = QCheckBox()
                chk.setChecked(r['enabled'])
                chk.toggled.connect(lambda v, idx=i: self._on_enable_toggled(idx, v))
                wrap = QWidget()
                wl = QHBoxLayout(wrap)
                wl.setContentsMargins(0, 0, 0, 0)
                wl.addStretch()
                wl.addWidget(chk)
                wl.addStretch()
                self.table.setCellWidget(i, 0, wrap)

                # Col 1: label (read-only)
                it1 = QTableWidgetItem(r['label'])
                it1.setFlags(it1.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, 1, it1)

                # Col 2: bbox (read-only)
                y1, x1, y2, x2 = r['bbox']
                it2 = QTableWidgetItem(f"({y1},{x1},{y2},{x2})")
                it2.setFlags(it2.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, 2, it2)

                # Col 3: frame range — EDITABLE per-NP (空=全部/全局)
                ftext = r['effective_range'] if r['effective_range'] else tr("all")
                it3 = QTableWidgetItem(ftext)
                it3.setToolTip(tr("Per-NP frame range, e.g. 0-100 or 0-30,50. "
                                  "Leave 'all' to use the global/all frames."))
                self.table.setItem(i, 3, it3)

                # Col 4: preview status (read-only) — 含 gamma/CLAHE 提示
                if r['has_preview']:
                    extra = ""
                    if abs(r.get('preview_gamma', 1.0) - 1.0) > 1e-6:
                        extra += f" γ={r['preview_gamma']:.2f}"
                    if r.get('preview_clahe'):
                        extra += " CLAHE"
                    ptxt = f"✓ min={r['preview_min']:.1f} max={r['preview_max']:.1f} LUT={r['preview_lut']}{extra}"
                else:
                    ptxt = tr("(none — uses napari default)")
                it4 = QTableWidgetItem(ptxt)
                it4.setFlags(it4.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, 4, it4)
        finally:
            self._populating = False

    # ---------- handlers ----------

    def _on_enable_toggled(self, idx, val):
        if 0 <= idx < len(self._roi_rows):
            self._roi_rows[idx]['enabled'] = bool(val)
            self._update_summary()
            self._refresh_preview()

    def _set_all(self, val: bool):
        for i, r in enumerate(self._roi_rows):
            r['enabled'] = val
            wrap = self.table.cellWidget(i, 0)
            if wrap is not None:
                chk = wrap.findChild(QCheckBox)
                if chk is not None:
                    chk.blockSignals(True)
                    chk.setChecked(val)
                    chk.blockSignals(False)
        self._update_summary()
        self._refresh_preview()

    def _invert_selection(self):
        for i, r in enumerate(self._roi_rows):
            r['enabled'] = not r['enabled']
            wrap = self.table.cellWidget(i, 0)
            if wrap is not None:
                chk = wrap.findChild(QCheckBox)
                if chk is not None:
                    chk.blockSignals(True)
                    chk.setChecked(r['enabled'])
                    chk.blockSignals(False)
        self._update_summary()
        self._refresh_preview()

    def _on_preview_frame_changed(self, v):
        """帧滑条拖动 → 更新预览帧(也是概览图导出帧)。"""
        self._preview_frame = int(v)
        self._refresh_preview()

    def _on_range_edited(self, item):
        """Frames 列被编辑 → 更新该 ROI 的帧范围 (空 / all / global → 全部帧)。"""
        if self._populating or item is None or item.column() != 3:
            return
        row = item.row()
        if not (0 <= row < len(self._roi_rows)):
            return
        text = item.text().strip()
        if text == "" or text.lower() in ("all", "global") or text == tr("all"):
            self._roi_rows[row]['effective_range'] = ""
            norm = tr("all")
        else:
            self._roi_rows[row]['effective_range'] = text
            norm = text
        if item.text() != norm:
            self._populating = True
            try:
                item.setText(norm)
            finally:
                self._populating = False
        self._update_summary()

    def _update_summary(self):
        # tier 复选框的 toggled 在 _build_ui 里 setChecked 时就会触发, 此时 summary_label /
        # btn_start 可能还没创建 → 先做存在性守卫, 避免构造期 AttributeError。
        if not hasattr(self, 'summary_label') or not hasattr(self, 'btn_start'):
            return
        selected = [r for r in self._roi_rows if r['enabled']]
        n_roi = len(selected)
        frame_files = 0
        for r in selected:
            frame_files += _estimate_frame_count(r['effective_range'], self.total_frames)
        # 选中的导出级数 (origin / contrasted / preview)；checkbox 可能还没建好时用 getattr 兜底
        n_tiers = sum([
            bool(getattr(self, 'chk_origin', None)) and self.chk_origin.isChecked(),
            bool(getattr(self, 'chk_contrasted', None)) and self.chk_contrasted.isChecked() and self._has_view_layer,
            bool(getattr(self, 'chk_preview', None)) and self.chk_preview.isChecked() and self._any_has_preview,
        ])
        if not n_roi:
            self.summary_label.setText(
                f"<span style='color:#FFB74D'>⚠ {tr('No ROI selected.')}</span>"
            )
            self.btn_start.setEnabled(False)
            return
        if n_tiers == 0:
            self.summary_label.setText(
                f"<span style='color:#FFB74D'>⚠ {tr('Select at least one export tier.')}</span>"
            )
            self.btn_start.setEnabled(False)
            return
        self.btn_start.setEnabled(True)
        total_files = frame_files * n_tiers
        # Rough disk: avg 200x200 uint8 PNG ~ 30KB; varies a lot
        est_mb = total_files * 0.04
        self.summary_label.setText(
            f"📊 {n_roi} {tr('ROI(s)')} × {n_tiers} {tr('tier(s)')} × ~{int(frame_files / max(n_roi, 1))} {tr('frames')} "
            f"= {total_files} {tr('files, ~')}{est_mb:.0f} MB ({tr('estimate')})"
        )

    def _refresh_preview(self):
        """Read viewer current frame, paint ROI bboxes, show as pixmap."""
        try:
            stack = self.viewer.layers[self.data_layer_name].data
            if stack.ndim == 3:
                # [Enhancement] 用滑条选的帧, 而不是 napari 的当前帧
                frame_idx = int(self._preview_frame)
                frame_idx = max(0, min(frame_idx, stack.shape[0] - 1))
                frame = np.asarray(stack[frame_idx])
            else:
                frame_idx = 0
                frame = np.asarray(stack)
        except Exception as e:
            self.preview_label.setText(f"{tr('Preview unavailable:')} {e}")
            return
        if hasattr(self, 'frame_slider_label'):
            self.frame_slider_label.setText(f"{frame_idx + 1}/{self.total_frames}")

        # Downscale large frames for preview only
        H, W = frame.shape[:2]
        max_dim = 720
        scale = min(1.0, max_dim / max(H, W))
        if scale < 1.0:
            new_h = max(1, int(H * scale))
            new_w = max(1, int(W * scale))
            try:
                import cv2
                small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except Exception:
                # Fallback: stride downsample
                stride = max(1, int(1 / scale))
                small = frame[::stride, ::stride]
        else:
            small = frame
        u8 = _normalize_to_uint8(small)
        u8 = np.ascontiguousarray(u8)
        qimg = _qimage_from_array(u8)
        pix = QPixmap.fromImage(qimg)

        # Paint ROI bboxes on top
        scale_y = pix.height() / max(H, 1)
        scale_x = pix.width() / max(W, 1)
        sel_idx = self._get_currently_focused_idx()
        painter = QPainter(pix)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        for i, r in enumerate(self._roi_rows):
            if not r['enabled']:
                continue
            y1, x1, y2, x2 = r['bbox']
            rx = int(x1 * scale_x)
            ry = int(y1 * scale_y)
            rw = max(1, int((x2 - x1) * scale_x))
            rh = max(1, int((y2 - y1) * scale_y))
            color = QColor("#FFEB3B") if i == sel_idx else QColor("#4CAF50")
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(rx, ry, rw, rh)
            painter.drawText(rx + 3, ry + 14, r['label'])
        painter.end()

        # Fit pixmap to label
        target = self.preview_label.size()
        if target.width() < 8 or target.height() < 8:
            target = pix.size()
        scaled = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

        n_sel = sum(1 for r in self._roi_rows if r['enabled'])
        self.preview_info.setText(
            f"{tr('Frame')} {frame_idx + 1}/{self.total_frames} &nbsp; "
            f"{tr('source')} {H}×{W} &nbsp; "
            f"{n_sel}/{len(self._roi_rows)} {tr('ROIs marked for export')}"
        )

    def _get_currently_focused_idx(self) -> int:
        sel = self.table.selectionModel().selectedRows()
        if sel:
            return sel[0].row()
        return -1

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview()

    # ---------- public API ----------

    def get_selected_indices(self) -> List[int]:
        return [i for i, r in enumerate(self._roi_rows) if r['enabled']]

    def get_effective_ranges(self) -> List[str]:
        """每个原始 ROI 索引的帧范围(用户可在表格 Frames 列编辑); 空串 = 全局/全部帧。"""
        return [r['effective_range'] for r in self._roi_rows]

    def get_overview_frame(self) -> int:
        """概览图导出使用的帧 (= 预览滑条当前帧)。"""
        return int(self._preview_frame)

    def get_preview_overrides(self) -> List[Optional[dict]]:
        """对齐原始 ROI 索引, 每个元素为 None 或 {'c_min', 'c_max', 'lut'}.
           注意: 返回的 list 长度 = 原始 ROI 数, 调用方根据 get_selected_indices 再 filter。"""
        out: List[Optional[dict]] = []
        for r in self._roi_rows:
            if r['has_preview']:
                out.append({
                    'c_min': r['preview_min'],
                    'c_max': r['preview_max'],
                    'lut': r['preview_lut'],
                    'gamma': r.get('preview_gamma', 1.0),
                    'clahe': r.get('preview_clahe', False),
                    'clahe_clip': r.get('preview_clahe_clip', 3.0),
                })
            else:
                out.append(None)
        return out

    @property
    def export_origin(self) -> bool:
        return self.chk_origin.isChecked()

    @property
    def export_contrasted(self) -> bool:
        return self.chk_contrasted.isChecked() and self._has_view_layer

    @property
    def export_preview(self) -> bool:
        return self.chk_preview.isChecked() and self._any_has_preview

    @property
    def apply_preview_contrast(self) -> bool:
        # 导出线程沿用的别名 = preview 这一级
        return self.export_preview

    @property
    def include_overview(self) -> bool:
        return self.chk_overview.isChecked()

    # ---------- [Fix 3] full liquid-cell export + missing-view warning ----------

    @staticmethod
    def _cfg_bool(val, default=True):
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() not in ('false', '0', 'no', '')

    def _full_liquid_default(self) -> bool:
        try:
            return self._cfg_bool(GlobalConfig.get("geo_export_full_liquid"), True)
        except Exception:
            return True

    def _warn_missing_view_enabled(self) -> bool:
        try:
            return self._cfg_bool(GlobalConfig.get("geo_warn_missing_view"), True)
        except Exception:
            return True

    def _on_full_liquid_toggled(self, val):
        try:
            GlobalConfig.set("geo_export_full_liquid", bool(val))
        except Exception:
            pass

    def _on_dont_warn_view_toggled(self, val):
        # 勾选「不再提醒」→ 关闭提醒
        try:
            GlobalConfig.set("geo_warn_missing_view", not bool(val))
        except Exception:
            pass

    @property
    def export_full_liquid(self) -> bool:
        chk = getattr(self, 'chk_full_liquid', None)
        return bool(chk is not None and chk.isChecked())

    @property
    def full_liquid_range(self) -> str:
        e = getattr(self, 'full_range_edit', None)
        return e.text().strip() if e is not None else ""

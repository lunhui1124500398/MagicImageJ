"""
ROI Video Export Dialog
=======================
基于 Batch_ROI 列表，对每个 ROI 单独做局域对比度后导出小区域视频。

设计要点:
- QSplitter 左表右预览, 预览随 spinbox 实时刷新, 不再起 napari。
- 底部参数/输出/按钮固定不收缩, 防止表格变长把导出按钮顶出可见区。
- WindowFlags 显式加 Min/Max 按钮, 去掉默认的 `?` (WindowContextHelp)。
- 每个 ROI 保留 c_min_orig/c_max_orig (首次 auto 算的值), 用户调的是 c_min/c_max。
- 导出三模式 (用户在 Export Mode 选):
    adjusted  → 输出用户调好的 contrast (默认)
    original  → 输出首次 auto 算的 contrast
    both      → 一个 ROI 出两个文件 (_adjusted.mp4 + _original.mp4)
"""
import numpy as np
from pathlib import Path
import gc

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QProgressBar, QGroupBox, QWidget, QApplication,
    QSplitter, QSizePolicy, QSlider, QFrame,
    QStyledItemDelegate, QStyleOptionViewItem, QStyle,
)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QImage, QPixmap, QBrush, QColor, QPalette

from widgets.settings_widget import tr
from utils.video_export import (
    export_video_smart, is_ffmpeg_available, get_all_codec_options
)
try:
    from utils.memory_utils import trim_working_set
except Exception:
    def trim_working_set():
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _polygon_to_bbox(roi_polygon, stack_shape):
    """polygon (N,2) -> (y1,x1,y2,x2) clipped to stack shape."""
    ys = roi_polygon[:, 0]
    xs = roi_polygon[:, 1]
    y1, y2 = int(min(ys)), int(max(ys))
    x1, x2 = int(min(xs)), int(max(xs))
    if len(stack_shape) == 3:
        H, W = stack_shape[1], stack_shape[2]
    else:
        H, W = stack_shape[0], stack_shape[1]
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(H, y2)
    x2 = min(W, x2)
    return y1, x1, y2, x2


def _crop_stack(stack, bbox):
    y1, x1, y2, x2 = bbox
    if stack.ndim == 3:
        return stack[:, y1:y2, x1:x2]
    return stack[y1:y2, x1:x2]


def _roi_auto_contrast(roi_stack):
    """采样式 percentile auto-contrast (memmap-safe)。"""
    arr = roi_stack
    if arr.ndim == 3 and arr.shape[0] > 10:
        idx = np.linspace(0, arr.shape[0] - 1, 10, dtype=int)
        chunks = [np.asarray(arr[int(i)]).ravel()[::5] for i in idx]
        sample = np.concatenate(chunks) if chunks else np.array([])
    else:
        sample = np.asarray(arr).ravel()[::5]
    if sample.size == 0:
        return 0.0, 255.0
    lo, hi = np.percentile(sample, [0.04, 99.96])
    if hi <= lo:
        try:
            lo, hi = float(np.asarray(arr).min()), float(np.asarray(arr).max())
        except Exception:
            lo, hi = 0.0, 255.0
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def _burn_to_uint8(roi_stack, c_min, c_max, frame_indices):
    """逐帧 burn 对比度 -> uint8。frame_indices: 帧索引列表 (对 3D)。"""
    rng = max(c_max - c_min, 1e-8)
    if roi_stack.ndim == 2:
        f = roi_stack.astype(np.float32)
        out = np.clip((f - c_min) / rng * 255, 0, 255).astype(np.uint8)
        return out[np.newaxis, ...]
    H, W = roi_stack.shape[1], roi_stack.shape[2]
    n = len(frame_indices)
    out = np.empty((n, H, W), dtype=np.uint8)
    for k, idx in enumerate(frame_indices):
        f = roi_stack[int(idx)].astype(np.float32)
        out[k] = np.clip((f - c_min) / rng * 255, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# LUT support — keep _LUT_NAMES / cmap names in sync with widgets/roi_magnifier.py
# ---------------------------------------------------------------------------
try:
    import matplotlib.cm as _cm
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

_LUT_NAMES = ['Gray', 'Inverted', 'Viridis', 'Inferno', 'Hot', 'Cool']
_LUT_TO_MPL = {'Viridis': 'viridis', 'Inferno': 'inferno', 'Hot': 'hot', 'Cool': 'cool'}


def _build_lut_table(lut_name):
    """Return a (256, 3) uint8 lookup table for the given LUT name."""
    if lut_name == 'Inverted':
        table = np.repeat((255 - np.arange(256))[:, None], 3, axis=1).astype(np.uint8)
        return table
    if lut_name == 'Gray' or not _HAS_MPL or lut_name not in _LUT_TO_MPL:
        return np.repeat(np.arange(256)[:, None], 3, axis=1).astype(np.uint8)
    cmap = _cm.get_cmap(_LUT_TO_MPL[lut_name])
    return (cmap(np.arange(256) / 255.0)[..., :3] * 255).astype(np.uint8)


def _apply_lut_to_frame(u8, lut_name):
    """uint8 灰度 (H,W) -> RGB uint8 (H,W,3)."""
    return _build_lut_table(lut_name)[u8]


def _apply_lut_to_stack(stack_u8, lut_name):
    """uint8 灰度堆 (T,H,W) -> RGB uint8 (T,H,W,3). 256-entry LUT 索引,内存安全。"""
    return _build_lut_table(lut_name)[stack_u8]


def _parse_frame_range(text, total_frames):
    """解析 '0-100, 150, 200-220' 风格的帧范围字符串。"""
    if total_frames <= 0:
        return []
    if not text or not text.strip() or text.strip().lower() == "all":
        return list(range(total_frames))
    indices = set()
    try:
        for part in text.split(','):
            p = part.strip()
            if not p:
                continue
            if '-' in p:
                a, b = p.split('-', 1)
                start = max(0, int(a.strip()))
                end = min(total_frames - 1, int(b.strip()))
                if start <= end:
                    indices.update(range(start, end + 1))
            else:
                v = int(p)
                if 0 <= v < total_frames:
                    indices.add(v)
        return sorted(indices) if indices else list(range(total_frames))
    except Exception:
        return list(range(total_frames))


# ---------------------------------------------------------------------------
# Export thread
# ---------------------------------------------------------------------------

class ROIVideoExportThread(QThread):
    progress = Signal(int, int, str)      # done_jobs, total_jobs, current_label
    one_done = Signal(str, str, bool)     # label_with_variant, output_path, success
    finished_all = Signal(int, int, list) # success_count, fail_count, out_paths
    error = Signal(str)

    def __init__(self, jobs, output_dir, fps, codec, backend, crf, preset, mode, parent=None):
        super().__init__(parent)
        self.jobs = jobs
        self.output_dir = Path(output_dir)
        self.fps = int(fps)
        self.codec = codec
        self.backend = backend
        self.crf = int(crf)
        self.preset = preset
        self.mode = mode  # 'adjusted' | 'original' | 'both'
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _encode_variant(self, label, suffix, cropped, c_min, c_max, frame_indices, lut_name):
        """Burn contrast + apply LUT + encode a single variant. Returns (success, out_path)."""
        if cropped.ndim == 3:
            n_src = cropped.shape[0]
            valid = [k for k in frame_indices if 0 <= k < n_src]
            if not valid:
                return False, ""
            burned = _burn_to_uint8(cropped, c_min, c_max, valid)
        else:
            burned = _burn_to_uint8(cropped, c_min, c_max, [0])

        if lut_name and lut_name != 'Gray':
            burned = _apply_lut_to_stack(burned, lut_name)

        out_path = self.output_dir / f"{label}{suffix}.mp4"
        ok = export_video_smart(
            burned, str(out_path),
            fps=self.fps,
            codec=self.codec,
            crf=self.crf,
            preset=self.preset,
            backend=self.backend,
            callback=None,
        )
        del burned
        return ok, str(out_path)

    def run(self):
        success_count = 0
        fail_count = 0
        out_paths = []
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            total = len(self.jobs)
            for i, job in enumerate(self.jobs):
                if self._cancelled:
                    break
                label = job['label']
                self.progress.emit(i, total, label)

                stack = job['stack']
                bbox = job['bbox']
                frame_indices = job['frame_indices']
                cropped = _crop_stack(stack, bbox)
                if cropped.size == 0:
                    self.one_done.emit(label, "", False)
                    fail_count += 1
                    continue

                lut_for_adj = job.get('lut_name', 'Gray')
                variants = []
                if self.mode == 'adjusted':
                    variants.append(("", job['c_min_adj'], job['c_max_adj'], "adjusted", lut_for_adj))
                elif self.mode == 'original':
                    # "original" variant 故意用 Gray:还原首次 auto-contrast 状态
                    variants.append(("", job['c_min_orig'], job['c_max_orig'], "original", 'Gray'))
                else:  # both
                    variants.append(("_adjusted", job['c_min_adj'], job['c_max_adj'], "adjusted", lut_for_adj))
                    variants.append(("_original", job['c_min_orig'], job['c_max_orig'], "original", 'Gray'))

                for suffix, c_min, c_max, variant_name, lut_name in variants:
                    if self._cancelled:
                        break
                    ok, out_path = self._encode_variant(
                        label, suffix, cropped, c_min, c_max, frame_indices, lut_name
                    )
                    tag = f"{label} [{variant_name}]" if self.mode == 'both' else label
                    if ok:
                        success_count += 1
                        out_paths.append(out_path)
                        self.one_done.emit(tag, out_path, True)
                    else:
                        fail_count += 1
                        self.one_done.emit(tag, "", False)

                gc.collect()
                try:
                    trim_working_set()
                except Exception:
                    pass

            self.progress.emit(total, total, "")
            self.finished_all.emit(success_count, fail_count, out_paths)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Item delegates — control per-column selection visual.
# ---------------------------------------------------------------------------

class _StripSelectionDelegate(QStyledItemDelegate):
    """Don't paint selection / focus visual. Use on columns that host
    cellWidgets (checkbox, spinbox, lineedit, buttons) so we don't see the
    table's row-selection background "bleed" through padding around the
    child widget."""
    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        opt.state &= ~QStyle.State_Selected
        opt.state &= ~QStyle.State_HasFocus
        super().paint(painter, opt, index)


class _WhiteSelectionDelegate(QStyledItemDelegate):
    """Paint selection bg as WHITE + dark text. Use on the label column so
    the selected row is visually identified there, and there alone.

    We MUST paint manually (fillRect + drawText) instead of going through
    super().paint() with a tweaked palette — napari's global stylesheet
    targets QTableWidget::item:selected and overrides palette-driven
    selection colors. Stylesheet > palette in Qt's painting order, so the
    only reliable way to force white is direct painter calls."""
    def paint(self, painter, option, index):
        if option.state & QStyle.State_Selected:
            painter.save()
            painter.fillRect(option.rect, QColor("#FFFFFF"))
            painter.setPen(QColor("#1a1a1a"))
            text = index.data(Qt.DisplayRole)
            # Match default QTableWidgetItem alignment (left + vcenter, 4px pad)
            text_rect = option.rect.adjusted(4, 0, -4, 0)
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, str(text or ""))
            painter.restore()
        else:
            opt = QStyleOptionViewItem(option)
            opt.state &= ~QStyle.State_HasFocus
            super().paint(painter, opt, index)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class ROIVideoExportDialog(QDialog):
    """ROI 视频导出对话框 (内嵌预览 + 三种导出模式)。"""

    def __init__(self, parent, viewer, data_layer_name, view_layer_name=None,
                 default_output_dir=None):
        super().__init__(parent)
        self.viewer = viewer
        self.data_layer_name = data_layer_name
        self.view_layer_name = view_layer_name
        self.default_output_dir = Path(default_output_dir) if default_output_dir else None

        # Add Min/Max buttons, drop the useless `?` (WindowContextHelpButtonHint).
        self.setWindowFlags(
            (self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self.setWindowTitle(tr("Export ROI Videos"))
        self.resize(1200, 720)

        self._thread = None
        self._rois = []  # see _reload_rois
        self._current_preview_idx = -1
        self._preview_pixmap = None  # keep last full-resolution pixmap for window resize

        self._build_ui()
        self._reload_rois()

    # -------------------- UI --------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ===== Top: horizontal splitter [left: source+table | right: preview] =====
        top_splitter = QSplitter(Qt.Horizontal)

        # ---- Left ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        src_box = QGroupBox(tr("Source"))
        src_layout = QFormLayout(src_box)
        self.lbl_data = QLabel(self.data_layer_name or "(None)")
        src_layout.addRow(QLabel(tr("Data Layer (Crop Source):")), self.lbl_data)
        if self.view_layer_name:
            src_layout.addRow(QLabel(tr("View Layer (info only):")),
                              QLabel(self.view_layer_name))
        btn_reload = QPushButton(tr("Refresh ROI List"))
        btn_reload.clicked.connect(self._reload_rois)
        src_layout.addRow(btn_reload)
        left_layout.addWidget(src_box)

        table_box = QGroupBox(tr("ROIs (uncheck to skip, click row to preview)"))
        table_layout = QVBoxLayout(table_box)
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            tr("Export"), tr("Label"), tr("Contrast Min"), tr("Contrast Max"),
            tr("Frame Range"), tr("Actions"),
        ])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        self.table.setColumnWidth(0, 56)
        self.table.setColumnWidth(1, 70)
        self.table.setColumnWidth(2, 110)
        self.table.setColumnWidth(3, 110)
        self.table.setColumnWidth(4, 110)
        self.table.setColumnWidth(5, 220)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(False)
        # Per-column delegates: selection visual appears ONLY in col 1 (label)
        # as a white highlight. Other columns (with cellWidgets) strip the
        # Selected state so we don't see "bleed" behind spinbox / lineedit /
        # buttons.
        self._strip_sel_delegate = _StripSelectionDelegate(self.table)
        self._white_sel_delegate = _WhiteSelectionDelegate(self.table)
        for _c in (0, 2, 3, 4, 5):
            self.table.setItemDelegateForColumn(_c, self._strip_sel_delegate)
        self.table.setItemDelegateForColumn(1, self._white_sel_delegate)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        table_layout.addWidget(self.table)

        ctrl_layout = QHBoxLayout()
        btn_all = QPushButton(tr("Select All"))
        btn_all.clicked.connect(lambda: self._set_all_export(True))
        btn_none = QPushButton(tr("Select None"))
        btn_none.clicked.connect(lambda: self._set_all_export(False))
        btn_only_selected = QPushButton(tr("Only Napari-Selected"))
        btn_only_selected.setToolTip(
            tr("Match the ROIs currently selected in the napari Batch_ROI layer")
        )
        btn_only_selected.clicked.connect(self._only_napari_selected)
        btn_auto_all = QPushButton(f"🔆 {tr('Auto Contrast All Enabled')}")
        btn_auto_all.clicked.connect(self._auto_contrast_all_enabled)
        ctrl_layout.addWidget(btn_all)
        ctrl_layout.addWidget(btn_none)
        ctrl_layout.addWidget(btn_only_selected)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(btn_auto_all)
        table_layout.addLayout(ctrl_layout)
        left_layout.addWidget(table_box, stretch=1)

        top_splitter.addWidget(left)

        # ---- Right: preview panel ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_header = QLabel(f"<b>{tr('Preview')}</b>")
        right_layout.addWidget(preview_header)

        self.preview_label = QLabel(tr("Click a row to preview"))
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 320)
        self.preview_label.setStyleSheet(
            "background-color: #1e1e1e; color: #888; border: 1px solid #444;"
        )
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.preview_label, stretch=1)

        self.preview_info = QLabel("")
        self.preview_info.setStyleSheet("color: #aaa; font-size: 11px;")
        self.preview_info.setWordWrap(True)
        right_layout.addWidget(self.preview_info)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel(tr("Preview frame:")))
        self.preview_frame_slider = QSlider(Qt.Horizontal)
        self.preview_frame_slider.setRange(0, 0)
        self.preview_frame_slider.valueChanged.connect(lambda _: self._refresh_preview())
        frame_row.addWidget(self.preview_frame_slider, stretch=1)
        self.preview_frame_label = QLabel("0/0")
        self.preview_frame_label.setMinimumWidth(60)
        frame_row.addWidget(self.preview_frame_label)
        right_layout.addLayout(frame_row)

        top_splitter.addWidget(right)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 2)
        top_splitter.setSizes([720, 460])

        root.addWidget(top_splitter, stretch=1)

        # Divider line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        root.addWidget(line)

        # ===== Bottom: params + output + mode + buttons (固定不收缩) =====
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 4, 0, 0)

        # --- Params (one row, compact) ---
        params_box = QGroupBox(tr("Video Parameters"))
        params_row = QHBoxLayout(params_box)
        params_row.addWidget(QLabel(tr("FPS:")))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(10)
        params_row.addWidget(self.fps_spin)
        params_row.addSpacing(10)
        self.chk_duration = QCheckBox(tr("Target Duration (s):"))
        self.chk_duration.setToolTip(tr("Auto-compute FPS from frame count and target duration"))
        self.chk_duration.toggled.connect(self._on_duration_toggled)
        params_row.addWidget(self.chk_duration)
        self.dur_spin = QDoubleSpinBox()
        self.dur_spin.setRange(0.5, 120.0)
        self.dur_spin.setSingleStep(0.5)
        self.dur_spin.setValue(3.0)
        self.dur_spin.setEnabled(False)
        self.dur_spin.valueChanged.connect(self._recalc_fps_from_duration)
        params_row.addWidget(self.dur_spin)
        params_row.addSpacing(16)
        params_row.addWidget(QLabel(tr("Codec:")))
        self.codec_combo = QComboBox()
        codec_options = get_all_codec_options()
        if not codec_options:
            codec_options = [('mp4v', 'mp4v (OpenCV)', 'opencv')]
        for cid, name, backend in codec_options:
            self.codec_combo.addItem(name, (cid, backend))
        params_row.addWidget(self.codec_combo)
        params_row.addSpacing(10)
        params_row.addWidget(QLabel(tr("CRF:")))
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(23)
        self.crf_spin.setToolTip(tr("18=best, 23=default, 28=smaller (FFmpeg only)"))
        params_row.addWidget(self.crf_spin)
        params_row.addSpacing(10)
        params_row.addWidget(QLabel(tr("Preset:")))
        self.preset_combo = QComboBox()
        for p in ['ultrafast', 'fast', 'medium', 'slow', 'veryslow']:
            self.preset_combo.addItem(p)
        self.preset_combo.setCurrentText('medium')
        params_row.addWidget(self.preset_combo)
        params_row.addStretch()
        bottom_layout.addWidget(params_box)

        # --- Output + Export Mode ---
        out_row = QHBoxLayout()
        out_box = QGroupBox(tr("Output"))
        out_form = QFormLayout(out_box)
        out_path_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        if self.default_output_dir:
            self.out_edit.setText(str(self.default_output_dir / "roi_videos"))
        else:
            self.out_edit.setText(str(Path.home() / "ROI_Videos"))
        btn_browse = QPushButton(tr("Browse..."))
        btn_browse.clicked.connect(self._browse_output_dir)
        out_path_row.addWidget(self.out_edit, stretch=1)
        out_path_row.addWidget(btn_browse)
        out_form.addRow(QLabel(tr("Directory:")), out_path_row)
        out_row.addWidget(out_box, stretch=3)

        mode_box = QGroupBox(tr("Export Mode"))
        mode_layout = QVBoxLayout(mode_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(
            tr("Adjusted only (one file per ROI)"), "adjusted"
        )
        self.mode_combo.addItem(
            tr("Original auto-contrast only"), "original"
        )
        self.mode_combo.addItem(
            tr("Both (adjusted + original)"), "both"
        )
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.setToolTip(tr(
            "Adjusted: use the values you tuned per ROI.\n"
            "Original: use the initial auto-contrast.\n"
            "Both: emit two files per ROI (_adjusted.mp4 + _original.mp4)."
        ))
        mode_layout.addWidget(self.mode_combo)
        out_row.addWidget(mode_box, stretch=2)
        bottom_layout.addLayout(out_row)

        # --- Progress + status ---
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        prog_row.addWidget(self.progress_bar, stretch=1)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        prog_row.addWidget(self.status_label)
        bottom_layout.addLayout(prog_row)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_export = QPushButton(f"💾 {tr('Export Videos')}")
        self.btn_export.setStyleSheet(
            "background-color: #2E7D32; color: white; font-weight: bold; padding: 6px 18px;"
        )
        self.btn_export.clicked.connect(self._on_export_clicked)
        self.btn_close = QPushButton(tr("Close"))
        self.btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_export)
        btn_row.addWidget(self.btn_close)
        bottom_layout.addLayout(btn_row)

        root.addWidget(bottom, stretch=0)

        if not is_ffmpeg_available():
            self.status_label.setText(tr("FFmpeg not detected — will use OpenCV fallback."))

    # -------------------- ROI loading --------------------

    def _reload_rois(self):
        """从 Batch_ROI layer 加载 ROI 列表，自动计算 auto-contrast。"""
        self._rois = []
        self.table.setRowCount(0)
        self._current_preview_idx = -1
        self.preview_label.setText(tr("Click a row to preview"))
        self.preview_label.setPixmap(QPixmap())
        self.preview_info.setText("")

        if "Batch_ROI" not in self.viewer.layers:
            self.status_label.setText(tr("No Batch_ROI layer found. Please draw ROIs first."))
            return
        if self.data_layer_name not in self.viewer.layers:
            self.status_label.setText(tr("Data layer not found in viewer."))
            return

        roi_layer = self.viewer.layers["Batch_ROI"]
        polygons = roi_layer.data
        if not len(polygons):
            self.status_label.setText(tr("No ROIs drawn."))
            return

        features = roi_layer.features
        labels = list(features.get("label", []))
        frame_ranges = list(features.get("frame_range", []))
        # Magnifier-saved overrides (preview_min/max/lut). Optional columns.
        preview_mins = features['preview_min'].tolist() if 'preview_min' in features.columns else []
        preview_maxs = features['preview_max'].tolist() if 'preview_max' in features.columns else []
        preview_luts = features['preview_lut'].tolist() if 'preview_lut' in features.columns else []

        stack = self.viewer.layers[self.data_layer_name].data
        stack_shape = stack.shape

        n_napari_selected = roi_layer.selected_data
        napari_selected_set = set(n_napari_selected) if n_napari_selected else set()

        self.status_label.setText(tr("Computing auto-contrast for each ROI..."))
        QApplication.processEvents()

        for i, poly in enumerate(polygons):
            label = labels[i] if i < len(labels) else f"NP{i+1}"
            frange = frame_ranges[i] if i < len(frame_ranges) else ""
            bbox = _polygon_to_bbox(poly, stack_shape)
            cropped = _crop_stack(stack, bbox)
            try:
                c_min_orig, c_max_orig = _roi_auto_contrast(cropped)
            except Exception:
                c_min_orig, c_max_orig = 0.0, 255.0
            # Saved magnifier values override defaults (NaN check)
            c_min, c_max = c_min_orig, c_max_orig
            if i < len(preview_mins) and i < len(preview_maxs):
                pm, px = preview_mins[i], preview_maxs[i]
                if pm == pm and px == px and px > pm:  # not NaN and valid
                    c_min, c_max = float(pm), float(px)
            lut_name = 'Gray'
            if i < len(preview_luts):
                lv = preview_luts[i]
                if isinstance(lv, str) and lv in _LUT_NAMES:
                    lut_name = lv

            self._rois.append({
                'label': str(label),
                'polygon': poly,
                'bbox': bbox,
                'c_min': c_min,
                'c_max': c_max,
                'c_min_orig': c_min_orig,
                'c_max_orig': c_max_orig,
                'lut_name': lut_name,
                'frame_range_text': str(frange) if frange else "",
                'enabled': True,
            })

        if napari_selected_set:
            for i in range(len(self._rois)):
                self._rois[i]['enabled'] = (i in napari_selected_set)

        self._rebuild_table()
        self.status_label.setText(f"{len(self._rois)} {tr('ROIs loaded.')}")

        # Auto-select first row to populate preview immediately
        if self._rois:
            self.table.selectRow(0)

    def _rebuild_table(self):
        self.table.setRowCount(len(self._rois))
        for i, roi in enumerate(self._rois):
            # Col 0: enable checkbox
            chk = QCheckBox()
            chk.setChecked(roi['enabled'])
            chk.toggled.connect(lambda v, idx=i: self._on_enable_toggled(idx, v))
            chk_wrap = QWidget()
            wrap_layout = QHBoxLayout(chk_wrap)
            wrap_layout.setContentsMargins(0, 0, 0, 0)
            wrap_layout.addWidget(chk)
            wrap_layout.addStretch()
            self.table.setCellWidget(i, 0, chk_wrap)

            # Col 1: label
            self.table.setItem(i, 1, QTableWidgetItem(roi['label']))

            # Local QSS for spinbox / lineedit cell controls:
            # main.py's tab_widget stylesheet sets `margin: 2px` on all
            # QSpinBox/QDoubleSpinBox/QLineEdit, which leaves a 2px gap
            # between the control's frame and the cell border — visible as
            # "vertical lines not aligning" between header and data rows.
            # Override margin:0 + keep frame styling consistent with main.py.
            _cell_input_qss = (
                "QDoubleSpinBox, QLineEdit {"
                "  margin: 0;"
                "  padding: 2px;"
                "  border: 1px solid #555;"
                "  border-radius: 3px;"
                "  background-color: #222;"
                "  min-height: 22px;"
                "}"
            )

            # Col 2/3: contrast min/max — Expanding vertical to fill cell height
            # (avoids dark "bleed" frame around spinbox when row > sizeHint)
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-1e9, 1e9)
            min_spin.setDecimals(2)
            min_spin.setValue(roi['c_min'])
            min_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            min_spin.setStyleSheet(_cell_input_qss)
            min_spin.valueChanged.connect(lambda v, idx=i: self._on_min_changed(idx, v))
            self.table.setCellWidget(i, 2, min_spin)

            max_spin = QDoubleSpinBox()
            max_spin.setRange(-1e9, 1e9)
            max_spin.setDecimals(2)
            max_spin.setValue(roi['c_max'])
            max_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            max_spin.setStyleSheet(_cell_input_qss)
            max_spin.valueChanged.connect(lambda v, idx=i: self._on_max_changed(idx, v))
            self.table.setCellWidget(i, 3, max_spin)

            # Col 4: frame range
            fr_edit = QLineEdit(roi['frame_range_text'])
            fr_edit.setPlaceholderText(tr("all"))
            fr_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            fr_edit.setStyleSheet(_cell_input_qss)
            fr_edit.editingFinished.connect(
                lambda idx=i, w=fr_edit: self._on_frange_changed(idx, w.text())
            )
            self.table.setCellWidget(i, 4, fr_edit)

            # Col 5: per-row actions (Auto / Reset)
            # - Zero horizontal margin so button right border touches cell border
            # - Local QPushButton stylesheet OVERRIDES main.py's tab_widget
            #   stylesheet which sets `padding: 6px 12px; margin: 2px` and eats
            #   ~30px per button (root cause of "曰动"/"审眷" truncation).
            # - Compact padding (2px 6px) + minWidth 100: button content area
            #   has enough room for "🔆 自动" / "↺ 重置" Chinese text.
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(0, 2, 0, 2)
            action_layout.setSpacing(4)
            _btn_qss = (
                "QPushButton {"
                "  padding: 2px 6px;"
                "  margin: 0;"
                "  border: 1px solid #555;"
                "  border-radius: 3px;"
                "  background-color: #333;"
                "}"
                "QPushButton:hover { background-color: #444; }"
                "QPushButton:pressed { background-color: #222; }"
            )
            btn_auto = QPushButton(f"🔆 {tr('Auto')}")
            btn_auto.setToolTip(tr("Recompute auto contrast from this ROI's data"))
            btn_auto.setMinimumWidth(95)
            btn_auto.setStyleSheet(_btn_qss)
            btn_auto.clicked.connect(lambda _, idx=i: self._on_per_roi_auto(idx))
            btn_reset = QPushButton(f"↺ {tr('Reset')}")
            btn_reset.setToolTip(tr("Reset to the initial auto-contrast value"))
            btn_reset.setMinimumWidth(95)
            btn_reset.setStyleSheet(_btn_qss)
            btn_reset.clicked.connect(lambda _, idx=i: self._on_per_roi_reset(idx))
            action_layout.addWidget(btn_auto)
            action_layout.addWidget(btn_reset)
            action_layout.addStretch()  # push buttons left, leave breathing room on right
            self.table.setCellWidget(i, 5, action_widget)

    # -------------------- Preview --------------------

    def _on_row_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if not (0 <= idx < len(self._rois)):
            return
        self._current_preview_idx = idx

        stack = self.viewer.layers[self.data_layer_name].data
        T = stack.shape[0] if stack.ndim == 3 else 1
        self.preview_frame_slider.blockSignals(True)
        self.preview_frame_slider.setRange(0, max(0, T - 1))
        # Pick middle of allowed frame range, fall back to T//2
        roi = self._rois[idx]
        indices = _parse_frame_range(roi['frame_range_text'], T)
        if indices:
            mid = indices[len(indices) // 2]
        else:
            mid = T // 2
        self.preview_frame_slider.setValue(int(mid))
        self.preview_frame_slider.blockSignals(False)
        self._refresh_preview()

    def _refresh_preview(self):
        idx = self._current_preview_idx
        if not (0 <= idx < len(self._rois)):
            return
        roi = self._rois[idx]
        stack = self.viewer.layers[self.data_layer_name].data
        y1, x1, y2, x2 = roi['bbox']
        T = stack.shape[0] if stack.ndim == 3 else 1
        frame_idx = int(self.preview_frame_slider.value()) if T > 1 else 0
        try:
            if stack.ndim == 3:
                cropped = stack[frame_idx, y1:y2, x1:x2]
            else:
                cropped = stack[y1:y2, x1:x2]
            cropped = np.asarray(cropped)
        except Exception as e:
            self.preview_label.setText(f"{tr('Crop error:')} {e}")
            self._preview_pixmap = None
            return

        if cropped.size == 0 or cropped.ndim != 2:
            self.preview_label.setText(tr("(empty or invalid ROI)"))
            self._preview_pixmap = None
            return

        c_min = float(roi['c_min'])
        c_max = float(roi['c_max'])
        lut_name = roi.get('lut_name', 'Gray')
        rng = max(c_max - c_min, 1e-8)
        u8 = np.clip((cropped.astype(np.float32) - c_min) / rng * 255.0, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(_apply_lut_to_frame(u8, lut_name))
        h, w, _ = rgb.shape
        # QImage shares the buffer; copy to detach before rgb goes out of scope.
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._preview_pixmap = QPixmap.fromImage(qimg)
        self._apply_preview_pixmap()
        self.preview_frame_label.setText(f"{frame_idx + 1}/{T}")
        self.preview_info.setText(
            f"<b>{roi['label']}</b> &nbsp; {tr('frame')} {frame_idx + 1}/{T} &nbsp; "
            f"bbox {h}×{w} &nbsp; LUT: {lut_name}<br>"
            f"{tr('contrast adjusted=')}[{c_min:.1f}, {c_max:.1f}] &nbsp; "
            f"{tr('original=')}[{roi['c_min_orig']:.1f}, {roi['c_max_orig']:.1f}]"
        )

    def _apply_preview_pixmap(self):
        if self._preview_pixmap is None or self._preview_pixmap.isNull():
            return
        target = self.preview_label.size()
        if target.width() < 8 or target.height() < 8:
            return
        scaled = self._preview_pixmap.scaled(
            target, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_preview_pixmap()

    # -------------------- Per-row handlers --------------------

    def _on_enable_toggled(self, idx, val):
        if 0 <= idx < len(self._rois):
            self._rois[idx]['enabled'] = bool(val)

    def _on_min_changed(self, idx, val):
        if 0 <= idx < len(self._rois):
            self._rois[idx]['c_min'] = float(val)
            if idx == self._current_preview_idx:
                self._refresh_preview()

    def _on_max_changed(self, idx, val):
        if 0 <= idx < len(self._rois):
            self._rois[idx]['c_max'] = float(val)
            if idx == self._current_preview_idx:
                self._refresh_preview()

    def _on_frange_changed(self, idx, text):
        if 0 <= idx < len(self._rois):
            self._rois[idx]['frame_range_text'] = text.strip()
            if idx == self._current_preview_idx:
                # Re-pick the middle frame from the new range
                self._on_row_selected()

    def _on_per_roi_auto(self, idx):
        if not (0 <= idx < len(self._rois)):
            return
        roi = self._rois[idx]
        stack = self.viewer.layers[self.data_layer_name].data
        cropped = _crop_stack(stack, roi['bbox'])
        try:
            c_min, c_max = _roi_auto_contrast(cropped)
        except Exception as e:
            self.status_label.setText(f"{tr('Auto contrast failed:')} {e}")
            return
        roi['c_min'] = c_min
        roi['c_max'] = c_max
        # Note: orig values stay frozen — Reset goes back to those.
        self._sync_row_spinboxes(idx)
        if idx == self._current_preview_idx:
            self._refresh_preview()
        self.status_label.setText(f"{roi['label']}: {tr('Auto')} [{c_min:.1f}, {c_max:.1f}]")

    def _on_per_roi_reset(self, idx):
        if not (0 <= idx < len(self._rois)):
            return
        roi = self._rois[idx]
        roi['c_min'] = roi['c_min_orig']
        roi['c_max'] = roi['c_max_orig']
        self._sync_row_spinboxes(idx)
        if idx == self._current_preview_idx:
            self._refresh_preview()
        self.status_label.setText(
            f"{roi['label']}: {tr('reset to original')} [{roi['c_min']:.1f}, {roi['c_max']:.1f}]"
        )

    def _sync_row_spinboxes(self, idx):
        roi = self._rois[idx]
        w_min = self.table.cellWidget(idx, 2)
        w_max = self.table.cellWidget(idx, 3)
        if w_min:
            w_min.blockSignals(True); w_min.setValue(roi['c_min']); w_min.blockSignals(False)
        if w_max:
            w_max.blockSignals(True); w_max.setValue(roi['c_max']); w_max.blockSignals(False)

    # -------------------- Bulk handlers --------------------

    def _set_all_export(self, val):
        for i, roi in enumerate(self._rois):
            roi['enabled'] = bool(val)
            w = self.table.cellWidget(i, 0)
            if w:
                chk = w.findChild(QCheckBox)
                if chk:
                    chk.blockSignals(True)
                    chk.setChecked(bool(val))
                    chk.blockSignals(False)

    def _only_napari_selected(self):
        if "Batch_ROI" not in self.viewer.layers:
            return
        sel = set(self.viewer.layers["Batch_ROI"].selected_data)
        if not sel:
            QMessageBox.information(
                self, tr("Selection"),
                tr("No ROI is currently selected in napari.")
            )
            return
        for i in range(len(self._rois)):
            self._rois[i]['enabled'] = (i in sel)
            w = self.table.cellWidget(i, 0)
            if w:
                chk = w.findChild(QCheckBox)
                if chk:
                    chk.blockSignals(True)
                    chk.setChecked(i in sel)
                    chk.blockSignals(False)

    def _auto_contrast_all_enabled(self):
        stack = self.viewer.layers[self.data_layer_name].data
        for i, roi in enumerate(self._rois):
            if not roi['enabled']:
                continue
            cropped = _crop_stack(stack, roi['bbox'])
            try:
                c_min, c_max = _roi_auto_contrast(cropped)
            except Exception:
                continue
            roi['c_min'] = c_min
            roi['c_max'] = c_max
            self._sync_row_spinboxes(i)
        if self._current_preview_idx >= 0:
            self._refresh_preview()
        self.status_label.setText(tr("Auto contrast recomputed for all enabled ROIs."))

    # -------------------- Duration <-> FPS --------------------

    def _on_duration_toggled(self, checked):
        self.dur_spin.setEnabled(checked)
        self.fps_spin.setEnabled(not checked)
        if checked:
            self._recalc_fps_from_duration()

    def _recalc_fps_from_duration(self):
        if not self.chk_duration.isChecked():
            return
        sample_n = 0
        for roi in self._rois:
            if not roi['enabled']:
                continue
            stack = self.viewer.layers[self.data_layer_name].data
            T = stack.shape[0] if stack.ndim == 3 else 1
            n = len(_parse_frame_range(roi['frame_range_text'], T))
            if n > 0:
                sample_n = n
                break
        if sample_n == 0:
            return
        dur = max(0.1, self.dur_spin.value())
        fps = max(1, int(round(sample_n / dur)))
        self.fps_spin.blockSignals(True)
        self.fps_spin.setValue(fps)
        self.fps_spin.blockSignals(False)
        self.status_label.setText(
            f"{tr('Computed FPS:')} {fps} ({sample_n} {tr('frames')} / {dur:.1f}s)"
        )

    # -------------------- Output --------------------

    def _browse_output_dir(self):
        start = self.out_edit.text().strip() or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, tr("Select Output Directory"), start)
        if d:
            self.out_edit.setText(d)

    # -------------------- Export --------------------

    def _on_export_clicked(self):
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()
            self.status_label.setText(tr("Cancelling..."))
            return

        enabled_rois = [r for r in self._rois if r['enabled']]
        if not enabled_rois:
            QMessageBox.information(
                self, tr("Nothing to export"), tr("No ROI is enabled for export.")
            )
            return

        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, tr("Output"), tr("Please choose an output directory."))
            return

        mode = self.mode_combo.currentData()

        # Validate min < max per active variant
        for r in enabled_rois:
            if mode in ('adjusted', 'both') and r['c_min'] >= r['c_max']:
                QMessageBox.warning(
                    self, tr("Invalid contrast"),
                    f"{r['label']} (adjusted): min ({r['c_min']:.2f}) >= max ({r['c_max']:.2f})"
                )
                return
            if mode in ('original', 'both') and r['c_min_orig'] >= r['c_max_orig']:
                QMessageBox.warning(
                    self, tr("Invalid contrast"),
                    f"{r['label']} (original): min ({r['c_min_orig']:.2f}) >= max ({r['c_max_orig']:.2f})"
                )
                return

        stack = self.viewer.layers[self.data_layer_name].data
        T = stack.shape[0] if stack.ndim == 3 else 1
        jobs = []
        for r in enabled_rois:
            frame_indices = _parse_frame_range(r['frame_range_text'], T)
            jobs.append({
                'label': r['label'],
                'bbox': r['bbox'],
                'c_min_adj': r['c_min'],
                'c_max_adj': r['c_max'],
                'c_min_orig': r['c_min_orig'],
                'c_max_orig': r['c_max_orig'],
                'lut_name': r.get('lut_name', 'Gray'),
                'frame_indices': frame_indices,
                'stack': stack,
            })

        codec_data = self.codec_combo.currentData()
        if codec_data is None:
            QMessageBox.warning(self, tr("Codec"), tr("No codec available."))
            return
        cid, backend = codec_data

        total_files = len(jobs) * (2 if mode == 'both' else 1)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(jobs))
        self.progress_bar.setValue(0)
        self.btn_export.setText(tr("Cancel"))
        self.btn_export.setStyleSheet(
            "background-color: #C62828; color: white; font-weight: bold; padding: 6px 18px;"
        )
        self.btn_close.setEnabled(False)
        self.status_label.setText(
            f"{tr('Exporting')} {len(jobs)} ROI × {1 if mode != 'both' else 2} = {total_files} {tr('file(s)')}..."
        )

        self._thread = ROIVideoExportThread(
            jobs=jobs,
            output_dir=out_dir,
            fps=self.fps_spin.value(),
            codec=cid,
            backend=backend,
            crf=self.crf_spin.value(),
            preset=self.preset_combo.currentText(),
            mode=mode,
        )
        self._thread.progress.connect(self._on_progress)
        self._thread.one_done.connect(self._on_one_done)
        self._thread.finished_all.connect(self._on_finished_all)
        self._thread.error.connect(self._on_thread_error)
        self._thread.start()

    def _on_progress(self, done, total, current_label):
        self.progress_bar.setValue(done)
        if current_label:
            self.status_label.setText(f"{done}/{total} — {current_label}")
        else:
            self.status_label.setText(f"{done}/{total}")

    def _on_one_done(self, tag, out_path, success):
        if success:
            self.status_label.setText(f"✅ {tag}: {Path(out_path).name}")
        else:
            self.status_label.setText(f"❌ {tag}: {tr('failed')}")

    def _on_finished_all(self, success_count, fail_count, out_paths):
        self.progress_bar.setVisible(False)
        self.btn_export.setText(f"💾 {tr('Export Videos')}")
        self.btn_export.setStyleSheet(
            "background-color: #2E7D32; color: white; font-weight: bold; padding: 6px 18px;"
        )
        self.btn_close.setEnabled(True)
        self._thread = None

        if success_count > 0 and fail_count == 0:
            msg = f"✅ {tr('All ROI videos exported.')} ({success_count})"
        elif success_count > 0:
            msg = f"⚠️ {success_count} {tr('succeeded')}, {fail_count} {tr('failed')}."
        elif fail_count > 0:
            msg = f"❌ {tr('All exports failed.')} ({fail_count})"
        else:
            msg = tr("Cancelled.")
        self.status_label.setText(msg)

        if success_count > 0:
            out_dir = self.out_edit.text().strip()
            QMessageBox.information(
                self, tr("Export complete"),
                f"{msg}\n\n{tr('Output:')} {out_dir}"
            )

    def _on_thread_error(self, err):
        self.progress_bar.setVisible(False)
        self.btn_export.setText(f"💾 {tr('Export Videos')}")
        self.btn_export.setStyleSheet(
            "background-color: #2E7D32; color: white; font-weight: bold; padding: 6px 18px;"
        )
        self.btn_close.setEnabled(True)
        self._thread = None
        self.status_label.setText(f"❌ {err}")
        QMessageBox.critical(self, tr("Export error"), err)

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            ret = QMessageBox.question(
                self, tr("Confirm"),
                tr("Export in progress. Cancel and close?"),
                QMessageBox.Yes | QMessageBox.No,
            )
            if ret == QMessageBox.Yes:
                self._thread.cancel()
                self._thread.wait(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

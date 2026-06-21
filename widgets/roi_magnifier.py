"""
ROI Magnifier — 浮动放大镜小窗口 (在 batch_crop 阶段使用).

设计目的: 用户在 napari 主视图画完 Batch_ROI 后, 想"看清"每个 ROI 内的颗粒、
调一个合适的对比度/LUT, 但又不想污染主视图也不想等 napari 子窗口启动。

特性:
- 始终置顶 + 任务栏可见 (不用 Qt.Tool, 防止用户找不到)
- 跟随 napari 选中的 ROI / 当前帧 / ROI 拖动改大小
- 50 ms 轮询 + napari 事件订阅 (双保险)
- 内置 min/max slider + LUT + Auto + Reset, 改的是放大镜内的视觉, 不动主视图
- Apply to ROI metadata: 把当前 Min/Max/LUT 写进 Batch_ROI.features 对应行
  (列名 preview_min / preview_max / preview_lut), 后续 ROI 视频导出对话框会读这些列
"""
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSlider, QSizePolicy, QSpinBox,
    QGroupBox, QCheckBox,
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QImage, QPixmap

try:
    import matplotlib.cm as _cm
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


LUT_NAMES = ['Gray', 'Inverted', 'Viridis', 'Inferno', 'Hot', 'Cool']
_LUT_TO_MPL = {
    'Viridis': 'viridis',
    'Inferno': 'inferno',
    'Hot': 'hot',
    'Cool': 'cool',
}


def _apply_lut(u8, lut_name):
    """u8 灰度 (HxW) -> RGB uint8 (HxWx3)."""
    if lut_name == 'Gray':
        return np.repeat(u8[..., None], 3, axis=2)
    if lut_name == 'Inverted':
        return np.repeat((255 - u8)[..., None], 3, axis=2)
    if not _HAS_MPL:
        return np.repeat(u8[..., None], 3, axis=2)
    cmap_name = _LUT_TO_MPL.get(lut_name, 'viridis')
    cmap = _cm.get_cmap(cmap_name)
    return (cmap(u8 / 255.0)[..., :3] * 255).astype(np.uint8)


def _auto_contrast(arr):
    flat = np.asarray(arr).ravel()
    sample = flat[::5] if flat.size > 50 else flat
    if sample.size == 0:
        return 0.0, 255.0
    lo, hi = np.percentile(sample, [0.04, 99.96])
    if hi <= lo:
        lo, hi = float(flat.min()), float(flat.max())
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def _parse_frame_indices(text, total_frames):
    """Match geometry_widget.parse_indices_helper: 0-indexed, 'a-b' inclusive.
    Empty / 'all' / 'global' -> all frames. Returns sorted list of valid indices."""
    if total_frames <= 0:
        return []
    s = str(text).strip().lower() if text is not None else ""
    if not s or s in ("all", "global"):
        return list(range(total_frames))
    indices = set()
    try:
        for p in s.split(','):
            p = p.strip()
            if not p:
                continue
            if '-' in p:
                a, b = p.split('-')
                a = max(0, int(a)); b = min(total_frames - 1, int(b))
                if a <= b:
                    indices.update(range(a, b + 1))
            else:
                i = int(p)
                if 0 <= i < total_frames:
                    indices.add(i)
    except Exception:
        return list(range(total_frames))
    return sorted(indices)


class ROIMagnifierWindow(QWidget):
    """Always-on-top magnifier mirroring a Batch_ROI.

    Two modes:
      - roi_idx is None (legacy): follow whatever ROI is selected in napari.
      - roi_idx is int (fixed):   bound to that ROI feature row; ignores selection.

    Multiple fixed-mode windows can coexist (one per ROI). The owner is
    responsible for closing them when the underlying ROI list changes size
    (indices would otherwise drift).
    """

    def __init__(self, viewer, roi_idx=None, parent=None):
        # Top-level window: Qt.Window keeps it in the taskbar, StaysOnTop floats it.
        super().__init__(None, Qt.Window | Qt.WindowStaysOnTopHint)
        self.viewer = viewer
        self._fixed_idx = roi_idx  # None = follow napari selection
        # Owner-managed back-references (set by geometry_widget._open_roi_magnifier)
        self._owner_dict = None
        self._owner_key = None
        # Optional back-ref to the geometry widget (set by _open_roi_magnifier),
        # used to read the global Slider Range Lock for frame clamping (Issue 3).
        self._geom = None
        self.setWindowTitle("ROI Magnifier")
        self.resize(320, 440)

        # State
        self._last_idx = -1
        self._last_bbox = None
        self._last_frame = -1
        self._last_stack_id = None
        self._current_data_layer = None
        self._last_T = 1
        self._c_min = 0.0
        self._c_max = 255.0
        self._c_min_orig = 0.0
        self._c_max_orig = 255.0
        self._lut_name = 'Gray'
        self._connected_layer = None
        self._suppress = False
        # Frame playback (independent of napari main viewer)
        self._frame_override = None  # None = follow main viewer's current_step
        self._playing = False
        # [Issue 3] Effective frame bounds (per-ROI Frame Filter, else global lock)
        self._frame_lo = 0
        self._frame_hi = 0
        # Phase 4 (2026-05-29): chained Auto memory + Advanced enhancements
        self._first_auto_min = None     # Initial auto values for this ROI; Reset goes here
        self._first_auto_max = None
        self._clahe_enabled = False
        self._gamma = 1.0
        self._clahe_clip = 3.0   # [Issue 2] tunable CLAHE clip limit (default 3.0)

        self._build_ui()

        # 50ms tick: detects ROI vertex drag (napari doesn't emit mid-drag events)
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start(50)

        # Play timer (decoupled from main viewer)
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        # [Issue #1] 防抖自动保存对比度/LUT 到 Batch_ROI.features，
        # 让导出无需点击 "Apply" 即可与放大镜预览一致。
        self._persist_timer = QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(180)
        self._persist_timer.timeout.connect(self._persist_preview)

        self._connect_signals()
        self._refresh()

    # ---------- UI ----------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(3)

        # Preview canvas (header info moved to window title)
        self.preview = QLabel(self)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(280, 280)
        self.preview.setStyleSheet(
            "background-color: #111; color: #888; border: 1px solid #444;"
        )
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setText("Draw or select a ROI in Batch_ROI")
        root.addWidget(self.preview, stretch=1)

        # Frame row: independent slider + <  Play  >  fps  Sync
        # Mirrors ImageJ's playback bar; uses _frame_override to decouple from main viewer.
        frame_row = QHBoxLayout()
        frame_row.setSpacing(3)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        frame_row.addWidget(self.frame_slider, stretch=1)
        self.btn_prev = QPushButton("<")
        self.btn_prev.setMaximumWidth(22)
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(lambda: self._step_frame(-1))
        frame_row.addWidget(self.btn_prev)
        self.btn_play = QPushButton("▶")
        self.btn_play.setMaximumWidth(28)
        self.btn_play.setCheckable(True)
        self.btn_play.setEnabled(False)
        self.btn_play.setToolTip("Play / Stop (decoupled from main viewer)")
        self.btn_play.toggled.connect(self._on_play_toggled)
        frame_row.addWidget(self.btn_play)
        self.btn_next = QPushButton(">")
        self.btn_next.setMaximumWidth(22)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(lambda: self._step_frame(+1))
        frame_row.addWidget(self.btn_next)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.setMaximumWidth(72)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.setToolTip("Playback rate (frames per second)")
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        frame_row.addWidget(self.fps_spin)
        self.btn_sync = QPushButton("Sync")
        self.btn_sync.setMaximumWidth(46)
        self.btn_sync.setToolTip("Re-sync frame to main viewer (cancel local frame override)")
        self.btn_sync.clicked.connect(self._on_sync_clicked)
        frame_row.addWidget(self.btn_sync)
        root.addLayout(frame_row)

        # LUT combo + Auto/Reset compact buttons
        lut_row = QHBoxLayout()
        lut_row.setSpacing(3)
        self.lut_combo = QComboBox()
        self.lut_combo.addItems(LUT_NAMES)
        self.lut_combo.setMinimumWidth(90)
        self.lut_combo.currentTextChanged.connect(self._on_lut_changed)
        lut_row.addWidget(self.lut_combo)
        lut_row.addStretch()
        btn_auto = QPushButton("Auto")
        btn_auto.setToolTip("Recompute auto-contrast from the current ROI/frame")
        btn_auto.setMaximumWidth(54)
        btn_auto.clicked.connect(self._on_auto)
        lut_row.addWidget(btn_auto)
        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Reset to the initial auto-contrast")
        btn_reset.setMaximumWidth(54)
        btn_reset.clicked.connect(self._on_reset)
        lut_row.addWidget(btn_reset)
        root.addLayout(lut_row)

        # Min slider+spin (label dropped — spinbox is self-descriptive)
        min_row = QHBoxLayout()
        min_row.setSpacing(3)
        lbl_min = QLabel("Min")
        lbl_min.setMinimumWidth(28)
        min_row.addWidget(lbl_min)
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, 1000)
        self.min_slider.valueChanged.connect(self._on_slider_changed)
        min_row.addWidget(self.min_slider, stretch=1)
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setDecimals(1)
        self.min_spin.setMaximumWidth(80)
        self.min_spin.valueChanged.connect(self._on_min_spin_changed)
        min_row.addWidget(self.min_spin)
        root.addLayout(min_row)

        # Max
        max_row = QHBoxLayout()
        max_row.setSpacing(3)
        lbl_max = QLabel("Max")
        lbl_max.setMinimumWidth(28)
        max_row.addWidget(lbl_max)
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setRange(0, 1000)
        self.max_slider.valueChanged.connect(self._on_slider_changed)
        max_row.addWidget(self.max_slider, stretch=1)
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setDecimals(1)
        self.max_spin.setMaximumWidth(80)
        self.max_spin.valueChanged.connect(self._on_max_spin_changed)
        max_row.addWidget(self.max_spin)
        root.addLayout(max_row)

        # Action row: Apply (writes preview_min/max/lut into ROI features) + force refresh
        btn_row = QHBoxLayout()
        btn_row.setSpacing(3)
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setToolTip(
            "Save current Min/Max/LUT to Batch_ROI features "
            "(preview_min, preview_max, preview_lut) so the ROI Video dialog reuses them."
        )
        self.btn_apply.clicked.connect(self._on_apply)
        btn_row.addWidget(self.btn_apply)
        btn_refresh = QPushButton("↻")
        btn_refresh.setMaximumWidth(28)
        btn_refresh.setToolTip("Force a refresh")
        btn_refresh.clicked.connect(lambda: self._refresh(force=True))
        btn_row.addWidget(btn_refresh)
        root.addLayout(btn_row)

        # Phase 4 (2026-05-29): Advanced (CLAHE + gamma). Collapsible — default off.
        self.adv_group = QGroupBox("Advanced (CLAHE / γ)")
        self.adv_group.setCheckable(True)
        self.adv_group.setChecked(False)
        # Slim vertical footprint when collapsed
        self.adv_group.toggled.connect(self._on_adv_toggled)
        adv_layout = QVBoxLayout()
        adv_layout.setContentsMargins(6, 6, 6, 6)
        adv_layout.setSpacing(4)

        # [Issue 2] CLAHE row: checkbox + tunable clip-limit spinbox (grid stays 4×4)
        clahe_row = QHBoxLayout()
        clahe_row.setSpacing(4)
        self.chk_clahe = QCheckBox("CLAHE")
        self.chk_clahe.setToolTip(
            "Contrast Limited Adaptive Histogram Equalization (grid 4×4). "
            "Applied AFTER min/max burn, BEFORE LUT. Useful for low-contrast TEM frames."
        )
        self.chk_clahe.toggled.connect(self._on_clahe_toggled)
        clahe_row.addWidget(self.chk_clahe)
        clahe_row.addStretch()
        clahe_row.addWidget(QLabel("clip"))
        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_spin.setRange(0.5, 20.0)
        self.clahe_clip_spin.setSingleStep(0.5)
        self.clahe_clip_spin.setDecimals(1)
        self.clahe_clip_spin.setValue(self._clahe_clip)
        self.clahe_clip_spin.setMaximumWidth(64)
        self.clahe_clip_spin.setToolTip(
            "CLAHE clip limit. Higher = stronger local contrast (more noise). Default 3.0."
        )
        self.clahe_clip_spin.valueChanged.connect(self._on_clahe_clip_changed)
        clahe_row.addWidget(self.clahe_clip_spin)
        adv_layout.addLayout(clahe_row)

        gamma_row = QHBoxLayout()
        gamma_row.setSpacing(4)
        gamma_row.addWidget(QLabel("γ"))
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 500)  # 0.10 .. 5.00 (×100)
        self.gamma_slider.setValue(100)
        self.gamma_slider.setToolTip(
            "Gamma correction: out = (in/255)^γ * 255. "
            "<1 brightens midtones, >1 darkens midtones."
        )
        self.gamma_slider.valueChanged.connect(self._on_gamma_changed)
        gamma_row.addWidget(self.gamma_slider, stretch=1)
        self.gamma_value_label = QLabel("1.00")
        self.gamma_value_label.setMinimumWidth(32)
        self.gamma_value_label.setStyleSheet("color: #BBB;")
        gamma_row.addWidget(self.gamma_value_label)
        self.btn_gamma_reset = QPushButton("⟲")
        self.btn_gamma_reset.setMaximumWidth(22)
        self.btn_gamma_reset.setToolTip("Reset γ to 1.0")
        self.btn_gamma_reset.clicked.connect(lambda: self.gamma_slider.setValue(100))
        gamma_row.addWidget(self.btn_gamma_reset)
        adv_layout.addLayout(gamma_row)

        self.adv_group.setLayout(adv_layout)
        # Start collapsed: hide content but keep header visible
        self._set_advanced_content_visible(False)
        root.addWidget(self.adv_group)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #888; font-size: 9px;")
        self.status.setMaximumHeight(14)
        root.addWidget(self.status)

    # ---------- napari signal wiring ----------

    def _connect_signals(self):
        try:
            self.viewer.layers.events.inserted.connect(self._on_layers_changed)
            self.viewer.layers.events.removed.connect(self._on_layers_changed)
            self.viewer.dims.events.current_step.connect(self._on_napari_changed)
        except Exception:
            pass
        self._bind_batch_roi_events()

    def _bind_batch_roi_events(self):
        try:
            if "Batch_ROI" in self.viewer.layers:
                layer = self.viewer.layers["Batch_ROI"]
                if self._connected_layer is layer:
                    return
                self._unbind_batch_roi_events()
                try:
                    layer.events.set_data.connect(self._on_napari_changed)
                except Exception:
                    pass
                try:
                    layer.events.highlight.connect(self._on_napari_changed)
                except Exception:
                    pass
                self._connected_layer = layer
        except Exception:
            pass

    def _unbind_batch_roi_events(self):
        if self._connected_layer is None:
            return
        for sig_name in ("set_data", "highlight"):
            try:
                getattr(self._connected_layer.events, sig_name).disconnect(self._on_napari_changed)
            except Exception:
                pass
        self._connected_layer = None

    def _on_layers_changed(self, _evt=None):
        if "Batch_ROI" not in self.viewer.layers:
            self._unbind_batch_roi_events()
        else:
            self._bind_batch_roi_events()
        self._refresh()

    def _on_napari_changed(self, _evt=None):
        self._refresh()

    # ---------- 50 ms tick ----------

    def _on_tick(self):
        if not self.isVisible():
            return
        if "Batch_ROI" not in self.viewer.layers:
            return
        roi_layer = self.viewer.layers["Batch_ROI"]

        # Fixed mode: bound to one ROI; close window if that ROI is gone.
        if self._fixed_idx is not None:
            if self._fixed_idx >= len(roi_layer.data):
                self.close()
                return
            idx = self._fixed_idx
        else:
            # Legacy follow-selection mode
            sel = roi_layer.selected_data
            idx = -1
            if sel:
                try:
                    idx = next(iter(sel))
                except Exception:
                    idx = -1
            if idx < 0 and self._last_idx >= 0 and self._last_idx < len(roi_layer.data):
                idx = self._last_idx
            if idx < 0 or idx >= len(roi_layer.data):
                return

        bbox = self._poly_to_bbox(roi_layer.data[idx])
        if self._frame_override is not None:
            frame = int(self._frame_override)
        else:
            try:
                frame = int(self.viewer.dims.current_step[0])
            except Exception:
                frame = 0
        data_layer = self._get_data_layer()
        stack_id = id(data_layer.data) if data_layer is not None else None

        if (idx != self._last_idx
                or bbox != self._last_bbox
                or frame != self._last_frame
                or stack_id != self._last_stack_id):
            self._refresh()

    # ---------- core refresh ----------

    def _get_data_layer(self):
        """Pick the most relevant image layer to crop from."""
        try:
            import napari
        except Exception:
            return None
        active = self.viewer.layers.selection.active
        if isinstance(active, napari.layers.Image):
            return active
        for l in reversed(list(self.viewer.layers)):
            if isinstance(l, napari.layers.Image) and l.visible:
                return l
        for l in self.viewer.layers:
            if isinstance(l, napari.layers.Image):
                return l
        return None

    def _poly_to_bbox(self, poly):
        ys = poly[:, 0]
        xs = poly[:, 1]
        return (int(min(ys)), int(min(xs)), int(max(ys)), int(max(xs)))

    def _refresh(self, force=False):
        if "Batch_ROI" not in self.viewer.layers:
            self.preview.setText("(no Batch_ROI layer — draw an ROI first)")
            self.preview.setPixmap(QPixmap())
            self.setWindowTitle("ROI Magnifier — (no Batch_ROI)")
            return
        roi_layer = self.viewer.layers["Batch_ROI"]
        if len(roi_layer.data) == 0:
            self.preview.setText("(no ROIs drawn yet)")
            self.preview.setPixmap(QPixmap())
            self.setWindowTitle("ROI Magnifier — (no ROIs)")
            return

        # Decide which ROI to show
        if self._fixed_idx is not None:
            if self._fixed_idx >= len(roi_layer.data):
                # ROI deleted while window was alive — the tick will close us
                self.preview.setText("(this ROI was deleted)")
                self.preview.setPixmap(QPixmap())
                self.setWindowTitle("ROI Magnifier — (deleted)")
                return
            idx = self._fixed_idx
        else:
            sel = roi_layer.selected_data
            idx = -1
            if sel:
                try:
                    idx = next(iter(sel))
                except Exception:
                    idx = -1
            if idx < 0:
                idx = self._last_idx if 0 <= self._last_idx < len(roi_layer.data) else 0

        poly = roi_layer.data[idx]
        bbox = self._poly_to_bbox(poly)
        y1, x1, y2, x2 = bbox

        data_layer = self._get_data_layer()
        if data_layer is None:
            self.preview.setText("(no image layer to crop from)")
            return
        stack = data_layer.data
        if stack.ndim == 3:
            T, H, W = stack.shape
        elif stack.ndim == 2:
            T = 1
            H, W = stack.shape
        else:
            self.preview.setText(f"(unsupported ndim={stack.ndim})")
            return

        # Sync frame slider range/state to current data layer
        self._sync_frame_controls(T)

        y1 = max(0, y1); x1 = max(0, x1)
        y2 = min(H, y2); x2 = min(W, x2)
        if y2 <= y1 or x2 <= x1:
            self.preview.setText("(empty ROI)")
            return

        if self._frame_override is not None:
            frame = int(self._frame_override)
        else:
            try:
                frame = int(self.viewer.dims.current_step[0])
            except Exception:
                frame = 0
        frame = max(0, min(frame, T - 1))
        # [Issue 3] Keep the displayed frame inside the active range (computed in
        # _sync_frame_controls just above) even when following the main viewer.
        frame = max(self._frame_lo, min(frame, self._frame_hi))
        try:
            if stack.ndim == 3:
                cropped = np.asarray(stack[frame, y1:y2, x1:x2])
            else:
                cropped = np.asarray(stack[y1:y2, x1:x2])
        except Exception as e:
            self.preview.setText(f"Crop error: {e}")
            return

        # ROI/layer/bbox changed → recompute orig auto-contrast and reset working values.
        roi_changed = (
            idx != self._last_idx
            or bbox != self._last_bbox
            or data_layer is not self._current_data_layer
            or self._last_bbox is None
        )
        if roi_changed:
            self._c_min_orig, self._c_max_orig = _auto_contrast(cropped)
            # Phase 4: clear chained-Auto memory so Reset on the new ROI works correctly
            self._first_auto_min = None
            self._first_auto_max = None
            # Preserve user-saved metadata if present
            features = roi_layer.features
            saved_min = None
            saved_max = None
            saved_lut = None
            try:
                if 'preview_min' in features.columns:
                    v = features['preview_min'].iloc[idx]
                    if v == v:  # not NaN
                        saved_min = float(v)
                if 'preview_max' in features.columns:
                    v = features['preview_max'].iloc[idx]
                    if v == v:
                        saved_max = float(v)
                if 'preview_lut' in features.columns:
                    v = features['preview_lut'].iloc[idx]
                    if isinstance(v, str) and v:
                        saved_lut = v
            except Exception:
                pass
            if saved_min is not None and saved_max is not None and saved_max > saved_min:
                self._c_min, self._c_max = saved_min, saved_max
            else:
                self._c_min, self._c_max = self._c_min_orig, self._c_max_orig
            if saved_lut and saved_lut in LUT_NAMES:
                self._lut_name = saved_lut
                self.lut_combo.blockSignals(True)
                self.lut_combo.setCurrentText(saved_lut)
                self.lut_combo.blockSignals(False)
            # [Issue 1] Restore Advanced (CLAHE + clip + gamma) for this ROI as well,
            # otherwise reopening the magnifier silently drops them to defaults while
            # export still applies the saved values (the reported divergence).
            self._restore_advanced_from_features(features, idx)
            self._sync_controls()

        self._last_idx = idx
        self._last_bbox = bbox
        self._last_frame = frame
        self._last_T = T
        self._last_stack_id = id(stack)
        self._current_data_layer = data_layer

        # Reflect the current frame back into the slider (without retriggering)
        if hasattr(self, 'frame_slider'):
            self.frame_slider.blockSignals(True)
            try:
                self.frame_slider.setValue(frame)
            finally:
                self.frame_slider.blockSignals(False)

        # Burn + post-burn (Phase 4: CLAHE/gamma) + LUT
        rng = max(self._c_max - self._c_min, 1e-8)
        u8 = np.clip(
            (cropped.astype(np.float32) - self._c_min) / rng * 255.0, 0, 255
        ).astype(np.uint8)
        u8 = self._apply_post_burn(u8)
        rgb = np.ascontiguousarray(_apply_lut(u8, self._lut_name))
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        target = self.preview.size()
        if target.width() > 8 and target.height() > 8:
            pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pix)

        # Header → window title (no HTML, single line)
        labels = roi_layer.features.get("label", [])
        try:
            label = labels.iloc[idx] if hasattr(labels, 'iloc') else (labels[idx] if idx < len(labels) else f"NP{idx + 1}")
        except Exception:
            label = f"NP{idx + 1}"
        self.setWindowTitle(
            f"{label} · {h}×{w} · frame {frame + 1}/{T} · {data_layer.name}"
        )

    def _sync_controls(self):
        self._suppress = True
        try:
            lo = self._c_min_orig
            hi = self._c_max_orig
            span = max(hi - lo, 1e-8)
            self.min_spin.blockSignals(True); self.min_spin.setValue(self._c_min); self.min_spin.blockSignals(False)
            self.max_spin.blockSignals(True); self.max_spin.setValue(self._c_max); self.max_spin.blockSignals(False)
            self.min_slider.blockSignals(True)
            self.min_slider.setValue(int(np.clip((self._c_min - lo) / span * 1000, 0, 1000)))
            self.min_slider.blockSignals(False)
            self.max_slider.blockSignals(True)
            self.max_slider.setValue(int(np.clip((self._c_max - lo) / span * 1000, 0, 1000)))
            self.max_slider.blockSignals(False)
        finally:
            self._suppress = False

    def _sync_frame_controls(self, T):
        """Enable / disable / range-set the frame playback row based on current data
        depth AND the active frame range (per-ROI Frame Filter, else global lock)."""
        if not hasattr(self, 'frame_slider'):
            return
        lo, hi = self._effective_frame_bounds(T)
        self._frame_lo, self._frame_hi = lo, hi
        has_frames = hi > lo
        self.frame_slider.blockSignals(True)
        try:
            cur = self.frame_slider.value()
            self.frame_slider.setRange(lo, hi)
            if cur < lo:
                self.frame_slider.setValue(lo)
            elif cur > hi:
                self.frame_slider.setValue(hi)
        finally:
            self.frame_slider.blockSignals(False)
        self.frame_slider.setEnabled(has_frames)
        self.btn_prev.setEnabled(has_frames)
        self.btn_next.setEnabled(has_frames)
        self.btn_play.setEnabled(has_frames)
        # Clamp any active frame override into the effective range
        if self._frame_override is not None:
            self._frame_override = max(lo, min(self._frame_override, hi))

    def _sync_sliders_from_spin(self):
        lo = self._c_min_orig
        hi = self._c_max_orig
        span = max(hi - lo, 1e-8)
        self.min_slider.blockSignals(True)
        self.min_slider.setValue(int(np.clip((self._c_min - lo) / span * 1000, 0, 1000)))
        self.min_slider.blockSignals(False)
        self.max_slider.blockSignals(True)
        self.max_slider.setValue(int(np.clip((self._c_max - lo) / span * 1000, 0, 1000)))
        self.max_slider.blockSignals(False)

    # ---------- frame playback handlers ----------

    def _on_frame_slider_changed(self, v):
        if self._suppress:
            return
        self._frame_override = int(v)
        self._refresh()

    def _step_frame(self, delta):
        lo = getattr(self, '_frame_lo', 0)
        hi = getattr(self, '_frame_hi', max(0, self._last_T - 1))
        span = hi - lo + 1
        if span <= 0:
            return
        if self._frame_override is None:
            try:
                cur = int(self.viewer.dims.current_step[0])
            except Exception:
                cur = lo
        else:
            cur = self._frame_override
        # Wrap within the effective range [lo, hi]
        self._frame_override = lo + ((cur - lo + delta) % span)
        self._refresh()

    def _on_play_toggled(self, checked):
        self._playing = bool(checked)
        if checked:
            self.btn_play.setText("⏸")
            interval = max(16, 1000 // max(1, self.fps_spin.value()))
            # Snapshot current frame so play starts from where we are
            if self._frame_override is None:
                try:
                    self._frame_override = int(self.viewer.dims.current_step[0])
                except Exception:
                    self._frame_override = 0
            self._play_timer.start(interval)
        else:
            self.btn_play.setText("▶")
            self._play_timer.stop()

    def _on_play_tick(self):
        lo = getattr(self, '_frame_lo', 0)
        hi = getattr(self, '_frame_hi', max(0, self._last_T - 1))
        span = hi - lo + 1
        if span <= 0:
            return
        if self._frame_override is None:
            self._frame_override = lo
        self._frame_override = lo + ((self._frame_override - lo + 1) % span)
        self._refresh()

    def _on_fps_changed(self, fps):
        if self._playing:
            self._play_timer.setInterval(max(16, 1000 // max(1, fps)))

    def _on_sync_clicked(self):
        """Cancel local frame override; resume following main viewer's current_step."""
        if self._playing:
            self.btn_play.setChecked(False)  # also stops timer via toggled
        self._frame_override = None
        self._refresh()

    # ---------- handlers ----------

    def _on_slider_changed(self, _v):
        if self._suppress:
            return
        lo = self._c_min_orig
        hi = self._c_max_orig
        span = max(hi - lo, 1e-8)
        self._c_min = lo + self.min_slider.value() / 1000.0 * span
        self._c_max = lo + self.max_slider.value() / 1000.0 * span
        if self._c_max <= self._c_min:
            self._c_max = self._c_min + max(1.0, span * 0.01)
        self.min_spin.blockSignals(True); self.min_spin.setValue(self._c_min); self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(True); self.max_spin.setValue(self._c_max); self.max_spin.blockSignals(False)
        self._render_current()
        self._schedule_persist()

    def _on_min_spin_changed(self, v):
        if self._suppress:
            return
        self._c_min = float(v)
        self._sync_sliders_from_spin()
        self._render_current()
        self._schedule_persist()

    def _on_max_spin_changed(self, v):
        if self._suppress:
            return
        self._c_max = float(v)
        self._sync_sliders_from_spin()
        self._render_current()
        self._schedule_persist()

    def _on_lut_changed(self, name):
        self._lut_name = name
        self._render_current()
        self._schedule_persist()

    def _render_current(self):
        """Re-render with current settings WITHOUT re-detecting ROI change.
        Used when only contrast/LUT changed."""
        if self._last_bbox is None or self._current_data_layer is None:
            return
        y1, x1, y2, x2 = self._last_bbox
        stack = self._current_data_layer.data
        try:
            if stack.ndim == 3:
                T = stack.shape[0]
                frame = max(0, min(self._last_frame, T - 1))
                cropped = np.asarray(stack[frame, y1:y2, x1:x2])
            else:
                cropped = np.asarray(stack[y1:y2, x1:x2])
        except Exception:
            return
        if cropped.size == 0:
            return
        rng = max(self._c_max - self._c_min, 1e-8)
        u8 = np.clip(
            (cropped.astype(np.float32) - self._c_min) / rng * 255.0, 0, 255
        ).astype(np.uint8)
        u8 = self._apply_post_burn(u8)
        rgb = np.ascontiguousarray(_apply_lut(u8, self._lut_name))
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        target = self.preview.size()
        if target.width() > 8 and target.height() > 8:
            pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pix)

    def _on_auto(self):
        """Phase 4: chained Auto.
        - First click on a fresh ROI: compute 0.04 / 99.96 percentile on the full crop
          and remember the result for Reset.
        - Subsequent clicks: re-compute percentile within the currently visible
          [_c_min, _c_max] window (ImageJ-style narrowing). Falls back to full crop
          if the visible region has too few pixels.
        """
        if self._last_bbox is None or self._current_data_layer is None:
            return
        y1, x1, y2, x2 = self._last_bbox
        stack = self._current_data_layer.data
        try:
            if stack.ndim == 3:
                T = stack.shape[0]
                frame = max(0, min(self._last_frame, T - 1))
                cropped = np.asarray(stack[frame, y1:y2, x1:x2])
            else:
                cropped = np.asarray(stack[y1:y2, x1:x2])
        except Exception as e:
            self.status.setText(f"Auto failed: {e}")
            return

        chained = self._first_auto_min is not None
        if chained:
            mask = (cropped >= self._c_min) & (cropped <= self._c_max)
            visible = cropped[mask]
            if visible.size >= 100:
                sample = visible.ravel()[::5] if visible.size > 50 else visible.ravel()
                lo, hi = np.percentile(sample, [0.04, 99.96])
                if hi <= lo:
                    lo = float(visible.min())
                    hi = float(visible.max())
                    if hi <= lo:
                        hi = lo + 1.0
            else:
                # Too sparse — fall back to full-crop auto
                lo, hi = _auto_contrast(cropped)
        else:
            lo, hi = _auto_contrast(cropped)
            self._first_auto_min = lo
            self._first_auto_max = hi

        self._c_min_orig, self._c_max_orig = lo, hi
        self._c_min, self._c_max = lo, hi
        self._sync_controls()
        self._render_current()
        self._schedule_persist()
        suffix = " (chained)" if chained else ""
        self.status.setText(f"Auto{suffix}: [{self._c_min:.1f}, {self._c_max:.1f}]")

    def _on_reset(self):
        """Phase 4: Reset goes back to the FIRST Auto's range (if any), not the latest."""
        if self._first_auto_min is not None and self._first_auto_max is not None:
            self._c_min_orig = self._first_auto_min
            self._c_max_orig = self._first_auto_max
            self._c_min = self._first_auto_min
            self._c_max = self._first_auto_max
        else:
            self._c_min, self._c_max = self._c_min_orig, self._c_max_orig
        self._sync_controls()
        self._render_current()
        self._schedule_persist()
        self.status.setText(f"Reset: [{self._c_min:.1f}, {self._c_max:.1f}]")

    def _restore_advanced_from_features(self, features, idx):
        """[Issue 1] Restore (or neutralize) the Advanced state for ROI `idx`.

        Reads preview_gamma / preview_clahe / preview_clahe_clip from the ROI
        features. If none are saved (or all neutral) the panel collapses to its
        defaults so switching ROIs never leaks the previous particle's settings.
        The advanced panel is auto-expanded when any non-neutral value exists so
        _apply_post_burn (which gates on adv_group.isChecked()) actually applies it.
        """
        gamma, clahe, clip = 1.0, False, 3.0
        try:
            n = len(features)
            if 'preview_gamma' in features.columns and idx < n:
                v = features['preview_gamma'].iloc[idx]
                if v == v:
                    gamma = float(v)
            if 'preview_clahe' in features.columns and idx < n:
                v = features['preview_clahe'].iloc[idx]
                if v == v:
                    clahe = bool(v)
            if 'preview_clahe_clip' in features.columns and idx < n:
                v = features['preview_clahe_clip'].iloc[idx]
                if v == v and float(v) > 0:
                    clip = float(v)
        except Exception:
            pass
        self._gamma = gamma if gamma > 0 else 1.0
        self._clahe_enabled = bool(clahe)
        self._clahe_clip = clip if clip > 0 else 3.0
        adv_on = self._clahe_enabled or abs(self._gamma - 1.0) > 1e-6

        widgets = [getattr(self, n, None) for n in
                   ('chk_clahe', 'gamma_slider', 'clahe_clip_spin', 'adv_group')]
        for w in widgets:
            if w is not None:
                w.blockSignals(True)
        try:
            if hasattr(self, 'chk_clahe'):
                self.chk_clahe.setChecked(self._clahe_enabled)
            if hasattr(self, 'clahe_clip_spin'):
                self.clahe_clip_spin.setValue(self._clahe_clip)
            if hasattr(self, 'gamma_slider'):
                self.gamma_slider.setValue(int(round(self._gamma * 100)))
            if hasattr(self, 'gamma_value_label'):
                self.gamma_value_label.setText(f"{self._gamma:.2f}")
            if hasattr(self, 'adv_group'):
                self.adv_group.setChecked(adv_on)
        finally:
            for w in widgets:
                if w is not None:
                    w.blockSignals(False)
        # Signals were blocked, so reflect the expand/collapse state manually.
        self._set_advanced_content_visible(adv_on)

    def _effective_frame_bounds(self, T):
        """[Issue 3] Compute [lo, hi] the magnifier frame slider should span.

        Precedence: this ROI's per-ROI Frame Filter (Batch_ROI.features
        'frame_range') > the global Slider Range Lock on the geometry widget >
        the full stack [0, T-1]. A range that resolves to ALL frames is ignored.
        """
        lo, hi = 0, max(0, T - 1)
        # 1) per-ROI Frame Filter
        try:
            idx = self._last_idx
            if idx is not None and idx >= 0 and "Batch_ROI" in self.viewer.layers:
                feats = self.viewer.layers["Batch_ROI"].features
                if 'frame_range' in feats.columns and idx < len(feats):
                    frames = _parse_frame_indices(feats['frame_range'].iloc[idx], T)
                    if frames and len(frames) < T:
                        return max(lo, min(frames)), min(hi, max(frames))
        except Exception:
            pass
        # 2) global Slider Range Lock (set via geometry widget In/Out/Lock)
        try:
            geom = getattr(self, '_geom', None)
            if geom is not None and getattr(geom, '_slider_lock_enabled', False):
                gmin = int(getattr(geom, '_slider_lock_min', 0))
                gmax = int(getattr(geom, '_slider_lock_max', 0))
                if gmax >= gmin:
                    nlo, nhi = max(lo, gmin), min(hi, gmax)
                    if nhi >= nlo:
                        return nlo, nhi
        except Exception:
            pass
        return lo, hi

    def _write_preview_to_features(self, idx):
        """把当前 _c_min/_c_max/_lut_name 写入 Batch_ROI.features 第 idx 行，返回该行 label。
        供手动 Apply 按钮与防抖自动保存共用。"""
        roi_layer = self.viewer.layers["Batch_ROI"]
        features = roi_layer.features
        for col, default in (('preview_min', np.nan),
                             ('preview_max', np.nan),
                             ('preview_lut', ''),
                             ('preview_gamma', 1.0),
                             ('preview_clahe', False),
                             ('preview_clahe_clip', 3.0)):
            if col not in features.columns:
                features[col] = default
        features.loc[idx, 'preview_min'] = float(self._c_min)
        features.loc[idx, 'preview_max'] = float(self._c_max)
        features.loc[idx, 'preview_lut'] = self._lut_name
        # [Issue] 保存 gamma/CLAHE/clip, 让导出的 _preview 与放大镜所见一致。
        # 仅 Advanced 面板开启时这些项生效, 否则存中性值(gamma=1, clahe=False, clip=3.0)。
        adv_on = bool(self.adv_group.isChecked()) if hasattr(self, 'adv_group') else False
        features.loc[idx, 'preview_gamma'] = float(self._gamma) if adv_on else 1.0
        features.loc[idx, 'preview_clahe'] = bool(getattr(self, '_clahe_enabled', False)) if adv_on else False
        features.loc[idx, 'preview_clahe_clip'] = float(getattr(self, '_clahe_clip', 3.0)) if adv_on else 3.0
        roi_layer.features = features
        labels = features.get('label', [])
        try:
            return labels.iloc[idx] if hasattr(labels, 'iloc') else (labels[idx] if idx < len(labels) else f"NP{idx + 1}")
        except Exception:
            return f"NP{idx + 1}"

    def _resolve_target_idx(self):
        """要写入的目标 ROI 行号；无有效目标返回 -1。
        _last_idx 在 _refresh 中已对齐 fixed/follow 两种模式。"""
        if "Batch_ROI" not in self.viewer.layers:
            return -1
        idx = self._last_idx
        if idx is None or idx < 0 or idx >= len(self.viewer.layers["Batch_ROI"].data):
            return -1
        return idx

    def _schedule_persist(self):
        """[Issue #1] 重启防抖计时器，合并连续的滑块/微调拖动写入。"""
        try:
            self._persist_timer.start()
        except Exception:
            pass

    def _persist_preview(self):
        """[Issue #1] 防抖自动保存：调完对比度/LUT 后无需点 Apply，导出即所见。
        静默执行——失败不打扰用户，Apply 按钮仍可手动重试。"""
        idx = self._resolve_target_idx()
        if idx < 0:
            return
        try:
            self._write_preview_to_features(idx)
        except Exception:
            pass

    def _on_apply(self):
        if "Batch_ROI" not in self.viewer.layers:
            self.status.setText("No Batch_ROI layer.")
            return
        idx = self._resolve_target_idx()
        if idx < 0:
            self.status.setText("No ROI selected.")
            return
        try:
            label = self._write_preview_to_features(idx)
            self.status.setText(
                f"✅ {label}: saved Min={self._c_min:.1f} Max={self._c_max:.1f} LUT={self._lut_name}"
            )
        except Exception as e:
            self.status.setText(f"Apply failed: {e}")

    # ---------- Phase 4 (2026-05-29): Advanced enhancements ----------

    def _on_adv_toggled(self, checked):
        """Show/hide the Advanced group contents when the header checkbox toggles."""
        self._set_advanced_content_visible(checked)
        # Re-render so any state change inside (CLAHE/gamma) reflects immediately
        self._render_current()

    def _set_advanced_content_visible(self, visible):
        """Hide/show all children inside the advanced GroupBox to collapse vertically."""
        if not hasattr(self, 'adv_group'):
            return
        layout = self.adv_group.layout()
        if layout is None:
            return
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.setVisible(visible)
            else:
                sub = item.layout()
                if sub is not None:
                    for j in range(sub.count()):
                        sub_item = sub.itemAt(j)
                        if sub_item and sub_item.widget() is not None:
                            sub_item.widget().setVisible(visible)

    def _on_clahe_toggled(self, checked):
        self._clahe_enabled = bool(checked)
        self._render_current()
        self._schedule_persist()

    def _on_clahe_clip_changed(self, v):
        self._clahe_clip = max(0.1, float(v))
        self._render_current()
        self._schedule_persist()

    def _on_gamma_changed(self, v):
        self._gamma = max(0.1, float(v) / 100.0)
        self.gamma_value_label.setText(f"{self._gamma:.2f}")
        self._render_current()

    def _apply_post_burn(self, u8):
        """Apply Advanced enhancements (CLAHE + gamma) to a uint8 grayscale frame.
        Order: CLAHE -> gamma. LUT is applied AFTER this (in the caller)."""
        if not self.adv_group.isChecked():
            return u8
        if getattr(self, '_clahe_enabled', False):
            try:
                import cv2
                clahe = cv2.createCLAHE(
                    clipLimit=float(getattr(self, '_clahe_clip', 3.0)),
                    tileGridSize=(4, 4),
                )
                u8 = clahe.apply(u8)
            except Exception:
                pass
        if abs(self._gamma - 1.0) > 1e-6:
            u8 = np.clip(
                (u8.astype(np.float32) / 255.0) ** self._gamma * 255.0,
                0, 255
            ).astype(np.uint8)
        return u8

    # ---------- cleanup ----------

    def closeEvent(self, event):
        try:
            self._tick.stop()
        except Exception:
            pass
        try:
            self._play_timer.stop()
        except Exception:
            pass
        self._unbind_batch_roi_events()
        try:
            self.viewer.dims.events.current_step.disconnect(self._on_napari_changed)
        except Exception:
            pass
        try:
            self.viewer.layers.events.inserted.disconnect(self._on_layers_changed)
        except Exception:
            pass
        try:
            self.viewer.layers.events.removed.disconnect(self._on_layers_changed)
        except Exception:
            pass
        # Remove self from owner's window dict (multi-window bookkeeping)
        if self._owner_dict is not None and self._owner_key is not None:
            try:
                if self._owner_dict.get(self._owner_key) is self:
                    self._owner_dict.pop(self._owner_key, None)
            except Exception:
                pass
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-fit current pixmap to new size
        self._render_current()

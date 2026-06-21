"""
导出控件 - v2 (FFmpeg 后端 + 目标时长 + 帧抽样 + 智能提示 + 导出后压缩)
"""
import math
import datetime
import json
from pathlib import Path

import numpy as np
import cv2
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSpinBox, QHBoxLayout,
    QComboBox, QCheckBox, QGroupBox, QFileDialog, QLineEdit,
    QProgressBar, QRadioButton, QButtonGroup, QScrollArea, QMessageBox,
    QDoubleSpinBox,
)
from qtpy.QtCore import Signal, QThread, QSettings, Qt

from widgets.settings_widget import tr
from utils.video_export import (
    export_video_smart, export_to_tiff_stack, export_to_gif,
    get_all_codec_options, is_ffmpeg_available, compress_video_ffmpeg,
)
from utils.utils import elide_text
from utils.ui_utils import setup_safe_scroll_all

MAX_PLAYER_FPS = 120
COMPRESS_THRESHOLD_MB = 100


# ───────────────────────────────────────────────────────────────────
# Threads
# ───────────────────────────────────────────────────────────────────

class ExportThread(QThread):
    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, image_stack, output_path, export_type, params):
        super().__init__()
        self.image_stack = image_stack
        self.output_path = output_path
        self.export_type = export_type
        self.params = params

    def run(self):
        try:
            success = False
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

            p = self.params.copy()
            p.pop('frame_range', None)

            if self.export_type == 'video':
                vp = {k: p[k] for k in (
                    'fps', 'codec', 'crf', 'preset', 'quality', 'backend',
                    'scale_bar_config', 'timestamp_config', 'frame_step',
                ) if k in p}
                success = export_video_smart(
                    self.image_stack, str(self.output_path),
                    callback=self._emit_progress, **vp,
                )

            elif self.export_type == 'tiff':
                success = export_to_tiff_stack(self.image_stack, str(self.output_path))
                self.progress.emit(100, 100)

            elif self.export_type == 'gif':
                gp = {k: p[k] for k in (
                    'fps', 'loop', 'colors',
                    'scale_bar_config', 'timestamp_config', 'frame_step',
                ) if k in p}
                success = export_to_gif(
                    self.image_stack, str(self.output_path),
                    callback=self._emit_progress, **gp,
                )

            elif self.export_type == 'image_sequence':
                success = self._export_image_sequence()

            if success:
                self.finished.emit(str(self.output_path))
            else:
                self.error.emit(tr("Export function returned False."))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def _emit_progress(self, current: int, total: int):
        self.progress.emit(current, total)

    def _export_image_sequence(self):
        output_path = Path(self.output_path)
        folder = output_path.parent / output_path.stem if output_path.suffix else output_path
        folder.mkdir(parents=True, exist_ok=True)

        fmt = self.params.get('format', 'png')
        pat = self.params.get('name_pattern', 'frame_{:04d}')
        total = len(self.image_stack)
        original_indices = self.params.get('original_indices', None)

        for i, frame in enumerate(self.image_stack):
            if frame.ndim == 3 and frame.shape[2] in [3, 4]:
                frame_out = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGBA2BGR if frame.shape[2] == 4 else cv2.COLOR_RGB2BGR,
                )
            else:
                frame_out = frame

            idx = original_indices[i] if (original_indices is not None and i < len(original_indices)) else i
            fname = folder / f"{pat.format(idx)}.{fmt}"
            cv2.imwrite(str(fname), frame_out)

            if i % 10 == 0:
                self.progress.emit(i + 1, total)

        self.progress.emit(total, total)
        return True


class CompressThread(QThread):
    finished = Signal(bool, float, float)  # success, orig_mb, compressed_mb

    def __init__(self, input_path, output_path, crf=28):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.crf = crf

    def run(self):
        ok, orig, comp = compress_video_ffmpeg(self.input_path, self.output_path, crf=self.crf)
        self.finished.emit(ok, orig, comp)


# ───────────────────────────────────────────────────────────────────
# Widget
# ───────────────────────────────────────────────────────────────────

class ExportWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.export_thread = None
        self.compress_thread = None
        self.output_path = None
        self.settings = QSettings("NapariUser", "ExportSettings")
        self.annotation_settings = QSettings("NapariUser", "AnnotationParams")

        self._setup_ui()
        self._restore_last_path()

        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

    # ── UI Setup ──────────────────────────────────────────────────

    def _setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content_widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel(f"<h3>💾 {tr('Export')}</h3>"))

        # 1. Source Layer
        h_lay = QHBoxLayout()
        h_lay.addWidget(QLabel(tr("Source:")))
        self.layer_combo = QComboBox()
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)
        h_lay.addWidget(self.layer_combo)
        layout.addLayout(h_lay)

        btn_refresh = QPushButton(f"🔄 {tr('Refresh Layers')}")
        btn_refresh.clicked.connect(self._refresh_layers)
        layout.addWidget(btn_refresh)

        # 2. Frame Range
        self.g_range = QGroupBox(tr("Frame Range"))
        l_range = QHBoxLayout()
        self.radio_all = QRadioButton(tr("All Frames"))
        self.radio_all.setChecked(True)
        self.radio_range = QRadioButton(tr("Range"))

        self.edit_frame_range = QLineEdit()
        self.edit_frame_range.setPlaceholderText(tr("e.g. 0-10, 15, 20-25"))
        self.edit_frame_range.setEnabled(False)
        self.edit_frame_range.setToolTip(
            f"{tr('Supported formats:')}\n- {tr('Range: 0-10')}\n- {tr('Single: 5')}\n- {tr('Mixed: 0-5, 8, 10-12')}"
        )
        self.radio_range.toggled.connect(self.edit_frame_range.setEnabled)
        self.radio_range.toggled.connect(lambda: self._update_fps_calculation())
        self.edit_frame_range.textChanged.connect(lambda: self._update_fps_calculation())

        l_range.addWidget(self.radio_all)
        l_range.addWidget(self.radio_range)
        l_range.addWidget(self.edit_frame_range)
        self.g_range.setLayout(l_range)
        layout.addWidget(self.g_range)

        # 3. Export Format
        g_type = QGroupBox(tr("Export Format"))
        l_type = QVBoxLayout()
        self.bg_type = QButtonGroup()

        self.radio_vid = QRadioButton(tr("Video (.mp4, .avi)"))
        self.radio_vid.setChecked(True)
        self.radio_gif = QRadioButton(tr("GIF Animation (.gif)"))
        self.radio_tiff = QRadioButton(tr("TIFF Stack (.tiff)"))
        self.radio_seq = QRadioButton(tr("Image Sequence (Folder)"))

        for r in (self.radio_vid, self.radio_gif, self.radio_tiff, self.radio_seq):
            self.bg_type.addButton(r)
            l_type.addWidget(r)
            r.toggled.connect(self._toggle_settings)

        g_type.setLayout(l_type)
        layout.addWidget(g_type)

        # Smart hint
        self.lbl_smart_hint = QLabel("")
        self.lbl_smart_hint.setStyleSheet(
            "color: #FFA726; font-size: 10px; font-style: italic; padding: 2px;"
        )
        self.lbl_smart_hint.setWordWrap(True)
        self.lbl_smart_hint.setVisible(False)
        layout.addWidget(self.lbl_smart_hint)

        # 4. Video / GIF shared: FPS & Duration & Sampling
        self.g_vid_set = QGroupBox(tr("Video Options"))
        l_vid = QVBoxLayout()

        # -- FPS mode radio --
        h_mode = QHBoxLayout()
        self.radio_manual_fps = QRadioButton(tr("Manual FPS"))
        self.radio_manual_fps.setChecked(True)
        self.radio_duration = QRadioButton(tr("Target Duration"))
        self.bg_fps_mode = QButtonGroup()
        self.bg_fps_mode.addButton(self.radio_manual_fps)
        self.bg_fps_mode.addButton(self.radio_duration)
        h_mode.addWidget(self.radio_manual_fps)
        h_mode.addWidget(self.radio_duration)
        l_vid.addLayout(h_mode)

        # -- Manual FPS row --
        self.w_fps_row = QWidget()
        h_fps = QHBoxLayout()
        h_fps.setContentsMargins(0, 0, 0, 0)
        h_fps.addWidget(QLabel(tr("FPS:")))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 99999)
        self.spin_fps.setValue(30)
        h_fps.addWidget(self.spin_fps)
        self.w_fps_row.setLayout(h_fps)
        l_vid.addWidget(self.w_fps_row)

        # -- Duration row --
        self.w_dur_row = QWidget()
        h_dur = QHBoxLayout()
        h_dur.setContentsMargins(0, 0, 0, 0)
        h_dur.addWidget(QLabel(tr("Duration (sec):")))
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.1, 9999.9)
        self.spin_duration.setSingleStep(0.5)
        self.spin_duration.setValue(15.0)
        self.spin_duration.setDecimals(1)
        h_dur.addWidget(self.spin_duration)
        self.w_dur_row.setLayout(h_dur)
        self.w_dur_row.setVisible(False)
        l_vid.addWidget(self.w_dur_row)

        # -- Calculated info --
        self.lbl_fps_info = QLabel("")
        self.lbl_fps_info.setStyleSheet("color: #81D4FA; font-size: 10px; padding-left: 4px;")
        self.lbl_fps_info.setWordWrap(True)
        l_vid.addWidget(self.lbl_fps_info)

        # -- High-FPS warning --
        self.lbl_fps_warn = QLabel("")
        self.lbl_fps_warn.setStyleSheet("color: #FF8A65; font-size: 10px; padding-left: 4px;")
        self.lbl_fps_warn.setWordWrap(True)
        self.lbl_fps_warn.setVisible(False)
        l_vid.addWidget(self.lbl_fps_warn)

        # -- Frame sampling --
        self.check_sampling = QCheckBox(tr("Frame Sampling"))
        self.check_sampling.setVisible(False)
        l_vid.addWidget(self.check_sampling)

        self.lbl_sampling_info = QLabel("")
        self.lbl_sampling_info.setStyleSheet("color: #A5D6A7; font-size: 10px; padding-left: 20px;")
        self.lbl_sampling_info.setWordWrap(True)
        self.lbl_sampling_info.setVisible(False)
        l_vid.addWidget(self.lbl_sampling_info)

        # -- Encoding section (video only, hidden for GIF) --
        self.w_encoding = QWidget()
        l_enc = QVBoxLayout()
        l_enc.setContentsMargins(0, 5, 0, 0)

        h_codec = QHBoxLayout()
        h_codec.addWidget(QLabel(tr("Encoder:")))
        self.combo_codec = QComboBox()
        self._populate_codecs()
        h_codec.addWidget(self.combo_codec)
        l_enc.addLayout(h_codec)

        # FFmpeg-specific options
        self.w_ffmpeg_opts = QWidget()
        l_ff = QVBoxLayout()
        l_ff.setContentsMargins(0, 0, 0, 0)

        h_crf = QHBoxLayout()
        h_crf.addWidget(QLabel(tr("Quality (CRF):")))
        self.spin_crf = QSpinBox()
        self.spin_crf.setRange(18, 35)
        self.spin_crf.setValue(23)
        self.spin_crf.setToolTip(tr("18=best quality  28=smaller file"))
        h_crf.addWidget(self.spin_crf)
        self.lbl_crf_hint = QLabel("18🔍←→📦28")
        self.lbl_crf_hint.setStyleSheet("color: #888; font-size: 9px;")
        h_crf.addWidget(self.lbl_crf_hint)
        l_ff.addLayout(h_crf)

        h_preset = QHBoxLayout()
        h_preset.addWidget(QLabel(tr("Encoding Speed:")))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(['ultrafast', 'fast', 'medium', 'slow', 'veryslow'])
        self.combo_preset.setCurrentText('medium')
        self.combo_preset.setToolTip(tr("Slower = better compression"))
        h_preset.addWidget(self.combo_preset)
        l_ff.addLayout(h_preset)

        self.w_ffmpeg_opts.setLayout(l_ff)
        l_enc.addWidget(self.w_ffmpeg_opts)

        self.lbl_backend = QLabel("")
        self.lbl_backend.setStyleSheet("color: #888; font-size: 9px; font-style: italic;")
        l_enc.addWidget(self.lbl_backend)

        self.w_encoding.setLayout(l_enc)
        l_vid.addWidget(self.w_encoding)

        self.g_vid_set.setLayout(l_vid)
        layout.addWidget(self.g_vid_set)

        # 5. Sequence Settings
        self.g_seq_set = QGroupBox(tr("Sequence Options"))
        l_seq = QVBoxLayout()

        self.check_auto_folder = QCheckBox(tr("Auto-create subfolder"))
        self.check_auto_folder.setChecked(True)
        self.check_auto_folder.setToolTip(
            tr("Automatically create a named subfolder (e.g. LayerName_20260129_143022)")
        )
        self.check_auto_folder.toggled.connect(self._update_folder_preview)
        l_seq.addWidget(self.check_auto_folder)

        self.lbl_folder_preview = QLabel("")
        self.lbl_folder_preview.setStyleSheet("color: #888; font-size: 9px; font-style: italic;")
        self.lbl_folder_preview.setWordWrap(True)
        l_seq.addWidget(self.lbl_folder_preview)

        h_fmt = QHBoxLayout()
        h_fmt.addWidget(QLabel(tr("Format:")))
        self.combo_img_fmt = QComboBox()
        self.combo_img_fmt.addItems(['png', 'jpg', 'bmp', 'tiff'])
        h_fmt.addWidget(self.combo_img_fmt)
        l_seq.addLayout(h_fmt)

        h_pat = QHBoxLayout()
        h_pat.addWidget(QLabel(tr("Pattern:")))
        self.edit_pattern = QLineEdit("frame_{:04d}")
        h_pat.addWidget(self.edit_pattern)
        l_seq.addLayout(h_pat)

        self.g_seq_set.setLayout(l_seq)
        self.g_seq_set.setVisible(False)
        layout.addWidget(self.g_seq_set)

        # 5.5. GIF Settings
        self.g_gif_set = QGroupBox(tr("GIF Options"))
        l_gif = QVBoxLayout()

        h_colors = QHBoxLayout()
        self.lbl_gif_colors = QLabel(tr("Colors (2-256):"))
        self.lbl_gif_colors.setToolTip(tr("Fewer colors = smaller file size, but lower quality"))
        h_colors.addWidget(self.lbl_gif_colors)
        self.spin_gif_colors = QSpinBox()
        self.spin_gif_colors.setRange(2, 256)
        self.spin_gif_colors.setValue(256)
        h_colors.addWidget(self.spin_gif_colors)
        l_gif.addLayout(h_colors)

        h_loop = QHBoxLayout()
        self.lbl_gif_loop = QLabel(tr("Loop (0=infinite):"))
        h_loop.addWidget(self.lbl_gif_loop)
        self.spin_gif_loop = QSpinBox()
        self.spin_gif_loop.setRange(0, 999)
        self.spin_gif_loop.setValue(0)
        h_loop.addWidget(self.spin_gif_loop)
        l_gif.addLayout(h_loop)

        self.g_gif_set.setLayout(l_gif)
        self.g_gif_set.setVisible(False)
        layout.addWidget(self.g_gif_set)

        # 6. Annotations
        self.g_anno = QGroupBox(tr("Overlay Annotations"))
        self.g_anno.setCheckable(True)
        self.g_anno.setChecked(False)
        l_anno = QVBoxLayout()

        self.check_sb = QCheckBox(tr("Scale Bar"))
        self.check_sb.setChecked(False)
        self.check_ts = QCheckBox(tr("Timestamp"))
        self.check_ts.setChecked(False)

        l_anno.addWidget(self.check_sb)
        l_anno.addWidget(self.check_ts)
        l_anno.addWidget(
            QLabel(f"<i style='color:gray'>{tr('Styles loaded from Annotation Tab')}</i>")
        )
        self.g_anno.setLayout(l_anno)
        layout.addWidget(self.g_anno)

        # 7. Output Path
        h_path = QHBoxLayout()
        self.lbl_path = QLabel(tr("No path selected"))
        self.lbl_path.setStyleSheet("font-size: 10px; color: gray;")
        self.lbl_path.setWordWrap(True)
        self.lbl_path.setMaximumWidth(200)

        btn_brow = QPushButton(f"📂 {tr('Browse')}")
        btn_brow.clicked.connect(self._browse)
        h_path.addWidget(self.lbl_path)
        h_path.addWidget(btn_brow)
        layout.addLayout(h_path)

        # 8. Progress
        self.pbar = QProgressBar()
        self.pbar.setVisible(False)
        self.pbar.setFormat("%p%")
        layout.addWidget(self.pbar)

        self.btn_run = QPushButton(f"🚀 {tr('Start Export')}")
        self.btn_run.clicked.connect(self._start_export)
        self.btn_run.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_run.setEnabled(False)
        layout.addWidget(self.btn_run)

        self.lbl_status = QLabel(tr("Ready"))
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        layout.addStretch()

        content_widget.setLayout(layout)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        self._refresh_layers()

        # Signal connections for FPS calculation
        self.radio_manual_fps.toggled.connect(self._on_fps_mode_changed)
        self.spin_fps.valueChanged.connect(self._update_fps_calculation)
        self.spin_duration.valueChanged.connect(self._update_fps_calculation)
        self.check_sampling.toggled.connect(self._update_fps_calculation)
        self.combo_codec.currentIndexChanged.connect(self._on_codec_changed)

        setup_safe_scroll_all(
            self.layer_combo,
            self.spin_fps, self.spin_duration, self.spin_crf,
            self.combo_codec, self.combo_preset, self.combo_img_fmt,
        )

        self._on_codec_changed()
        self._update_backend_label()

    # ── Codec population ──────────────────────────────────────────

    def _populate_codecs(self):
        self.combo_codec.clear()
        for cid, display, backend in get_all_codec_options():
            self.combo_codec.addItem(display, {'id': cid, 'backend': backend})
        if self.combo_codec.count() > 0:
            self.combo_codec.setCurrentIndex(0)

    def _on_codec_changed(self):
        data = self.combo_codec.currentData()
        is_ff = data and data.get('backend') == 'ffmpeg'
        self.w_ffmpeg_opts.setVisible(is_ff)

    def _update_backend_label(self):
        if is_ffmpeg_available():
            self.lbl_backend.setText(f"✓ {tr('FFmpeg available')}")
            self.lbl_backend.setStyleSheet("color: #4CAF50; font-size: 9px; font-style: italic;")
        else:
            self.lbl_backend.setText(f"⚠ {tr('FFmpeg not available, using OpenCV')}")
            self.lbl_backend.setStyleSheet("color: #FF8A65; font-size: 9px; font-style: italic;")

    # ── Toggle visibility ─────────────────────────────────────────

    def _toggle_settings(self):
        is_video = self.radio_vid.isChecked()
        is_gif = self.radio_gif.isChecked()
        is_seq = self.radio_seq.isChecked()

        self.g_vid_set.setVisible(is_video or is_gif)
        self.w_encoding.setVisible(is_video)
        self.g_gif_set.setVisible(is_gif)
        self.g_seq_set.setVisible(is_seq)

        if is_seq:
            self._update_folder_preview()

        self._update_fps_calculation()
        self._update_smart_hint()

    # ── FPS mode toggle ───────────────────────────────────────────

    def _on_fps_mode_changed(self):
        manual = self.radio_manual_fps.isChecked()
        self.w_fps_row.setVisible(manual)
        self.w_dur_row.setVisible(not manual)
        self._update_fps_calculation()

    # ── FPS / Duration calculation ────────────────────────────────

    def _get_total_frames(self):
        name = self.layer_combo.currentData()
        if not name or name not in self.viewer.layers:
            return 0
        return self.viewer.layers[name].data.shape[0]

    def _get_export_frame_count(self):
        total = self._get_total_frames()
        if total <= 0:
            return 0
        if self.radio_range.isChecked():
            text = self.edit_frame_range.text()
            indices = self._parse_frame_indices(text, total)
            if indices:
                return len(indices)
        return total

    def _update_fps_calculation(self):
        n = self._get_export_frame_count()
        if n <= 0:
            self.lbl_fps_info.setText("")
            self.lbl_fps_warn.setVisible(False)
            self.check_sampling.setVisible(False)
            self.lbl_sampling_info.setVisible(False)
            return

        if self.radio_duration.isChecked():
            dur = self.spin_duration.value()
            raw_fps = n / dur if dur > 0 else 30

            if raw_fps > MAX_PLAYER_FPS:
                self.check_sampling.setVisible(True)
                self.lbl_fps_warn.setText(
                    f"⚠ {tr('FPS too high for most players')} ({int(raw_fps)} fps > {MAX_PLAYER_FPS})"
                )
                self.lbl_fps_warn.setVisible(True)

                if self.check_sampling.isChecked():
                    step = math.ceil(raw_fps / MAX_PLAYER_FPS)
                    eff = math.ceil(n / step)
                    actual_fps = max(1, round(eff / dur))
                    self.lbl_fps_info.setText(
                        f"→ {tr('Sampling: export %d of %d frames @ %dfps') % (eff, n, actual_fps)}"
                    )
                    self.lbl_sampling_info.setText(
                        f"  {tr('Every %d frames, duration %.1fs') % (step, eff / actual_fps)}"
                    )
                    self.lbl_sampling_info.setVisible(True)
                else:
                    self.lbl_fps_info.setText(f"→ {n} {tr('frames')} @ {int(raw_fps)} fps")
                    self.lbl_sampling_info.setVisible(False)
            else:
                self.check_sampling.setVisible(False)
                self.lbl_sampling_info.setVisible(False)
                self.lbl_fps_warn.setVisible(False)
                self.lbl_fps_info.setText(f"→ {n} {tr('frames')} @ {int(raw_fps)} fps")
        else:
            fps = self.spin_fps.value()
            dur = n / fps if fps > 0 else 0
            self.lbl_fps_info.setText(f"→ {tr('Playback duration')}: {dur:.1f}s")
            if fps > MAX_PLAYER_FPS:
                self.lbl_fps_warn.setText(
                    f"⚠ {tr('FPS too high for most players')} ({fps} > {MAX_PLAYER_FPS})"
                )
                self.lbl_fps_warn.setVisible(True)
            else:
                self.lbl_fps_warn.setVisible(False)
            self.check_sampling.setVisible(False)
            self.lbl_sampling_info.setVisible(False)

    # ── Smart hint ────────────────────────────────────────────────

    def _update_smart_hint(self):
        n = self._get_total_frames()
        name = self.layer_combo.currentData()
        if n <= 0 or not name or name not in self.viewer.layers:
            self.lbl_smart_hint.setVisible(False)
            return

        data = self.viewer.layers[name].data
        H, W = data.shape[1], data.shape[2]

        if n > 500 or (H * W) > 1920 * 1080:
            if n > 2000:
                hint = tr("Recommended: H.264 + Target Duration") + f" ({n} {tr('frames')})"
            else:
                hint = tr("Recommended: Video (H.264)")
        elif n <= 150 and max(H, W) <= 800:
            hint = tr("Small data, GIF or Video both work")
        else:
            hint = tr("Recommended: Video (H.264)")

        self.lbl_smart_hint.setText(f"💡 {hint}")
        self.lbl_smart_hint.setVisible(True)

    # ── Layer management ──────────────────────────────────────────

    def _refresh_layers(self, event=None):
        self._restore_last_path()
        curr_data = self.layer_combo.currentData()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for l in self.viewer.layers:
            if hasattr(l, 'data') and isinstance(l.data, np.ndarray) and l.data.ndim in [3, 4]:
                short = elide_text(l.name, 25)
                self.layer_combo.addItem(short, l.name)
                self.layer_combo.setItemData(self.layer_combo.count() - 1, l.name, Qt.ToolTipRole)

        index_set = False
        active = self.viewer.layers.selection.active
        if active:
            idx = self.layer_combo.findData(active.name)
            if idx >= 0:
                self.layer_combo.setCurrentIndex(idx)
                index_set = True
        if not index_set and curr_data:
            idx = self.layer_combo.findData(curr_data)
            if idx >= 0:
                self.layer_combo.setCurrentIndex(idx)
        self.layer_combo.blockSignals(False)
        self._on_layer_changed(self.layer_combo.currentData())

        s = self.annotation_settings
        def str2bool(v):
            return str(v).lower() == 'true'
        self.check_sb.setChecked(str2bool(s.value("scale/enable", "false")))
        self.check_ts.setChecked(str2bool(s.value("label/enable", "false")))

    def _on_active_layer_changed(self, event=None):
        active = self.viewer.layers.selection.active
        if active:
            idx = self.layer_combo.findData(active.name)
            if idx >= 0:
                self.layer_combo.setCurrentIndex(idx)

    def _on_layer_changed(self, txt=None):
        real_name = self.layer_combo.currentData()
        if not real_name or real_name not in self.viewer.layers:
            return
        layer = self.viewer.layers[real_name]
        if not hasattr(layer, 'data'):
            return

        num_frames = layer.data.shape[0]
        valid_range_str = f"0-{num_frames - 1}"
        self.edit_frame_range.setPlaceholderText(f"All (Default) or e.g. {valid_range_str}")
        self.edit_frame_range.setToolTip(
            f"Valid frames: 0 to {num_frames - 1}\nSupported syntax:\n- Range: 0-10\n- Single: 5\n- Mixed: 0-5, 8, 10-12"
        )

        is_rgb = (layer.data.ndim == 4) or (hasattr(layer, 'rgb') and layer.rgb)
        if is_rgb:
            self.g_anno.setChecked(False)
            self.g_anno.setTitle("Overlay (Disabled for RGB)")
            self.g_anno.setEnabled(False)
        else:
            self.g_anno.setEnabled(True)
            self.g_anno.setTitle(tr("Overlay Annotations"))

        self._update_fps_calculation()
        self._update_smart_hint()

    # ── Browse ────────────────────────────────────────────────────

    def _browse(self):
        d = self.settings.value("last_dir", str(Path.home()))

        if self.radio_vid.isChecked():
            f, _ = QFileDialog.getSaveFileName(self, "Save Video", d, "Video (*.mp4 *.avi)")
        elif self.radio_gif.isChecked():
            f, _ = QFileDialog.getSaveFileName(self, "Save GIF", d, "GIF (*.gif)")
        elif self.radio_tiff.isChecked():
            f, _ = QFileDialog.getSaveFileName(self, "Save TIFF", d, "TIFF (*.tiff)")
        else:
            f = QFileDialog.getExistingDirectory(self, tr("Select Parent Folder for Sequence"), d)

        if f:
            self.output_path = f
            self.lbl_path.setText(f)
            self.lbl_path.setStyleSheet("color: #E0E0E0; font-size: 10px;")

            p = Path(f)
            save_dir = str(p.parent) if p.suffix else str(p)
            self.settings.setValue("last_dir", save_dir)
            self.btn_run.setEnabled(True)

            if self.radio_seq.isChecked():
                self._update_folder_preview()

    def _restore_last_path(self):
        archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
        if archive_path and Path(archive_path).exists():
            self.output_path = archive_path
            self.lbl_path.setText(f"📂 Auto-Archive: {Path(archive_path).name}")
            self.lbl_path.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11px;")
            self.btn_run.setEnabled(True)
        else:
            last = self.settings.value("last_output", "")
            if last and Path(last).exists():
                self.output_path = last
                self.lbl_path.setText(last)
                self.btn_run.setEnabled(True)
            else:
                self.lbl_path.setText(tr("No path selected"))
                self.btn_run.setEnabled(False)

    # ── Export params ─────────────────────────────────────────────

    def _compute_fps_and_step(self):
        """Return (fps, frame_step) based on current settings."""
        n = self._get_export_frame_count()
        frame_step = 1

        if self.radio_duration.isChecked() and n > 0:
            dur = self.spin_duration.value()
            raw_fps = n / dur if dur > 0 else 30

            if raw_fps > MAX_PLAYER_FPS and self.check_sampling.isChecked():
                frame_step = math.ceil(raw_fps / MAX_PLAYER_FPS)
                eff = math.ceil(n / frame_step)
                fps = max(1, round(eff / dur))
            else:
                fps = max(1, round(raw_fps))
        else:
            fps = self.spin_fps.value()

        return fps, frame_step

    def _get_export_params(self):
        params = {}
        fps, frame_step = self._compute_fps_and_step()

        if self.radio_vid.isChecked():
            params['fps'] = fps
            params['frame_step'] = frame_step

            codec_data = self.combo_codec.currentData()
            if codec_data:
                params['codec'] = codec_data['id']
                params['backend'] = codec_data['backend']
            else:
                params['codec'] = 'mp4v'
                params['backend'] = 'opencv'

            if params.get('backend') == 'ffmpeg':
                params['crf'] = self.spin_crf.value()
                params['preset'] = self.combo_preset.currentText()
            else:
                params['quality'] = 100

            if self.g_anno.isChecked():
                s = self.annotation_settings
                if self.check_sb.isChecked():
                    try:
                        params['scale_bar_config'] = {
                            'enable': True,
                            'ratio': float(s.value("scale/ratio", 1.0)),
                            'unit': s.value("scale/unit", "nm"),
                            'length': float(s.value("scale/length", 100.0)),
                            'height': int(s.value("scale/height", 80)),
                            'thickness': int(s.value("scale/thickness", 8)),
                            'font_size': int(s.value("scale/font_size", 36)),
                            'padding': int(s.value("scale/padding", 10)),
                            'color': s.value("scale/color", (1, 1, 1, 1)),
                            'bg_color': s.value("scale/bg_color", (0, 0, 0, 1)),
                            'bg_alpha': int(s.value("scale/bg_alpha", 100)),
                            'use_bg': s.value("scale/use_bg", "true") == "true",
                            'position': s.value("scale/position", (50, 50)),
                        }
                    except Exception:
                        print("Warning: Could not load scale bar settings.")

                if self.check_ts.isChecked():
                    try:
                        ts_cfg = {
                            'enable': True,
                            'format': s.value("label/format", "0.00"),
                            'custom_fmt': s.value("label/custom_fmt", ""),
                            'font_size': int(s.value("label/font_size", 32)),
                            'color': s.value("label/color", (1, 1, 1, 1)),
                            'position': s.value("label/position", (10, 40)),
                            'start': float(s.value("label/start", 0.0)),
                            'interval': float(s.value("label/interval", 1.0)),
                        }
                        # interval 不需要调整: 导出函数使用原始帧索引(0, 3, 6...)
                        # 乘以原始 interval 即可得到正确的物理时间
                        params['timestamp_config'] = ts_cfg
                    except Exception:
                        print("Warning: Could not load timestamp settings.")

        elif self.radio_gif.isChecked():
            params['fps'] = fps
            params['frame_step'] = frame_step
            params['colors'] = self.spin_gif_colors.value()
            params['loop'] = self.spin_gif_loop.value()

        elif self.radio_seq.isChecked():
            params['format'] = self.combo_img_fmt.currentText()
            params['name_pattern'] = self.edit_pattern.text()

        return params

    # ── Start export ──────────────────────────────────────────────

    def _start_export(self):
        if not self.layer_combo.currentData() or not self.output_path:
            self.lbl_status.setText(f"❌ {tr('Check inputs')}")
            return

        layer_name = self.layer_combo.currentData()

        # Annotation warning
        is_burned = "burned" in layer_name.lower() or "annotated" in layer_name.lower()
        if self.radio_vid.isChecked() and not is_burned:
            has_sb = self.check_sb.isChecked()
            has_ts = self.check_ts.isChecked()
            anno_enabled = self.g_anno.isChecked()
            if not anno_enabled or (not has_sb and not has_ts):
                reply = QMessageBox.question(
                    self, tr("Missing Annotations"),
                    tr("You are exporting a video WITHOUT Scale Bar or Timestamp.\n\nAre you sure?"),
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    return

        self.btn_run.setEnabled(False)
        self.pbar.setValue(0)
        self.pbar.setVisible(True)
        self.lbl_status.setText(f"⏳ {tr('Exporting...')}")

        # Resolve output path
        final_path = Path(self.output_path)
        archive_path = QSettings("NapariUser", "Global").value("archive_path", "")

        if archive_path and final_path == Path(archive_path):
            ts = datetime.datetime.now().strftime("%H%M%S")
            safe_layer = "".join(c if c.isalnum() or c in "-_" else "_" for c in layer_name)
            fname = f"{safe_layer}_{ts}"

            if self.radio_vid.isChecked():
                final_path = final_path / "Exported_Videos" / f"{fname}.mp4"
            elif self.radio_gif.isChecked():
                final_path = final_path / "Exported_GIFs" / f"{fname}.gif"
            elif self.radio_tiff.isChecked():
                final_path = final_path / "Exported_Stacks" / f"{fname}.tiff"
            else:
                final_path = final_path / "Exported_Sequences" / fname
            final_path.parent.mkdir(parents=True, exist_ok=True)

        elif self.radio_seq.isChecked() and self.check_auto_folder.isChecked():
            final_path = final_path / self._generate_auto_folder_name()

        # Slice data
        layer = self.viewer.layers[layer_name]
        data = layer.data
        total_frames = data.shape[0]

        selected_indices = None
        if self.radio_range.isChecked():
            text = self.edit_frame_range.text()
            indices = self._parse_frame_indices(text, total_frames)
            if not indices:
                self.lbl_status.setText(f"❌ {tr('Invalid frame range syntax')}")
                self.btn_run.setEnabled(True)
                return
            data_slice = data[indices]
            selected_indices = text
        else:
            data_slice = data
            selected_indices = "All"

        etype = (
            'video' if self.radio_vid.isChecked()
            else 'gif' if self.radio_gif.isChecked()
            else 'tiff' if self.radio_tiff.isChecked()
            else 'image_sequence'
        )

        params = self._get_export_params()
        params['frame_range'] = selected_indices

        original_indices = layer.metadata.get('original_indices', None)
        if original_indices is not None:
            params['original_indices'] = original_indices

        self.export_thread = ExportThread(data_slice, str(final_path), etype, params)
        self.export_thread.progress.connect(lambda c, t: self.pbar.setValue(int(c / t * 100)))
        self.export_thread.finished.connect(lambda p: self._on_done(p, params))
        self.export_thread.error.connect(self._on_error)
        self.export_thread.start()

    # ── Export callbacks ──────────────────────────────────────────

    def _on_done(self, path, params):
        self.lbl_status.setText(f"✅ {tr('Done: %s') % Path(path).name}")
        self.pbar.setVisible(False)
        self.btn_run.setEnabled(True)
        self._log_export(path, params)

        # Offer compression for large video files
        if self.radio_vid.isChecked() and is_ffmpeg_available():
            try:
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                if size_mb > COMPRESS_THRESHOLD_MB:
                    self._offer_compression(path, size_mb)
            except Exception:
                pass

    def _on_error(self, err):
        self.lbl_status.setText(f"❌ {tr('Error:')} {err}")
        self.pbar.setVisible(False)
        self.btn_run.setEnabled(True)

    def _log_export(self, path, params):
        try:
            from utils.session_logger import get_logger
            get_logger().log_action("export", "export_media", {
                "file": str(Path(path).name),
                "type": self.export_thread.export_type,
                "params": {k: v for k, v in params.items() if k not in ('scale_bar_config', 'timestamp_config')},
            })
        except Exception as e:
            print(f"Log failed: {e}")

    # ── Post-export compression ───────────────────────────────────

    def _offer_compression(self, path, size_mb):
        reply = QMessageBox.question(
            self,
            tr("Large File Detected"),
            tr("Exported file is %.1f MB.\n\nCompress with FFmpeg H.264?\n(Typically 50-80%% smaller)") % size_mb,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        self.btn_run.setEnabled(False)
        self.pbar.setValue(0)
        self.pbar.setVisible(True)
        self.pbar.setMaximum(0)  # indeterminate
        self.lbl_status.setText(f"⏳ {tr('Compressing...')}")

        p = Path(path)
        temp_out = str(p.parent / f"{p.stem}_compressed{p.suffix}")

        self.compress_thread = CompressThread(path, temp_out, crf=28)
        self._compress_original_path = path
        self._compress_temp_path = temp_out
        self.compress_thread.finished.connect(self._on_compress_done)
        self.compress_thread.start()

    def _on_compress_done(self, success, orig_mb, comp_mb):
        self.pbar.setMaximum(100)
        self.pbar.setVisible(False)
        self.btn_run.setEnabled(True)

        if not success:
            self.lbl_status.setText(f"❌ {tr('Compression failed')}")
            return

        ratio = (1 - comp_mb / orig_mb) * 100 if orig_mb > 0 else 0

        reply = QMessageBox.question(
            self,
            tr("Compression Complete"),
            tr("Compressed: %.1f MB → %.1f MB (%.0f%% saved)\n\nReplace original file?")
            % (orig_mb, comp_mb, ratio),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        orig = Path(self._compress_original_path)
        temp = Path(self._compress_temp_path)

        if reply == QMessageBox.Yes:
            try:
                orig.unlink()
                temp.rename(orig)
                self.lbl_status.setText(
                    f"✅ {tr('Compressed: %.1f MB → %.1f MB') % (orig_mb, comp_mb)}"
                )
            except Exception as e:
                self.lbl_status.setText(f"❌ {tr('Replace failed:')} {e}")
        else:
            self.lbl_status.setText(
                f"✅ {tr('Compressed saved as:')} {temp.name} ({comp_mb:.1f} MB)"
            )

    # ── Helpers ───────────────────────────────────────────────────

    def _parse_frame_indices(self, text, total_frames):
        indices = set()
        try:
            parts = [p.strip() for p in text.split(',')]
            for p in parts:
                if not p:
                    continue
                if '-' in p:
                    start, end = map(int, p.split('-'))
                    start = max(0, start)
                    end = min(total_frames - 1, end)
                    if start <= end:
                        indices.update(range(start, end + 1))
                else:
                    idx = int(p)
                    if 0 <= idx < total_frames:
                        indices.add(idx)
            return sorted(list(indices))
        except ValueError:
            return []

    def _generate_auto_folder_name(self):
        layer_name = self.layer_combo.currentData() or "Sequence"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in layer_name)[:30]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe}_{ts}"

    def _update_folder_preview(self):
        if not self.radio_seq.isChecked() or not self.check_auto_folder.isChecked():
            self.lbl_folder_preview.setText("")
            return
        if not self.output_path:
            self.lbl_folder_preview.setText(tr("Select a folder first"))
            return
        name = self._generate_auto_folder_name()
        self.lbl_folder_preview.setText(f"📂 {tr('Will create')}: {name}/")

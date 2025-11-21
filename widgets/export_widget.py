"""
导出控件
修复：双重弹窗Bug，路径记忆功能增强（启动自动填入上次路径）
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QSpinBox, QHBoxLayout, QComboBox,
                            QCheckBox, QGroupBox, QFileDialog, QLineEdit,
                            QProgressBar, QRadioButton, QButtonGroup, QScrollArea)
from qtpy.QtCore import Signal, QThread, QSettings
import numpy as np
from pathlib import Path
import os
from utils.video_export import (export_to_video, export_to_tiff_stack, 
                                get_available_codecs)

class ExportThread(QThread):
    """导出线程"""
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
            if self.export_type == 'video':
                success = export_to_video(
                    self.image_stack,
                    self.output_path,
                    **self.params
                )
            elif self.export_type == 'tiff':
                success = export_to_tiff_stack(
                    self.image_stack,
                    self.output_path
                )
            elif self.export_type == 'image_sequence':
                success = self._export_image_sequence()
            
            if success:
                self.finished.emit(self.output_path)
            else:
                self.error.emit("Export failed")
        except Exception as e:
            self.error.emit(str(e))

    def _export_image_sequence(self):
        import cv2
        output_path = Path(self.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        format_ext = self.params.get('format', 'png')
        name_pattern = self.params.get('name_pattern', 'frame_{:04d}')
        
        total = len(self.image_stack)
        for i, frame in enumerate(self.image_stack):
            filename = output_path / f"{name_pattern.format(i)}.{format_ext}"
            
            if frame.ndim == 3 and frame.shape[2] in [3, 4]:
                if frame.shape[2] == 4:
                    frame_out = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    frame_out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_out = frame
                
            cv2.imwrite(str(filename), frame_out)
            self.progress.emit(i + 1, total)
        return True

class ExportWidget(QWidget):
    """导出控件"""
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.export_thread = None
        self.output_path = None
        
        self.settings = QSettings("NapariUser", "ExportSettings")
        self.annotation_settings = QSettings("NapariUser", "AnnotationParams")
        
        self._setup_ui()
        
        # === 新增：启动时恢复上次的文件路径 ===
        self._restore_last_path()
        
        self.viewer.layers.events.inserted.connect(self._refresh_layers_silently)
        self.viewer.layers.events.removed.connect(self._refresh_layers_silently)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

    def _setup_ui(self):
        # 1. 创建最外层布局 (用于放滚动条)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 2. 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True) # 关键：让内容自适应宽度
        # 去掉滚动区域的边框，使其看起来像原生界面
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        # 3. 创建内容容器 (原本的控件都加到这里面)
        content_widget = QWidget()
        layout = QVBoxLayout()
        title = QLabel("<h3>💾 Export Data</h3>")
        layout.addWidget(title)

        # 图层选择
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Source Layer:"))
        self.layer_combo = QComboBox()
        self.layer_combo.currentTextChanged.connect(self._on_combo_changed)
        layer_layout.addWidget(self.layer_combo)
        layout.addLayout(layer_layout)

        refresh_btn = QPushButton("🔄 Refresh Layers")
        refresh_btn.clicked.connect(self._refresh_layers)
        layout.addWidget(refresh_btn)

        # Type
        type_group = QGroupBox("Export Type")
        type_layout = QVBoxLayout()
        self.export_type_group = QButtonGroup()
        
        self.video_radio = QRadioButton("Video File (.mp4, .avi)")
        self.video_radio.setChecked(True)
        self.video_radio.toggled.connect(self._on_type_changed)
        self.export_type_group.addButton(self.video_radio, 0)
        type_layout.addWidget(self.video_radio)
        
        self.tiff_radio = QRadioButton("TIFF Stack (.tiff)")
        self.tiff_radio.toggled.connect(self._on_type_changed)
        self.export_type_group.addButton(self.tiff_radio, 1)
        type_layout.addWidget(self.tiff_radio)
        
        self.sequence_radio = QRadioButton("Image Sequence (folder)")
        self.sequence_radio.toggled.connect(self._on_type_changed)
        self.export_type_group.addButton(self.sequence_radio, 2)
        type_layout.addWidget(self.sequence_radio)
        
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Video Settings
        self.video_settings_group = QGroupBox("Video Settings")
        video_layout = QVBoxLayout()
        
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120); self.fps_spin.setValue(30)
        fps_layout.addWidget(self.fps_spin)
        video_layout.addLayout(fps_layout)
        
        codec_layout = QHBoxLayout()
        codec_layout.addWidget(QLabel("Codec:"))
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(get_available_codecs())
        codec_layout.addWidget(self.codec_combo)
        video_layout.addLayout(codec_layout)
        
        q_layout = QHBoxLayout()
        q_layout.addWidget(QLabel("Quality:"))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(0, 100); self.quality_spin.setValue(95)
        q_layout.addWidget(self.quality_spin)
        video_layout.addLayout(q_layout)
        
        self.video_settings_group.setLayout(video_layout)
        layout.addWidget(self.video_settings_group)

        # Sequence Settings
        self.sequence_settings_group = QGroupBox("Sequence Settings")
        seq_layout = QVBoxLayout()
        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(['png', 'jpg', 'bmp', 'tiff'])
        seq_layout.addWidget(QLabel("Format:"))
        seq_layout.addWidget(self.image_format_combo)
        self.name_pattern_edit = QLineEdit("frame_{:04d}")
        seq_layout.addWidget(QLabel("Pattern:"))
        seq_layout.addWidget(self.name_pattern_edit)
        self.sequence_settings_group.setLayout(seq_layout)
        self.sequence_settings_group.setVisible(False)
        layout.addWidget(self.sequence_settings_group)

        # Output Path
        path_layout = QHBoxLayout()
        self.path_label = QLabel("Output: Not selected")
        self.path_label.setStyleSheet("color: gray; font-size: 10px;")
        self.path_label.setWordWrap(True)
        browse_btn = QPushButton("📂 Browse...")
        browse_btn.clicked.connect(self._browse_output)
        path_layout.addWidget(self.path_label)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # Range
        range_group = QGroupBox("Frame Range")
        r_layout = QHBoxLayout()
        self.export_all_radio = QRadioButton("All")
        self.export_all_radio.setChecked(True)
        self.export_range_radio = QRadioButton("Range")
        r_layout.addWidget(self.export_all_radio)
        r_layout.addWidget(self.export_range_radio)
        self.start_frame_spin = QSpinBox(); self.start_frame_spin.setEnabled(False)
        self.end_frame_spin = QSpinBox(); self.end_frame_spin.setEnabled(False)
        r_layout.addWidget(self.start_frame_spin)
        r_layout.addWidget(QLabel("-"))
        r_layout.addWidget(self.end_frame_spin)
        self.export_range_radio.toggled.connect(
            lambda c: (self.start_frame_spin.setEnabled(c), self.end_frame_spin.setEnabled(c))
        )
        range_group.setLayout(r_layout)
        layout.addWidget(range_group)

        # Annotations
        self.annotation_group = QGroupBox("Overlay Annotations (Draw on Export)")
        self.annotation_group.setCheckable(True)
        self.annotation_group.setChecked(True)
        a_layout = QVBoxLayout()
        self.include_scale_bar_check = QCheckBox("Scale Bar")
        self.include_timestamp_check = QCheckBox("Timestamp")
        a_layout.addWidget(self.include_scale_bar_check)
        a_layout.addWidget(self.include_timestamp_check)
        self.annotation_info = QLabel("ℹ️ Styles loaded from Annotation tab")
        self.annotation_info.setStyleSheet("color: gray; font-style: italic;")
        a_layout.addWidget(self.annotation_info)
        self.annotation_group.setLayout(a_layout)
        layout.addWidget(self.annotation_group)

        # Progress & Action
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.export_btn = QPushButton("🚀 Start Export")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._start_export)
        self.export_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.export_btn)
        
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch() # 建议在最后加一个弹簧，防止内容分散
        content_widget.setLayout(layout)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
        self._refresh_layers()

    def _refresh_layers_silently(self, event=None):
        current_text = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and isinstance(layer.data, np.ndarray):
                if layer.data.ndim == 3:
                    self.layer_combo.addItem(layer.name)
                elif layer.data.ndim == 4 and layer.data.shape[-1] in [3, 4]:
                    self.layer_combo.addItem(layer.name)
        
        idx = self.layer_combo.findText(current_text)
        if idx >= 0: self.layer_combo.setCurrentIndex(idx)
        self.layer_combo.blockSignals(False)

    def _refresh_layers(self):
        self._refresh_layers_silently()
        self._on_active_layer_changed()

    def _on_active_layer_changed(self, event=None):
        active_layer = self.viewer.layers.selection.active
        if active_layer:
            idx = self.layer_combo.findText(active_layer.name)
            if idx >= 0:
                self.layer_combo.blockSignals(True)
                self.layer_combo.setCurrentIndex(idx)
                self.layer_combo.blockSignals(False)
                self._check_layer_type(active_layer)
                self._update_frame_range()

    def _on_combo_changed(self, text):
        if not text: return
        if text in self.viewer.layers:
            layer = self.viewer.layers[text]
            self._check_layer_type(layer)
            self._update_frame_range()

    def _check_layer_type(self, layer):
        is_rgb = False
        if layer.data.ndim == 4 and layer.data.shape[-1] in [3, 4]:
            is_rgb = True
        if hasattr(layer, 'rgb') and layer.rgb:
            is_rgb = True
        
        if is_rgb:
            self.annotation_group.setChecked(False)
            self.annotation_group.setTitle("Overlay (Disabled: Layer is already RGB/Burned)")
            self.annotation_group.setEnabled(False)
            self.status_label.setText("ℹ️ Selected layer is already burned (RGB). Exporting directly.")
        else:
            self.annotation_group.setEnabled(True)
            self.annotation_group.setTitle("Overlay Annotations (Draw on Export)")
            self.status_label.setText("Ready")

    def _update_frame_range(self):
        text = self.layer_combo.currentText()
        if not text or text not in self.viewer.layers: return
        layer = self.viewer.layers[text]
        n = len(layer.data)
        self.start_frame_spin.setMaximum(n-1)
        self.end_frame_spin.setMaximum(n-1)
        self.end_frame_spin.setValue(n-1)

    def _on_type_changed(self):
        """切换导出类型时，仅更新UI可见性，不再自动弹窗"""
        # 修复：移除了 self._browse_output()
        if self.video_radio.isChecked():
            self.video_settings_group.setVisible(True)
            self.sequence_settings_group.setVisible(False)
        elif self.tiff_radio.isChecked():
            self.video_settings_group.setVisible(False)
            self.sequence_settings_group.setVisible(False)
        else:
            self.video_settings_group.setVisible(False)
            self.sequence_settings_group.setVisible(True)

    def _restore_last_path(self):
        """启动时恢复上次的导出路径"""
        last_output = self.settings.value("last_output", "")
        # 检查路径的父目录是否存在，防止路径失效
        if last_output and (os.path.isdir(last_output) or os.path.isdir(os.path.dirname(last_output))):
            self.output_path = last_output
            self.path_label.setText(last_output)
            self.export_btn.setEnabled(True)

    def _browse_output(self):
        last_dir = self.settings.value("last_dir", str(Path.home()))
        if self.video_radio.isChecked():
            path, _ = QFileDialog.getSaveFileName(self, "Save Video", last_dir, "Video (*.mp4 *.avi)")
        elif self.tiff_radio.isChecked():
            path, _ = QFileDialog.getSaveFileName(self, "Save TIFF", last_dir, "TIFF (*.tiff)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Folder", last_dir)
            
        if path:
            self.output_path = path
            self.path_label.setText(path)
            self.export_btn.setEnabled(True)
            
            # 记忆逻辑
            self.settings.setValue("last_output", path) # 记忆完整文件/目录路径
            
            p = Path(path)
            save_dir = str(p.parent) if p.suffix else str(p)
            self.settings.setValue("last_dir", save_dir) # 记忆打开目录

    def _get_export_params(self):
        params = {}
        if self.video_radio.isChecked():
            params['fps'] = self.fps_spin.value()
            params['codec'] = self.codec_combo.currentText()
            params['quality'] = self.quality_spin.value()
            
            if self.annotation_group.isEnabled() and self.annotation_group.isChecked():
                s = self.annotation_settings
                if self.include_scale_bar_check.isChecked():
                    params['scale_bar_config'] = {
                        'enable': True,
                        'ratio': float(s.value("scale/ratio", 1.0)),
                        'unit': s.value("scale/unit", "nm"),
                        'length': float(s.value("scale/length", 100.0)),
                        'height': int(s.value("scale/height", 80)),
                        'thickness': int(s.value("scale/thickness", 8)),
                        'font_size': int(s.value("scale/font_size", 36)),
                        'padding': int(s.value("scale/padding", 10)),
                        'color': s.value("scale/color", (1,1,1,1)),
                        'bg_color': s.value("scale/bg_color", (0,0,0,1)),
                        'bg_alpha': int(s.value("scale/bg_alpha", 100)),
                        'use_bg': s.value("scale/use_bg", "true") == "true",
                        'position': s.value("scale/position", (50, 50))
                    }
                if self.include_timestamp_check.isChecked():
                    params['timestamp_config'] = {
                        'enable': True,
                        'format': s.value("label/format", "0.00"),
                        'custom_fmt': s.value("label/custom_fmt", ""),
                        'font_size': int(s.value("label/font_size", 32)),
                        'color': s.value("label/color", (1,1,1,1)),
                        'position': s.value("label/position", (10, 40)),
                        'start': float(s.value("label/start", 0.0)),
                        'interval': float(s.value("label/interval", 1.0))
                    }
        elif self.sequence_radio.isChecked():
            params['format'] = self.image_format_combo.currentText()
            params['name_pattern'] = self.name_pattern_edit.text()
        return params

    def _start_export(self):
        layer_name = self.layer_combo.currentText()
        if not layer_name or not self.output_path: 
            self.status_label.setText("❌ Please select layer and output path")
            return
        
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("⏳ Exporting...")
        
        layer = self.viewer.layers[layer_name]
        data = layer.data
        
        if self.export_range_radio.isChecked():
            s = self.start_frame_spin.value()
            e = self.end_frame_spin.value()
            data = data[s:e+1]
            
        if self.video_radio.isChecked():
            export_type = 'video'
        elif self.tiff_radio.isChecked():
            export_type = 'tiff'
        else:
            export_type = 'image_sequence'
            
        params = self._get_export_params()
            
        self.export_thread = ExportThread(data, self.output_path, export_type, params)
        self.export_thread.progress.connect(lambda c, t: (self.progress_bar.setMaximum(t), self.progress_bar.setValue(c)))
        self.export_thread.finished.connect(self._on_finished)
        self.export_thread.error.connect(self._on_error)
        self.export_thread.start()

    def _on_finished(self, path):
        self.status_label.setText(f"✅ Exported successfully to:\n{path}")
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)

    def _on_error(self, msg):
        self.status_label.setText(f"❌ Export Error: {msg}")
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)
"""
导出控件 - 完整修复版
包含：
1. 视频/序列/TIFF 导出
2. 帧范围选择 (Frame Range) - [修复]
3. 视频参数设置 (FPS, Codec, Quality) - [确认]
4. 导出前安全检查 (Missing Annotations)
5. 归档路径自动集成
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QSpinBox, QHBoxLayout, QComboBox,
                            QCheckBox, QGroupBox, QFileDialog, QLineEdit,
                            QProgressBar, QRadioButton, QButtonGroup, QScrollArea, QMessageBox)
from qtpy.QtCore import Signal, QThread, QSettings
import numpy as np
from pathlib import Path
import json
import datetime
import cv2

# 引入工具函数
from utils.video_export import export_to_video, export_to_tiff_stack, get_available_codecs

class ExportThread(QThread):
    """导出后台线程"""
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
            # 确保父目录存在
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if self.export_type == 'video':
                # Video 导出
                success = export_to_video(
                    self.image_stack, str(self.output_path), **self.params
                )
                # 视频导出进度目前封装在 utils 里，这里发送完成信号
                self.progress.emit(100, 100)
                
            elif self.export_type == 'tiff':
                # TIFF Stack 导出
                success = export_to_tiff_stack(self.image_stack, str(self.output_path))
                self.progress.emit(100, 100)
                
            elif self.export_type == 'image_sequence':
                # 序列导出
                success = self._export_image_sequence()
            
            if success:
                self.finished.emit(str(self.output_path))
            else:
                self.error.emit("Export function returned False.")
                
        except Exception as e:
            self.error.emit(str(e))

    def _export_image_sequence(self):
        output_path = Path(self.output_path)
        # 如果路径有后缀（如 .png），则视为文件名模板的父文件夹；
        # 或者我们在上一级创建一个同名文件夹
        if output_path.suffix: 
            folder = output_path.parent / output_path.stem
        else:
            folder = output_path
            
        folder.mkdir(parents=True, exist_ok=True)
        
        fmt = self.params.get('format', 'png')
        pat = self.params.get('name_pattern', 'frame_{:04d}')
        total = len(self.image_stack)
        
        for i, frame in enumerate(self.image_stack):
            # 简单的 RGB 处理
            if frame.ndim == 3 and frame.shape[2] in [3, 4]:
                if frame.shape[2] == 4:
                    frame_out = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    frame_out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_out = frame
            
            fname = folder / f"{pat.format(i)}.{fmt}"
            cv2.imwrite(str(fname), frame_out)
            
            if i % 10 == 0: 
                self.progress.emit(i + 1, total)
                
        self.progress.emit(total, total)
        return True

class ExportWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.export_thread = None
        self.output_path = None
        self.settings = QSettings("NapariUser", "ExportSettings")
        self.annotation_settings = QSettings("NapariUser", "AnnotationParams")
        
        self._setup_ui()
        self._restore_last_path()
        
        # 监听图层变化
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

    def _setup_ui(self):
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<h3>💾 Export Data</h3>"))
        
        # 1. Source Layer
        h_lay = QHBoxLayout()
        h_lay.addWidget(QLabel("Source:"))
        self.layer_combo = QComboBox()
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)
        h_lay.addWidget(self.layer_combo)
        layout.addLayout(h_lay)
        
        # 刷新按钮
        btn_refresh = QPushButton("🔄 Refresh Layers")
        btn_refresh.clicked.connect(self._refresh_layers)
        layout.addWidget(btn_refresh)
        
        # 2. Frame Range (修复：找回了遗漏的帧范围选择)
        self.g_range = QGroupBox("Frame Range")
        l_range = QHBoxLayout()
        self.radio_all = QRadioButton("All Frames")
        self.radio_all.setChecked(True)
        self.radio_range = QRadioButton("Range")
        
        self.spin_start = QSpinBox(); self.spin_start.setEnabled(False)
        self.spin_end = QSpinBox(); self.spin_end.setEnabled(False)
        
        # 联动逻辑
        self.radio_range.toggled.connect(lambda c: (self.spin_start.setEnabled(c), self.spin_end.setEnabled(c)))
        
        l_range.addWidget(self.radio_all)
        l_range.addWidget(self.radio_range)
        l_range.addWidget(QLabel("From:"))
        l_range.addWidget(self.spin_start)
        l_range.addWidget(QLabel("To:"))
        l_range.addWidget(self.spin_end)
        self.g_range.setLayout(l_range)
        layout.addWidget(self.g_range)

        # 3. Export Format Type
        g_type = QGroupBox("Export Format")
        l_type = QVBoxLayout()
        self.bg_type = QButtonGroup()
        
        self.radio_vid = QRadioButton("Video (.mp4, .avi)")
        self.radio_vid.setChecked(True)
        self.radio_tiff = QRadioButton("TIFF Stack (.tiff)")
        self.radio_seq = QRadioButton("Image Sequence (Folder)")
        
        self.bg_type.addButton(self.radio_vid)
        self.bg_type.addButton(self.radio_tiff)
        self.bg_type.addButton(self.radio_seq)
        
        l_type.addWidget(self.radio_vid)
        l_type.addWidget(self.radio_tiff)
        l_type.addWidget(self.radio_seq)
        
        self.radio_vid.toggled.connect(self._toggle_settings)
        self.radio_tiff.toggled.connect(self._toggle_settings)
        self.radio_seq.toggled.connect(self._toggle_settings)
        
        g_type.setLayout(l_type)
        layout.addWidget(g_type)
        
        # 4. Video Settings (FPS, Codec, Quality)
        self.g_vid_set = QGroupBox("Video Options")
        l_vid = QVBoxLayout()
        
        h_fps = QHBoxLayout()
        h_fps.addWidget(QLabel("FPS:"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 120)
        self.spin_fps.setValue(30)
        h_fps.addWidget(self.spin_fps)
        l_vid.addLayout(h_fps)
        
        h_codec = QHBoxLayout()
        h_codec.addWidget(QLabel("Codec:"))
        self.combo_codec = QComboBox()
        self.combo_codec.addItems(get_available_codecs())
        h_codec.addWidget(self.combo_codec)
        l_vid.addLayout(h_codec)
        
        h_qual = QHBoxLayout()
        h_qual.addWidget(QLabel("Quality (0-100):"))
        self.spin_qual = QSpinBox()
        self.spin_qual.setRange(1, 100)
        self.spin_qual.setValue(95)
        h_qual.addWidget(self.spin_qual)
        l_vid.addLayout(h_qual)
        
        self.g_vid_set.setLayout(l_vid)
        layout.addWidget(self.g_vid_set)
        
        # 5. Sequence Settings
        self.g_seq_set = QGroupBox("Sequence Options")
        l_seq = QVBoxLayout()
        
        h_fmt = QHBoxLayout()
        h_fmt.addWidget(QLabel("Format:"))
        self.combo_img_fmt = QComboBox()
        self.combo_img_fmt.addItems(['png', 'jpg', 'bmp', 'tiff'])
        h_fmt.addWidget(self.combo_img_fmt)
        l_seq.addLayout(h_fmt)
        
        h_pat = QHBoxLayout()
        h_pat.addWidget(QLabel("Pattern:"))
        self.edit_pattern = QLineEdit("frame_{:04d}")
        h_pat.addWidget(self.edit_pattern)
        l_seq.addLayout(h_pat)
        
        self.g_seq_set.setLayout(l_seq)
        self.g_seq_set.setVisible(False) # Default hidden
        layout.addWidget(self.g_seq_set)
        
        # 6. Annotations Check
        self.g_anno = QGroupBox("Overlay Annotations")
        self.g_anno.setCheckable(True)
        self.g_anno.setChecked(True)
        l_anno = QVBoxLayout()
        
        self.check_sb = QCheckBox("Scale Bar")
        self.check_sb.setChecked(True)
        self.check_ts = QCheckBox("Timestamp")
        self.check_ts.setChecked(True)
        
        l_anno.addWidget(self.check_sb)
        l_anno.addWidget(self.check_ts)
        l_anno.addWidget(QLabel("<i style='color:gray'>(Styles loaded from Annotation Tab)</i>"))
        self.g_anno.setLayout(l_anno)
        layout.addWidget(self.g_anno)
        
        # 7. Output Path
        h_path = QHBoxLayout()
        self.lbl_path = QLabel("No path selected")
        self.lbl_path.setStyleSheet("font-size: 10px; color: gray;")
        self.lbl_path.setWordWrap(True)
        
        btn_brow = QPushButton("📂 Browse")
        btn_brow.clicked.connect(self._browse)
        
        h_path.addWidget(self.lbl_path)
        h_path.addWidget(btn_brow)
        layout.addLayout(h_path)
        
        # 8. Progress & Action
        self.pbar = QProgressBar()
        self.pbar.setVisible(False)
        self.pbar.setFormat("%p%")
        layout.addWidget(self.pbar)
        
        self.btn_run = QPushButton("🚀 Start Export")
        self.btn_run.clicked.connect(self._start_export)
        self.btn_run.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px;")
        self.btn_run.setEnabled(False)
        layout.addWidget(self.btn_run)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)
        
        layout.addStretch()
        
        content_widget.setLayout(layout)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
        self._refresh_layers()

    def _toggle_settings(self):
        """根据选择的格式显示/隐藏对应设置"""
        is_video = self.radio_vid.isChecked()
        is_seq = self.radio_seq.isChecked()
        self.g_vid_set.setVisible(is_video)
        self.g_seq_set.setVisible(is_seq)
        
    def _refresh_layers(self, event=None):
        curr = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for l in self.viewer.layers:
            if hasattr(l, 'data') and isinstance(l.data, np.ndarray):
                if l.data.ndim in [3, 4]: # 3D stack or 4D RGB
                    self.layer_combo.addItem(l.name)
        
        if curr: 
            idx = self.layer_combo.findText(curr)
            if idx >= 0: self.layer_combo.setCurrentIndex(idx)
        self.layer_combo.blockSignals(False)
        self._on_layer_changed(self.layer_combo.currentText())

    def _on_active_layer_changed(self, event=None):
        active = self.viewer.layers.selection.active
        if active:
            idx = self.layer_combo.findText(active.name)
            if idx >= 0: 
                self.layer_combo.setCurrentIndex(idx)

    def _on_layer_changed(self, txt):
        if not txt: return
        if txt not in self.viewer.layers: return
        layer = self.viewer.layers[txt]
        
        # 更新 Frame Range 上限
        num_frames = layer.data.shape[0]
        self.spin_start.setMaximum(num_frames - 1)
        self.spin_end.setMaximum(num_frames - 1)
        self.spin_end.setValue(num_frames - 1)
        self.spin_start.setValue(0)
        
        # 如果是RGB层，禁用Annotation覆盖 (因为Annotation模块通常处理单通道叠加)
        is_rgb = (layer.data.ndim == 4) or (hasattr(layer, 'rgb') and layer.rgb)
        if is_rgb:
            self.g_anno.setChecked(False)
            self.g_anno.setTitle("Overlay (Disabled for RGB)")
            self.g_anno.setEnabled(False)
        else:
            self.g_anno.setEnabled(True)
            self.g_anno.setTitle("Overlay Annotations")

    def _browse(self):
        d = self.settings.value("last_dir", str(Path.home()))
        
        if self.radio_vid.isChecked():
            f, _ = QFileDialog.getSaveFileName(self, "Save Video", d, "Video (*.mp4 *.avi)")
        elif self.radio_tiff.isChecked():
            f, _ = QFileDialog.getSaveFileName(self, "Save TIFF", d, "TIFF (*.tiff)")
        else:
            f = QFileDialog.getExistingDirectory(self, "Select Output Folder", d)
            
        if f:
            self.output_path = f
            self.lbl_path.setText(f)
            self.lbl_path.setStyleSheet("color: #E0E0E0; font-size: 10px;")
            
            p = Path(f)
            # 如果是文件，保存其父目录；如果是文件夹，保存该目录
            save_dir = str(p.parent) if p.suffix else str(p)
            self.settings.setValue("last_dir", save_dir)
            self.btn_run.setEnabled(True)

    def _restore_last_path(self):
        # 检查全局归档路径 (from Import Widget)
        archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
        if archive_path and Path(archive_path).exists():
            self.output_path = archive_path
            self.lbl_path.setText(f"📂 Auto-Archive: {Path(archive_path).name}")
            self.lbl_path.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11px;")
            self.btn_run.setEnabled(True)
        else:
            # 回退到本地记忆
            last = self.settings.value("last_output", "")
            if last:
                self.output_path = last
                self.lbl_path.setText(last)
                self.btn_run.setEnabled(True)

    def _get_export_params(self):
        params = {}
        
        if self.radio_vid.isChecked():
            params['fps'] = self.spin_fps.value()
            params['codec'] = self.combo_codec.currentText()
            params['quality'] = self.spin_qual.value()
            
            # Load Annotations from Settings
            if self.g_anno.isChecked():
                s = self.annotation_settings
                
                if self.check_sb.isChecked():
                    # 读取 AnnotationWidget 保存的配置
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
                            'color': s.value("scale/color", (1,1,1,1)),
                            'bg_color': s.value("scale/bg_color", (0,0,0,1)),
                            'bg_alpha': int(s.value("scale/bg_alpha", 100)),
                            'use_bg': s.value("scale/use_bg", "true") == "true",
                            'position': s.value("scale/position", (50, 50))
                        }
                    except:
                        print("Warning: Could not load scale bar settings.")

                if self.check_ts.isChecked():
                    try:
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
                    except:
                        print("Warning: Could not load timestamp settings.")
        
        elif self.radio_seq.isChecked():
            params['format'] = self.combo_img_fmt.currentText()
            params['name_pattern'] = self.edit_pattern.text()
            
        return params

    def _start_export(self):
        if not self.layer_combo.currentText() or not self.output_path: 
            self.lbl_status.setText("❌ Check inputs")
            return
        layer_name = self.layer_combo.currentText()
        # === [Fix] Warning Check (Smart) ===
        # 如果图层名包含 "Burned" 或 "Annotated"，我们假设用户已经烧录好了，跳过检查
        # 否则，如果未启用Annotation或未勾选子项，提示警告
        is_burned = "burned" in layer_name.lower() or "annotated" in layer_name.lower()

        # === 导出前检查 (Warning Check) ===
        if self.radio_vid.isChecked() and not is_burned:
            # 如果启用了Annotation Group 但并没有勾选任何子项，或者直接未启用
            # 这里逻辑是：如果用户要做视频，通常需要比例尺和时间戳。如果都没选，提示一下。
            has_sb = self.check_sb.isChecked()
            has_ts = self.check_ts.isChecked()
            anno_enabled = self.g_anno.isChecked()
            
            if not anno_enabled or (not has_sb and not has_ts):
                reply = QMessageBox.question(
                    self, "Missing Annotations",
                    "You are exporting a video WITHOUT Scale Bar or Timestamp.\n\nAre you sure?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply == QMessageBox.No: return
        
        self.btn_run.setEnabled(False)
        self.pbar.setValue(0)
        self.pbar.setVisible(True)
        self.lbl_status.setText("⏳ Exporting...")
        
        # 1. 自动处理文件名 (如果使用了归档路径)
        final_path = Path(self.output_path)
        archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
        layer_name = self.layer_combo.currentText()
        
        if archive_path and final_path == Path(archive_path):
            # 自动归档模式：生成文件名
            ts = datetime.datetime.now().strftime("%H%M%S")
            safe_layer = "".join([c if c.isalnum() or c in "-_" else "_" for c in layer_name])
            fname = f"{safe_layer}_{ts}"
            
            if self.radio_vid.isChecked():
                final_path = final_path / "Exported_Videos" / f"{fname}.mp4"
            elif self.radio_tiff.isChecked():
                final_path = final_path / "Exported_Stacks" / f"{fname}.tiff"
            else:
                final_path = final_path / "Exported_Sequences" / fname
            
            final_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 2. 准备数据 (切片处理)
        layer = self.viewer.layers[layer_name]
        data = layer.data
        
        if self.radio_range.isChecked():
            start = self.spin_start.value()
            end = self.spin_end.value()
            # 确保范围有效
            if start > end: start, end = end, start
            data_slice = data[start:end+1]
        else:
            data_slice = data
            
        etype = 'video' if self.radio_vid.isChecked() else 'tiff' if self.radio_tiff.isChecked() else 'image_sequence'
        params = self._get_export_params()
        
        # 3. 启动线程
        self.export_thread = ExportThread(data_slice, str(final_path), etype, params)
        self.export_thread.progress.connect(lambda c, t: self.pbar.setValue(int(c/t*100)))
        self.export_thread.finished.connect(lambda p: self._on_done(p, params))
        self.export_thread.error.connect(self._on_error)
        self.export_thread.start()

    def _on_done(self, path, params):
        self.lbl_status.setText(f"✅ Done: {Path(path).name}")
        self.pbar.setVisible(False)
        self.btn_run.setEnabled(True)
        
        # Log to JSON
        self._log_export(path, params)

    def _on_error(self, err):
        self.lbl_status.setText(f"❌ Error: {err}")
        self.pbar.setVisible(False)
        self.btn_run.setEnabled(True)

    def _log_export(self, path, params):
        try:
            archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
            if archive_path:
                log_file = Path(archive_path) / "processing_log.json"
                data = {}
                if log_file.exists():
                    with open(log_file, 'r') as f: data = json.load(f)
                
                if "exports" not in data: data["exports"] = []
                entry = {
                    "timestamp": str(datetime.datetime.now()),
                    "file": str(Path(path).name),
                    "type": self.export_thread.export_type,
                    "params": params
                }
                data["exports"].append(entry)
                
                with open(log_file, 'w') as f: json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Log failed: {e}")
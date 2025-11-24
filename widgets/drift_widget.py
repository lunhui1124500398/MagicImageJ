"""
漂移矫正控件 (完整版)
功能：
1. 自动切换绘制工具
2. 鼠标释放后触发计算 (带进度条)
3. 归档集成：应用矫正后，自动将 ROI、Template、Kernel 等参数写入 processing_log.json。
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QSpinBox, QHBoxLayout, QComboBox,
                            QCheckBox, QMessageBox, QProgressBar, QScrollArea)
from qtpy.QtCore import Signal, QThread, Qt, QSettings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import napari
import json
from pathlib import Path
import datetime

# 导入核心算法
from core.drift_correction import (calculate_drift_curve, 
                                   apply_drift_correction,
                                   validate_roi)

# JSON Encoder for Numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)): return int(obj)
        elif isinstance(obj, (np.floating, float)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

class DriftCalculationThread(QThread):
    """漂移计算线程"""
    finished = Signal(np.ndarray)  # drifts
    progress = Signal(int, int)    # current, total
    error = Signal(str)
    
    def __init__(self, image_stack, roi_bbox, template_idx, max_workers, kernel_size):
        super().__init__()
        self.image_stack = image_stack
        self.roi_bbox = roi_bbox
        self.template_idx = template_idx
        self.max_workers = max_workers
        self.kernel_size = kernel_size
        
    def run(self):
        try:
            # 进度回调适配器
            def cb(c, t):
                self.progress.emit(c, t)
                
            drifts = calculate_drift_curve(
                self.image_stack,
                self.roi_bbox,
                self.template_idx,
                self.max_workers,
                self.kernel_size,
                progress_callback=cb
            )
            self.finished.emit(drifts)
        except Exception as e:
            self.error.emit(str(e))

class DriftApplyThread(QThread):
    """漂移应用线程"""
    finished = Signal(np.ndarray)
    progress = Signal(int, int)
    error = Signal(str)
    
    def __init__(self, image_stack, drifts, max_workers):
        super().__init__()
        self.image_stack = image_stack
        self.drifts = drifts
        self.max_workers = max_workers
        
    def run(self):
        try:
            def cb(c, t):
                self.progress.emit(c, t)
                
            corrected = apply_drift_correction(
                self.image_stack,
                self.drifts,
                self.max_workers,
                progress_callback=cb
            )
            self.finished.emit(corrected)
        except Exception as e:
            self.error.emit(str(e))

class DriftCorrectionWidget(QWidget):
    """漂移矫正控件"""
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_drifts = None
        self.calc_thread = None
        self.apply_thread = None
        self.correction_history = []  # [新增] 用于管理生成的校正图层
        self._setup_ui()
        
        # 监听图层事件
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

    def _setup_ui(self):
        # 1. 创建最外层布局 (用于放滚动条)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True) 
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        title = QLabel("<h3>🔧 Drift Correction</h3>")
        layout.addWidget(title)

        # 图层选择
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Source Layer:"))
        self.layer_combo = QComboBox()
        self.layer_combo.currentTextChanged.connect(self._update_template_range)
        layer_layout.addWidget(self.layer_combo)
        layout.addLayout(layer_layout)
        
        refresh_btn = QPushButton("🔄 Refresh Layers")
        refresh_btn.clicked.connect(self._refresh_layers)
        layout.addWidget(refresh_btn)

        # 模板帧选择
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Template Frame:"))
        self.template_spin = QSpinBox()
        self.template_spin.setMinimum(0)
        template_layout.addWidget(self.template_spin)
        template_layout.addStretch()
        layout.addLayout(template_layout)

        # 参数设置
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Kernel:"))
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(3, 51)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.setValue(11)
        param_layout.addWidget(self.kernel_spin)
        
        param_layout.addWidget(QLabel("Workers:"))
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 32)
        self.max_workers_spin.setValue(8)
        param_layout.addWidget(self.max_workers_spin)
        layout.addLayout(param_layout)

        # 步骤1：绘制ROI
        layout.addWidget(QLabel("<b>Step 1: Draw ROI</b>"))
        draw_roi_btn = QPushButton("✏️ Draw ROI (Add Shapes Layer)")
        draw_roi_btn.clicked.connect(self._add_shapes_layer)
        layout.addWidget(draw_roi_btn)

        # 步骤2：预览漂移曲线
        layout.addWidget(QLabel("<b>Step 2: Preview Drift</b>"))
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        ctrl_layout = QHBoxLayout()
        self.auto_calc_cb = QCheckBox("Auto Preview (Calc & Apply)")
        self.auto_calc_cb.setChecked(True)
        self.auto_calc_cb.setToolTip("Calculate and show corrected result immediately after drawing ROI.")
        ctrl_layout.addWidget(self.auto_calc_cb)
        
        preview_btn = QPushButton("📊 Manual Recalc")
        preview_btn.clicked.connect(self._preview_drift)
        ctrl_layout.addWidget(preview_btn)
        layout.addLayout(ctrl_layout)

        self.drift_figure = plt.Figure(figsize=(5, 3))
        self.drift_canvas = FigureCanvasQTAgg(self.drift_figure)
        self.drift_canvas.setVisible(False)
        layout.addWidget(self.drift_canvas)

        # 步骤3：应用矫正
        layout.addWidget(QLabel("<b>Step 3: Apply Correction</b>"))
        self.apply_btn = QPushButton("✅ Apply / Commit")
        self.apply_btn.clicked.connect(self._apply_correction)
        self.apply_btn.setEnabled(False)
        layout.addWidget(self.apply_btn)

        redraw_btn = QPushButton("🔄 Reset / Clear ROI")
        redraw_btn.clicked.connect(self._redraw_roi)
        layout.addWidget(redraw_btn)

        # [新增] 提示标签
        self.hint_label = QLabel("Tip: Press 'Z' on corrected layer to Undo/Retry.")
        self.hint_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        layout.addWidget(self.hint_label)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()
        content_widget.setLayout(layout)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        self._refresh_layers()

    def _refresh_layers(self, event=None):
        current_text = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if (hasattr(layer, 'data') and isinstance(layer.data, np.ndarray) and len(layer.data.shape) == 3):
                self.layer_combo.addItem(layer.name)
        
        active_layer = self.viewer.layers.selection.active
        if active_layer and self.layer_combo.findText(active_layer.name) >= 0:
            self.layer_combo.setCurrentText(active_layer.name)
        elif self.layer_combo.findText(current_text) >= 0:
            self.layer_combo.setCurrentText(current_text)
        self.layer_combo.blockSignals(False)
        self._update_template_range()

    def _on_active_layer_changed(self, event=None):
        active_layer = self.viewer.layers.selection.active
        if active_layer:
            index = self.layer_combo.findText(active_layer.name)
            if index >= 0: self.layer_combo.setCurrentIndex(index)

    def _update_template_range(self):
        layer_name = self.layer_combo.currentText()
        if layer_name:
            layer = self.viewer.layers[layer_name]
            self.template_spin.setMaximum(len(layer.data) - 1)
            # 默认中间帧
            self.template_spin.setValue((len(layer.data) - 1) // 2)

    def _add_shapes_layer(self):
        roi_layer_name = "Drift_ROI"
        if roi_layer_name in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[roi_layer_name]
            layer = self.viewer.layers[roi_layer_name]
        else:
            layer = self.viewer.add_shapes(name=roi_layer_name, edge_color='red', edge_width=2, face_color=[1, 0, 0, 0.01])
            layer.mouse_drag_callbacks.append(self._on_roi_interaction)
            
            @layer.bind_key('z')
            def clear_roi(layer):
                if len(layer.data) > 0:
                    layer.data = []
                    self.status_label.setText("🗑️ ROI cleared.")
                    layer.mode = 'add_rectangle'
                    self.drift_canvas.setVisible(False)
                    self.apply_btn.setEnabled(False)
                    self.progress_bar.setVisible(False)
        self.viewer.layers.selection.active = layer
        layer.mode = 'add_rectangle'
        self.status_label.setText("✏️ Mode: Draw Rectangle (Auto-Preview ON)")

    def _on_roi_interaction(self, layer, event):
        yield
        while event.type == 'mouse_move': yield
        if len(layer.data) == 0: return
        if len(layer.data) > 1: layer.data = layer.data[-1:]
        if layer.mode == 'add_rectangle': layer.mode = 'select'
        
        if self.auto_calc_cb.isChecked():
            self.status_label.setText("✋ ROI updated. Auto-calculating...")
            self._preview_drift()
        else:
            self.status_label.setText("✋ ROI updated. Click 'Recalculate'.")

    def _get_roi_bbox(self):
        if "Drift_ROI" not in self.viewer.layers: return None
        roi_layer = self.viewer.layers["Drift_ROI"]
        if len(roi_layer.data) == 0: return None
        shape_data = roi_layer.data[0]
        ys, xs = shape_data[:, 0], shape_data[:, 1]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    def _preview_drift(self):
        layer_name = self.layer_combo.currentText()
        if not layer_name: return
        image_stack = self.viewer.layers[layer_name].data
        roi_bbox = self._get_roi_bbox()
        if roi_bbox is None: 
            self.status_label.setText("❌ Draw ROI first.")
            return
        if not validate_roi(image_stack.shape[1:], roi_bbox):
            self.status_label.setText("❌ Invalid ROI (too small or out of bounds).")
            return
        
        self.status_label.setText("⏳ Calculating drift...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.drift_canvas.setVisible(False)
        self.apply_btn.setEnabled(False)
        self.auto_calc_cb.setEnabled(False)
        
        self.calc_thread = DriftCalculationThread(
            image_stack, roi_bbox, self.template_spin.value(),
            self.max_workers_spin.value(), self.kernel_spin.value()
        )
        self.calc_thread.progress.connect(lambda c, t: self.progress_bar.setValue(int(c/t*100)))
        self.calc_thread.finished.connect(self._on_drift_calculated)
        self.calc_thread.error.connect(self._on_drift_error)
        self.calc_thread.start()

    def _on_drift_calculated(self, drifts):
        self.progress_bar.setVisible(False)
        self.auto_calc_cb.setEnabled(True)
        if "Drift_ROI" in self.viewer.layers and len(self.viewer.layers["Drift_ROI"].data) == 0: return
        
        self.current_drifts = drifts
        self.drift_figure.clear()
        ax = self.drift_figure.add_subplot(111)
        frames = np.arange(len(drifts))
        ax.plot(frames, drifts[:, 0], label='X', color='blue')
        ax.plot(frames, drifts[:, 1], label='Y', color='red')
        ax.set_title('Drift Curve')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        self.drift_figure.tight_layout()
        self.drift_canvas.draw()
        self.drift_canvas.setVisible(True)
        self.apply_btn.setEnabled(True)
        self.status_label.setText(f"✅ Max X: {np.max(np.abs(drifts[:,0])):.1f}, Y: {np.max(np.abs(drifts[:,1])):.1f}")
        # === [核心修改] 自动应用校正 ===
        if self.auto_calc_cb.isChecked():
            self.status_label.setText("⚡ Auto-applying correction...")
            self._apply_correction(auto_mode=True)
        else:
            self.status_label.setText("✅ Calculated. Click 'Apply' to see result.")


    def _on_drift_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.auto_calc_cb.setEnabled(True)
        self.status_label.setText(f"❌ Error: {error_msg}")

    def _apply_correction(self, auto_mode=False):
        if self.current_drifts is None: return
        layer_name = self.layer_combo.currentText()
        layer = self.viewer.layers[layer_name]

        if not auto_mode:
            self.status_label.setText("⏳ Applying correction...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.apply_btn.setEnabled(False)
        
        self.apply_thread = DriftApplyThread(layer.data, self.current_drifts, self.max_workers_spin.value())
        self.apply_thread.progress.connect(lambda c, t: self.progress_bar.setValue(int(c/t*100)))
        self.apply_thread.finished.connect(lambda res: self._on_apply_finished(res, layer_name))
        self.apply_thread.error.connect(self._on_drift_error)
        self.apply_thread.start()

    def _on_apply_finished(self, corrected_stack, source_layer_name):
        self.progress_bar.setVisible(False)
        self.apply_btn.setEnabled(True)
        try:
            count = len(self.correction_history) + 1
            new_layer_name = f"Corrected_v{count}_{source_layer_name}"
            new_layer = self.viewer.add_image(
                corrected_stack, 
                name=new_layer_name, 
                colormap='gray', 
                metadata={'source': source_layer_name, 'is_drift_result': True}
            )

            self.correction_history.append(new_layer)
            if len(self.correction_history) > 3:
                oldest_layer = self.correction_history.pop(0)
                if oldest_layer in self.viewer.layers:
                    self.viewer.layers.remove(oldest_layer)

            @new_layer.bind_key('z')
            def undo_correction(layer):
                self._undo_last_correction(layer)
            
            if source_layer_name in self.viewer.layers:
                self.viewer.layers[source_layer_name].visible = False
            self.viewer.layers.selection.active = new_layer
            self.status_label.setText(f"✅ Previewing: {new_layer_name}. Press 'Z' to Undo.")

            self._log_drift_action(source_layer_name)
        
        except Exception as e:
            self.status_label.setText(f"❌ Apply Error: {str(e)}")
    
    def _undo_last_correction(self, layer_to_remove):
        """撤销操作：删除图层，显示原图，激活ROI层"""
        # 1. 删除图层
        if layer_to_remove in self.viewer.layers:
            self.viewer.layers.remove(layer_to_remove)
        
        if layer_to_remove in self.correction_history:
            self.correction_history.remove(layer_to_remove)
        
        
        # 2. 恢复原图可见性
        source_name = layer_to_remove.metadata.get('source')
        if source_name and source_name in self.viewer.layers:
            self.viewer.layers[source_name].visible = True
            
        # 3. 激活 ROI 图层以便重画
        if "Drift_ROI" in self.viewer.layers:
            roi_layer = self.viewer.layers["Drift_ROI"]
            self.viewer.layers.selection.active = roi_layer
            roi_layer.mode = 'select' # 或者 'add_rectangle' 根据偏好
            
        self.status_label.setText("↩️ Undone. Adjust ROI and try again.")

    def _log_drift_action(self, source_layer):
        """Save parameters to processing_log.json"""
        try:
            archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
            if not archive_path: return
            
            log_path = Path(archive_path) / "processing_log.json"
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f: data = json.load(f)
                except: data = {}
            else: data = {}

            if "drift_correction" not in data: data["drift_correction"] = []
            
            max_drift = np.max(np.abs(self.current_drifts), axis=0)
            
            entry = {
                "timestamp": str(datetime.datetime.now()),
                "source_layer": source_layer,
                "template_frame": self.template_spin.value(),
                "roi_bbox": self._get_roi_bbox(),
                "kernel_size": self.kernel_spin.value(),
                "max_shift_x": max_drift[0],
                "max_shift_y": max_drift[1]
            }
            
            data["drift_correction"].append(entry)
            
            with open(log_path, 'w') as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
                
        except Exception as e:
            print(f"Failed to log drift: {e}")

    def _redraw_roi(self):
        # 清理所有历史
        for l in self.correction_history:
            if l in self.viewer.layers: # 必须检查存在性
                self.viewer.layers.remove(l)
        self.correction_history.clear()
        
        if "Drift_ROI" in self.viewer.layers: 
            self.viewer.layers.remove("Drift_ROI")
            
        self.current_drifts = None
        self.drift_canvas.setVisible(False)
        self._add_shapes_layer()
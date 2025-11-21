"""
漂移矫正控件 (完整版 - 交互优化)
包含功能：
1. 自动切换绘制工具
2. 鼠标释放后触发计算 (避免拖拽时频繁卡顿)
3. 快捷键 'Z' 撤销/清除 ROI
4. 计算进度条显示
5. [NEW] 自动计算开关 (Auto-calc Checkbox) - 让用户决定是否在调整后立即计算
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QSpinBox, QHBoxLayout, QComboBox,
                            QCheckBox, QMessageBox, QProgressBar)
from qtpy.QtCore import Signal, QThread, Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import napari

# 请确保您的项目中存在 core.drift_correction 模块
from core.drift_correction import (calculate_drift_curve, 
                                   apply_drift_correction,
                                   validate_roi)

class DriftCalculationThread(QThread):
    """漂移计算线程"""
    finished = Signal(np.ndarray)  # drifts
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
            drifts = calculate_drift_curve(
                self.image_stack,
                self.roi_bbox,
                self.template_idx,
                self.max_workers,
                self.kernel_size
            )
            self.finished.emit(drifts)
        except Exception as e:
            self.error.emit(str(e))


class DriftCorrectionWidget(QWidget):
    """漂移矫正控件"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_drifts = None
        self.drift_thread = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # 标题
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
        
        # Kernel
        param_layout.addWidget(QLabel("Kernel:"))
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(3, 51)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.setValue(11)
        param_layout.addWidget(self.kernel_spin)
        
        # Workers
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
        
        # === 进度条 ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 繁忙模式
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { height: 10px; }")
        layout.addWidget(self.progress_bar)
        
        # === 自动计算开关 ===
        # 用户建议：避免频繁触发。通过这个开关控制。
        ctrl_layout = QHBoxLayout()
        self.auto_calc_cb = QCheckBox("Auto-calc on release")
        self.auto_calc_cb.setChecked(True) # 默认开启，方便新手
        self.auto_calc_cb.setToolTip("If checked, drift is calculated immediately after drawing or resizing the ROI.\nUncheck this for large datasets to adjust ROI freely.")
        ctrl_layout.addWidget(self.auto_calc_cb)
        
        # 手动计算按钮
        preview_btn = QPushButton("📊 Recalculate")
        preview_btn.clicked.connect(self._preview_drift)
        ctrl_layout.addWidget(preview_btn)
        
        layout.addLayout(ctrl_layout)
        
        # 漂移曲线显示
        self.drift_figure = plt.Figure(figsize=(5, 3))
        self.drift_canvas = FigureCanvasQTAgg(self.drift_figure)
        self.drift_canvas.setVisible(False)
        layout.addWidget(self.drift_canvas)
        
        # 步骤3：应用矫正
        layout.addWidget(QLabel("<b>Step 3: Apply Correction</b>"))
        self.apply_btn = QPushButton("✅ Apply Drift Correction")
        self.apply_btn.clicked.connect(self._apply_correction)
        self.apply_btn.setEnabled(False)
        layout.addWidget(self.apply_btn)
        
        # 重绘ROI按钮
        redraw_btn = QPushButton("🔄 Reset / Clear ROI")
        redraw_btn.clicked.connect(self._redraw_roi)
        layout.addWidget(redraw_btn)
        
        # 状态信息
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self._refresh_layers()
    
    def _refresh_layers(self):
        """刷新图层列表并自动选中活跃图层"""
        # 记录当前选中项，以便在没有更好选择时恢复
        current_text = self.layer_combo.currentText()
        
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        
        for layer in self.viewer.layers:
            if (hasattr(layer, 'data') and 
                isinstance(layer.data, np.ndarray) and 
                len(layer.data.shape) == 3):
                self.layer_combo.addItem(layer.name)
        
        # 尝试选中当前 Napari 激活的图层
        active_layer = self.viewer.layers.selection.active
        if active_layer and self.layer_combo.findText(active_layer.name) >= 0:
            self.layer_combo.setCurrentText(active_layer.name)
        # 如果没有激活图层或不在列表中，尝试恢复之前的选择
        elif self.layer_combo.findText(current_text) >= 0:
            self.layer_combo.setCurrentText(current_text)
            
        self.layer_combo.blockSignals(False)
        
        # 触发一次更新逻辑
        self._update_template_range()
    
    def _on_active_layer_changed(self, event=None):
        """响应 Napari 图层选择变化"""
        active_layer = self.viewer.layers.selection.active
        if active_layer:
            index = self.layer_combo.findText(active_layer.name)
            if index >= 0:
                self.layer_combo.setCurrentIndex(index)

    def _update_template_range(self):
        layer_name = self.layer_combo.currentText()
        if layer_name:
            layer = self.viewer.layers[layer_name]
            self.template_spin.setMaximum(len(layer.data) - 1)
            self.template_spin.setValue((len(layer.data) - 1) // 2)
    
    def _add_shapes_layer(self):
        roi_layer_name = "Drift_ROI"
        
        if roi_layer_name in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[roi_layer_name]
            layer = self.viewer.layers[roi_layer_name]
        else:
            layer = self.viewer.add_shapes(
                name=roi_layer_name,
                edge_color='red',
                edge_width=2,
                face_color=[1, 0, 0, 0.01],
            )
            
            # === 鼠标回调：处理拖拽逻辑 ===
            layer.mouse_drag_callbacks.append(self._on_roi_interaction)
            
            # === 快捷键：Z键撤销 ===
            @layer.bind_key('z')
            def clear_roi(layer):
                if len(layer.data) > 0:
                    layer.data = []
                    self.status_label.setText("🗑️ ROI cleared. Draw again.")
                    layer.mode = 'add_rectangle'
                    self.drift_canvas.setVisible(False)
                    self.apply_btn.setEnabled(False)
                    self.progress_bar.setVisible(False)

        self.viewer.layers.selection.active = layer
        layer.mode = 'add_rectangle'
        self.status_label.setText("✏️ Mode: Draw Rectangle. (Press 'Z' to undo)")

    def _on_roi_interaction(self, layer, event):
        """
        鼠标交互回调：Yield模式
        """
        # --- Mouse Press ---
        yield
        
        # --- Mouse Drag ---
        while event.type == 'mouse_move':
            yield
            
        # --- Mouse Release (Interaction Finished) ---
        
        if len(layer.data) == 0:
            return

        # 限制单个ROI
        if len(layer.data) > 1:
            layer.data = layer.data[-1:]
        
        # 自动切换模式
        if layer.mode == 'add_rectangle':
            layer.mode = 'select'
            
        # === 关键修改：检查自动计算开关 ===
        if self.auto_calc_cb.isChecked():
            self.status_label.setText("✋ ROI updated. Auto-calculating...")
            self._preview_drift()
        else:
            self.status_label.setText("✋ ROI updated. Click 'Recalculate' when ready.")

    def _get_roi_bbox(self):
        roi_layer_name = "Drift_ROI"
        if roi_layer_name not in self.viewer.layers:
            return None
        roi_layer = self.viewer.layers[roi_layer_name]
        if len(roi_layer.data) == 0:
            return None
        
        shape_data = roi_layer.data[0]
        ys = shape_data[:, 0]
        xs = shape_data[:, 1]
        
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        return (x1, y1, x2, y2)
    
    def _preview_drift(self):
        """预览漂移曲线"""
        # 基本校验
        layer_name = self.layer_combo.currentText()
        if not layer_name:
            self.status_label.setText("❌ Please select a layer first.")
            return
        
        layer = self.viewer.layers[layer_name]
        image_stack = layer.data
        
        roi_bbox = self._get_roi_bbox()
        if roi_bbox is None:
            self.status_label.setText("❌ Please draw a ROI rectangle first.")
            return
        
        if not validate_roi(image_stack.shape[1:], roi_bbox):
            self.status_label.setText("❌ Invalid ROI. Please redraw.")
            return
        
        # 准备参数
        template_idx = self.template_spin.value()
        kernel_size = self.kernel_spin.value()
        max_workers = self.max_workers_spin.value()
        
        # 更新UI状态
        self.status_label.setText("⏳ Calculating drift...")
        self.progress_bar.setVisible(True)
        self.drift_canvas.setVisible(False)
        self.apply_btn.setEnabled(False)
        self.auto_calc_cb.setEnabled(False) # 计算时暂时禁用开关，防止状态混乱
        
        # 启动线程
        self.drift_thread = DriftCalculationThread(
            image_stack, roi_bbox, template_idx, max_workers, kernel_size
        )
        self.drift_thread.finished.connect(self._on_drift_calculated)
        self.drift_thread.error.connect(self._on_drift_error)
        self.drift_thread.start()
    
    def _on_drift_calculated(self, drifts):
        self.progress_bar.setVisible(False)
        self.auto_calc_cb.setEnabled(True) # 恢复开关
        
        # Z键保护
        roi_layer_name = "Drift_ROI"
        if roi_layer_name in self.viewer.layers:
            roi_layer = self.viewer.layers[roi_layer_name]
            if len(roi_layer.data) == 0:
                self.status_label.setText("⚠️ Calculation ignored (ROI cleared).")
                return

        self.current_drifts = drifts
        
        # 绘图
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
        self.status_label.setText(f"✅ Calculated. Max X: {np.max(np.abs(drifts[:,0])):.1f}, Y: {np.max(np.abs(drifts[:,1])):.1f}")
    
    def _on_drift_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.auto_calc_cb.setEnabled(True)
        self.status_label.setText(f"❌ Error: {error_msg}")
    
    def _apply_correction(self):
        if self.current_drifts is None: return
        
        layer_name = self.layer_combo.currentText()
        layer = self.viewer.layers[layer_name]
        image_stack = layer.data
        
        self.status_label.setText("⏳ Applying...")
        self.progress_bar.setVisible(True)
        
        try:
            max_workers = self.max_workers_spin.value()
            corrected_stack = apply_drift_correction(
                image_stack, self.current_drifts, max_workers
            )
            if not corrected_stack.flags['C_CONTIGUOUS']:
                corrected_stack = np.ascontiguousarray(corrected_stack)

            new_layer_name = f"Corrected_{layer_name}"
            self.viewer.add_image(
                corrected_stack,
                name=new_layer_name,
                colormap='gray',
                metadata={'source': layer_name}
            )
            
            self.status_label.setText(f"✅ Done! Layer: {new_layer_name}")
            
            # 隐藏旧图层，选中新图层
            for l in self.viewer.layers:
                if isinstance(l, napari.layers.Image) and l.name != new_layer_name:
                    l.visible = False
            self.viewer.layers.selection.active = self.viewer.layers[new_layer_name]
            self._refresh_layers()
            self.layer_combo.setCurrentText(new_layer_name)
            
            if "Drift_ROI" in self.viewer.layers:
                self.viewer.layers["Drift_ROI"].visible = False
                
        except Exception as e:
            self.status_label.setText(f"❌ Apply Error: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    
    def _redraw_roi(self):
        roi_layer_name = "Drift_ROI"
        if roi_layer_name in self.viewer.layers:
            self.viewer.layers.remove(roi_layer_name)
        
        self.current_drifts = None
        self.apply_btn.setEnabled(False)
        self.drift_canvas.setVisible(False)
        self.progress_bar.setVisible(False)
        self._add_shapes_layer()
"""
图像增强控件 - 增强版
包含：高斯模糊、滚动平均（自动裁切边缘）、仿ImageJ对比度调节（直方图+曲线）
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QSpinBox, QHBoxLayout, QComboBox,
                            QCheckBox, QSlider, QGroupBox, QDoubleSpinBox,
                            QMessageBox, QProgressBar)
from qtpy.QtCore import Qt, Signal, QThread
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from core.image_enhance import enhance_image_stack
import napari


class EnhanceThread(QThread):
    """增强处理线程"""
    finished = Signal(np.ndarray)
    error = Signal(str)
    
    def __init__(self, image_stack, **kwargs):
        super().__init__()
        self.image_stack = image_stack
        self.params = kwargs
    
    def run(self):
        try:
            # 注意：这里假设 enhance_image_stack 返回的是并未裁切边缘的数据
            result = enhance_image_stack(self.image_stack, **self.params)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class EnhanceWidget(QWidget):
    """图像增强控件 - 完整版"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.enhance_thread = None
        
        self._setup_ui()
        
        # 监听图层变化以更新直方图
        # self.viewer.layers.selection.events.changed.connect(self._on_layer_selection_changed)

        # 核心修复：监听图层列表的插入/移除事件，实现自动刷新
        self.viewer.layers.events.inserted.connect(self._refresh_layers_silently)
        self.viewer.layers.events.removed.connect(self._refresh_layers_silently)
        
        # 核心修复：监听激活图层的变化，实现下拉框自动同步
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # 标题
        title = QLabel("<h3>✨ Image Enhancement</h3>")
        layout.addWidget(title)
        
        # 图层选择
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Target Layer:"))
        self.layer_combo = QComboBox()
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)
        layer_layout.addWidget(self.layer_combo)
        layout.addLayout(layer_layout)
        
        refresh_btn = QPushButton("🔄 Refresh Layers")
        refresh_btn.clicked.connect(self._refresh_layers)
        layout.addWidget(refresh_btn)
        
        # ==========================================
        # Part 1: 滤波器设置 (Gaussian & Rolling)
        # ==========================================
        filter_group = QGroupBox("1. Filters")
        filter_layout = QVBoxLayout()

        # --- 高斯模糊 ---
        gaussian_layout = QHBoxLayout()
        self.use_gaussian_check = QCheckBox("Gaussian Blur")
        self.use_gaussian_check.stateChanged.connect(self._toggle_gaussian)
        gaussian_layout.addWidget(self.use_gaussian_check)
        
        self.ksize_slider = QSlider(Qt.Horizontal)
        self.ksize_slider.setRange(1, 25)
        self.ksize_slider.setValue(3)
        self.ksize_slider.setSingleStep(2)
        self.ksize_slider.valueChanged.connect(self._ensure_odd_ksize)
        gaussian_layout.addWidget(QLabel("Kernel:"))
        gaussian_layout.addWidget(self.ksize_slider)
        self.ksize_label = QLabel("3")
        gaussian_layout.addWidget(self.ksize_label)
        
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 10.0)
        self.sigma_spin.setValue(0.8)
        self.sigma_spin.setSingleStep(0.1)
        gaussian_layout.addWidget(QLabel("Sigma:"))
        gaussian_layout.addWidget(self.sigma_spin)
        filter_layout.addLayout(gaussian_layout)
        
        # --- 滚动平均 ---
        avg_layout = QHBoxLayout()
        self.use_average_check = QCheckBox("Roll Avg")
        self.use_average_check.stateChanged.connect(self._toggle_average)
        avg_layout.addWidget(self.use_average_check)
        
        avg_layout.addWidget(QLabel("Win:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(3, 51)
        self.window_spin.setSingleStep(2)
        self.window_spin.setValue(3)
        self.window_spin.valueChanged.connect(self._update_frame_loss_info)
        avg_layout.addWidget(self.window_spin)
        
        # 自动丢弃边缘帧选项
        self.crop_edges_check = QCheckBox("Auto-drop black edges")
        self.crop_edges_check.setChecked(True)
        self.crop_edges_check.setToolTip("Remove empty frames at the beginning and end caused by rolling average.")
        avg_layout.addWidget(self.crop_edges_check)
        
        filter_layout.addLayout(avg_layout)
        
        self.frame_loss_label = QLabel("")
        self.frame_loss_label.setStyleSheet("color: gray; font-size: 10px;")
        filter_layout.addWidget(self.frame_loss_label)

        # --- 线程数 ---
        thread_layout = QHBoxLayout()
        thread_layout.addWidget(QLabel("Workers:"))
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 32)
        self.max_workers_spin.setValue(8)
        thread_layout.addWidget(self.max_workers_spin)
        filter_layout.addLayout(thread_layout)

        # --- 应用按钮 ---
        self.apply_btn = QPushButton("Run Filters (Create Layer)")
        self.apply_btn.clicked.connect(self._apply_enhancement)
        filter_layout.addWidget(self.apply_btn)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # ==========================================
        # Part 2: 对比度调节 (仿 ImageJ)
        # ==========================================
        contrast_group = QGroupBox("2. Contrast & Brightness (Post-Process)")
        contrast_layout = QVBoxLayout()
        
        # 直方图画布
        self.hist_figure = Figure(figsize=(4, 2), dpi=100)
        self.hist_figure.patch.set_facecolor('#f0f0f0') # 浅灰背景
        self.hist_canvas = FigureCanvasQTAgg(self.hist_figure)
        self.hist_ax = self.hist_figure.add_subplot(111)
        # 移除四周留白
        self.hist_figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.hist_ax.axis('off') # 隐藏坐标轴，仿ImageJ简洁风格
        contrast_layout.addWidget(self.hist_canvas)
        
        # 控制项
        ctrl_layout = QHBoxLayout()
        
        # Min / Max Controls
        grid_layout = QVBoxLayout()
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Min:"))
        self.contrast_min_spin = QDoubleSpinBox()
        self.contrast_min_spin.setRange(-65535, 65535) # 适应不同位深
        self.contrast_min_spin.setDecimals(0)
        self.contrast_min_spin.valueChanged.connect(self._update_contrast_from_spin)
        row1.addWidget(self.contrast_min_spin)
        grid_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Max:"))
        self.contrast_max_spin = QDoubleSpinBox()
        self.contrast_max_spin.setRange(-65535, 65535)
        self.contrast_max_spin.setDecimals(0)
        self.contrast_max_spin.valueChanged.connect(self._update_contrast_from_spin)
        row2.addWidget(self.contrast_max_spin)
        grid_layout.addLayout(row2)
        
        ctrl_layout.addLayout(grid_layout)
        
        # Buttons
        btn_layout = QVBoxLayout()
        self.auto_contrast_btn = QPushButton("Auto")
        self.auto_contrast_btn.clicked.connect(self._auto_contrast)
        btn_layout.addWidget(self.auto_contrast_btn)
        
        self.reset_contrast_btn = QPushButton("Reset")
        self.reset_contrast_btn.clicked.connect(self._reset_contrast)
        btn_layout.addWidget(self.reset_contrast_btn)
        ctrl_layout.addLayout(btn_layout)
        
        contrast_layout.addLayout(ctrl_layout)
        
        # Burn 按钮 (将对比度固化到像素值)
        self.apply_contrast_btn = QPushButton("🔥 Apply (Burn to New Layer)")
        self.apply_contrast_btn.setToolTip("Create a new layer with pixel values clipped and rescaled based on current contrast settings.")
        self.apply_contrast_btn.clicked.connect(self._apply_contrast_burn)
        contrast_layout.addWidget(self.apply_contrast_btn)
        
        contrast_group.setLayout(contrast_layout)
        layout.addWidget(contrast_group)

        # ==========================================
        # Common: 进度条与状态
        # ==========================================
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { height: 10px; }")
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # 初始化
        self._refresh_layers()
        self._toggle_gaussian()
        self._toggle_average()
        self._update_frame_loss_info()

    # ... (Helper functions: _refresh_layers, _ensure_odd_ksize, etc. 保持不变) ...
    # --- 修复：自动同步逻辑 ---
    def _refresh_layers_silently(self, event=None):
        """仅刷新列表内容，保持当前选中项（如果还在）"""
        current_text = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True) # 暂时阻塞信号防止触发计算
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and isinstance(layer.data, np.ndarray):
                # 仅允许 3D (T,Y,X) 灰度图用于增强，RGB图通常不进行此类增强
                if layer.data.ndim == 3: 
                    self.layer_combo.addItem(layer.name)
        
        # 尝试恢复选中
        index = self.layer_combo.findText(current_text)
        if index >= 0:
            self.layer_combo.setCurrentIndex(index)
        self.layer_combo.blockSignals(False)

    def _on_active_layer_changed(self, event=None):
        """当 Napari 左侧图层被点击选中时，同步下拉框"""
        active_layer = self.viewer.layers.selection.active
        if active_layer:
            # 临时阻塞信号，只改变显示，不触发重算
            self.layer_combo.blockSignals(True)
            index = self.layer_combo.findText(active_layer.name)
            if index >= 0:
                self.layer_combo.setCurrentIndex(index)
            self.layer_combo.blockSignals(False)
            # 手动触发直方图更新
            self._update_histogram()

    # --- 修改：手动刷新按钮逻辑 ---
    def _refresh_layers(self):
        """手动刷新：刷新列表并选中当前激活图层"""
        self._refresh_layers_silently()
        self._on_active_layer_changed()

    def _ensure_odd_ksize(self):
        value = self.ksize_slider.value()
        if value % 2 == 0:
            value += 1
            self.ksize_slider.setValue(value)
        self.ksize_label.setText(str(value))
        # 更新sigma
        sigma = 0.3 * ((value - 1) * 0.5 - 1) + 0.8
        self.sigma_spin.setValue(sigma)

    def _toggle_gaussian(self):
        enabled = self.use_gaussian_check.isChecked()
        self.ksize_slider.setEnabled(enabled)
        self.sigma_spin.setEnabled(enabled)

    def _toggle_average(self):
        enabled = self.use_average_check.isChecked()
        self.window_spin.setEnabled(enabled)
        self.crop_edges_check.setEnabled(enabled)
        self._update_frame_loss_info()

    def _on_layer_changed(self, text):
        """当下拉框选择改变时，尝试更新对比度面板"""
        self._update_histogram()
        
    def _on_layer_selection_changed(self, event=None):
        """当napari图层列表选择改变时，同步更新"""
        active_layer = self.viewer.layers.selection.active
        if active_layer and active_layer.name != self.layer_combo.currentText():
            # 尝试在下拉框中选中当前激活的图层
            index = self.layer_combo.findText(active_layer.name)
            if index >= 0:
                self.layer_combo.setCurrentIndex(index)
        
        # 更新直方图
        self._update_histogram()

    def _update_frame_loss_info(self):
        """更新帧损失信息提示"""
        if not self.use_average_check.isChecked():
            self.frame_loss_label.setText("")
            return
        
        layer_name = self.layer_combo.currentText()
        if not layer_name or layer_name not in self.viewer.layers:
            return
        
        total_frames = len(self.viewer.layers[layer_name].data)
        window_size = self.window_spin.value()
        
        # 计算裁切量
        # 滚动平均通常导致两端各 (window-1)//2 帧无效
        cut_one_side = (window_size - 1) // 2
        lost_frames = cut_one_side * 2
        
        if self.crop_edges_check.isChecked():
            output_frames = total_frames - lost_frames
            self.frame_loss_label.setText(
                f"ℹ️ Cropping: {total_frames} → {output_frames} frames "
                f"(removing {cut_one_side} from start/end)"
            )
        else:
            self.frame_loss_label.setText(
                f"⚠️ Keeping Edges: {total_frames} frames (first/last {cut_one_side} will be black)"
            )

    # ============================
    # Core: Enhancement Logic
    # ============================

    def _get_enhancement_params(self):
        return {
            'use_gaussian': self.use_gaussian_check.isChecked(),
            'ksize': self.ksize_slider.value(),
            'sigma': self.sigma_spin.value(),
            'use_average': self.use_average_check.isChecked(),
            'average_window': self.window_spin.value(),
            'num_threads': self.max_workers_spin.value()
        }

    def _apply_enhancement(self):
        layer_name = self.layer_combo.currentText()
        if not layer_name: return
        
        image_stack = self.viewer.layers[layer_name].data
        params = self._get_enhancement_params()
        
        if not params['use_gaussian'] and not params['use_average']:
            self.status_label.setText("❌ Select at least one filter.")
            return

        # UI Update
        self.status_label.setText("⏳ Running filters...")
        self.progress_bar.setRange(0, 0) # 不确定模式
        self.progress_bar.setVisible(True)
        self.apply_btn.setEnabled(False)
        
        self.enhance_thread = EnhanceThread(image_stack, **params)
        self.enhance_thread.finished.connect(self._on_enhance_finished)
        self.enhance_thread.error.connect(self._on_enhance_error)
        self.enhance_thread.start()

    # --- 修复：操作完成后的自动切换 ---
    def _on_enhance_finished(self, enhanced_stack):
        self.progress_bar.setVisible(False)
        self.apply_btn.setEnabled(True)
        try:
            layer_name = self.layer_combo.currentText()
            notes = []
            
            # 检查帧数变化
            input_layer = self.viewer.layers[layer_name]
            input_frames = len(input_layer.data)
            output_frames = len(enhanced_stack)
            
            if output_frames < input_frames:
                loss = input_frames - output_frames
                notes.append(f"Len: {input_frames}->{output_frames}")

            if not enhanced_stack.flags['C_CONTIGUOUS']:
                enhanced_stack = np.ascontiguousarray(enhanced_stack)

            new_layer_name = f"Enh_{layer_name}"
            
            # 添加新图层
            new_layer = self.viewer.add_image(
                enhanced_stack,
                name=new_layer_name,
                colormap='gray'
            )
            
            msg = f"✅ Done. Layer: {new_layer_name}."
            if notes: msg += f" ({', '.join(notes)})"
            self.status_label.setText(msg)

            # === 关键修复：自动切换焦点并隐藏旧图层 ===
            # 隐藏原图层，避免因长度不一致导致的最后几帧黑屏/闪烁问题
            input_layer.visible = False 
            
            # 将焦点切换到新图层
            self.viewer.layers.selection.active = new_layer
            
            # 这里的自动同步会由 _on_active_layer_changed 触发，无需手动调用下拉框设置

        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")

    # def _switch_to_layer(self, layer_name):
    #     """辅助函数：切换焦点到指定图层"""
    #     if layer_name in self.viewer.layers:
    #         # 隐藏其他 Image 图层 (可选，根据用户习惯)
    #         for l in self.viewer.layers:
    #             if isinstance(l, napari.layers.Image):
    #                 l.visible = (l.name == layer_name)
    #         # 设置选中
    #         self.viewer.layers.selection.active = self.viewer.layers[layer_name]
    #         # 刷新下拉框
    #         self._refresh_layers()
    #         self.layer_combo.setCurrentText(layer_name)

    def _on_enhance_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.apply_btn.setEnabled(True)
        self.status_label.setText(f"❌ Filter Error: {error_msg}")

    def _update_layer_focus(self, new_layer_name):
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image) and layer.name != new_layer_name:
                layer.visible = False
        self.viewer.layers.selection.active = self.viewer.layers[new_layer_name]
        self._refresh_layers()
        self.layer_combo.setCurrentText(new_layer_name)

    # ============================
    # New: ImageJ-style Contrast
    # ============================

    def _update_histogram(self):
        """更新直方图和对比度控件"""
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, napari.layers.Image):
            self.hist_ax.clear()
            self.hist_ax.text(0.5, 0.5, "No Image Selected", ha='center', va='center')
            self.hist_canvas.draw()
            return
        
        # 1. 更新SpinBox的值为当前图层的 contrast_limits
        # 避免触发 valueChanged 信号导致循环调用
        self.contrast_min_spin.blockSignals(True)
        self.contrast_max_spin.blockSignals(True)
        
        c_min, c_max = layer.contrast_limits
        self.contrast_min_spin.setValue(c_min)
        self.contrast_max_spin.setValue(c_max)
        
        self.contrast_min_spin.blockSignals(False)
        self.contrast_max_spin.blockSignals(False)

        # 2. 计算直方图 (仅取当前帧以提高性能，或者降采样)
        # 如果数据很大，建议 layer.data[current_step]
        current_step = self.viewer.dims.current_step[0]
        if len(layer.data.shape) == 3 and current_step < len(layer.data):
            display_data = layer.data[current_step]
        else:
            display_data = layer.data
            
        # 简单的降采样以加快直方图计算
        if display_data.size > 1000000:
            sample = display_data.ravel()[::10] # 10倍降采样
        else:
            sample = display_data.ravel()
            
        # 3. 绘图
        self.hist_ax.clear()
        
        # 绘制灰度直方图
        # ImageJ风格：灰色填充
        self.hist_ax.hist(sample, bins=64, color='#888888', alpha=0.6, density=True)
        
        # 绘制转换曲线 (Gamma/Linear)
        # 这是一个示意的对角线，表示 Min -> Max 的映射
        # x轴范围
        d_min, d_max = np.min(sample), np.max(sample)
        # 稍微扩展视图以便看清Min/Max线
        view_min = min(d_min, c_min)
        view_max = max(d_max, c_max)
        pad = (view_max - view_min) * 0.1
        self.hist_ax.set_xlim(view_min - pad, view_max + pad)
        
        # 绘制 Min 和 Max 的垂直线
        self.hist_ax.axvline(c_min, color='blue', linestyle='--', linewidth=1, label='Min')
        self.hist_ax.axvline(c_max, color='red', linestyle='--', linewidth=1, label='Max')
        
        # 绘制斜线 (Map function)
        # 当 x <= Min 时 y=0 (bottom), 当 x >= Max 时 y=1 (top)
        x_vals = np.linspace(view_min - pad, view_max + pad, 100)
        # 简单的线性映射可视化
        y_vals = np.clip((x_vals - c_min) / (c_max - c_min + 1e-5), 0, 1)
        # 为了显示在直方图上，我们需要缩放y轴 (density直方图最大值通常较小)
        ylim = self.hist_ax.get_ylim()
        y_max_vis = ylim[1]
        self.hist_ax.plot(x_vals, y_vals * y_max_vis, color='black', linewidth=1.5)
        
        self.hist_canvas.draw()

    def _update_contrast_from_spin(self):
        """当SpinBox数值改变时，直接更新Napari显示"""
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, napari.layers.Image): return
        
        new_min = self.contrast_min_spin.value()
        new_max = self.contrast_max_spin.value()
        
        if new_min >= new_max:
            return # 防止错误
            
        # 实时更新 Napari 显示
        layer.contrast_limits = [new_min, new_max]
        
        # 实时更新直方图上的线
        self._update_histogram()

    def _auto_contrast(self):
        """ImageJ风格的Auto: 基于百分位数"""
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, napari.layers.Image): return
        
        data = layer.data
        current_lims = list(layer.contrast_limits)
        full_range = [data.min(), data.max()]
        # 为了速度，使用子采样
        if data.size > 1000000:
            sample = data.ravel()[::100]
        else:
            sample = data.ravel()
            
        # 排除 0 值 (对于荧光图像通常很有用，看情况)
        # sample = sample[sample > 0] 
        
        if len(sample) == 0: return

        # 1. 计算标准的 Auto 范围 (基准)
        auto_lims = np.percentile(sample, [0.04, 99.96])
        
        # 判断逻辑：
        # 如果当前范围比 auto_lims 还要宽（或者接近全范围），说明还没做过 Auto -> 执行标准 Auto
        # 如果当前范围已经在 auto_lims 附近或更窄 -> 执行“进一步收缩”
        
        range_width = current_lims[1] - current_lims[0]
        auto_width = auto_lims[1] - auto_lims[0]

        # 阈值判定：如果当前宽度明显大于标准Auto宽度的 1.1 倍，或者接近全范围
        is_fresh = (range_width > auto_width * 1.1) or \
                   (abs(current_lims[0] - full_range[0]) < 1e-5 and abs(current_lims[1] - full_range[1]) < 1e-5)

        if is_fresh:
            # 第一次点击：标准 Auto
            new_min, new_max = auto_lims
            self.status_label.setText("✨ Auto Contrast: Standard (0.04%-99.96%)")
        else:
            # 后续点击：累进增强 (Squeeze)
            # 每次向内收缩 5% (一共缩窄 10%)
            shrink_factor = 0.05 
            margin = range_width * shrink_factor
            new_min = current_lims[0] + margin
            new_max = current_lims[1] - margin
            
            # 边界保护：不能交叉
            if new_min >= new_max:
                new_min = (current_lims[0] + current_lims[1]) / 2 - 1e-5
                new_max = (current_lims[0] + current_lims[1]) / 2 + 1e-5
            
            self.status_label.setText(f"✨ Auto Contrast: Boosted (+{int(shrink_factor*100)}%)")

        # 更新 UI 和 图层
        self.contrast_min_spin.blockSignals(True)
        self.contrast_max_spin.blockSignals(True)
        self.contrast_min_spin.setValue(new_min)
        self.contrast_max_spin.setValue(new_max)
        self.contrast_min_spin.blockSignals(False)
        self.contrast_max_spin.blockSignals(False)
        
        layer.contrast_limits = [new_min, new_max]
        self._update_histogram()
        
    def _reset_contrast(self):
        """重置为数据全范围"""
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, napari.layers.Image): return
        
        data = layer.data
        c_min, c_max = data.min(), data.max()
        
        self.contrast_min_spin.setValue(c_min)
        self.contrast_max_spin.setValue(c_max)

    # --- 修复：对比度 Burn-in 后的自动切换 ---
    def _apply_contrast_burn(self):
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, napari.layers.Image): 
            self.status_label.setText("❌ No image selected.")
            return
            
        c_min = self.contrast_min_spin.value()
        c_max = self.contrast_max_spin.value()
        
        self.status_label.setText("⏳ Applying contrast...")
        self.progress_bar.setVisible(True)
        
        try:
            data = layer.data
            range_width = c_max - c_min
            if range_width < 1e-9: range_width = 1e-9
            
            normalized = (data - c_min) / range_width
            normalized = np.clip(normalized, 0, 1)
            
            dtype = data.dtype
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                burnt_data = (normalized * info.max).astype(dtype)
            else:
                burnt_data = normalized.astype(np.float32)
                
            new_layer_name = f"Contrast_{layer.name}"
            new_layer = self.viewer.add_image(
                burnt_data,
                name=new_layer_name,
                colormap='gray'
            )
            self.status_label.setText(f"✅ Applied. New layer: {new_layer_name}")
            
            # 自动切换：隐藏旧图层，激活新图层
            layer.visible = False
            self.viewer.layers.selection.active = new_layer
            
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
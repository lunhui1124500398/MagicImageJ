"""
图像增强控件 - 完整版
包含：高斯模糊、滚动平均（带进度条）、仿ImageJ对比度调节（直方图+曲线）
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QSpinBox, QHBoxLayout, QComboBox,
                            QCheckBox, QGroupBox, QDoubleSpinBox, QSlider,
                            QMessageBox, QProgressBar, QScrollArea)
from qtpy.QtCore import Qt, Signal, QThread, QSettings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from core.image_enhance import enhance_image_stack
import napari
import json
from pathlib import Path
import datetime
from widgets.settings_widget import GlobalConfig, tr
from utils.utils import elide_text
from utils.ui_utils import setup_safe_scroll_all

# JSON Encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)): return int(obj)
        elif isinstance(obj, (np.floating, float)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

class EnhanceThread(QThread):
    """增强处理线程 (带进度条)"""
    finished = Signal(np.ndarray)
    progress = Signal(int) # 0-100%
    error = Signal(str)
    def __init__(self, image_stack, **kwargs):
        super().__init__()
        self.image_stack = image_stack
        self.params = kwargs
    def run(self):
        try:
            def cb(percent):
                self.progress.emit(percent)
            
            result = enhance_image_stack(self.image_stack, **self.params, progress_callback=cb)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class ContrastBurnThread(QThread):
    progress = Signal(int)
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, data, c_min, c_max):
        super().__init__()
        self.data = data
        self.c_min = c_min
        self.c_max = c_max

    def run(self):
        try:
            self.progress.emit(10)
            # 1. 转换类型 (Float32) 以便计算
            # 如果数据量极大，这里也可以分块，但通常 numpy 整体运算够快，
            # 主要是搬移到子线程不卡 UI
            data_f = self.data.astype(np.float32, copy=False)
            
            self.progress.emit(30)
            range_width = self.c_max - self.c_min
            if range_width < 1e-9: range_width = 1e-9
            
            # 2. 归一化
            normalized = (data_f - self.c_min) / range_width
            self.progress.emit(60)
            
            # 3. Clip
            normalized = np.clip(normalized, 0, 1)
            self.progress.emit(80)
            
            # 4. 映射回 uint8 (通常 burn 都是为了可视化导出，转为 8bit 最通用)
            # 如果原图是 16bit，这里也可以保留，但通常 ImageJ 风格的 Adjust Contrast 
            # 意味着 "Apply Window/Level"，结果通常预期是视觉一致的 8bit。
            # 这里我们根据原图类型智能判断：如果是整数类型，通常映射回相同类型或 uint8
            
            dtype = self.data.dtype
            if np.issubdtype(dtype, np.integer):
                # 如果原图是整数，我们映射到该类型的最大值
                # 但通常 Apply Contrast 后的目的是为了看清楚，转 uint8 (0-255) 是最常用的
                # 这里为了稳妥，映射到 0-255 uint8，这也是 "Burn" 的通常含义
                res = (normalized * 255).astype(np.uint8)
            else:
                # 浮点图保持浮点
                res = normalized.astype(np.float32)
                
            self.progress.emit(100)
            self.finished.emit(res)
            
        except Exception as e:
            self.error.emit(str(e))

class EnhanceWidget(QWidget):
    """图像增强控件 - 完整版"""
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.enhance_thread = None
        self._setup_ui()
        self.viewer.layers.events.inserted.connect(self._refresh_layers_silently)
        self.viewer.layers.events.removed.connect(self._refresh_layers_silently)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)
        self._load_params_from_config()
        GlobalConfig.signals.config_updated.connect(self._load_params_from_config)

    def _setup_ui(self):
        # 1. 创建最外层布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        title = QLabel(f"<h3>✨ {tr('Enhancement')}</h3>")
        layout.addWidget(title)

        # 图层选择
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel(tr("Target Layer:")))
        self.layer_combo = QComboBox()
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)
        layer_layout.addWidget(self.layer_combo)
        layout.addLayout(layer_layout)
        
        refresh_btn = QPushButton(f"🔄 {tr('Refresh Layers')}")
        refresh_btn.clicked.connect(self._refresh_layers)
        layout.addWidget(refresh_btn)

        # ==========================================
        # Part 1: 滤波器设置 (Gaussian & Rolling)
        # ==========================================
        filter_group = QGroupBox(tr("1. Filters"))
        filter_layout = QVBoxLayout()
        
        # --- 高斯模糊 ---
        gaussian_layout = QHBoxLayout()
        self.use_gaussian_check = QCheckBox(tr("Gaussian Blur"))
        #self.use_gaussian_check.stateChanged.connect(self._toggle_gaussian)
        gaussian_layout.addWidget(self.use_gaussian_check)
        
        self.ksize_slider = QSlider(Qt.Horizontal)
        self.ksize_slider.setRange(1, 25)
        self.ksize_slider.setValue(3)
        self.ksize_slider.setSingleStep(2)
        self.ksize_slider.valueChanged.connect(self._ensure_odd_ksize)
        gaussian_layout.addWidget(QLabel(tr("Kernel:")))
        gaussian_layout.addWidget(self.ksize_slider)
        
        self.ksize_label = QLabel("3")
        gaussian_layout.addWidget(self.ksize_label)
        
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 10.0)
        self.sigma_spin.setValue(0.8)
        self.sigma_spin.setSingleStep(0.1)
        gaussian_layout.addWidget(QLabel(tr("Sigma:")))
        gaussian_layout.addWidget(self.sigma_spin)
        filter_layout.addLayout(gaussian_layout)

        # --- 滚动平均 ---
        avg_layout = QHBoxLayout()
        self.use_average_check = QCheckBox(tr("Roll Avg"))
        # self.use_average_check.stateChanged.connect(self._toggle_average)
        avg_layout.addWidget(self.use_average_check)
        avg_layout.addWidget(QLabel(tr("Win:")))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(3, 51)
        self.window_spin.setSingleStep(2)
        self.window_spin.setValue(3)
        self.window_spin.valueChanged.connect(self._update_frame_loss_info)
        avg_layout.addWidget(self.window_spin)
        
        self.frame_loss_label = QLabel("")
        self.frame_loss_label.setStyleSheet("color: gray; font-size: 10px;")
        avg_layout.addWidget(self.frame_loss_label)
        filter_layout.addLayout(avg_layout)

        # --- 线程数 ---
        thread_layout = QHBoxLayout()
        thread_layout.addWidget(QLabel(tr("Workers:")))
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 32)
        self.max_workers_spin.setValue(8)
        thread_layout.addWidget(self.max_workers_spin)
        filter_layout.addLayout(thread_layout)

        # --- 应用按钮 ---
        self.apply_btn = QPushButton(tr("Run Filters (Create Layer)"))
        self.apply_btn.clicked.connect(self._apply_enhancement)
        filter_layout.addWidget(self.apply_btn)
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # ==========================================
        # Part 2: 对比度调节 (仿 ImageJ)
        # ==========================================
        contrast_group = QGroupBox(tr("2. Contrast & Brightness (Post-Process)"))
        contrast_layout = QVBoxLayout()
        
        self.hist_figure = Figure(figsize=(4, 2), dpi=100)
        self.hist_figure.patch.set_facecolor('#f0f0f0')
        self.hist_canvas = FigureCanvasQTAgg(self.hist_figure)
        self.hist_ax = self.hist_figure.add_subplot(111)
        self.hist_figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.hist_ax.axis('off')
        contrast_layout.addWidget(self.hist_canvas)

        ctrl_layout = QHBoxLayout()
        grid_layout = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Min:"))
        self.contrast_min_spin = QDoubleSpinBox()
        self.contrast_min_spin.setRange(-65535, 65535)
        self.contrast_min_spin.setDecimals(0)
        self.contrast_min_spin.valueChanged.connect(self._update_contrast_from_spin)
        row1.addWidget(self.contrast_min_spin)
        grid_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel(tr("Max:")))
        self.contrast_max_spin = QDoubleSpinBox()
        self.contrast_max_spin.setRange(-65535, 65535)
        self.contrast_max_spin.setDecimals(0)
        self.contrast_max_spin.valueChanged.connect(self._update_contrast_from_spin)
        row2.addWidget(self.contrast_max_spin)
        grid_layout.addLayout(row2)
        ctrl_layout.addLayout(grid_layout)

        btn_layout = QVBoxLayout()
        self.auto_contrast_btn = QPushButton(tr("Auto"))
        self.auto_contrast_btn.clicked.connect(self._auto_contrast)
        btn_layout.addWidget(self.auto_contrast_btn)
        self.reset_contrast_btn = QPushButton(tr("Reset"))
        self.reset_contrast_btn.clicked.connect(self._reset_contrast)
        btn_layout.addWidget(self.reset_contrast_btn)
        ctrl_layout.addLayout(btn_layout)
        contrast_layout.addLayout(ctrl_layout)

        self.apply_contrast_btn = QPushButton(f"🔥 {tr('Apply (Burn to New Layer)')}")
        self.apply_contrast_btn.clicked.connect(self._apply_contrast_burn)
        contrast_layout.addWidget(self.apply_contrast_btn)
        contrast_group.setLayout(contrast_layout)
        layout.addWidget(contrast_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch()
        content_widget.setLayout(layout)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
        self._refresh_layers()
        self._toggle_gaussian()
        self._toggle_average()
        self._update_frame_loss_info()

        # 绑定保存逻辑：当数值改变时，立即保存到 GlobalConfig
        # ==========================================
        self.use_gaussian_check.stateChanged.connect(lambda v: (self._toggle_gaussian(), GlobalConfig.set("enh_use_gaussian", bool(v))))
        self.use_average_check.stateChanged.connect(lambda v: (self._toggle_average(), GlobalConfig.set("enh_use_average", bool(v))))
        self.sigma_spin.valueChanged.connect(lambda v: GlobalConfig.set("enh_sigma", v))
        self.window_spin.valueChanged.connect(lambda v: GlobalConfig.set("enh_window", v))
        self.max_workers_spin.valueChanged.connect(lambda v: GlobalConfig.set("enh_workers", v))
        
        # [新增] 防止滚轮误触更改数值
        setup_safe_scroll_all(
            self.sigma_spin, self.window_spin, self.max_workers_spin,
            self.contrast_min_spin, self.contrast_max_spin, self.layer_combo
        )

    def _refresh_layers_silently(self, event=None):
        current_data = self.layer_combo.currentData()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and isinstance(layer.data, np.ndarray):
                if layer.data.ndim == 3: 
                    short = elide_text(layer.name, 25)
                    self.layer_combo.addItem(short, layer.name)
                    self.layer_combo.setItemData(self.layer_combo.count()-1, layer.name, Qt.ToolTipRole)
        
        # 优先选择激活图层
        index_set = False
        active_layer = self.viewer.layers.selection.active
        if active_layer:
            index = self.layer_combo.findData(active_layer.name)
            if index >= 0:
                self.layer_combo.setCurrentIndex(index)
                index_set = True
        
        if not index_set and current_data:
            index = self.layer_combo.findData(current_data)
            if index >= 0: self.layer_combo.setCurrentIndex(index)
        self.layer_combo.blockSignals(False)

    def _on_active_layer_changed(self, event=None):
        active_layer = self.viewer.layers.selection.active
        if active_layer:
            self.layer_combo.blockSignals(True)
            index = self.layer_combo.findData(active_layer.name)
            if index >= 0: self.layer_combo.setCurrentIndex(index)
            self.layer_combo.blockSignals(False)
            self._update_histogram()

    def _refresh_layers(self):
        self._refresh_layers_silently()
        self._on_active_layer_changed()

    def _ensure_odd_ksize(self):
        value = self.ksize_slider.value()
        if value % 2 == 0:
            value += 1
            self.ksize_slider.setValue(value)
        self.ksize_label.setText(str(value))
        sigma = 0.3 * ((value - 1) * 0.5 - 1) + 0.8
        self.sigma_spin.setValue(sigma)

    def _toggle_gaussian(self):
        enabled = self.use_gaussian_check.isChecked()
        self.ksize_slider.setEnabled(enabled)
        self.sigma_spin.setEnabled(enabled)

    def _toggle_average(self):
        enabled = self.use_average_check.isChecked()
        self.window_spin.setEnabled(enabled)
        self._update_frame_loss_info()

    def _on_layer_changed(self, text):
        self._update_histogram()

    def _update_frame_loss_info(self):
        if not self.use_average_check.isChecked():
            self.frame_loss_label.setText("")
            return
        layer_name = self.layer_combo.currentData()
        if not layer_name or layer_name not in self.viewer.layers: return
        total_frames = len(self.viewer.layers[layer_name].data)
        window_size = self.window_spin.value()
        # Core逻辑: output = T - window + 1
        output_frames = max(0, total_frames - window_size + 1)
        self.frame_loss_label.setText(tr("Output: %s frames (Loss: %s)") % (output_frames, window_size-1))

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
        layer_name = self.layer_combo.currentData()
        if not layer_name: return
        image_stack = self.viewer.layers[layer_name].data
        params = self._get_enhancement_params()
        if not params['use_gaussian'] and not params['use_average']:
            self.status_label.setText(f"❌ {tr('Select at least one filter.')}")
            return
        
        self.status_label.setText(f"⏳ {tr('Running filters...')}")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.apply_btn.setEnabled(False)
        
        self.enhance_thread = EnhanceThread(image_stack, **params)
        self.enhance_thread.progress.connect(self.progress_bar.setValue)
        self.enhance_thread.finished.connect(lambda res: self._on_enhance_finished(res, params))
        self.enhance_thread.error.connect(self._on_enhance_error)
        self.enhance_thread.start()

    def _on_enhance_finished(self, enhanced_stack, params):
        self.progress_bar.setVisible(False)
        self.apply_btn.setEnabled(True)
        try:
            layer_name = self.layer_combo.currentData()
            input_layer = self.viewer.layers[layer_name]
            
            if not enhanced_stack.flags['C_CONTIGUOUS']:
                enhanced_stack = np.ascontiguousarray(enhanced_stack)
            new_layer_name = f"Enh_{layer_name}"
            new_layer = self.viewer.add_image(enhanced_stack, name=new_layer_name, colormap='gray', metadata={'source_layer': layer_name})
            
            # === Log params and save action_id ===
            action_id = self._log_action("filter_enhancement", {
                "source": layer_name,
                "params": params
            })
            new_layer.metadata['action_id'] = action_id

            self.status_label.setText(f"✅ {tr('Done. Layer: %s.') % new_layer_name}")
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].visible = False
            self.viewer.layers.selection.active = new_layer

            # [新增] 绑定 Undo
            undo_key = GlobalConfig.get_napari_shortcut("shortcut_undo_drift")
            @new_layer.bind_key(undo_key, overwrite=True)
            def undo_enhance(layer):
                # 1. 恢复原图层
                src = layer.metadata.get('source_layer')
                if src and src in self.viewer.layers:
                    self.viewer.layers[src].visible = True
                    self.viewer.layers.selection.active = self.viewer.layers[src]
                # 2. 删除当前层
                self.viewer.layers.remove(layer)
                self.status_label.setText(f"↩️ {tr('Enhancement Undone.')}")
                # 3. 记录撤回
                aid = layer.metadata.get('action_id')
                if aid:
                    try:
                        from utils.session_logger import get_logger
                        get_logger().log_undo(aid)
                    except: pass
        except Exception as e:
            self.status_label.setText(f"❌ {tr('Error:')} {str(e)}")

    def _on_enhance_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.apply_btn.setEnabled(True)
        self.status_label.setText(f"❌ {tr('Filter Error:')} {error_msg}")

    def _update_histogram(self):
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, napari.layers.Image):
            self.hist_ax.clear()
            self.hist_ax.text(0.5, 0.5, "No Image", ha='center')
            self.hist_canvas.draw()
            return
        
        self.contrast_min_spin.blockSignals(True)
        self.contrast_max_spin.blockSignals(True)
        c_min, c_max = layer.contrast_limits
        self.contrast_min_spin.setValue(c_min)
        self.contrast_max_spin.setValue(c_max)
        self.contrast_min_spin.blockSignals(False)
        self.contrast_max_spin.blockSignals(False)
        
        current_step = self.viewer.dims.current_step[0]
        if len(layer.data.shape) == 3 and current_step < len(layer.data):
            display_data = layer.data[current_step]
        else:
            display_data = layer.data
        
        if display_data.size > 1000000: sample = display_data.ravel()[::100]
        else: sample = display_data.ravel()
        
        self.hist_ax.clear()
        self.hist_ax.hist(sample, bins=64, color='#888888', alpha=0.6, density=True)
        d_min, d_max = np.min(sample), np.max(sample)
        view_min, view_max = min(d_min, c_min), max(d_max, c_max)
        pad = (view_max - view_min) * 0.1
        self.hist_ax.set_xlim(view_min - pad, view_max + pad)
        self.hist_ax.axvline(c_min, color='blue', linestyle='--')
        self.hist_ax.axvline(c_max, color='red', linestyle='--')
        self.hist_canvas.draw()

    def _update_contrast_from_spin(self):
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, napari.layers.Image): return
        new_min = self.contrast_min_spin.value()
        new_max = self.contrast_max_spin.value()
        if new_min >= new_max: return
        layer.contrast_limits = [new_min, new_max]
        self._update_histogram()

    def _auto_contrast(self):
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, napari.layers.Image): return
        data = layer.data
        if data.size > 1000000: sample = data.ravel()[::100]
        else: sample = data.ravel()
        if len(sample) == 0: return
        
        current_lims = list(layer.contrast_limits)
        auto_lims = np.percentile(sample, [0.04, 99.96])
        range_width = current_lims[1] - current_lims[0]
        auto_width = auto_lims[1] - auto_lims[0]
        
        if (range_width > auto_width * 1.1):
            new_min, new_max = auto_lims
        else:
            shrink = 0.05 
            margin = range_width * shrink
            new_min = current_lims[0] + margin
            new_max = current_lims[1] - margin
            if new_min >= new_max:
                new_min = (current_lims[0]+current_lims[1])/2 - 1e-5
                new_max = (current_lims[0]+current_lims[1])/2 + 1e-5
        
        self.contrast_min_spin.setValue(new_min)
        self.contrast_max_spin.setValue(new_max)
        layer.contrast_limits = [new_min, new_max]
        self._update_histogram()

    def _reset_contrast(self):
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, napari.layers.Image): return
        data = layer.data
        self.contrast_min_spin.setValue(data.min())
        self.contrast_max_spin.setValue(data.max())

    def _apply_contrast_burn(self):
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, napari.layers.Image): 
            self.status_label.setText(f"❌ {tr('No image selected.')}")
            return
        c_min = self.contrast_min_spin.value()
        c_max = self.contrast_max_spin.value()
        self.status_label.setText(f"⏳ {tr('Applying contrast...')}")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.apply_contrast_btn.setEnabled(False)

        self.burn_thread = ContrastBurnThread(layer.data, c_min, c_max)
        self.burn_thread.progress.connect(self.progress_bar.setValue)
        self.burn_thread.finished.connect(lambda res: self._on_burn_finished(res, layer, c_min, c_max))
        self.burn_thread.error.connect(lambda err: self._on_enhance_error(err)) # 复用错误处理
        self.burn_thread.start()
        
        # try:
        #     data = layer.data
        #     range_width = c_max - c_min
        #     if range_width < 1e-9: range_width = 1e-9
        #     normalized = (data - c_min) / range_width
        #     normalized = np.clip(normalized, 0, 1)
        #     dtype = data.dtype
            
        #     # 映射回原类型范围 (如 uint8 0-255)
        #     if np.issubdtype(dtype, np.integer):
        #         info = np.iinfo(dtype)
        #         burnt_data = (normalized * info.max).astype(dtype)
        #     else:
        #         burnt_data = normalized.astype(np.float32)
            
        #     new_layer_name = f"Contrast_{layer.name}"
        #     new_layer = self.viewer.add_image(burnt_data, name=new_layer_name, colormap='gray')
            
        #     self._log_action("contrast_adjustment", {
        #         "source": layer.name,
        #         "min": c_min,
        #         "max": c_max,
        #         "clipped": True
        #     })

        #     self.status_label.setText(f"✅ Applied. New layer: {new_layer_name}")
        #     layer.visible = False
        #     self.viewer.layers.selection.active = new_layer
        # except Exception as e:
        #     self.status_label.setText(f"❌ Error: {str(e)}")
        # finally:
        #     self.progress_bar.setVisible(False)
    def _on_burn_finished(self, burnt_data, original_layer, c_min, c_max):
        self.progress_bar.setVisible(False)
        self.apply_contrast_btn.setEnabled(True)
        try:
            new_layer_name = f"Contrast_{original_layer.name}"
            
            # 添加新图层
            # [新增] 绑定 source_layer 元数据以便 Undo
            new_layer = self.viewer.add_image(
                burnt_data, 
                name=new_layer_name, 
                colormap='gray',
                metadata={'source_layer': original_layer.name}
            )
            
            # 记录日志并保存 action_id
            action_id = self._log_action("contrast_adjustment", {
                "source": original_layer.name,
                "min": c_min,
                "max": c_max,
                "clipped": True
            })
            new_layer.metadata['action_id'] = action_id

            self.status_label.setText(f"✅ {tr('Applied. New layer: %s') % new_layer_name}")
            
            # 隐藏原图层
            original_layer.visible = False
            self.viewer.layers.selection.active = new_layer
            
            # [新增] 绑定 Undo
            from widgets.settings_widget import GlobalConfig
            undo_key = GlobalConfig.get_napari_shortcut("shortcut_undo_drift")
            @new_layer.bind_key(undo_key, overwrite=True)
            def undo_burn(layer):
                src = layer.metadata.get('source_layer')
                if src and src in self.viewer.layers:
                    self.viewer.layers[src].visible = True
                    self.viewer.layers.selection.active = self.viewer.layers[src]
                self.viewer.layers.remove(layer)
                self.status_label.setText(f"↩️ {tr('Contrast Undo.')}")
                # 记录撤回
                aid = layer.metadata.get('action_id')
                if aid:
                    try:
                        from utils.session_logger import get_logger
                        get_logger().log_undo(aid)
                    except: pass

        except Exception as e:
            self.status_label.setText(f"❌ {tr('Error:')} {str(e)}")

    def _log_action(self, key, info):
        """使用 SessionLogger 记录操作，并返回 action_id"""
        try:
            from utils.session_logger import get_logger
            return get_logger().log_action("enhance", key, info)
        except Exception as e:
            print(f"Log error: {e}")
            return None
    
    def _load_params_from_config(self):
        self.use_gaussian_check.blockSignals(True)
        self.use_average_check.blockSignals(True)
        self.sigma_spin.blockSignals(True)
        self.window_spin.blockSignals(True)
        self.max_workers_spin.blockSignals(True)

        self.use_gaussian_check.setChecked(bool(GlobalConfig.get("enh_use_gaussian")))
        self.use_average_check.setChecked(bool(GlobalConfig.get("enh_use_average")))
        self.sigma_spin.setValue(float(GlobalConfig.get("enh_sigma")))
        self.window_spin.setValue(int(GlobalConfig.get("enh_window")))
        self.max_workers_spin.setValue(int(GlobalConfig.get("enh_workers")))

        self.use_gaussian_check.blockSignals(False)
        self.use_average_check.blockSignals(False)
        self.sigma_spin.blockSignals(False)
        self.window_spin.blockSignals(False)
        self.max_workers_spin.blockSignals(False)
        
        # Refresh UI state (enable/disable)
        self._toggle_gaussian()
        self._toggle_average()
"""
Napari 标注工具 - 终极交互版 (Lazy Render & Smart Snapping)
Version: 7.2

更新日志 v7.2:
1. [性能] 懒加载机制 (Lazy Rendering)：
   - 初始化时不再渲染所有帧，仅创建数据结构。
   - 滚动时间轴时实时渲染当前帧 (On-the-fly)，极大提升大图操作流畅度。
2. [交互] 智能边缘吸附 (Smart Snapping)：
   - 快捷位置按钮现在会计算文字/比例尺的真实包围盒。
   - 自动计算坐标，使其紧贴图像边缘 (默认 20px 间距)，避免“不够靠边”或“出界”。
3. [UI] 优化快捷键图标和布局。
"""

from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QSpinBox, QHBoxLayout, QComboBox,
                            QCheckBox, QGroupBox, QDoubleSpinBox, QLineEdit,
                            QColorDialog, QTabWidget, QMessageBox, QSlider, QGridLayout,
                            QProgressDialog, QApplication)
from qtpy.QtCore import Qt, QTimer, QSettings
from qtpy.QtGui import QColor
import numpy as np
import napari
import cv2
from PIL import Image, ImageDraw, ImageFont
import os

class AnnotationWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        
        self.settings = QSettings("NapariUser", "AnnotationParams") # 初始化 Settings
        # === 图层句柄 ===
        self.preview_overlay_layer = None  
        self.interaction_layer = None      
        
        self._updating = False 
        self._dragging = False 
        
        self.current_source_layer = None
        self.interaction_map = {} 
        
        # 监听维度变化 (滚动滑条时触发懒加载)
        self.viewer.dims.events.current_step.connect(self._on_frame_change)
        
        self._setup_ui()
        # 监听激活图层，自动选中
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)
    
    def _on_active_layer_changed(self, event=None):
        """当外部选中图层时同步"""
        active_layer = self.viewer.layers.selection.active
        if active_layer:
            idx = self.layer_combo.findText(active_layer.name)
            if idx >= 0:
                self.layer_combo.setCurrentIndex(idx)

    def _setup_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("<h3>📏 ImageJ-Style Annotation (Lazy & Smart)</h3>")
        layout.addWidget(title)
        
        # 图层选择
        layer_box = QHBoxLayout()
        layer_box.addWidget(QLabel("Source:"))
        self.layer_combo = QComboBox()
        self.layer_combo.currentTextChanged.connect(self._on_layer_selected)
        layer_box.addWidget(self.layer_combo)
        layout.addLayout(layer_box)
        
        btn_refresh = QPushButton("🔄 Refresh Layers")
        btn_refresh.clicked.connect(self._refresh_layers)
        layout.addWidget(btn_refresh)
        
        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_scale_bar_tab(), "📏 Scale Bar")
        self.tabs.addTab(self._create_label_tab(), "🏷️ Label")
        layout.addWidget(self.tabs)
        
        # 底部按钮
        btn_box = QVBoxLayout()
        
        self.btn_preview = QPushButton("👁️ Initialize / Reset Preview")
        self.btn_preview.clicked.connect(self._create_preview)
        self.btn_preview.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        btn_box.addWidget(self.btn_preview)
        
        self.btn_burn = QPushButton("🔥 Burn-in to New Layer")
        self.btn_burn.clicked.connect(self._apply_to_new_layer)
        self.btn_burn.setStyleSheet("background-color: #FF5722; color: white; font-weight: bold; padding: 8px;")
        btn_box.addWidget(self.btn_burn)
        
        self.btn_clear = QPushButton("🗑️ Clear All")
        self.btn_clear.clicked.connect(self._clear_preview)
        btn_box.addWidget(self.btn_clear)
        
        layout.addLayout(btn_box)
        
        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
        self._refresh_layers()

    def _create_scale_bar_tab(self):
        w = QWidget()
        l = QVBoxLayout()
        
        # Enable
        h_top = QHBoxLayout()
        self.use_scale_bar_check = QCheckBox("Enable")
        self.use_scale_bar_check.setChecked(True)
        self.use_scale_bar_check.stateChanged.connect(self._on_ui_param_change)
        h_top.addWidget(self.use_scale_bar_check)
        h_top.addStretch()
        l.addLayout(h_top)
        
        # Quick Position Buttons (Smart Snap)
        g_quick = QGroupBox("Quick Snap")
        l_quick = QGridLayout()
        # 使用更直观的图标
        btn_ul = QPushButton("◤ Top-Left"); btn_ul.clicked.connect(lambda: self._snap_pos('scale', 'TL'))
        btn_ur = QPushButton("Top-Right ◥"); btn_ur.clicked.connect(lambda: self._snap_pos('scale', 'TR'))
        btn_ll = QPushButton("◣ Bottom-Left"); btn_ll.clicked.connect(lambda: self._snap_pos('scale', 'BL'))
        btn_lr = QPushButton("Bottom-Right ◢"); btn_lr.clicked.connect(lambda: self._snap_pos('scale', 'BR'))
        l_quick.addWidget(btn_ul, 0, 0); l_quick.addWidget(btn_ur, 0, 1)
        l_quick.addWidget(btn_ll, 1, 0); l_quick.addWidget(btn_lr, 1, 1)
        g_quick.setLayout(l_quick)
        l.addWidget(g_quick)

        # Ratio
        g_ratio = QGroupBox("Scale Ratio")
        l_ratio = QHBoxLayout()
        l_ratio.addWidget(QLabel("1 px ="))
        self.scale_ratio_spin = QDoubleSpinBox()
        self.scale_ratio_spin.setRange(0.000001, 100000.0)
        self.scale_ratio_spin.setValue(0.36) 
        self.scale_ratio_spin.setDecimals(6)
        self.scale_ratio_spin.valueChanged.connect(self._on_ui_param_change)
        l_ratio.addWidget(self.scale_ratio_spin)
        
        self.scale_unit_edit = QLineEdit("nm")
        self.scale_unit_edit.setFixedWidth(60)
        self.scale_unit_edit.textChanged.connect(self._on_ui_param_change)
        l_ratio.addWidget(self.scale_unit_edit)
        g_ratio.setLayout(l_ratio)
        l.addWidget(g_ratio)
        
        # Appearance
        g_app = QGroupBox("Appearance")
        l_app = QVBoxLayout()
        
        h_len = QHBoxLayout()
        h_len.addWidget(QLabel("Bar Length:"))
        self.scale_length_spin = QDoubleSpinBox()
        self.scale_length_spin.setRange(0.001, 1000000)
        self.scale_length_spin.setValue(100.0) 
        self.scale_length_spin.setDecimals(2)
        self.scale_length_spin.setSuffix(" unit") 
        self.scale_length_spin.valueChanged.connect(self._on_ui_param_change)
        h_len.addWidget(self.scale_length_spin)
        l_app.addLayout(h_len)

        self.auto_size_check = QCheckBox("Auto BG Size (Smart)")
        self.auto_size_check.setChecked(True)
        self.auto_size_check.stateChanged.connect(self._on_auto_size_toggled)
        l_app.addWidget(self.auto_size_check)

        def add_spin(label, min_v, max_v, val):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            s = QSpinBox(); s.setRange(min_v, max_v); s.setValue(val)
            s.valueChanged.connect(self._on_ui_param_change)
            h.addWidget(s)
            l_app.addLayout(h)
            return s
            
        self.scale_thickness_spin = add_spin("Bar Thickness:", 1, 100, 8)
        self.scale_font_spin = add_spin("Font Size (pt):", 6, 500, 36)
        self.scale_padding_spin = add_spin("Padding:", 0, 100, 10)
        self.scale_height_spin = add_spin("BG Height (px):", 10, 1000, 80)
        
        self._on_auto_size_toggled()

        h_col = QHBoxLayout()
        self.scale_color_btn = QPushButton("Text/Bar Color"); self.scale_color_btn.clicked.connect(self._choose_scale_color)
        self.scale_bg_color_btn = QPushButton("BG Color"); self.scale_bg_color_btn.clicked.connect(self._choose_scale_bg_color)
        h_col.addWidget(self.scale_color_btn); h_col.addWidget(self.scale_bg_color_btn)
        l_app.addLayout(h_col)
        
        self.scale_use_bg_check = QCheckBox("Show Background")
        self.scale_use_bg_check.setChecked(True)
        self.scale_use_bg_check.stateChanged.connect(self._on_ui_param_change)
        l_app.addWidget(self.scale_use_bg_check)
        
        h_alpha = QHBoxLayout()
        h_alpha.addWidget(QLabel("BG Opacity:"))
        self.scale_bg_alpha_slider = QSlider(Qt.Horizontal); self.scale_bg_alpha_slider.setRange(0, 100)
        self.scale_bg_alpha_slider.setValue(100)
        self.scale_bg_alpha_val_label = QLabel("100%")
        self.scale_bg_alpha_slider.valueChanged.connect(lambda v: (self.scale_bg_alpha_val_label.setText(f"{v}%"), self._on_ui_param_change()))
        h_alpha.addWidget(self.scale_bg_alpha_slider)
        h_alpha.addWidget(self.scale_bg_alpha_val_label)
        l_app.addLayout(h_alpha)
        g_app.setLayout(l_app)
        l.addWidget(g_app)
        
        # Position (Hidden)
        self.scale_x_spin = QSpinBox(); self.scale_x_spin.setRange(-9999, 99999); self.scale_x_spin.setValue(50)
        self.scale_y_spin = QSpinBox(); self.scale_y_spin.setRange(-9999, 99999); self.scale_y_spin.setValue(50)
        self.scale_width_spin = QSpinBox(); self.scale_width_spin.setRange(10, 99999); self.scale_width_spin.setValue(100)
        self.scale_x_spin.valueChanged.connect(self._on_ui_param_change)
        self.scale_y_spin.valueChanged.connect(self._on_ui_param_change)
        
        l.addStretch()
        w.setLayout(l)
        
        self.scale_color = (1.0, 1.0, 1.0, 1.0)
        self.scale_bg_color = (0.0, 0.0, 0.0)
        self._update_btn_style(self.scale_color_btn, self.scale_color)
        self._update_btn_style(self.scale_bg_color_btn, self.scale_bg_color)
        return w

    def _create_label_tab(self):
        w = QWidget()
        l = QVBoxLayout()
        
        h_top = QHBoxLayout()
        self.use_label_check = QCheckBox("Enable Label")
        self.use_label_check.setChecked(True)
        self.use_label_check.stateChanged.connect(self._on_ui_param_change)
        h_top.addWidget(self.use_label_check)
        h_top.addStretch()
        l.addLayout(h_top)
        
        # Quick Pos
        g_quick = QGroupBox("Quick Snap")
        l_quick = QGridLayout()
        btn_ul = QPushButton("◤ Top-Left"); btn_ul.clicked.connect(lambda: self._snap_pos('label', 'TL'))
        btn_ur = QPushButton("Top-Right ◥"); btn_ur.clicked.connect(lambda: self._snap_pos('label', 'TR'))
        btn_ll = QPushButton("◣ Bottom-Left"); btn_ll.clicked.connect(lambda: self._snap_pos('label', 'BL'))
        btn_lr = QPushButton("Bottom-Right ◢"); btn_lr.clicked.connect(lambda: self._snap_pos('label', 'BR'))
        l_quick.addWidget(btn_ul, 0, 0); l_quick.addWidget(btn_ur, 0, 1)
        l_quick.addWidget(btn_ll, 1, 0); l_quick.addWidget(btn_lr, 1, 1)
        g_quick.setLayout(l_quick)
        l.addWidget(g_quick)
        
        g_fmt = QGroupBox("Format")
        l_fmt = QVBoxLayout()
        self.label_format_combo = QComboBox()
        self.label_format_combo.addItems(["00:00", "0", "0.0", "0.00", "Custom"]) 
        self.label_format_combo.currentTextChanged.connect(self._on_ui_param_change)
        self.label_custom_edit = QLineEdit("{:.0f}"); self.label_custom_edit.setVisible(False)
        self.label_custom_edit.textChanged.connect(self._on_ui_param_change)
        self.label_format_combo.currentTextChanged.connect(lambda t: self.label_custom_edit.setVisible(t == "Custom"))
        l_fmt.addWidget(self.label_format_combo)
        l_fmt.addWidget(self.label_custom_edit)
        
        h_val = QHBoxLayout()
        self.label_start_spin = QDoubleSpinBox(); self.label_start_spin.setRange(-1e6, 1e6); self.label_start_spin.setDecimals(6)
        self.label_interval_spin = QDoubleSpinBox(); self.label_interval_spin.setRange(-1e6, 1e6)
        self.label_interval_spin.setDecimals(6)
        self.label_interval_spin.setValue(0.1594) 
        
        self.label_start_spin.valueChanged.connect(self._on_ui_param_change)
        self.label_interval_spin.valueChanged.connect(self._on_ui_param_change)
        h_val.addWidget(QLabel("Start:")); h_val.addWidget(self.label_start_spin)
        h_val.addWidget(QLabel("Step:")); h_val.addWidget(self.label_interval_spin)
        l_fmt.addLayout(h_val)
        g_fmt.setLayout(l_fmt)
        l.addWidget(g_fmt)
        
        g_app = QGroupBox("Appearance")
        l_app = QVBoxLayout()
        
        h_font = QHBoxLayout()
        h_font.addWidget(QLabel("Font Size (pt):"))
        self.label_font_spin = QSpinBox(); self.label_font_spin.setRange(6, 500); self.label_font_spin.setValue(32)
        self.label_font_spin.valueChanged.connect(self._on_ui_param_change)
        h_font.addWidget(self.label_font_spin)
        l_app.addLayout(h_font)
        
        self.label_color_btn = QPushButton("Text Color"); self.label_color_btn.clicked.connect(self._choose_label_color)
        l_app.addWidget(self.label_color_btn)
        g_app.setLayout(l_app)
        l.addWidget(g_app)
        
        # Hidden Pos
        self.label_x_spin = QSpinBox(); self.label_x_spin.setRange(-9999, 99999); self.label_x_spin.setValue(10)
        self.label_y_spin = QSpinBox(); self.label_y_spin.setRange(-9999, 99999); self.label_y_spin.setValue(40)
        self.label_x_spin.valueChanged.connect(self._on_ui_param_change)
        self.label_y_spin.valueChanged.connect(self._on_ui_param_change)
        
        l.addStretch()
        w.setLayout(l)
        
        self.label_color = (1.0, 1.0, 1.0, 1.0)
        self._update_btn_style(self.label_color_btn, self.label_color)
        return w

    # ========== Logic: Smart Auto-Calc ==========

    def _on_auto_size_toggled(self):
        """切换自动计算模式"""
        is_auto = self.auto_size_check.isChecked()
        self.scale_height_spin.setReadOnly(is_auto)
        self.scale_padding_spin.setReadOnly(is_auto)
        self.scale_height_spin.setEnabled(not is_auto)
        self.scale_padding_spin.setEnabled(not is_auto)
        if is_auto: self._perform_auto_calc()
            
    def _perform_auto_calc(self):
        """执行智能尺寸计算"""
        if not self.auto_size_check.isChecked(): return
        font_size = self.scale_font_spin.value()
        thickness = self.scale_thickness_spin.value()
        padding = max(4, int(font_size * 0.3))
        gap = max(2, int(font_size * 0.2))
        bg_height = padding + font_size + gap + thickness + padding
        self._updating = True
        self.scale_padding_spin.setValue(padding)
        self.scale_height_spin.setValue(bg_height)
        self._updating = False

    def _get_pixel_width(self):
        """根据物理长度(unit)计算像素宽度(px)"""
        length_unit = self.scale_length_spin.value()
        ratio = self.scale_ratio_spin.value()
        if ratio == 0: return 100
        return int(length_unit / ratio)

    # ========== Logic: Pillow Drawing ==========
    
    def _get_font(self, size):
        try: return ImageFont.truetype("arial.ttf", size)
        except:
            try: return ImageFont.truetype("DejaVuSans.ttf", size)
            except: return ImageFont.load_default()

    def _draw_text_pil(self, img_rgba, text, x, y, font_size, color_rgb, anchor):
        pil_img = Image.fromarray(img_rgba)
        draw = ImageDraw.Draw(pil_img)
        font = self._get_font(font_size)
        fill_color = (color_rgb[0], color_rgb[1], color_rgb[2], 255)
        draw.text((int(x), int(y)), text, font=font, fill=fill_color, anchor=anchor)
        return np.array(pil_img)

    # ========== Logic: Preview & Interaction (Lazy Loading) ==========

    # === 修复：Preview 增加进度条提示 ===
    def _create_preview(self):
        if not self.current_source_layer: return
        
        data = self.current_source_layer.data
        ndim = data.ndim
        if ndim == 3: n_frames, H, W = data.shape
        elif ndim == 4: n_frames, H, W, C = data.shape
        else: return

        # 使用模态进度条，防止界面假死感
        progress = QProgressDialog("Initializing Preview Layer...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0) # 立即显示
        progress.show()
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # 1. Create Empty Overlay
            # 注意：对于非常大的 stack，np.zeros 可能会耗内存，但通常这是最快的方法
            overlay_data = np.zeros((n_frames, H, W, 4), dtype=np.uint8)
            
            if 'Preview_Overlay' in self.viewer.layers: 
                self.viewer.layers.remove('Preview_Overlay')
            
            self.preview_overlay_layer = self.viewer.add_image(
                overlay_data, name='Preview_Overlay', blending='translucent_no_depth', 
                scale=self.current_source_layer.scale, translate=self.current_source_layer.translate
            )
            self.preview_overlay_layer.editable = False
            
            # 2. Render Current Frame
            self._refresh_overlay_only()
            
            # 3. Interaction Box
            self._create_interaction_box()
            
            self.status_label.setText("Preview Active.")
        finally:
            QApplication.restoreOverrideCursor()
            progress.close()

    def _on_frame_change(self, event):
        """懒加载核心：当时间轴滚动时，实时渲染新的一帧"""
        if self.preview_overlay_layer is None: return
        # 为了性能，不重复创建 Interaction Box，只更新 Overlay
        self._refresh_overlay_only()

    def _refresh_overlay_only(self):
        """渲染当前帧的 Overlay"""
        if not self.preview_overlay_layer: return
        
        current_step = self.viewer.dims.current_step[0]
        # 检查索引是否越界 (针对不同长度的图层切换)
        if current_step >= self.preview_overlay_layer.data.shape[0]: return

        H, W = self.preview_overlay_layer.data.shape[1], self.preview_overlay_layer.data.shape[2]
        
        # Render On-the-fly
        empty = np.zeros((H, W, 4), dtype=np.uint8)
        new_frame = self._render_single_frame_overlay(empty, current_step, H, W)
        
        # Update Data Slice
        self.preview_overlay_layer.data[current_step] = new_frame
        self.preview_overlay_layer.refresh()

    def _render_single_frame_overlay(self, canvas_rgba, frame_idx, H, W):
        """渲染单帧 (Pillow)"""
        # Label
        if self.use_label_check.isChecked():
            x, y = self.label_x_spin.value(), self.label_y_spin.value()
            size = self.label_font_spin.value()
            val = self.label_start_spin.value() + frame_idx * self.label_interval_spin.value()
            txt = self._format_val(val)
            c = self.label_color
            col = (int(c[0]*255), int(c[1]*255), int(c[2]*255))
            canvas_rgba = self._draw_text_pil(canvas_rgba, txt, x, y, size, col, 'lt')

        # Scale Bar
        if self.use_scale_bar_check.isChecked():
            x, y = self.scale_x_spin.value(), self.scale_y_spin.value()
            w = self._get_pixel_width() 
            h = self.scale_height_spin.value()
            thick, size = self.scale_thickness_spin.value(), self.scale_font_spin.value()
            pad = self.scale_padding_spin.value()
            
            if self.scale_use_bg_check.isChecked():
                bg_c = self.scale_bg_color
                bg_alpha = int(self.scale_bg_alpha_slider.value() / 100.0 * 255)
                bg_rgba = (int(bg_c[0]*255), int(bg_c[1]*255), int(bg_c[2]*255), bg_alpha)
                bg_x, bg_w = x - pad, w + 2 * pad
                
                pil_img = Image.fromarray(canvas_rgba)
                draw = ImageDraw.Draw(pil_img, 'RGBA')
                draw.rectangle([bg_x, y, bg_x+bg_w, y+h], fill=bg_rgba)
                canvas_rgba = np.array(pil_img)

            real_len = self.scale_length_spin.value()
            unit = self.scale_unit_edit.text()
            txt = f"{int(real_len)} {unit}" if real_len == int(real_len) else f"{real_len:.2f} {unit}"
            
            gap = int(size * 0.4)
            total_h = thick + gap + size
            start_y = y + (h - total_h) // 2
            c = self.scale_color
            col_rgb = (int(c[0]*255), int(c[1]*255), int(c[2]*255))
            
            pil_img = Image.fromarray(canvas_rgba)
            draw = ImageDraw.Draw(pil_img)
            draw.rectangle([x, start_y, x+w, start_y+thick], fill=(*col_rgb, 255))
            canvas_rgba = np.array(pil_img)
            
            text_y = start_y + thick + gap
            text_x = x + w / 2
            canvas_rgba = self._draw_text_pil(canvas_rgba, txt, text_x, text_y, size, col_rgb, 'mt')
            
        return canvas_rgba

    def _create_interaction_box(self):
        """创建交互矩形框"""
        if 'Interaction_Box' in self.viewer.layers: self.viewer.layers.remove('Interaction_Box')
        rects, edge_colors = [], []
        self.interaction_map = {} 
        idx = 0

        # Scale Box
        if self.use_scale_bar_check.isChecked():
            x, y = self.scale_x_spin.value(), self.scale_y_spin.value()
            w = self._get_pixel_width()
            h = self.scale_height_spin.value()
            pad = self.scale_padding_spin.value()
            
            bg_x, bg_w = x - pad, w + 2 * pad
            # Rect: [TL, TR, BR, BL]
            box = np.array([[y, bg_x], [y, bg_x+bg_w], [y+h, bg_x+bg_w], [y+h, bg_x]])
            rects.append(box); edge_colors.append('red')
            self.interaction_map[idx] = 'scale'; idx += 1
            
        # Label Box
        if self.use_label_check.isChecked():
            x, y = self.label_x_spin.value(), self.label_y_spin.value()
            size = self.label_font_spin.value()
            
            current_step = self.viewer.dims.current_step[0]
            val = self.label_start_spin.value() + current_step * self.label_interval_spin.value()
            txt = self._format_val(val)
            
            font = self._get_font(size)
            bbox = font.getbbox(txt)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 5
            
            box = np.array([[y-pad, x-pad], [y-pad, x+w+pad], [y+h+pad, x+w+pad], [y+h+pad, x-pad]])
            rects.append(box); edge_colors.append('yellow')
            self.interaction_map[idx] = 'label'; idx += 1
        
        if rects:
            self.interaction_layer = self.viewer.add_shapes(
                rects, shape_type='rectangle', name='Interaction_Box',
                edge_width=2, edge_color=edge_colors, face_color=[0,0,0,0]
            )
            self.interaction_layer.mode = 'select'
            self.interaction_layer.events.data.connect(self._on_interaction_change)

    # ========== Interaction ==========

    def _on_ui_param_change(self):
        if self._updating: return
        # === 新增：保存参数到 QSettings ===
        # Scale Bar Settings
        self.settings.setValue("scale/enable", self.use_scale_bar_check.isChecked())
        self.settings.setValue("scale/ratio", self.scale_ratio_spin.value())
        self.settings.setValue("scale/unit", self.scale_unit_edit.text())
        self.settings.setValue("scale/length", self.scale_length_spin.value())
        self.settings.setValue("scale/height", self.scale_height_spin.value())
        self.settings.setValue("scale/thickness", self.scale_thickness_spin.value())
        self.settings.setValue("scale/font_size", self.scale_font_spin.value())
        self.settings.setValue("scale/padding", self.scale_padding_spin.value())
        self.settings.setValue("scale/color", self.scale_color)  # Tuple (r,g,b,a)
        self.settings.setValue("scale/bg_color", self.scale_bg_color)
        self.settings.setValue("scale/bg_alpha", self.scale_bg_alpha_slider.value())
        self.settings.setValue("scale/use_bg", self.scale_use_bg_check.isChecked())
        self.settings.setValue("scale/position", (self.scale_x_spin.value(), self.scale_y_spin.value()))

        # Timestamp Settings
        self.settings.setValue("label/enable", self.use_label_check.isChecked())
        self.settings.setValue("label/format", self.label_format_combo.currentText())
        self.settings.setValue("label/custom_fmt", self.label_custom_edit.text())
        self.settings.setValue("label/font_size", self.label_font_spin.value())
        self.settings.setValue("label/color", self.label_color)
        self.settings.setValue("label/position", (self.label_x_spin.value(), self.label_y_spin.value()))
        self.settings.setValue("label/start", self.label_start_spin.value())
        self.settings.setValue("label/interval", self.label_interval_spin.value())
        # =================================

        if self.auto_size_check.isChecked(): self._perform_auto_calc()
        if self._dragging: self._refresh_overlay_only()
        else:
            self._refresh_overlay_only()
            self._create_interaction_box()

    def _on_interaction_change(self, event):
        if self._updating: return
        self._dragging = True 
        
        try:
            if not self.current_source_layer: return
            data_shape = self.current_source_layer.data.shape
            if len(data_shape) >= 2:
                IMG_H, IMG_W = data_shape[-2], data_shape[-1]
            else:
                return

            data = self.interaction_layer.data
            for i, rect in enumerate(data):
                if i not in self.interaction_map: continue
                type_ = self.interaction_map[i]
                
                ys, xs = rect[:, 0], rect[:, 1]
                min_y, max_y = np.min(ys), np.max(ys)
                min_x, max_x = np.min(xs), np.max(xs)
                curr_w, curr_h = max_x - min_x, max_y - min_y
                
                self._updating = True
                
                if type_ == 'scale':
                    pad = self.scale_padding_spin.value()
                    
                    # Boundary Clamp
                    new_x = max(0, min(min_x + pad, IMG_W))
                    new_y = max(0, min(min_y, IMG_H))
                    new_w_px_raw = curr_w - 2*pad
                    
                    # Anti-Drift
                    current_w_px = self._get_pixel_width()
                    if abs(new_w_px_raw - current_w_px) > 1.0:
                        ratio = self.scale_ratio_spin.value()
                        new_len_unit = max(0.001, new_w_px_raw * ratio)
                        self.scale_length_spin.setValue(new_len_unit)
                    
                    self.scale_x_spin.setValue(int(new_x))
                    self.scale_y_spin.setValue(int(new_y))
                    if not self.auto_size_check.isChecked():
                        self.scale_height_spin.setValue(max(10, int(curr_h)))
                    
                elif type_ == 'label':
                    pad = 5
                    new_x = max(0, min(min_x + pad, IMG_W))
                    new_y = max(0, min(min_y + pad, IMG_H))
                    
                    current_size = self.label_font_spin.value()
                    new_size_raw = curr_h - 2*pad
                    
                    if abs(new_size_raw - current_size) > 1.0:
                        self.label_font_spin.setValue(max(6, int(new_size_raw)))
                        
                    self.label_x_spin.setValue(int(new_x))
                    self.label_y_spin.setValue(int(new_y))
                
                self._updating = False
            self._refresh_overlay_only()
        finally:
            self._dragging = False

    def _snap_pos(self, target, corner):
        """
        智能边缘吸附 (Smart Snapping)
        target: 'scale' or 'label'
        corner: 'TL', 'TR', 'BL', 'BR'
        """
        if not self.current_source_layer: return
        data = self.current_source_layer.data
        if hasattr(data, 'ndim') and data.ndim >= 2: 
            H, W = data.shape[-2], data.shape[-1]
        else: return
        
        MARGIN = 20 # 吸附间距 (ImageJ 风格)
        
        self._updating = True
        
        # 1. 计算目标的包围盒尺寸
        obj_w, obj_h = 0, 0
        
        if target == 'scale':
            # 比例尺的视觉包围盒 (Background Box)
            # 宽度 = BarPixelWidth + 2*Padding
            # 高度 = BG Height
            bar_w = self._get_pixel_width()
            pad = self.scale_padding_spin.value()
            obj_w = bar_w + 2 * pad
            obj_h = self.scale_height_spin.value()
            
            # Scale 的 X, Y 是 BG Box 的左上角 (因为我们的绘制逻辑是 x-pad)
            # 修正：我们 UI 里的 X, Y 是 Bar 的起点，但背景是从 X-pad 开始的
            # 所以如果我们要把背景吸附到边缘，我们需要调整 X
            
            if corner == 'TL':
                # 左边缘：X - pad = Margin => X = Margin + pad
                new_x = MARGIN + pad
                new_y = MARGIN
            elif corner == 'TR':
                # 右边缘：X - pad + obj_w = W - Margin => X = W - Margin - obj_w + pad
                # obj_w = bar_w + 2*pad
                # X = W - Margin - (bar_w + 2*pad) + pad = W - Margin - bar_w - pad
                new_x = W - MARGIN - bar_w - pad
                new_y = MARGIN
            elif corner == 'BL':
                new_x = MARGIN + pad
                new_y = H - MARGIN - obj_h
            elif corner == 'BR':
                new_x = W - MARGIN - bar_w - pad
                new_y = H - MARGIN - obj_h
                
            self.scale_x_spin.setValue(int(new_x))
            self.scale_y_spin.setValue(int(new_y))

        elif target == 'label':
            # Label 尺寸计算 (Text BBox)
            size = self.label_font_spin.value()
            # 获取当前帧的文本做估算
            current_step = self.viewer.dims.current_step[0]
            val = self.label_start_spin.value() + current_step * self.label_interval_spin.value()
            txt = self._format_val(val)
            
            font = self._get_font(size)
            bbox = font.getbbox(txt)
            obj_w = bbox[2] - bbox[0]
            obj_h = bbox[3] - bbox[1]
            
            # Label 的绘制锚点是 'lt' (左上角)
            if corner == 'TL':
                new_x = MARGIN
                new_y = MARGIN
            elif corner == 'TR':
                new_x = W - MARGIN - obj_w
                new_y = MARGIN
            elif corner == 'BL':
                new_x = MARGIN
                new_y = H - MARGIN - obj_h
            elif corner == 'BR':
                new_x = W - MARGIN - obj_w
                new_y = H - MARGIN - obj_h
                
            self.label_x_spin.setValue(int(new_x))
            self.label_y_spin.setValue(int(new_y))
            
        self._updating = False
        self._refresh_overlay_only()
        self._create_interaction_box()

    # ========== Burn-in ==========

    def _apply_to_new_layer(self):
        if not self.current_source_layer: return
        data = self.current_source_layer.data
        if data.ndim == 3: n_frames, H, W = data.shape
        elif data.ndim == 4: n_frames, H, W, C = data.shape
        else: return

        # === 新增：获取当前图层的显示对比度 (所见即所得) ===
        # Napari 的 contrast_limits 决定了屏幕上怎么显示像素值
        contrast_limits = None
        if hasattr(self.current_source_layer, 'contrast_limits'):
            contrast_limits = self.current_source_layer.contrast_limits

        progress = QProgressDialog("Burning annotations...", "Cancel", 0, n_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        out_frames = []
        for i in range(n_frames):
            if progress.wasCanceled(): return
            # --- 核心逻辑 ---
            frame = np.asarray(data[i])
            # === 核心修复：应用对比度映射 ===
            # 如果是单通道灰度图，必须应用 contrast_limits 才能还原用户在屏幕上看到的亮度/对比度
            if frame.ndim == 2:
                if contrast_limits is not None:
                    c_min, c_max = contrast_limits
                    if c_max > c_min:
                        # 1. 转 float 避免精度丢失
                        # 2. 线性映射: (val - min) / (max - min)
                        # 3. Clip 到 0-1
                        # 4. 缩放到 0-255
                        frame_f = frame.astype(np.float32)
                        frame_f = (frame_f - c_min) / (c_max - c_min)
                        frame_f = np.clip(frame_f, 0, 1)
                        frame = (frame_f * 255).astype(np.uint8)
                    else:
                        # 异常情况：max <= min，回退到普通归一化
                        if frame.dtype != np.uint8:
                            f_min, f_max = frame.min(), frame.max()
                            frame = ((frame - f_min) / (f_max - f_min + 1e-6) * 255).astype(np.uint8)
                else:
                    # 没有对比度信息时的默认处理
                    if frame.dtype != np.uint8:
                        f_min, f_max = frame.min(), frame.max()
                        frame = ((frame - f_min) / (f_max - f_min + 1e-6) * 255).astype(np.uint8)
                
                # 转为 BGR (OpenCV格式) 准备绘图
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            elif frame.ndim == 3 and frame.shape[2] in [3, 4]:
                # RGB/RGBA 图层通常已经是 0-255 的 uint8，或者 0-1 的 float
                # Napari 对 RGB 图层通常不使用 contrast_limits 进行同样的映射
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            overlay_rgba = np.zeros((H, W, 4), dtype=np.uint8)
            overlay_rgba = self._render_single_frame_overlay(overlay_rgba, i, H, W)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb).convert("RGBA")
            pil_overlay = Image.fromarray(overlay_rgba)
            pil_out = Image.alpha_composite(pil_frame, pil_overlay)
            
            # frame_out = cv2.cvtColor(np.array(pil_out.convert("RGB")), cv2.COLOR_RGB2BGR)
            # 转回 Numpy (RGB) 供 Napari 显示
            frame_out = np.array(pil_out.convert("RGB"))
            out_frames.append(frame_out)

            progress.setValue(i)
            QApplication.processEvents() 
            
        progress.setValue(n_frames)
        new_data = np.stack(out_frames)
        # self.viewer.add_image(new_data, name=f"Burned_{self.current_source_layer.name}")
        new_layer_name = f"Burned_{self.current_source_layer.name}"
        # self.viewer.add_image(new_data, name=new_layer_name)
        
       # 添加新图层
        new_layer = self.viewer.add_image(new_data, name=new_layer_name)
        self.status_label.setText("Done.")

        # === 修复：自动切换逻辑 ===
        # 隐藏源图层 (Annotation 通常也是针对特定图层操作的)
        if self.current_source_layer:
            self.current_source_layer.visible = False
        # 激活新图层 (这会自动触发其他 Widget 的联动刷新)
        self.viewer.layers.selection.active = new_layer
        # 本地刷新
        self._refresh_layers()
        self.layer_combo.setCurrentText(new_layer_name)
    
    def _switch_to_layer(self, layer_name):
        """自动切换焦点"""
        if layer_name in self.viewer.layers:
            # 隐藏其他 Image 图层
            for l in self.viewer.layers:
                if isinstance(l, napari.layers.Image):
                    l.visible = (l.name == layer_name)
            self.viewer.layers.selection.active = self.viewer.layers[layer_name]
            # 更新本控件的图层选择
            self._refresh_layers()
            self.layer_combo.setCurrentText(layer_name)

    # ========== Utils ==========
    def _refresh_layers(self):
        """刷新图层并自动选中"""
        current_text = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        
        for layer in self.viewer.layers:
            if (hasattr(layer, 'data') and 
                isinstance(layer.data, np.ndarray) and 
                len(layer.data.shape) >= 3):
                self.layer_combo.addItem(layer.name)
        
        # 智能选择逻辑
        active_layer = self.viewer.layers.selection.active
        if active_layer and self.layer_combo.findText(active_layer.name) >= 0:
            self.layer_combo.setCurrentText(active_layer.name)
        elif self.layer_combo.findText(current_text) >= 0:
            self.layer_combo.setCurrentText(current_text)
            
        self.layer_combo.blockSignals(False)
        # 强制更新状态
        self._on_layer_selected(self.layer_combo.currentText())
                
    def _on_layer_selected(self, name):
        if name: self.current_source_layer = self.viewer.layers[name]
        
    def _format_val(self, val):
        fmt = self.label_format_combo.currentText()
        if fmt == "Custom": fmt = self.label_custom_edit.text()
        if fmt == "0": return f"{val:.0f}"
        if fmt == "0.0": return f"{val:.1f}"
        if fmt == "0.00": return f"{val:.2f}"
        if fmt == "00:00": return f"{int(val)//60:02d}:{int(val)%60:02d}"
        try: return fmt.format(val)
        except: return str(val)
        
    def _update_btn_style(self, btn, col):
        c_hex = self._rgba_to_hex(col)
        fg = "black" if (col[0]*0.299 + col[1]*0.587 + col[2]*0.114) > 0.5 else "white"
        btn.setStyleSheet(f"background-color: {c_hex}; color: {fg};")
        
    def _choose_scale_color(self):
        c = QColorDialog.getColor()
        if c.isValid(): 
            self.scale_color = c.getRgbF()
            self._update_btn_style(self.scale_color_btn, self.scale_color)
            self._on_ui_param_change() 
            
    def _choose_scale_bg_color(self):
        c = QColorDialog.getColor()
        if c.isValid(): 
            self.scale_bg_color = c.getRgbF()[:3]
            self._update_btn_style(self.scale_bg_color_btn, self.scale_bg_color)
            self._on_ui_param_change()

    def _choose_label_color(self):
        c = QColorDialog.getColor()
        if c.isValid(): 
            self.label_color = c.getRgbF()
            self._update_btn_style(self.label_color_btn, self.label_color)
            self._on_ui_param_change()

    def _rgba_to_hex(self, rgba):
        return '#{:02X}{:02X}{:02X}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
        
    def _clear_preview(self):
        for l in ['Preview_Overlay', 'Interaction_Box']:
            if l in self.viewer.layers: self.viewer.layers.remove(l)
        self.preview_overlay_layer = None
        self.interaction_layer = None
        self.status_label.setText("Cleared.")
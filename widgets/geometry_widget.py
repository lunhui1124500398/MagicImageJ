"""
几何变换控件 (交互优化版)
功能：
1. 旋转：画线后自动计算角度，自动切换到编辑模式，自动清理旧图层。
2. 裁剪：画框后自动切换工具，应用后只显示结果图层，隐藏所有干扰项。
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QHBoxLayout, QComboBox, QGroupBox, QDoubleSpinBox,QScrollArea)
import numpy as np
# 假设 core.geometry 已经存在于项目中
from core.geometry import (calculate_rotation_angle, rotate_image_stack, 
                           crop_image_stack, validate_bbox)
import napari

class GeometryWidget(QWidget):
    """几何变换控件"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._setup_ui()
    
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
        
        title = QLabel("<h3>📐 Geometry Tools</h3>")
        layout.addWidget(title)
        
        # 图层选择
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Target Layer:"))
        self.layer_combo = QComboBox()
        layer_layout.addWidget(self.layer_combo)
        layout.addLayout(layer_layout)
        
        refresh_btn = QPushButton("🔄 Refresh Layers")
        refresh_btn.clicked.connect(self._refresh_layers)
        layout.addWidget(refresh_btn)
        
        # ========== 旋转模块 ==========
        rotate_group = QGroupBox("Rotation (Horizon Correction)")
        rotate_layout = QVBoxLayout()
        
        rotate_layout.addWidget(QLabel("1. Draw a line that should be horizontal:"))
        draw_line_btn = QPushButton("✏️ Draw Horizon Line (Auto-tool)")
        draw_line_btn.clicked.connect(self._draw_rotation_line)
        rotate_layout.addWidget(draw_line_btn)
        
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Calculated Angle:"))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-360, 360)
        self.angle_spin.setDecimals(2)
        angle_layout.addWidget(self.angle_spin)
        
        # 虽然支持自动计算，保留按钮以备手动修正
        calc_btn = QPushButton("📐 Recalculate")
        calc_btn.clicked.connect(self._calculate_angle)
        angle_layout.addWidget(calc_btn)
        rotate_layout.addLayout(angle_layout)
        
        apply_rotate_btn = QPushButton("✅ Apply Rotation")
        apply_rotate_btn.clicked.connect(self._apply_rotation)
        rotate_layout.addWidget(apply_rotate_btn)
        
        rotate_group.setLayout(rotate_layout)
        layout.addWidget(rotate_group)
        
        # ========== 裁剪模块 ==========
        crop_group = QGroupBox("Crop")
        crop_layout = QVBoxLayout()
        
        crop_layout.addWidget(QLabel("1. Draw a rectangle to crop:"))
        draw_rect_btn = QPushButton("✏️ Draw Crop Rectangle (Auto-tool)")
        draw_rect_btn.clicked.connect(self._draw_crop_rect)
        crop_layout.addWidget(draw_rect_btn)
        
        apply_crop_btn = QPushButton("✂️ Apply Crop (Clean View)")
        apply_crop_btn.clicked.connect(self._apply_crop)
        crop_layout.addWidget(apply_crop_btn)
        
        crop_group.setLayout(crop_layout)
        layout.addWidget(crop_group)
        
        # 状态信息
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch() # 建议在最后加一个弹簧，防止内容分散
        content_widget.setLayout(layout)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        self._refresh_layers()
        
    def _refresh_layers(self):
        """刷新图层列表并自动选中活跃图层"""
        current_text = self.layer_combo.currentText()
        
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        
        for layer in self.viewer.layers:
            if (hasattr(layer, 'data') and isinstance(layer.data, np.ndarray) 
                and len(layer.data.shape) == 3):
                self.layer_combo.addItem(layer.name)
        
        # 智能选择：优先选 Active Layer
        active_layer = self.viewer.layers.selection.active
        if active_layer and self.layer_combo.findText(active_layer.name) >= 0:
            self.layer_combo.setCurrentText(active_layer.name)
        elif self.layer_combo.findText(current_text) >= 0:
            self.layer_combo.setCurrentText(current_text)
            
        self.layer_combo.blockSignals(False)

    def _on_active_layer_changed(self, event=None):
        """响应 Napari 图层选择变化"""
        active_layer = self.viewer.layers.selection.active
        if active_layer:
            index = self.layer_combo.findText(active_layer.name)
            if index >= 0:
                self.layer_combo.setCurrentIndex(index)

    def _clear_residue(self, target_names):
        """清理残留的辅助图层，保持视野干净"""
        for name in target_names:
            if name in self.viewer.layers:
                # 直接移除，彻底清理
                self.viewer.layers.remove(name)

    def _draw_rotation_line(self):
        """添加用于画线的Shapes层 (自动清理 + 自动工具)"""
        # 1. 清理漂移矫正残留 和 裁剪框
        self._clear_residue(["Drift_ROI", "Crop_ROI", "Rotation_Line"])
        
        layer_name = "Rotation_Line"
        
        # 添加新图层
        layer = self.viewer.add_shapes(
            name=layer_name,
            shape_type='line',
            edge_color='cyan',
            edge_width=4,
            face_color='transparent'
        )
        
        # 2. 自动绑定计算事件：画完线立即计算角度
        layer.events.data.connect(self._auto_calculate_angle)
        
        self.viewer.layers.selection.active = layer
        
        # 3. 自动切换到画线工具
        layer.mode = 'add_line'
        
        self.status_label.setText("✏️ Mode: Draw Line. Angle will be auto-calculated.")

    def _auto_calculate_angle(self, event=None):
        """事件回调：数据变化时自动计算"""
        # === UX优化：画完线自动切换到编辑模式 ===
        layer_name = "Rotation_Line"
        if layer_name in self.viewer.layers:
            layer = self.viewer.layers[layer_name]
            # 检查是否刚画完（add_line模式）且已有数据
            if layer.mode == 'add_line' and len(layer.data) > 0:
                layer.mode = 'select'
                self.status_label.setText("✋ Line drawn. Switched to edit mode.")
        
        self._calculate_angle()

    def _calculate_angle(self):
        """计算角度"""
        layer_name = "Rotation_Line"
        if layer_name not in self.viewer.layers:
            return
        
        layer = self.viewer.layers[layer_name]
        if len(layer.data) == 0:
            return
            
        # 获取最后画的一条线
        line_data = layer.data[-1] # [[y1, x1], [y2, x2]]
        p1 = (line_data[0][1], line_data[0][0]) # x, y
        p2 = (line_data[1][1], line_data[1][0])
        
        angle = calculate_rotation_angle((p1, p2))
        self.angle_spin.setValue(angle)
        self.status_label.setText(f"ℹ️ Auto-calculated angle: {angle:.2f}°")

    def _apply_rotation(self):
        """应用旋转"""
        layer_name = self.layer_combo.currentText()
        if not layer_name: return
        
        angle = self.angle_spin.value()
        image_stack = self.viewer.layers[layer_name].data
        
        try:
            rotated = rotate_image_stack(image_stack, angle)
            
            new_layer_name = f"Rotated_{layer_name}"
            self.viewer.add_image(
                rotated,
                name=new_layer_name,
                colormap='gray'
            )
            self.status_label.setText(f"✅ Rotated by {angle:.2f}°")
            
            # 清理画线层
            self._clear_residue(["Rotation_Line"])
            
            # 聚焦新图层
            self._update_layer_focus(new_layer_name)
        except Exception as e:
            self.status_label.setText(f"❌ Rotation Error: {str(e)}")

    def _draw_crop_rect(self):
        """添加裁剪ROI层 (自动清理 + 自动工具)"""
        # 1. 清理漂移残留 和 旋转线
        self._clear_residue(["Drift_ROI", "Rotation_Line", "Crop_ROI"])
        
        layer_name = "Crop_ROI"
        
        # 2. 使用带透明度的填充色，方便拖拽 (借鉴漂移模块的经验)
        layer = self.viewer.add_shapes(
            name=layer_name,
            shape_type='rectangle',
            edge_color='yellow',
            face_color=[1, 1, 0, 0.01], # 黄色，1%透明度
            edge_width=3
        )
        
        # 3. 自动切换到画矩形工具
        layer.mode = 'add_rectangle'
        
        # 4. 绑定事件：画完后自动切换到选择模式方便调整
        layer.events.data.connect(self._on_crop_rect_drawn)
        
        self.viewer.layers.selection.active = layer
        self.status_label.setText("✏️ Mode: Draw Crop Rectangle.")

    def _on_crop_rect_drawn(self, event=None):
        """画完矩形后，切换到选择模式"""
        layer_name = "Crop_ROI"
        if layer_name in self.viewer.layers:
            layer = self.viewer.layers[layer_name]
            # 只有在刚画完(add_rectangle模式)且有数据时才切换
            if layer.mode == 'add_rectangle' and len(layer.data) > 0:
                layer.mode = 'select'
                self.status_label.setText("✋ Rectangle drawn. Adjust edges if needed.")

    def _apply_crop(self):
        """应用裁剪"""
        target_layer = self.layer_combo.currentText()
        roi_layer = "Crop_ROI"
        
        if not target_layer or roi_layer not in self.viewer.layers:
            self.status_label.setText("❌ Missing layer or ROI.")
            return
            
        shapes = self.viewer.layers[roi_layer].data
        if len(shapes) == 0:
            return
            
        # 获取边界框
        data = shapes[-1]
        ys = data[:, 0]
        xs = data[:, 1]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        
        image_stack = self.viewer.layers[target_layer].data
        
        if not validate_bbox(image_stack.shape[1:], (x1, y1, x2, y2)):
            self.status_label.setText("❌ Invalid ROI.")
            return
            
        try:
            cropped = crop_image_stack(image_stack, (x1, y1, x2, y2))
            
            new_layer_name = f"Cropped_{target_layer}"
            self.viewer.add_image(
                cropped,
                name=new_layer_name,
                colormap='gray'
            )
            self.status_label.setText("✅ Crop applied.")
            
            # === 关键修改：强制清理所有ROI图层，确保视野干净 ===
            self._clear_residue(["Crop_ROI", "Rotation_Line", "Drift_ROI"])
            
            # 聚焦新图层
            self._update_layer_focus(new_layer_name)
            
        except Exception as e:
            self.status_label.setText(f"❌ Crop Error: {str(e)}")
        
    def _update_layer_focus(self, new_layer_name):
        """通用：更新图层焦点、可见性和下拉框"""
        # 1. 隐藏其他所有 Image 图层，确保只看到结果
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image) and layer.name != new_layer_name:
                layer.visible = False
        
        # 2. 将焦点（Active）设置为新图层
        if new_layer_name in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[new_layer_name]
        
        # 3. 刷新下拉框并自动选中新图层
        self._refresh_layers()
        self.layer_combo.setCurrentText(new_layer_name)
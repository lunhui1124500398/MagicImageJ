"""
几何变换控件 (交互优化版 + 批量ROI提取)
修复日志:
- [Fix] draw_rect 颜色改为淡白色 [1, 1, 1, 0.01]。
- [Fix] 旋转后自动切换 Simple Crop 的目标图层为旋转后的图层。
- [Fix] 批量/单次裁剪时强制清理 Interaction_Box / Preview_Overlay。
- [Fix] 增加越界检查 (Clamp to image bounds)。
- [New] 增加 "Peek Data" 按钮，按住可临时查看 Data Layer。
- [UX] 导出时增加模态进度条。
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QHBoxLayout, QComboBox, QGroupBox, 
                            QDoubleSpinBox, QScrollArea, QLineEdit, QFileDialog, 
                            QMessageBox, QCheckBox, QProgressDialog, QSpinBox)
from qtpy.QtCore import Qt, QTimer, QSettings
import numpy as np
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
from core.geometry import (calculate_rotation_angle, rotate_image_stack, 
                           crop_image_stack, validate_bbox)
from utils.video_export import export_to_tiff_stack
import napari
import json
from widgets.settings_widget import GlobalConfig

class GeometryWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._is_updating = False 
        self._force_view_active = False # 是否强制锁定视图层
        self._setup_ui()
        
        # 监听图层可见性变化，用于强制显示逻辑
        self.viewer.layers.events.reordered.connect(self._enforce_view_visibility)
        # 监听图层增减，自动刷新列表
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        # 监听激活图层变化 (用于自动选择)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

        apply_key = GlobalConfig.get_napari_shortcut("shortcut_apply_crop")
        switch_key = GlobalConfig.get_napari_shortcut("shortcut_switch_mode")
        
        self.viewer.bind_key(apply_key, self._on_shortcut_apply)
        self.viewer.bind_key(switch_key, self._on_shortcut_switch)

    def _setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        content_widget = QWidget()
        layout = QVBoxLayout()

        title = QLabel("<h3>📐 Geometry & Batch Crop</h3>")
        layout.addWidget(title)

        # 全局图层刷新
        refresh_btn = QPushButton("🔄 Refresh All Layers")
        refresh_btn.clicked.connect(self._refresh_layers)
        layout.addWidget(refresh_btn)

        # ========== 1. 旋转模块 ==========
        rotate_group = QGroupBox("1. Rotation (Horizon)")
        rotate_layout = QVBoxLayout()
        
        # 图层选择
        h_rot_layer = QHBoxLayout()
        h_rot_layer.addWidget(QLabel("Target:"))
        self.rotate_layer_combo = QComboBox()
        h_rot_layer.addWidget(self.rotate_layer_combo)
        rotate_layout.addLayout(h_rot_layer)

        rotate_layout.addWidget(QLabel("Draw a line to define horizon:"))
        draw_line_btn = QPushButton("✏️ Draw Horizon Line")
        draw_line_btn.clicked.connect(self._draw_rotation_line)
        rotate_layout.addWidget(draw_line_btn)

        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Angle:"))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-360, 360)
        self.angle_spin.setDecimals(2)
        angle_layout.addWidget(self.angle_spin)
        calc_btn = QPushButton("📐 Recalc")
        calc_btn.clicked.connect(self._calculate_angle)
        angle_layout.addWidget(calc_btn)
        rotate_layout.addLayout(angle_layout)

        # Enlarge 选项
        self.enlarge_check = QCheckBox("Enlarge Canvas (Fit All)")
        self.enlarge_check.setToolTip("Expand image size to fit rotated content without cropping")
        self.enlarge_check.setChecked(True)
        rotate_layout.addWidget(self.enlarge_check)

        apply_rotate_btn = QPushButton("✅ Apply Rotation")
        apply_rotate_btn.clicked.connect(self._apply_rotation)
        rotate_layout.addWidget(apply_rotate_btn)
        rotate_group.setLayout(rotate_layout)
        layout.addWidget(rotate_group)

        # ========== 2. 单次裁剪模块 ==========
        crop_group = QGroupBox("2. Simple Crop (Single)")
        crop_layout = QVBoxLayout()
        
        # 图层选择
        h_crop_layer = QHBoxLayout()
        h_crop_layer.addWidget(QLabel("Target:"))
        self.simple_crop_combo = QComboBox()
        h_crop_layer.addWidget(self.simple_crop_combo)
        crop_layout.addLayout(h_crop_layer)
        
        draw_rect_btn = QPushButton("✏️ Draw Rect")
        draw_rect_btn.clicked.connect(self._draw_crop_rect)
        crop_layout.addWidget(draw_rect_btn)
        apply_crop_btn = QPushButton("✂️ Apply Crop (New Layer)")
        apply_crop_btn.clicked.connect(self._apply_crop)
        crop_layout.addWidget(apply_crop_btn)
        crop_group.setLayout(crop_layout)
        layout.addWidget(crop_group)

        # ========== 3. 批量ROI提取 (增强版) ==========
        batch_group = QGroupBox("3. Batch Extraction (Multi-ROI)")
        batch_group.setStyleSheet("QGroupBox { border: 1px solid #4CAF50; margin-top: 10px; } QGroupBox::title { color: #4CAF50; }")
        batch_layout = QVBoxLayout()

        # --- View vs Data Layer Logic ---
        layer_grid = QVBoxLayout()
        
        # Row 1: Data Layer (实际裁剪的层)
        h_data = QHBoxLayout()
        h_data.addWidget(QLabel("Data Layer (Crop Source):"))
        self.batch_data_combo = QComboBox()
        h_data.addWidget(self.batch_data_combo)
        layer_grid.addLayout(h_data)

        # Row 2: View Layer (参考显示的层)
        h_view = QHBoxLayout()
        h_view.addWidget(QLabel("View Layer (Reference):"))
        self.batch_view_combo = QComboBox()
        h_view.addWidget(self.batch_view_combo)
        layer_grid.addLayout(h_view)
        
        # Row 3: Controls & Peek
        h_sync = QHBoxLayout()
        self.sync_layers_btn = QPushButton("🔗 Sync Select")
        self.sync_layers_btn.setToolTip("Set View Layer same as Data Layer")
        self.sync_layers_btn.clicked.connect(self._sync_batch_layers)
        h_sync.addWidget(self.sync_layers_btn)

        # Peek Button
        self.peek_btn = QPushButton("👁️ Peek Data (Hold)")
        self.peek_btn.setToolTip("Hold to temporarily show Data Layer to check alignment")
        self.peek_btn.pressed.connect(self._peek_data_layer_show)
        self.peek_btn.released.connect(self._peek_data_layer_hide)
        h_sync.addWidget(self.peek_btn)
        
        layer_grid.addLayout(h_sync)
        
        self.lock_view_check = QCheckBox("🔒 Lock View Layer (Prevent auto-switching)")
        self.lock_view_check.setChecked(True)
        layer_grid.addWidget(self.lock_view_check)
        
        batch_layout.addLayout(layer_grid)
        batch_layout.addWidget(QLabel("<hr>")) 

        # 物质名输入
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Substance Name:"))
        self.sample_name_edit = QLineEdit("CRY2")
        name_layout.addWidget(self.sample_name_edit)

        # === [新增功能 4] 自定义后缀输入 ===
        name_layout.addWidget(QLabel("Suffix:"))
        self.suffix_edit = QLineEdit("-NP{}")
        self.suffix_edit.setPlaceholderText("e.g. -NP{} or -{}")
        self.suffix_edit.setFixedWidth(80)
        self.suffix_edit.setToolTip("Use {} as placeholder for number.\nExample: '-NP{}' -> '-NP1'")
        name_layout.addWidget(self.suffix_edit)
        # ================================
        batch_layout.addLayout(name_layout)

        # 格式选择
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Export Format:"))
        self.batch_format_combo = QComboBox()
        self.batch_format_combo.addItems(["PNG Sequence (Folder)","TIFF Stack (.tiff)"])
        format_layout.addWidget(self.batch_format_combo)
        batch_layout.addLayout(format_layout)

        # === [新增] 帧范围过滤 ===
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame Filter:"))
        self.batch_frame_edit = QLineEdit()
        self.batch_frame_edit.setPlaceholderText("All (Default) or 0-10, 15...")
        self.batch_frame_edit.setToolTip("Leave empty for All frames.\nOr use: 0-10, 15, 20-25")
        frame_layout.addWidget(self.batch_frame_edit)
        batch_layout.addLayout(frame_layout)
        # ========================

        # === [新增功能] 序列命名设置 (Keep Index & Padding) ===
        naming_layout = QHBoxLayout()

        # 1. 保留原始帧号
        self.keep_index_check = QCheckBox("Keep Original Frame Index")
        self.keep_index_check.setChecked(True) # 默认勾选，符合用户现在的需求
        self.keep_index_check.setToolTip("Checked: Frame 48 -> 00048.png\nUnchecked: Frame 48 -> 00000.png")
        naming_layout.addWidget(self.keep_index_check)

        # 2. 数字位数设置
        naming_layout.addWidget(QLabel("Padding:"))
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(1, 12)
        self.padding_spin.setValue(5) # 默认5位
        self.padding_spin.setSuffix(" digits")
        self.padding_spin.setToolTip("Example: 5 digits -> 00048.png; 8 digits -> 00000048.png")
        naming_layout.addWidget(self.padding_spin)
        
        # 弹簧撑开
        naming_layout.addStretch()
        batch_layout.addLayout(naming_layout)
        # ===================================================
        
        # 强制正方形选项
        self.force_square_check = QCheckBox("Force Square Crops")
        self.force_square_check.setChecked(True) 
        batch_layout.addWidget(self.force_square_check)

        # 工具按钮
        tools_layout = QHBoxLayout()
        self.start_batch_btn = QPushButton("✏️ Start Draw")
        self.start_batch_btn.clicked.connect(self._start_batch_mode)
        self.start_batch_btn.setStyleSheet("background-color: #444; font-weight: bold;")
        tools_layout.addWidget(self.start_batch_btn)

        self.adjust_batch_btn = QPushButton("🖐️ Adjust")
        self.adjust_batch_btn.clicked.connect(self._switch_to_select_mode)
        tools_layout.addWidget(self.adjust_batch_btn)
        batch_layout.addLayout(tools_layout)

        # 导出按钮
        self.export_batch_btn = QPushButton("💾 Export Crops & Map")
        self.export_batch_btn.clicked.connect(self._export_batch_crops)
        self.export_batch_btn.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 6px;")
        batch_layout.addWidget(self.export_batch_btn)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        # 1. Suffix (Batch Extraction 部分创建的)
        self.suffix_edit.setText(GlobalConfig.get("geo_suffix"))
        self.suffix_edit.textChanged.connect(lambda t: GlobalConfig.set("geo_suffix", t))
        
        # 2. Keep Index (Batch Extraction 部分创建的)
        # 注意：QSettings 有时返回字符串 'true'/'false'，有时返回 bool，这里做个兼容处理
        val_keep = GlobalConfig.get("geo_keep_index")
        is_checked_keep = (val_keep == 'true') if isinstance(val_keep, str) else bool(val_keep)
        self.keep_index_check.setChecked(is_checked_keep)
        self.keep_index_check.stateChanged.connect(lambda v: GlobalConfig.set("geo_keep_index", bool(v)))
        
        # 3. Padding (Batch Extraction 部分创建的)
        self.padding_spin.setValue(int(GlobalConfig.get("geo_padding")))
        self.padding_spin.valueChanged.connect(lambda v: GlobalConfig.set("geo_padding", v))
        
        # 4. Force Square (Batch Extraction 部分创建的)
        val_sq = GlobalConfig.get("geo_force_square")
        is_checked_sq = (val_sq == 'true') if isinstance(val_sq, str) else bool(val_sq)
        self.force_square_check.setChecked(is_checked_sq)
        self.force_square_check.stateChanged.connect(lambda v: GlobalConfig.set("geo_force_square", bool(v)))

        # 5. Enlarge Canvas (Rotation 部分创建的)
        val_enl = GlobalConfig.get("geo_enlarge")
        is_checked_enl = (val_enl == 'true') if isinstance(val_enl, str) else bool(val_enl)
        self.enlarge_check.setChecked(is_checked_enl)
        self.enlarge_check.stateChanged.connect(lambda v: GlobalConfig.set("geo_enlarge", bool(v)))

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #AAA; font-size: 11px;")
        layout.addWidget(self.status_label)
        layout.addStretch()
        content_widget.setLayout(layout)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
        self._refresh_layers()
        # 初始化 Batch combos
        self._sync_batch_layers()

    def _on_shortcut_apply(self, viewer):
        """按下 Enter 键：根据当前激活的 Tab 区域决定执行什么"""
        # 判断当前哪个GroupBox是逻辑焦点比较难，但通常如果是 Batch 模式下有 ROI，就导出 Batch
        if "Batch_ROI" in self.viewer.layers and len(self.viewer.layers["Batch_ROI"].data) > 0:
            self._export_batch_crops()
            self.status_label.setText("⚡ Shortcut: Batch Export Triggered")
        elif "Crop_ROI" in self.viewer.layers and len(self.viewer.layers["Crop_ROI"].data) > 0:
            self._apply_crop()
            self.status_label.setText("⚡ Shortcut: Single Crop Triggered")

    def _on_shortcut_switch(self, viewer):
        """按下 M 键：切换绘制/选择模式"""
        # 针对 Batch ROI
        if "Batch_ROI" in self.viewer.layers:
            layer = self.viewer.layers["Batch_ROI"]
            if layer.mode == 'add_rectangle':
                layer.mode = 'select'
                self.status_label.setText("⚡ Mode: Select/Adjust")
            else:
                layer.mode = 'add_rectangle'
                self.status_label.setText("⚡ Mode: Draw")

    # --- Layer Management ---
    def _refresh_layers(self, event=None):
        """刷新所有下拉框，并保持当前选中项"""
        # 过滤有效 Image 图层
        layers = [
            l.name for l in self.viewer.layers 
            if hasattr(l, 'data') and isinstance(l.data, np.ndarray) and l.data.ndim == 3
        ]
        
        for combo in [self.rotate_layer_combo, self.simple_crop_combo, 
                      self.batch_data_combo, self.batch_view_combo]:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(layers)
            
            # 恢复逻辑：如果原来的选中项还在，就选原来的；如果不在，尝试选 Active Layer
            if current in layers:
                combo.setCurrentText(current)
            # 这里的 active layer fallback 逻辑保留给 rotate/simple crop
            elif layers and combo in [self.rotate_layer_combo, self.simple_crop_combo]:
                active = self.viewer.layers.selection.active
                if active and active.name in layers:
                    combo.setCurrentText(active.name)
            
            combo.blockSignals(False)
        
        # === [新增功能 2] 自动选择 Batch Data/View Layer ===
        # 逻辑：Data Layer 优先找 Rotated/Cropped (原始数据)，View Layer 优先找 Contrast/Enh (增强数据)
        self.batch_data_combo.blockSignals(True)
        self.batch_view_combo.blockSignals(True)

        # 1. 自动选择 Data Layer (Crop Source)
        data_candidates = [l for l in layers if l.startswith("Cropped_Rotated")]
        # Fallback: 如果没有严格匹配的，尝试找包含 Rotated 且不含 Enh/Contrast 的
        if not data_candidates:
             data_candidates = [l for l in layers if "Rotated" in l and "Enh" not in l and "Contrast" not in l and "Burned" not in l]
        # 如果当前没选或者选的不在列表中，且有推荐候选，则自动选择最新的一个
        if data_candidates:
            self.batch_data_combo.setCurrentText(data_candidates[-1])

        # 2. 自动选择 View Layer (Reference) - 优先找对比度增强过的
        view_candidates = [l for l in layers if l.startswith("Contrast_Enh")]
        if not view_candidates:
            view_candidates = [l for l in layers if l.startswith("Enh")]
        # 如果有增强图，选最新的增强图；如果没有，默认跟 Data Layer 一样
        if view_candidates:
            self.batch_view_combo.setCurrentText(view_candidates[-1])
        elif self.batch_data_combo.currentText():
            self.batch_view_combo.setCurrentText(self.batch_data_combo.currentText())

        self.batch_data_combo.blockSignals(False)
        self.batch_view_combo.blockSignals(False)

        # === [新增功能 3] 尝试从归档读取 Substance Name ===
        self._try_load_archived_substance()

    def _try_load_archived_substance(self):
        """尝试从全局归档路径读取 readme.txt 中的物质名"""
        try:
            archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
            if not archive_path: return
            
            readme_path = Path(archive_path) / "readme.txt"
            if readme_path.exists():
                with open(readme_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith("Substance:"):
                            # 提取 Substance: 后的内容
                            name = line.split(":", 1)[1].strip()
                            if name: 
                                self.sample_name_edit.setText(name)
                            break
        except Exception:
            pass

    def _on_active_layer_changed(self, event=None):
        """外部图层切换时，自动更新下拉框（除非正在操作）"""
        if self._force_view_active: return # 批量模式下不跟随
        active = self.viewer.layers.selection.active
        if active and hasattr(active, 'data') and isinstance(active.data, np.ndarray) and active.data.ndim == 3:
            name = active.name
            # 更新 Simple Crop 和 Rotation 的目标
            self.rotate_layer_combo.setCurrentText(name)
            self.simple_crop_combo.setCurrentText(name)

    def _sync_batch_layers(self):
        txt = self.batch_data_combo.currentText()
        if txt: self.batch_view_combo.setCurrentText(txt)

    def _enforce_view_visibility(self, event=None):
        """强制显示 View Layer (Batch模式下)"""
        if not self._force_view_active: return
        
        view_name = self.batch_view_combo.currentText()
        if not view_name or view_name not in self.viewer.layers: return
        
        try:
            self.viewer.layers.events.reordered.disconnect(self._enforce_view_visibility)
        except: pass
        
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                layer.visible = (layer.name == view_name)
        
        self.viewer.layers.events.reordered.connect(self._enforce_view_visibility)

    def _clear_residue(self, target_names):
        """清理指定的临时图层"""
        for name in target_names:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

    # --- Peek Data Logic ---
    def _peek_data_layer_show(self):
        """按下 Peek 按钮：显示 Data Layer"""
        data_name = self.batch_data_combo.currentText()
        if not data_name or data_name not in self.viewer.layers: return
        
        self._force_view_active = False # 临时解锁
        # 隐藏所有 Image，只显示 Data Layer
        for l in self.viewer.layers:
            if isinstance(l, napari.layers.Image):
                l.visible = (l.name == data_name)
                
    def _peek_data_layer_hide(self):
        """松开 Peek 按钮：恢复 View Layer"""
        if self.lock_view_check.isChecked():
            self._force_view_active = True
            self._enforce_view_visibility() # 强制切回
        else:
            # 如果没锁，手动切回 View Layer
            view_name = self.batch_view_combo.currentText()
            if view_name in self.viewer.layers:
                for l in self.viewer.layers:
                    if isinstance(l, napari.layers.Image):
                        l.visible = (l.name == view_name)

    # --- Rotation ---
    def _draw_rotation_line(self):
        # 清理所有不相关图层
        self._clear_residue(["Rotation_Line", "Batch_ROI", "Crop_ROI", "Interaction_Box", "Preview_Overlay", "Drift_ROI"])
        layer = self.viewer.add_shapes(name="Rotation_Line", shape_type='line', edge_color='cyan', edge_width=4)
        layer.events.data.connect(self._auto_calculate_angle)
        layer.mode = 'add_line'
        self.status_label.setText("✏️ Draw Horizon Line.")

    def _auto_calculate_angle(self, event=None):
        layer = self.viewer.layers["Rotation_Line"]
        if layer.mode == 'add_line' and len(layer.data) > 0:
            layer.mode = 'select'
        if len(layer.data) > 0:
            line_data = layer.data[-1]
            p1, p2 = (line_data[0][1], line_data[0][0]), (line_data[1][1], line_data[1][0])
            angle = calculate_rotation_angle((p1, p2))
            self.angle_spin.setValue(angle)

    def _calculate_angle(self):
        if "Rotation_Line" in self.viewer.layers: self._auto_calculate_angle()

    def _apply_rotation(self):
        layer_name = self.rotate_layer_combo.currentText()
        if not layer_name: return
        angle = self.angle_spin.value()
        expand = self.enlarge_check.isChecked()
        image_stack = self.viewer.layers[layer_name].data
        try:
            rotated = rotate_image_stack(image_stack, angle, expand=expand)
            new_name = f"Rotated_{layer_name}"
            self.viewer.add_image(rotated, name=new_name, colormap='gray')
            
            # 清理 Line
            self._clear_residue(["Rotation_Line"])
            
            # [Fix 1.2] 自动切换 Simple Crop 的目标图层为新图层
            self.simple_crop_combo.setCurrentText(new_name)
            self.viewer.layers.selection.active = self.viewer.layers[new_name]
            
            self.status_label.setText(f"✅ Rotated {angle:.1f}° (Expand={expand})")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")

    # --- Simple Crop ---
    def _draw_crop_rect(self):
        self._clear_residue(["Crop_ROI", "Rotation_Line", "Batch_ROI", "Interaction_Box", "Preview_Overlay", "Drift_ROI"])
        # [Fix 1.3] 淡白色填充
        layer = self.viewer.add_shapes(
            name="Crop_ROI", shape_type='rectangle', 
            edge_color='yellow', edge_width=3,
            face_color=[1, 1, 1, 0.01] 
        )
        layer.mode = 'add_rectangle'
        self.status_label.setText("✏️ Draw Single Crop Rect.")

    def _apply_crop(self):
        target = self.simple_crop_combo.currentText()
        if not target or "Crop_ROI" not in self.viewer.layers: return
        shapes = self.viewer.layers["Crop_ROI"].data
        if not shapes: return
        
        data = shapes[-1]
        ys, xs = data[:, 0], data[:, 1]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        
        stack = self.viewer.layers[target].data
        cropped = crop_image_stack(stack, bbox)
        new_name = f"Cropped_{target}"
        self.viewer.add_image(cropped, name=new_name, colormap='gray')
        
        # [Fix 1.4] 清理 Crop ROI
        self._clear_residue(["Crop_ROI"])
        self.viewer.layers.selection.active = self.viewer.layers[new_name]

        # === [新增功能 1] Crop之后清理显示，只显示Crop出的图层 ===
        # 只隐藏 Image 图层，且不要隐藏本图层
        # 也不要隐藏预览层（虽然 clear_residue 已经清理了，但为了健壮性）
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image) and layer.name != new_name:
                # 排除掉一些不想被误伤的辅助层（可选）
                if "Preview" not in layer.name: 
                    layer.visible = False
        # =====================================================

        self.status_label.setText(f"✅ Crop applied: {new_name}")

    # --- Batch Crop Logic ---
    def _start_batch_mode(self):
        view_layer = self.batch_view_combo.currentText()
        if not view_layer: return
        
        # [Fix 1.5] 清理所有干扰图层
        self._clear_residue(["Batch_ROI", "Rotation_Line", "Crop_ROI", "Interaction_Box", "Preview_Overlay", "Drift_ROI"])
        
        # 1. 确保 View Layer 可见
        if self.lock_view_check.isChecked():
            self._force_view_active = True
            self._enforce_view_visibility()
        else:
            self._force_view_active = False
            for l in self.viewer.layers:
                if isinstance(l, napari.layers.Image): l.visible = (l.name == view_layer)

        # 2. 创建 ROI 层
        roi_layer = self.viewer.add_shapes(
            name="Batch_ROI",
            shape_type='rectangle',
            edge_color='#00FF00', 
            face_color=[0, 1, 0, 0.05],
            edge_width=2,
            text={'string': '{label}', 'size': 12, 'color': 'white', 'anchor': 'upper_left', 'translation': [-5, -5]}
        )
        roi_layer.events.data.connect(self._on_batch_data_change)
        roi_layer.mode = 'add_rectangle'
        self.status_label.setText(f"✏️ Drawing on '{view_layer}'. Data source: '{self.batch_data_combo.currentText()}'")

    def _switch_to_select_mode(self):
        if "Batch_ROI" in self.viewer.layers:
            self.viewer.layers["Batch_ROI"].mode = 'select'
            self.status_label.setText("🖐️ Adjust Mode.")

    def _on_batch_data_change(self, event=None):
        if self._is_updating: return
        if "Batch_ROI" not in self.viewer.layers: return
        
        view_layer_name = self.batch_view_combo.currentText()
        if view_layer_name not in self.viewer.layers: return
        
        # 获取图像尺寸用于 Clamp
        img_layer = self.viewer.layers[view_layer_name]
        IMG_H, IMG_W = img_layer.data.shape[-2], img_layer.data.shape[-1]
        
        layer = self.viewer.layers["Batch_ROI"]
        if len(layer.data) == 0: return

        self._is_updating = True
        try:
            # 1. Force Square Logic & [Fix 1.6] Clamp to Bounds
            new_data_list = []
            modified = False
            
            for roi in layer.data:
                ys, xs = roi[:, 0], roi[:, 1]
                y1, y2 = np.min(ys), np.max(ys)
                x1, x2 = np.min(xs), np.max(xs)
                
                h, w = y2 - y1, x2 - x1
                
                needs_reshape = False
                
                # 正方形逻辑
                if self.force_square_check.isChecked() and abs(w - h) > 1.0:
                    side = int(max(w, h))
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    ny1, ny2 = int(cy - side / 2), int(cy - side / 2) + side
                    nx1, nx2 = int(cx - side / 2), int(cx - side / 2) + side
                    needs_reshape = True
                else:
                    ny1, ny2, nx1, nx2 = y1, y2, x1, x2
                
                # [Fix 1.6] Clamp 越界处理
                ny1 = max(0, min(ny1, IMG_H)); ny2 = max(0, min(ny2, IMG_H))
                nx1 = max(0, min(nx1, IMG_W)); nx2 = max(0, min(nx2, IMG_W))
                
                # 如果被 Clamp 导致变形，不再是正方形，但在边界处只能妥协
                
                if needs_reshape or (ny1 != y1 or ny2 != y2 or nx1 != x1 or nx2 != x2):
                    new_rect = np.array([[ny1, nx1], [ny2, nx1], [ny2, nx2], [ny1, nx2]])
                    new_data_list.append(new_rect)
                    modified = True
                else:
                    new_data_list.append(roi)
            
            if modified:
                layer.data = new_data_list

            # 2. Update Labels
            labels = [str(i+1) for i in range(len(layer.data))]
            if hasattr(layer, 'features'): layer.features = {'label': labels}
            elif hasattr(layer, 'properties'): layer.properties = {'label': labels}
            
        finally:
            self._is_updating = False

    def _export_batch_crops(self):
        data_layer_name = self.batch_data_combo.currentText()
        view_layer_name = self.batch_view_combo.currentText()
        
        if "Batch_ROI" not in self.viewer.layers or not len(self.viewer.layers["Batch_ROI"].data):
            self.status_label.setText("❌ No ROIs defined.")
            return
        if not data_layer_name or data_layer_name not in self.viewer.layers:
            self.status_label.setText("❌ Invalid Data Layer.")
            return

        # 检查尺寸匹配
        data_stack = self.viewer.layers[data_layer_name].data
        view_stack = self.viewer.layers[view_layer_name].data

        # === [修改] 应用帧过滤 ===
        raw_range_text = self.batch_frame_edit.text()
        total_frames = data_stack.shape[0]
        selected_indices = self._parse_frame_indices(raw_range_text, total_frames)
        
        if not selected_indices:
            self.status_label.setText("❌ No valid frames selected.")
            return

        # 这一步很关键：先筛选帧，减少数据量，且排除坏帧
        # 注意：使用 fancy indexing 会创建副本，内存占用会暂时增加
        filtered_data_stack = data_stack[selected_indices]
        keep_original_index = self.keep_index_check.isChecked()
        pad_width = self.padding_spin.value()
        # View stack 也要同步筛选，用于生成 overview map (虽然 map 只取中间帧，但最好取筛选后的中间帧)
        filtered_view_stack = view_stack[selected_indices]
        # =======================
        
        if data_stack.shape[-2:] != view_stack.shape[-2:]:
            QMessageBox.warning(self, "Mismatch", "Data Layer and View Layer sizes do not match! Crops may be misaligned.")

        sub_name = self.sample_name_edit.text().strip() or "sample"
        
        # 归档路径检测
        archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
        if archive_path and Path(archive_path).exists():
            output_dir = Path(archive_path)
        else:
            d = QFileDialog.getExistingDirectory(self, "Select Output")
            if not d: return
            output_dir = Path(d) / sub_name
            output_dir.mkdir(parents=True, exist_ok=True)

        is_tiff = "TIFF" in self.batch_format_combo.currentText()
        rois = self.viewer.layers["Batch_ROI"].data
        log_crops = []
        count = len(rois)

        # [Fix 3] 添加进度条
        progress = QProgressDialog("Exporting Crops...", "Cancel", 0, count, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        for i, roi in enumerate(rois):
            if progress.wasCanceled(): break
            
            ys, xs = roi[:, 0], roi[:, 1]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            
            # 使用 Data Layer 进行裁剪
            crop = crop_image_stack(filtered_data_stack, bbox)

            # === [新增功能 4] 使用自定义后缀格式 ===
            suffix_fmt = self.suffix_edit.text()
            if "{}" not in suffix_fmt:
                # 如果用户没写 {}，自动补上数字
                suffix_str = f"{suffix_fmt}{i+1}"
            else:
                # 格式化字符串
                suffix_str = suffix_fmt.replace("{}", str(i+1))
            
            fname = f"{sub_name}{suffix_str}"
            # ===================================
            
            if is_tiff:
                export_to_tiff_stack(crop, str(output_dir / f"{fname}.tiff"))
            else:
                p = output_dir / fname
                p.mkdir(exist_ok=True)
                for k, img in enumerate(crop):
                    # 简单归一化以便预览
                    if img.dtype in [np.float32, np.float64]:
                        mn, mx = img.min(), img.max()
                        if mx > mn: img = ((img - mn)/(mx - mn)*255).astype(np.uint8)
                        else: img = img.astype(np.uint8)
                    
                    # === [核心修改] 计算文件名索引 ===
                    if keep_original_index:
                        # 使用 selected_indices 中的真实原始帧号
                        # 注意：crop 的第 k 帧对应 selected_indices 的第 k 个元素
                        file_idx = selected_indices[k]
                    else:
                        # 重置为 0, 1, 2...
                        file_idx = k
                    
                    # 使用 f-string 动态填充零: {file_idx:0{pad_width}d}
                    file_name = f"{file_idx:0{pad_width}d}.png"
                    cv2.imwrite(str(p / file_name), img)
            
            log_crops.append({"id": i+1, "bbox": bbox, "filename": fname})
            progress.setValue(i + 1)
            
        # 生成 Overview Map (使用 View Layer + 矩形框)
        self._create_overview_map(filtered_view_stack, rois, sub_name, output_dir)
        
        # Log to JSON
        json_path = output_dir / "processing_log.json"
        data = {}
        if json_path.exists():
            try: 
                with open(json_path, 'r') as f: data = json.load(f)
            except: pass
            
        data["batch_crop"] = {
            "data_layer": data_layer_name,
            "view_layer": view_layer_name,
            "count": count,
            "frame_filter": raw_range_text if raw_range_text else "All", # [修改] 记录筛选参数
            "rois": log_crops
        }
        with open(json_path, 'w') as f: json.dump(data, f, indent=2)

        progress.setValue(count)
        progress.close()
        
        self.status_label.setText(f"✅ Exported {count} crops to {output_dir.name}")
        self._force_view_active = False # 解锁视图
        
        QMessageBox.information(self, "Success", f"Successfully exported {count} crops!\nSaved to: {output_dir.name}")

    def _create_overview_map(self, image_stack, rois, sample_name, output_dir):
        """保存 Overview Map (可视层 + 矩形)"""
        if len(image_stack) == 0: return
        # 取中间帧
        bg_img = image_stack[len(image_stack)//2]
        
        # 归一化转 RGB
        if bg_img.dtype != np.uint8:
            mn, mx = bg_img.min(), bg_img.max()
            if mx > mn: bg_img = ((bg_img - mn)/(mx - mn)*255).astype(np.uint8)
            else: bg_img = bg_img.astype(np.uint8)
            
        pil_img = Image.fromarray(bg_img).convert("RGB")
        draw = ImageDraw.Draw(pil_img)
        try: font = ImageFont.truetype("arial.ttf", 24)
        except: font = ImageFont.load_default()

        for i, roi in enumerate(rois):
            ys, xs = roi[:, 0], roi[:, 1]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
            draw.text((x1, y1 - 25 if y1 > 25 else y1+5), f"NP{i+1}", fill="yellow", font=font)
            
        pil_img.save(output_dir / f"{sample_name}_Overview.png")
    
    def _parse_frame_indices(self, text, total_frames):
        """解析帧范围 (复用逻辑)"""
        if not text.strip(): return list(range(total_frames)) # 空字符串返回所有
        indices = set()
        try:
            parts = [p.strip() for p in text.split(',')]
            for p in parts:
                if not p: continue
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
            return list(range(total_frames)) # 解析失败回退到所有，或者抛错
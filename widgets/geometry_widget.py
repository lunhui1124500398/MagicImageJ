"""
几何变换控件 (交互优化版 + 批量ROI提取)
功能：
1. 旋转：画线后自动计算角度，自动切换到编辑模式。
2. 裁剪：单次裁剪模式。
3. [NEW] 批量ROI提取：支持一次绘制多个矩形，自动命名导出 (at-NP1, at-NP2...) 并生成索引图。
   - [Update] 支持选择导出格式：TIFF Stack 或 PNG Sequence (文件夹)。
   - [Fix] 智能选择逻辑优化：优先选择 Contrast_Enh > Contrast_ > Enh_，并在未找到时发出警告。
   - [Fix] 强制正方形功能 & 快速切换调整模式按钮。
   - [Fix] 自动清理 Annotation 工具遗留的 Interaction_Box 和 Preview_Overlay。
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QHBoxLayout, QComboBox, QGroupBox, 
                            QDoubleSpinBox, QScrollArea, QLineEdit, QFileDialog, QMessageBox, QCheckBox)
from qtpy.QtCore import Qt
import numpy as np
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
# 假设 core.geometry 已经存在于项目中
from core.geometry import (calculate_rotation_angle, rotate_image_stack, 
                           crop_image_stack, validate_bbox)
# 复用 utils 中的导出功能
from utils.video_export import export_to_tiff_stack
import napari
import json

class GeometryWidget(QWidget):
    """几何变换控件"""
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._is_updating = False # 防止信号递归调用
        self._setup_ui()

    def _setup_ui(self):
        # 1. 创建最外层布局 (用于放滚动条)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 2. 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        # 3. 创建内容容器
        content_widget = QWidget()
        layout = QVBoxLayout()

        title = QLabel("<h3>📐 Geometry & Batch Crop</h3>")
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

        # ========== 1. 旋转模块 ==========
        rotate_group = QGroupBox("1. Rotation (Horizon)")
        rotate_layout = QVBoxLayout()
        
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

        apply_rotate_btn = QPushButton("✅ Apply Rotation")
        apply_rotate_btn.clicked.connect(self._apply_rotation)
        rotate_layout.addWidget(apply_rotate_btn)

        rotate_group.setLayout(rotate_layout)
        layout.addWidget(rotate_group)

        # ========== 2. 单次裁剪模块 ==========
        crop_group = QGroupBox("2. Simple Crop (Single)")
        crop_layout = QVBoxLayout()
        draw_rect_btn = QPushButton("✏️ Draw Rect")
        draw_rect_btn.clicked.connect(self._draw_crop_rect)
        crop_layout.addWidget(draw_rect_btn)
        
        apply_crop_btn = QPushButton("✂️ Apply Crop (New Layer)")
        apply_crop_btn.clicked.connect(self._apply_crop)
        crop_layout.addWidget(apply_crop_btn)
        
        crop_group.setLayout(crop_layout)
        layout.addWidget(crop_group)

        # ========== 3. 批量ROI提取 (新功能) ==========
        batch_group = QGroupBox("3. Batch Extraction (Multi-ROI)")
        batch_group.setStyleSheet("QGroupBox { border: 1px solid #4CAF50; margin-top: 10px; } QGroupBox::title { color: #4CAF50; }")
        batch_layout = QVBoxLayout()

        # 物质名输入
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Substance Name:"))
        self.sample_name_edit = QLineEdit("at")
        self.sample_name_edit.setPlaceholderText("e.g. at, Au, sample1")
        name_layout.addWidget(self.sample_name_edit)
        batch_layout.addLayout(name_layout)

        # 格式选择
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Export Format:"))
        self.batch_format_combo = QComboBox()
        self.batch_format_combo.addItems(["TIFF Stack (.tiff)", "PNG Sequence (Folder)"])
        format_layout.addWidget(self.batch_format_combo)
        batch_layout.addLayout(format_layout)

        # 强制正方形选项
        self.force_square_check = QCheckBox("Force Square Crops")
        self.force_square_check.setChecked(True) # 默认开启
        self.force_square_check.setToolTip("If checked, newly drawn rectangles will automatically snap to a square shape.")
        batch_layout.addWidget(self.force_square_check)

        # 工具按钮行
        tools_layout = QHBoxLayout()
        
        # 开始绘制按钮
        self.start_batch_btn = QPushButton("✏️ Start Draw")
        self.start_batch_btn.setToolTip("Automatically selects the best layer and enters drawing mode.")
        self.start_batch_btn.clicked.connect(self._start_batch_mode)
        self.start_batch_btn.setStyleSheet("background-color: #444; font-weight: bold;")
        tools_layout.addWidget(self.start_batch_btn)

        # 调整按钮 (切换到 Select 模式)
        self.adjust_batch_btn = QPushButton("🖐️ Adjust / Select")
        self.adjust_batch_btn.setToolTip("Switch to Select mode to move or resize ROIs.")
        self.adjust_batch_btn.clicked.connect(self._switch_to_select_mode)
        tools_layout.addWidget(self.adjust_batch_btn)

        batch_layout.addLayout(tools_layout)
        batch_layout.addWidget(QLabel("<i>Draw multiple rectangles. Use 'Adjust' to tweak.</i>"))

        # 导出按钮
        self.export_batch_btn = QPushButton("💾 Export Crops & Map")
        self.export_batch_btn.clicked.connect(self._export_batch_crops)
        self.export_batch_btn.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 6px;")
        batch_layout.addWidget(self.export_batch_btn)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        # 状态信息
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

    def _switch_to_select_mode(self):
        """切换到选择模式，方便用户调整"""
        if "Batch_ROI" in self.viewer.layers:
            self.viewer.layers["Batch_ROI"].mode = 'select'
            self.status_label.setText("🖐️ Mode: Select/Adjust. Drag ROIs to move/resize.")
        else:
            self.status_label.setText("❌ No Batch ROI layer found.")

    # ... [保留原有的 _refresh_layers, _on_active_layer_changed, _clear_residue, _update_layer_focus 等辅助函数不变] ...
    def _refresh_layers(self):
        """刷新图层列表并自动选中活跃图层"""
        current_text = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if (hasattr(layer, 'data') and isinstance(layer.data, np.ndarray) 
                and len(layer.data.shape) == 3):
                self.layer_combo.addItem(layer.name)
        
        active_layer = self.viewer.layers.selection.active
        if active_layer and self.layer_combo.findText(active_layer.name) >= 0:
            self.layer_combo.setCurrentText(active_layer.name)
        elif self.layer_combo.findText(current_text) >= 0:
            self.layer_combo.setCurrentText(current_text)
        self.layer_combo.blockSignals(False)

    def _on_active_layer_changed(self, event=None):
        active_layer = self.viewer.layers.selection.active
        if active_layer:
            index = self.layer_combo.findText(active_layer.name)
            if index >= 0:
                self.layer_combo.setCurrentIndex(index)

    def _clear_residue(self, target_names):
        for name in target_names:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

    def _update_layer_focus(self, new_layer_name):
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                layer.visible = (layer.name == new_layer_name)
        if new_layer_name in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[new_layer_name]
        self._refresh_layers()
        self.layer_combo.setCurrentText(new_layer_name)

    # ... [保留原有的旋转和单次裁剪函数不变] ...
    def _draw_rotation_line(self):
        # FIX: Added Interaction_Box and Preview_Overlay to cleanup list
        self._clear_residue(["Drift_ROI", "Crop_ROI", "Rotation_Line", "Batch_ROI", "Interaction_Box", "Preview_Overlay"])
        layer = self.viewer.add_shapes(
            name="Rotation_Line", shape_type='line', edge_color='cyan', edge_width=4, face_color='transparent'
        )
        layer.events.data.connect(self._auto_calculate_angle)
        self.viewer.layers.selection.active = layer
        layer.mode = 'add_line'
        self.status_label.setText("✏️ Mode: Draw Line.")

    def _auto_calculate_angle(self, event=None):
        layer = self.viewer.layers["Rotation_Line"]
        if layer.mode == 'add_line' and len(layer.data) > 0:
            layer.mode = 'select'
        self._calculate_angle()

    def _calculate_angle(self):
        if "Rotation_Line" not in self.viewer.layers: return
        layer = self.viewer.layers["Rotation_Line"]
        if len(layer.data) == 0: return
        line_data = layer.data[-1]
        p1 = (line_data[0][1], line_data[0][0])
        p2 = (line_data[1][1], line_data[1][0])
        angle = calculate_rotation_angle((p1, p2))
        self.angle_spin.setValue(angle)
        self.status_label.setText(f"ℹ️ Angle: {angle:.2f}°")

    def _apply_rotation(self):
        layer_name = self.layer_combo.currentText()
        if not layer_name: return
        angle = self.angle_spin.value()
        image_stack = self.viewer.layers[layer_name].data
        try:
            rotated = rotate_image_stack(image_stack, angle)
            new_layer_name = f"Rotated_{layer_name}"
            self.viewer.add_image(rotated, name=new_layer_name, colormap='gray')
            # FIX: Cleanup extra layers
            self._clear_residue(["Rotation_Line", "Interaction_Box", "Preview_Overlay"])
            self._update_layer_focus(new_layer_name)
            self.status_label.setText(f"✅ Rotated by {angle:.2f}°")
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")

    def _draw_crop_rect(self):
        # FIX: Added Interaction_Box and Preview_Overlay to cleanup list
        self._clear_residue(["Drift_ROI", "Rotation_Line", "Crop_ROI", "Batch_ROI", "Interaction_Box", "Preview_Overlay"])
        layer = self.viewer.add_shapes(
            name="Crop_ROI", shape_type='rectangle', edge_color='yellow', face_color=[1, 1, 0, 0.01], edge_width=3
        )
        layer.mode = 'add_rectangle'
        layer.events.data.connect(self._on_crop_rect_drawn)
        self.viewer.layers.selection.active = layer
        self.status_label.setText("✏️ Mode: Draw Crop Rectangle.")

    def _on_crop_rect_drawn(self, event=None):
        layer = self.viewer.layers["Crop_ROI"]
        if layer.mode == 'add_rectangle' and len(layer.data) > 0:
            layer.mode = 'select'

    def _apply_crop(self):
        target_layer = self.layer_combo.currentText()
        if not target_layer or "Crop_ROI" not in self.viewer.layers: return
        shapes = self.viewer.layers["Crop_ROI"].data
        if len(shapes) == 0: return
        data = shapes[-1]
        ys, xs = data[:, 0], data[:, 1]
        x1, x2, y1, y2 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
        image_stack = self.viewer.layers[target_layer].data
        try:
            cropped = crop_image_stack(image_stack, (x1, y1, x2, y2))
            new_layer_name = f"Cropped_{target_layer}"
            self.viewer.add_image(cropped, name=new_layer_name, colormap='gray')
            # FIX: Cleanup extra layers
            self._clear_residue(["Crop_ROI", "Rotation_Line", "Drift_ROI", "Batch_ROI", "Interaction_Box", "Preview_Overlay"])
            self._update_layer_focus(new_layer_name)
            self.status_label.setText("✅ Crop applied.")
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")

    # =========================================================
    # NEW: Batch ROI Extraction Logic
    # =========================================================
    def _start_batch_mode(self):
        target_layer = None
        priorities = ["Contrast_Enh", "Contrast_", "Enh_", "Corrected_", "Rotated_"]
        
        all_layers = [self.layer_combo.itemText(i) for i in range(self.layer_combo.count())]
        for p in priorities:
            for name in all_layers:
                if name.startswith(p):
                    target_layer = name
                    break
            if target_layer: break
        
        if not target_layer:
            target_layer = self.layer_combo.currentText()
        
        if not target_layer:
            self.status_label.setText("❌ No image layer found.")
            return

        if target_layer in self.viewer.layers:
            self._update_layer_focus(target_layer)
            self.layer_combo.setCurrentText(target_layer)
            
            if not (target_layer.startswith("Contrast_") or target_layer.startswith("Enh_") or target_layer.startswith("Contrast_Enh")):
                reply = QMessageBox.warning(
                    self, 
                    "Enhancement Check", 
                    f"Selected layer '{target_layer}' does not appear to be enhanced (Contrast/Enh).\n\n"
                    "Do you want to continue?",
                    QMessageBox.Yes | QMessageBox.No, 
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
        
        # FIX: Added Interaction_Box and Preview_Overlay to cleanup list
        self._clear_residue(["Drift_ROI", "Crop_ROI", "Rotation_Line", "Batch_ROI", "Interaction_Box", "Preview_Overlay"])
        
        roi_layer = self.viewer.add_shapes(
            name="Batch_ROI",
            shape_type='rectangle',
            edge_color='#00FF00', 
            face_color=[0, 1, 0, 0.05],
            edge_width=2,
            text={
                'string': '{label}', 
                'size': 12,
                'color': 'white',
                'anchor': 'upper_left',
                'translation': [-5, -5]
            }
        )
        
        roi_layer.events.data.connect(self._on_batch_data_change)
        
        roi_layer.mode = 'add_rectangle'
        self.viewer.layers.selection.active = roi_layer
        
        self.status_label.setText(f"✏️ Batch: Draw rects on '{target_layer}'. Auto-Square: {self.force_square_check.isChecked()}")

    def _on_batch_data_change(self, event=None):
        """当批量ROI数据变化时：1. 强制正方形 (如果开启); 2. 更新标号"""
        if self._is_updating: return
        if "Batch_ROI" not in self.viewer.layers: return
        
        layer = self.viewer.layers["Batch_ROI"]
        n_shapes = len(layer.data)
        if n_shapes == 0: return
        
        self._is_updating = True
        
        try:
            # --- 1. 强制正方形逻辑 ---
            if self.force_square_check.isChecked():
                data_list = layer.data
                new_data_list = []
                modified = False
                
                for roi in data_list:
                    # ROI: [[y1, x1], [y2, x1], [y2, x2], [y1, x2]] (approx)
                    ys, xs = roi[:, 0], roi[:, 1]
                    y1, y2 = np.min(ys), np.max(ys)
                    x1, x2 = np.min(xs), np.max(xs)
                    
                    h, w = y2 - y1, x2 - x1
                    
                    # 允许 1px 误差
                    if abs(w - h) > 1.0:
                        # 取最大边长
                        side = max(w, h)
                        # 中心点
                        cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
                        
                        # 新坐标
                        ny1, ny2 = cy - side/2, cy + side/2
                        nx1, nx2 = cx - side/2, cx + side/2
                        
                        # 构造矩形 (Napari 顺序: TL, BL, BR, TR - 或类似，只要四个角对就行)
                        new_rect = np.array([
                            [ny1, nx1], [ny2, nx1], [ny2, nx2], [ny1, nx2]
                        ])
                        new_data_list.append(new_rect)
                        modified = True
                    else:
                        new_data_list.append(roi)
                
                if modified:
                    layer.data = new_data_list
                    # 注意：设置 layer.data 会再次触发事件，所以必须有 _is_updating 锁

            # --- 2. 更新标号逻辑 ---
            labels = [str(i+1) for i in range(len(layer.data))]
            new_features = {'label': labels}
            
            if hasattr(layer, 'features'):
                layer.features = new_features
            elif hasattr(layer, 'properties'):
                layer.properties = new_features
            
            # 不需要调用 layer.refresh()，设置 features/data 会自动刷新

        finally:
            self._is_updating = False

    def _export_batch_crops(self):
        """Modified: Uses Archive Path and updates JSON log."""
        if "Batch_ROI" not in self.viewer.layers or len(self.viewer.layers["Batch_ROI"].data) == 0:
            self.status_label.setText("❌ No ROIs.")
            return
        target = self.layer_combo.currentText()
        if target not in self.viewer.layers: return
        
        stack = self.viewer.layers[target].data
        sub_name = self.sample_name_edit.text().strip() or "sample"

        # === Auto-detect Archive ===
        from qtpy.QtCore import QSettings
        archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
        # archive_path = self.viewer.metadata.get('archive_path')
        
        if archive_path and Path(archive_path).exists():
            # If archived, save to root of archive (as requested)
            output_dir = Path(archive_path)
            self.status_label.setText(f"📂 Saving to Archive: {output_dir.name}")
        else:
            d = QFileDialog.getExistingDirectory(self, "Select Output")
            if not d: return
            output_dir = Path(d) / sub_name
            output_dir.mkdir(parents=True, exist_ok=True)

        is_tiff = "TIFF" in self.batch_format_combo.currentText()
        
        try:
            rois = self.viewer.layers["Batch_ROI"].data
            count = 0
            log_crops = [] # For JSON

            for i, roi in enumerate(rois):
                ys, xs = roi[:, 0], roi[:, 1]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                if not validate_bbox(stack.shape[1:], bbox): continue
                
                crop = crop_image_stack(stack, bbox)
                fname = f"{sub_name}-NP{i+1}"
                
                if is_tiff:
                    export_to_tiff_stack(crop, str(output_dir / f"{fname}.tiff"))
                else:
                    p = output_dir / fname
                    p.mkdir(exist_ok=True)
                    for f, img in enumerate(crop):
                        if img.dtype in [np.float32, np.float64]:
                            mn, mx = img.min(), img.max()
                            img = ((img - mn)/(mx - mn)*255).astype(np.uint8) if mx > mn else img.astype(np.uint8)
                        cv2.imwrite(str(p / f"{f:05d}.png"), img)
                
                log_crops.append({"id": i+1, "bbox": bbox, "filename": fname})
                count += 1
            
            self._create_overview_map(stack, rois, sub_name, output_dir)
            
            # === Update JSON Log ===
            json_path = output_dir / "processing_log.json"
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f: data = json.load(f)
                except: data = {}
            else: data = {}
            
            data["geometry_actions"] = {
                "source_layer": target,
                "rotation_angle": self.angle_spin.value(),
                "crop_count": count,
                "rois": log_crops
            }
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)

            QMessageBox.information(self, "Done", f"Saved {count} crops to:\n{output_dir}")
            self.status_label.setText(f"✅ Saved {count} crops.")
            
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            print(e)

    def _create_overview_map(self, image_stack, rois, sample_name, output_dir):
        if len(image_stack) > 0:
            mid_idx = len(image_stack) // 2
            bg_img = image_stack[mid_idx]
        else:
            return

        if bg_img.dtype != np.uint8:
            img_min, img_max = bg_img.min(), bg_img.max()
            if img_max > img_min:
                bg_img = ((bg_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                bg_img = bg_img.astype(np.uint8)
        
        pil_img = Image.fromarray(bg_img).convert("RGB")
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        for i, roi in enumerate(rois):
            ys, xs = roi[:, 0], roi[:, 1]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
            
            label = f"NP{i+1}"
            text_pos = (x1, y1 - 25 if y1 > 25 else y1 + 5)
            draw.text(text_pos, label, fill="yellow", font=font)

        map_filename = f"{sample_name}_Overview.png"
        pil_img.save(output_dir / map_filename)
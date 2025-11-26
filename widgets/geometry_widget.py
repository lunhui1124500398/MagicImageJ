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
from qtpy.QtCore import Qt, QTimer, QSettings, QThread, Signal
import numpy as np
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
from core.geometry import (calculate_rotation_angle, rotate_image_stack, flip_image_stack,
                           crop_image_stack, validate_bbox)
from utils.video_export import export_to_tiff_stack
import napari
import json
from widgets.settings_widget import GlobalConfig

class BatchExportThread(QThread):
    """
    后台导出线程 (修复版)
    修复日志:
    - [Fix] 解决 Windows 下 cv2.imwrite 无法保存中文/特殊字符路径的问题。
    - [Fix] 增加 np.asarray 确保数据不是 Dask 格式。
    """
    progress = Signal(int)          # 发送进度信号
    finished = Signal(int, str)     # 完成信号 (数量, 路径名)
    error = Signal(str)             # 错误信号

    def __init__(self, params):
        super().__init__()
        self.p = params

    def run(self):
        try:
            # --- 解包参数 ---
            data_stack = self.p['data_stack']
            rois = self.p['rois']
            frame_ranges_list = self.p['frame_ranges_list']
            global_range_text = self.p['global_range_text']
            output_dir = self.p['output_dir']
            sub_name = self.p['sub_name']
            suffix_fmt = self.p['suffix_fmt']
            is_tiff = self.p['is_tiff']
            keep_idx = self.p['keep_idx']
            pad = self.p['pad']
            
            total_frames = data_stack.shape[0]
            log_crops = []
            count = len(rois)

            for i, roi in enumerate(rois):
                if self.isInterruptionRequested(): break

                # 1. 计算坐标
                ys, xs = roi[:, 0], roi[:, 1]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                
                # 2. 确定帧范围
                specific_range = str(frame_ranges_list[i]).strip() if i < len(frame_ranges_list) else ""
                range_to_use = specific_range if specific_range else global_range_text
                
                # 3. 解析需要导出的帧索引
                selected_indices = parse_indices_helper(range_to_use, total_frames)
                
                if not selected_indices:
                    continue 
                    
                # 4. 动态切片 (使用 np.asarray 确保转为内存中的 Numpy 数组，防止 Dask 懒加载导致写入失败)
                filtered_data_stack = np.asarray(data_stack[selected_indices])
                
                # 5. 执行裁剪
                crop = crop_image_stack(filtered_data_stack, bbox)

                # 6. 生成文件名
                if "{}" not in suffix_fmt:
                    suffix_str = f"{suffix_fmt}{i+1}"
                else:
                    suffix_str = suffix_fmt.replace("{}", str(i+1))
                fname = f"{sub_name}{suffix_str}"
                
                # 7. 保存文件
                if is_tiff:
                    # TIFF 格式
                    export_to_tiff_stack(crop, str(output_dir / f"{fname}.tiff"))
                else:
                    # PNG 序列格式
                    p = output_dir / fname
                    p.mkdir(exist_ok=True) # 文件夹创建成功，说明路径没问题
                    
                    for k, img in enumerate(crop):
                        # 数据归一化与转换
                        if img.dtype in [np.float32, np.float64]:
                            mn, mx = img.min(), img.max()
                            if mx > mn: img = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
                            else: img = img.astype(np.uint8)
                        elif img.dtype != np.uint8:
                            # 16bit 转 8bit (可选，为了兼容性)
                            # 如果你想保留 16bit png，可以注释掉这行，但部分看图软件看不了
                            pass 
                        
                        # 编号逻辑
                        file_idx = selected_indices[k] if keep_idx else k
                        file_name = f"{file_idx:0{pad}d}.png"
                        save_path = str(p / file_name)

                        # === [核心修复] 使用 imencode + tofile 支持中文/特殊路径 ===
                        try:
                            # cv2.imwrite 不支持中文路径，改用这种写法：
                            is_success, im_buf = cv2.imencode(".png", img)
                            if is_success:
                                im_buf.tofile(save_path)
                            else:
                                print(f"Warning: Failed to encode frame {k}")
                        except Exception as save_err:
                            print(f"Save Error: {save_err}")
                        # =======================================================
                
                # 8. 记录日志
                log_crops.append({
                    "id": i+1, "bbox": bbox, "filename": fname, 
                    "frame_range_used": range_to_use if range_to_use else "All"
                })
                
                self.progress.emit(i + 1)
            
            # 写入日志
            json_path = output_dir / "processing_log.json"
            log_data = {}
            if json_path.exists():
                try: 
                    with open(json_path, 'r') as f: log_data = json.load(f)
                except: 
                    pass
            
            log_data["batch_crop"] = {
                "data_layer": self.p['data_layer_name'],
                "count": count,
                "global_filter": global_range_text,
                "rois": log_crops
            }
            with open(json_path, 'w') as f: json.dump(log_data, f, indent=2)
            
            self.finished.emit(count, str(output_dir.name))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

def parse_indices_helper(text, total_frames):
    if not text.strip() or text.strip().lower() == "all": 
        return list(range(total_frames))
    indices = set()
    try:
        parts = [p.strip() for p in text.split(',')]
        for p in parts:
            if not p: continue
            if '-' in p:
                start, end = map(int, p.split('-'))
                start = max(0, start); end = min(total_frames - 1, end)
                if start <= end: indices.update(range(start, end + 1))
            else:
                idx = int(p)
                if 0 <= idx < total_frames: indices.add(idx)
        return sorted(list(indices))
    except:
        return list(range(total_frames))

class RotationThread(QThread):
    progress = Signal(int, int)
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, stack, angle, expand):
        super().__init__()
        self.stack = stack
        self.angle = angle
        self.expand = expand

    def run(self):
        try:
            def cb(c, t):
                self.progress.emit(c, t)
            
            # 调用核心算法
            res = rotate_image_stack(self.stack, self.angle, expand=self.expand, progress_callback=cb)
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))

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

        # --- 翻转按钮区 (新增) ---
        h_flip = QHBoxLayout()
        btn_flip_h = QPushButton("↔️ Flip Horz")
        btn_flip_h.clicked.connect(lambda: self._apply_flip('horizontal'))
        btn_flip_v = QPushButton("↕️ Flip Vert")
        btn_flip_v.clicked.connect(lambda: self._apply_flip('vertical'))
        h_flip.addWidget(btn_flip_h)
        h_flip.addWidget(btn_flip_v)
        rotate_layout.addLayout(h_flip)
        
        # --- 旋转区 (保留但简化) ---
        h_angle = QHBoxLayout()
        h_angle.addWidget(QLabel("Angle:"))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-360, 360); self.angle_spin.setDecimals(2)
        h_angle.addWidget(self.angle_spin)
        
        # Recalc 功能保留在小按钮里，以防万一还需要
        calc_btn = QPushButton("📏 Line-Calc") 
        calc_btn.setToolTip("Draw a line to calculate angle")
        calc_btn.clicked.connect(self._draw_rotation_line)
        h_angle.addWidget(calc_btn)
        rotate_layout.addLayout(h_angle)
    
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
        name_layout.addWidget(self.sample_name_edit, 1)

        # === [新增功能 4] 自定义后缀输入 ===
        name_layout.addWidget(QLabel("Suffix:"))
        self.suffix_edit = QLineEdit("-NP{}")
        self.suffix_edit.setPlaceholderText("e.g. -NP{} or -{}")
        # self.suffix_edit.setFixedWidth(80)
        self.suffix_edit.setMinimumWidth(100)
        self.suffix_edit.setToolTip("Use {} as placeholder for number.\nExample: '-NP{}' -> '-NP1'")
        name_layout.addWidget(self.suffix_edit, 2)
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

        self.btn_set_specific_range = QPushButton("📌 Set for Selected")
        self.btn_set_specific_range.setToolTip("Apply the text in the box to the CURRENTLY SELECTED ROI only.\nExample: Select ROI 2, type '0-50', click this button.")
        self.btn_set_specific_range.clicked.connect(self._set_range_for_selected_roi)
        # 稍微改个样式区分一下
        self.btn_set_specific_range.setStyleSheet("background-color: #555; font-size: 10px; padding: 4px;")
        frame_layout.addWidget(self.btn_set_specific_range)
        
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
        # UI 锁定
        self.status_label.setText("⏳ Rotating...")
        
        # 进度条
        self.rot_progress = QProgressDialog(f"Rotating {angle:.1f}°...", "Cancel", 0, len(image_stack), self)
        self.rot_progress.setWindowModality(Qt.WindowModal)
        self.rot_progress.show()
        
        # 启动线程
        self.rot_thread = RotationThread(image_stack, angle, expand)
        self.rot_thread.progress.connect(lambda c, t: self.rot_progress.setValue(c))
        
        def on_finished(rotated):
            self.rot_progress.close()
            try:
                new_name = f"Rotated_{layer_name}"
                new_layer = self.viewer.add_image(rotated, name=new_name, colormap='gray', metadata={'source_layer': layer_name})
                if layer_name in self.viewer.layers:
                    self.viewer.layers[layer_name].visible = False
                
                # 清理 Line
                self._clear_residue(["Rotation_Line"])
                
                # 自动切换 Simple Crop 的目标图层
                self.simple_crop_combo.setCurrentText(new_name)
                self.viewer.layers.selection.active = new_layer
                undo_key = GlobalConfig.get_napari_shortcut("shortcut_undo_drift")
                @new_layer.bind_key(undo_key, overwrite=True)
                def undo_rotation(layer):
                    # 1. 恢复原图层
                    src = layer.metadata.get('source_layer')
                    if src and src in self.viewer.layers:
                        self.viewer.layers[src].visible = True
                        self.viewer.layers.selection.active = self.viewer.layers[src]
                    # 2. 删除当前层
                    self.viewer.layers.remove(layer)
                    self.status_label.setText("↩️ Rotation Undone.")
                
                self.status_label.setText(f"✅ Rotated {angle:.1f}° (Expand={expand})")
            except Exception as e:
                self.status_label.setText(f"Error showing result: {e}")
        
        def on_error(err):
            self.rot_progress.close()
            self.status_label.setText(f"❌ Rotation Error: {err}")

        self.rot_thread.finished.connect(on_finished)
        self.rot_thread.error.connect(on_error)
        self.rot_thread.start()

    def _apply_flip(self, direction):
        layer_name = self.rotate_layer_combo.currentText()
        if not layer_name: return
        image_stack = self.viewer.layers[layer_name].data
        try:
            flipped = flip_image_stack(image_stack, direction)
            suffix = "FlipH" if direction == 'horizontal' else "FlipV"
            new_name = f"{suffix}_{layer_name}"
            self.viewer.add_image(flipped, name=new_name, colormap='gray')
            
            # 自动切换下拉框目标
            self.rotate_layer_combo.setCurrentText(new_name)
            self.simple_crop_combo.setCurrentText(new_name)
            self.viewer.layers.selection.active = self.viewer.layers[new_name]
            
            self.status_label.setText(f"✅ Applied {direction} flip.")
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
        def on_data_change(event):
            # 只有在 add_rectangle 模式且有数据时才切换，防止循环触发
            if layer.mode == 'add_rectangle' and len(layer.data) > 0:
                layer.mode = 'select' # 切换到选择/编辑模式
                self.viewer.layers.selection.active = layer # 确保图层被选中以便编辑
                self.status_label.setText("🖐️ Mode: Adjust Crop Rect (Drag corners to resize)")

        layer.events.data.connect(on_data_change)
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
        
        new_layer = self.viewer.add_image(cropped, name=new_name, colormap='gray')
        
        # 清理 Crop ROI
        self._clear_residue(["Crop_ROI"])
        self.viewer.layers.selection.active = new_layer

        # === [新增功能 1] Crop之后清理显示，只显示Crop出的图层 ===
        # 只隐藏 Image 图层，且不要隐藏本图层
        # 也不要隐藏预览层（虽然 clear_residue 已经清理了，但为了健壮性）
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image) and layer.name != new_name:
                # 排除掉一些不想被误伤的辅助层（可选）
                if "Preview" not in layer.name: 
                    layer.visible = False
        # =====================================================
        undo_key = GlobalConfig.get_napari_shortcut("shortcut_undo_drift") 
        
        @new_layer.bind_key(undo_key, overwrite=True)
        def undo_crop(layer):
            # 1. 删除当前裁剪层
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
            
            # 2. 恢复原图层可见性
            if target in self.viewer.layers:
                self.viewer.layers[target].visible = True
                self.viewer.layers.selection.active = self.viewer.layers[target]
            
            # 3. 恢复 ROI 绘制层 (方便重画)
            self._draw_crop_rect()
            self.status_label.setText("↩️ Crop Undone.")

        self.status_label.setText(f"✅ Crop applied. Press '{undo_key}' to Undo.")
        # self.status_label.setText(f"✅ Crop applied: {new_name}")

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
       # [修复 Dtype 报错] 显式初始化 features 为字符串类型
        # 即使是空列表，也声明一下这列是放字符串的，防止 pandas 推断错误
        roi_layer = self.viewer.add_shapes(
            name="Batch_ROI",
            shape_type='rectangle',
            edge_color='#00FF00', 
            face_color=[0, 1, 0, 0.05],
            edge_width=2,
            text={
                'string': '{label}\n{frame_info}', 
                'size': 10, 
                'color': '#00FF00', 
                'anchor': 'upper_left', 
                'translation': [-5, -5]
            },
            features={
                'label': [], 
                'frame_range': [], 
                'frame_info': []
            }
        )
        roi_layer.events.data.connect(self._on_batch_data_change)
        roi_layer.mode = 'add_rectangle'
        
        # [新增] 绑定 Undo (删除上一个画的框)
        undo_key = GlobalConfig.get_napari_shortcut("shortcut_undo_drift")
        @roi_layer.bind_key(undo_key)
        def undo_batch_rect(layer):
            if layer.mode == 'add_rectangle' and len(layer.data) > 0:
                # 移除最后一个数据
                layer.data = layer.data[:-1]
                # features 会触发 _on_batch_data_change 自动重建，不用手动删
                self.status_label.setText("↩️ Last ROI removed.")
            elif layer.mode == 'select':
                self.status_label.setText("ℹ️ Undo available in Draw Mode.")

        self.status_label.setText(f"✏️ Drawing on '{view_layer}'. (Ctrl+Z to Undo last)")

    def _set_range_for_selected_roi(self):
        """
        设置选中 ROI 的特定帧范围
        [Fix] 使用 Python List 中转，彻底解决 Pandas ChainedAssignmentError 和 Dtype 问题
        """
        if "Batch_ROI" not in self.viewer.layers: return
        layer = self.viewer.layers["Batch_ROI"]
        
        # 1. 获取选中项
        selected_idxs = list(layer.selected_data)
        if not selected_idxs:
            self.status_label.setText("⚠️ No ROI selected. Select a green box first.")
            return
        
        # 2. 获取输入文本
        range_str = self.batch_frame_edit.text().strip()
        
        # === [核心修复] 数据中转 ===
        # 不直接操作 layer.features (DataFrame)，而是提取为纯 Python 列表
        # 这样做既快又安全，完全避开 Pandas 的警告
        
        current_features = layer.features
        n_shapes = len(layer.data)
        
        # 提取列，如果列不存在则初始化为空列表
        # list(...) 强制转换，切断与 Pandas 的引用关联
        labels = list(current_features.get('label', []))
        ranges = list(current_features.get('frame_range', []))
        infos = list(current_features.get('frame_info', []))
        
        # 3. 对齐数据长度 (防御性编程)
        # 防止因手动删除等操作导致 features 长度滞后
        while len(labels) < n_shapes: labels.append(str(len(labels)+1))
        while len(ranges) < n_shapes: ranges.append("")
        while len(infos) < n_shapes: infos.append("")
            
        # 4. 修改 Python List (安全操作)
        for idx in selected_idxs:
            if idx < len(ranges): # 再次检查越界，确保安全
                if not range_str or range_str.lower() == "global":
                    ranges[idx] = "" # 空字符串表示使用全局设置
                    infos[idx] = ""
                else:
                    ranges[idx] = str(range_str) # 强制转为字符串
                    infos[idx] = f"[{range_str}]"
        
        # 5. 整体赋值回 Features
        # Napari 会接收这个字典并自动更新底层的 DataFrame
        layer.features = {
            'label': labels,
            'frame_range': ranges,
            'frame_info': infos
        }
        
        # 6. 刷新界面
        layer.refresh()
        self.status_label.setText(f"✅ Set range '{range_str}' for {len(selected_idxs)} ROI(s).")

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
            n_shapes = len(layer.data)
            labels = [str(i+1) for i in range(n_shapes)]
            current_ranges = layer.features.get('frame_range', [])
            current_infos = layer.features.get('frame_info', [])
            range_list = [str(x) for x in current_ranges]
            info_list = [str(x) for x in current_infos]
            # 补齐长度
            while len(range_list) < n_shapes:
                range_list.append("")
                info_list.append("")
            # 截断
            range_list = range_list[:n_shapes]
            info_list = info_list[:n_shapes]
            
            layer.features = {
                'label': labels,
                'frame_range': range_list,
                'frame_info': info_list
            }
            # if hasattr(layer, 'features'): layer.features = {'label': labels}
            # elif hasattr(layer, 'properties'): layer.properties = {'label': labels}
            
        finally:
            self._is_updating = False

    def _export_batch_crops(self):
        # 1. 基础校验
        data_layer_name = self.batch_data_combo.currentText()
        view_layer_name = self.batch_view_combo.currentText()
        if "Batch_ROI" not in self.viewer.layers or not len(self.viewer.layers["Batch_ROI"].data):
            self.status_label.setText("❌ No ROIs defined.")
            return
        if not data_layer_name or data_layer_name not in self.viewer.layers: return

        # 2. 准备参数 (主线程只负责收集数据)
        # 提取 Numpy 数组数据，传递给线程是安全的（引用传递，不占额外内存）
        data_stack = self.viewer.layers[data_layer_name].data
        view_stack = self.viewer.layers[view_layer_name].data # 留给回调函数画 Map 用
        
        layer = self.viewer.layers["Batch_ROI"]
        rois = layer.data # 这是一个 List[np.ndarray]，拷贝给线程
        
        # 安全提取 features (防止线程运行时被修改)
        roi_features = layer.features
        frame_ranges_list = list(roi_features.get('frame_range', [""] * len(rois)))
        global_range_text = self.batch_frame_edit.text().strip()

        # 3. 路径处理
        sub_name = self.sample_name_edit.text().strip() or "sample"
        archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
        if archive_path and Path(archive_path).exists():
            output_dir = Path(archive_path)
        else:
            last_import_folder = QSettings("NapariUser", "Importer").value("last_folder", "")
            start_dir = ""
            
            if last_import_folder and Path(last_import_folder).exists():
                # 设为导入文件夹的"同级目录" (即父目录)
                # 例如导入的是 D:/Data/Session1/Raw，默认打开 D:/Data/Session1
                start_dir = str(Path(last_import_folder).parent)
            d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if not d: return
            output_dir = Path(d) / sub_name
            output_dir.mkdir(parents=True, exist_ok=True)

        # 4. 打包参数字典 (传递给线程)
        params = {
            'data_stack': data_stack,
            'rois': rois,
            'frame_ranges_list': frame_ranges_list,
            'global_range_text': global_range_text,
            'output_dir': output_dir,
            'sub_name': sub_name,
            'suffix_fmt': self.suffix_edit.text(),
            'is_tiff': "TIFF" in self.batch_format_combo.currentText(),
            'keep_idx': self.keep_index_check.isChecked(),
            'pad': self.padding_spin.value(),
            'data_layer_name': data_layer_name 
        }

        # 5. 显示模态进度条 (这下主界面不会卡死了，进度条也能刷新了)
        self.batch_progress = QProgressDialog("Exporting Crops...", "Cancel", 0, len(rois), self)
        self.batch_progress.setWindowModality(Qt.WindowModal)
        self.batch_progress.setMinimumDuration(0)
        self.batch_progress.canceled.connect(self._on_export_cancel)
        self.batch_progress.show()

        # 6. 创建并启动线程
        self.export_thread = BatchExportThread(params)
        
        # 绑定信号：线程发出的进度 -> 更新进度条
        self.export_thread.progress.connect(self.batch_progress.setValue)
        
        # 绑定信号：线程完成 -> 执行后续操作 (画 Map，弹提示)
        # 注意：这里用 lambda 把 view_stack 传给回调，因为画 map 很快，可以在主线程做
        self.export_thread.finished.connect(lambda c, path: self._on_export_finished(c, path, view_stack, rois, sub_name, output_dir))
        self.export_thread.error.connect(self._on_export_error)
        
        self.export_thread.start() # 🚀 启动！
        
    def _on_export_cancel(self):
        """用户点击取消按钮时触发"""
        if self.export_thread.isRunning():
            self.export_thread.requestInterruption()
            self.status_label.setText("⚠️ Export canceled.")

    def _on_export_finished(self, count, path_name, view_stack, rois, sub_name, output_dir):
        """线程任务完成后触发"""
        self.batch_progress.close()
        
        # 生成 Map (Pillow 画图很快，不阻塞 UI，放在主线程没问题)
        self._create_overview_map(view_stack, rois, sub_name, output_dir)
        
        self.status_label.setText(f"✅ Exported {count} crops.")
        self._force_view_active = False
        QMessageBox.information(self, "Success", f"Exported {count} crops!\nSaved to: {path_name}")

    def _on_export_error(self, err):
        """线程报错时触发"""
        self.batch_progress.close()
        self.status_label.setText(f"❌ Error: {err}")
        QMessageBox.critical(self, "Export Error", str(err))

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
    
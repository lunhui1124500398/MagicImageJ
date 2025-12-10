"""
几何变换控件 (交互优化版 + 批量ROI提取 + 归档集成)
修改日志:
- [Fix] 文件夹命名逻辑更新：辅助文件夹 (lrtem/mask等) 现在会包含主后缀 (MainSuffix)。
  例如：如果主后缀是 "_contrasted"，生成的去噪文件夹将是 "..._NP1_contrasted_lrtem"。
- [Req 3] 文件夹命名: Date_Datasetn_Substance_NPx_{MainSuffix}
- [Req 4/5] 辅助文件夹使用配置的后缀。
- [Req 6] Overview Map 使用当前显示帧。
- [Fix Date] 新增日期输入框，默认从归档信息加载，导出时优先使用输入框中的日期。
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
import os
import datetime
from widgets.settings_widget import GlobalConfig, tr

class BatchExportThread(QThread):
    """
    后台导出线程 (增强版)
    支持自动创建多层级文件夹结构，无硬编码后缀
    """
    progress = Signal(int)
    finished = Signal(int, str)
    error = Signal(str)

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
            
            # Naming components
            date_str = self.p['date_str']
            dataset_id = self.p['dataset_id']
            sub_name = self.p['sub_name']
            
            # === Suffixes ===
            suffix_main = self.p['suffix_main'] # 来自 UI 输入框 (e.g., "_contrasted" or "_origin")
            aux_suffixes = self.p['aux_suffixes'] # 来自 Settings (e.g., {'lrtem': '_lrtem', ...})
            
            # Flags
            create_denoise = self.p['create_denoise']
            create_refine = self.p['create_refine']
            
            is_tiff = self.p['is_tiff']
            keep_idx = self.p['keep_idx']
            pad = self.p['pad']
            
            total_frames = data_stack.shape[0]
            log_crops = []
            count = len(rois)

            for i, roi in enumerate(rois):
                if self.isInterruptionRequested(): break

                # 1. 计算 ROI 坐标
                ys, xs = roi[:, 0], roi[:, 1]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                
                # 2. 确定帧范围
                specific_range = str(frame_ranges_list[i]).strip() if i < len(frame_ranges_list) else ""
                range_to_use = specific_range if specific_range else global_range_text
                selected_indices = parse_indices_helper(range_to_use, total_frames)
                
                if not selected_indices: continue 
                    
                # 3. 动态切片 & 裁剪
                filtered_data_stack = np.asarray(data_stack[selected_indices])
                crop = crop_image_stack(filtered_data_stack, bbox)

                # 4. 构建基础名称
                # 格式: 20251104_Dataset3_CRY2_NP1
                base_name = f"{date_str}_{dataset_id}_{sub_name}_NP{i+1}"
                
                # 主数据文件夹 (加上 UI 输入的主后缀)
                # e.g. ..._NP1_contrasted
                folder_main = output_dir / f"{base_name}{suffix_main}"
                folder_main.mkdir(parents=True, exist_ok=True)
                
                # === [修改点] 辅助文件夹现在包含 suffix_main ===
                # e.g. ..._NP1_contrasted_lrtem
                if create_denoise:
                    (output_dir / f"{base_name}{suffix_main}{aux_suffixes['lrtem']}").mkdir(exist_ok=True)
                    (output_dir / f"{base_name}{suffix_main}{aux_suffixes['hrtem']}").mkdir(exist_ok=True)
                    # (output_dir / f"{base_name}{suffix_main}{aux_suffixes['mask']}").mkdir(exist_ok=True)
                
                if create_refine:
                    (output_dir / f"{base_name}{suffix_main}{aux_suffixes['mask']}").mkdir(exist_ok=True)
                    (output_dir / f"{base_name}{suffix_main}{aux_suffixes['mask_new']}").mkdir(exist_ok=True)
                # ===============================================

                # 5. 保存文件到 main 文件夹
                if is_tiff:
                    export_to_tiff_stack(crop, str(folder_main / f"{base_name}.tiff"))
                else:
                    for k, img in enumerate(crop):
                        # 归一化
                        if img.dtype in [np.float32, np.float64]:
                            mn, mx = img.min(), img.max()
                            if mx > mn: img = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
                            else: img = img.astype(np.uint8)
                        
                        # 命名
                        file_idx = selected_indices[k] if keep_idx else k
                        file_name = f"{file_idx:0{pad}d}.png"
                        save_path = str(folder_main / file_name)

                        try:
                            # 兼容中文路径
                            is_success, im_buf = cv2.imencode(".png", img)
                            if is_success: im_buf.tofile(save_path)
                        except Exception as save_err:
                            print(f"Save Error: {save_err}")
                
                # 6. 日志
                log_crops.append({
                    "id": i+1, "bbox": bbox, "folder": folder_main.name, 
                    "frame_range_used": range_to_use if range_to_use else "All"
                })
                self.progress.emit(i + 1)
            
            # 写入日志
            json_path = output_dir / "processing_log.json"
            log_data = {}
            if json_path.exists():
                try: 
                    with open(json_path, 'r') as f: 
                        log_data = json.load(f)
                except: 
                    pass
            
            log_data["batch_crop"] = {
                "data_layer": self.p['data_layer_name'],
                "count": count,
                "global_filter": global_range_text,
                "suffixes_used": {
                    "main": suffix_main,
                    "aux": aux_suffixes
                },
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
            res = rotate_image_stack(self.stack, self.angle, expand=self.expand, progress_callback=cb)
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))

class GeometryWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._is_updating = False 
        self._force_view_active = False 
        self._setup_ui()

        # 初始化与热更新
        self._load_params_from_config()
        GlobalConfig.signals.config_updated.connect(self._load_params_from_config)
        
        self.viewer.layers.events.reordered.connect(self._enforce_view_visibility)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
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

        title = QLabel(f"<h3>📐 {tr('Geometry & Batch Extraction (Multi-ROI)')}</h3>")
        layout.addWidget(title)

        refresh_btn = QPushButton(f"🔄 {tr('Refresh All Layers')}")
        refresh_btn.clicked.connect(self._refresh_layers)
        layout.addWidget(refresh_btn)

        # ========== 1. 旋转模块 ==========
        rotate_group = QGroupBox(tr("1. Rotation (Horizon)"))
        rotate_layout = QVBoxLayout()
        h_rot_layer = QHBoxLayout()
        h_rot_layer.addWidget(QLabel(tr("Target:")))
        self.rotate_layer_combo = QComboBox()
        h_rot_layer.addWidget(self.rotate_layer_combo)
        rotate_layout.addLayout(h_rot_layer)

        rotate_layout.addWidget(QLabel(tr("Draw a line to define horizon:")))
        draw_line_btn = QPushButton(f"✏️ {tr('Draw Horizon Line')}")
        draw_line_btn.clicked.connect(self._draw_rotation_line)
        rotate_layout.addWidget(draw_line_btn)

        h_flip = QHBoxLayout()
        btn_flip_h = QPushButton(f"↔️ {tr('Flip Horz')}")
        btn_flip_h.clicked.connect(lambda: self._apply_flip('horizontal'))
        btn_flip_v = QPushButton(f"↕️ {tr('Flip Vert')}")
        btn_flip_v.clicked.connect(lambda: self._apply_flip('vertical'))
        h_flip.addWidget(btn_flip_h)
        h_flip.addWidget(btn_flip_v)
        rotate_layout.addLayout(h_flip)
        
        h_angle = QHBoxLayout()
        h_angle.addWidget(QLabel(tr("Angle:")))
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-360, 360); self.angle_spin.setDecimals(2)
        h_angle.addWidget(self.angle_spin)
        calc_btn = QPushButton(f"📏 {tr('Line-Calc')}") 
        calc_btn.setToolTip(tr("Draw a line to calculate angle"))
        calc_btn.clicked.connect(self._draw_rotation_line)
        h_angle.addWidget(calc_btn)
        rotate_layout.addLayout(h_angle)
    
        self.enlarge_check = QCheckBox(tr("Enlarge Canvas (Fit All)"))
        self.enlarge_check.setToolTip(tr("Expand image size to fit rotated content without cropping"))
        self.enlarge_check.stateChanged.connect(lambda v: GlobalConfig.set("geo_enlarge", bool(v)))
        rotate_layout.addWidget(self.enlarge_check)

        apply_rotate_btn = QPushButton(f"✅ {tr('Apply Rotation')}")
        apply_rotate_btn.clicked.connect(self._apply_rotation)
        rotate_layout.addWidget(apply_rotate_btn)
        rotate_group.setLayout(rotate_layout)
        layout.addWidget(rotate_group)

        # ========== 2. 单次裁剪模块 ==========
        crop_group = QGroupBox(tr("2. Simple Crop (Single)"))
        crop_layout = QVBoxLayout()
        
        h_crop_layer = QHBoxLayout()
        h_crop_layer.addWidget(QLabel(tr("Target:")))
        self.simple_crop_combo = QComboBox()
        h_crop_layer.addWidget(self.simple_crop_combo)
        crop_layout.addLayout(h_crop_layer)
        
        draw_rect_btn = QPushButton(f"✏️ {tr('Draw Rect')}")
        draw_rect_btn.clicked.connect(self._draw_crop_rect)
        crop_layout.addWidget(draw_rect_btn)
        apply_crop_btn = QPushButton(f"✂️ {tr('Apply Crop (New Layer)')}")
        apply_crop_btn.clicked.connect(self._apply_crop)
        crop_layout.addWidget(apply_crop_btn)
        crop_group.setLayout(crop_layout)
        layout.addWidget(crop_group)

        # ========== 3. 批量ROI提取 (增强版) ==========
        batch_group = QGroupBox(tr("3. Batch Extraction (Multi-ROI)"))
        batch_group.setStyleSheet("QGroupBox { border: 1px solid #4CAF50; margin-top: 10px; } QGroupBox::title { color: #4CAF50; }")
        batch_layout = QVBoxLayout()

        # View vs Data Layer
        layer_grid = QVBoxLayout()
        h_data = QHBoxLayout()
        h_data.addWidget(QLabel(tr("Data Layer (Crop Source):")))
        self.batch_data_combo = QComboBox()
        h_data.addWidget(self.batch_data_combo)
        layer_grid.addLayout(h_data)

        h_view = QHBoxLayout()
        h_view.addWidget(QLabel(tr("View Layer (Reference):")))
        self.batch_view_combo = QComboBox()
        h_view.addWidget(self.batch_view_combo)
        layer_grid.addLayout(h_view)
        
        h_sync = QHBoxLayout()
        self.sync_layers_btn = QPushButton(f"🔗 {tr('Sync Select')}")
        self.sync_layers_btn.setToolTip(tr("Set View Layer same as Data Layer"))
        self.sync_layers_btn.clicked.connect(self._sync_batch_layers)
        h_sync.addWidget(self.sync_layers_btn)

        self.peek_btn = QPushButton(f"👁️ {tr('Peek Data (Hold)')}")
        self.peek_btn.setToolTip("Hold to temporarily show Data Layer to check alignment")
        self.peek_btn.pressed.connect(self._peek_data_layer_show)
        self.peek_btn.released.connect(self._peek_data_layer_hide)
        h_sync.addWidget(self.peek_btn)
        
        layer_grid.addLayout(h_sync)
        
        self.lock_view_check = QCheckBox(f"🔒 {tr('Lock View Layer (Prevent auto-switching)')}")
        self.lock_view_check.setChecked(True)
        layer_grid.addWidget(self.lock_view_check)
        
        batch_layout.addLayout(layer_grid)
        batch_layout.addWidget(QLabel("<hr>")) 

        # Naming & Format
        name_layout = QHBoxLayout()
        
        # [Req 6.1] Date Input Field
        name_layout.addWidget(QLabel(tr("Date:")))
        self.date_edit = QLineEdit()
        # 尝试从配置加载日期，否则默认今天
        default_date = QSettings("NapariUser", "Global").value("current_date", datetime.datetime.now().strftime("%Y%m%d"))
        self.date_edit.setText(default_date)
        self.date_edit.setFixedWidth(75) 
        self.date_edit.setToolTip(tr("Date prefix (YYYYMMDD). Loaded from Archive or Today."))
        name_layout.addWidget(self.date_edit)

        name_layout.addWidget(QLabel(tr("Sub:")))
        self.sample_name_edit = QLineEdit("CRY2")
        name_layout.addWidget(self.sample_name_edit, 1)

        # [Req 3] Suffix Input (Flexible)
        name_layout.addWidget(QLabel(tr("Suffix:")))
        self.suffix_edit = QLineEdit("_origin")
        self.suffix_edit.setPlaceholderText("e.g. _origin, _contrasted")
        self.suffix_edit.setMinimumWidth(80)
        name_layout.addWidget(self.suffix_edit, 1)
        batch_layout.addLayout(name_layout)

        # [Req 4 & 5] Checkboxes for extra folders
        h_checks = QHBoxLayout()
        self.check_denoise = QCheckBox(tr("Gen Denoise Folders"))
        self.check_denoise.setToolTip(tr("Creates empty folders with Main Suffix + Configured Suffix (e.g. _contrasted_lrtem)"))
        

        self.check_refine = QCheckBox(tr("Gen Refine Folder"))
        self.check_refine.setToolTip(tr("Creates empty folder with Main Suffix + Configured Suffix (e.g. _contrasted_mask_new)"))
        

        h_checks.addWidget(self.check_denoise)
        h_checks.addWidget(self.check_refine)
        batch_layout.addLayout(h_checks)

        # 添加信号
        self.check_denoise.stateChanged.connect(lambda v: GlobalConfig.set("geo_create_denoise", bool(v)))
        self.check_refine.stateChanged.connect(lambda v: GlobalConfig.set("geo_create_refine", bool(v)))
        self.suffix_edit.editingFinished.connect(lambda: GlobalConfig.set("geo_suffix", self.suffix_edit.text()))

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel(tr("Export Format:")))
        self.batch_format_combo = QComboBox()
        self.batch_format_combo.addItems(["PNG Sequence (Folder)","TIFF Stack (.tiff)"])
        format_layout.addWidget(self.batch_format_combo)
        batch_layout.addLayout(format_layout)

        # Frame Filter
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel(tr("Frame Filter:")))
        self.batch_frame_edit = QLineEdit()
        self.batch_frame_edit.setPlaceholderText(tr("All (Default) or 0-10, 15..."))
        self.batch_frame_edit.setToolTip(tr("Leave empty for All frames.\nOr use: 0-10, 15, 20-25"))
        frame_layout.addWidget(self.batch_frame_edit)

        self.btn_set_specific_range = QPushButton(f"📌 {tr('Set for Selected')}")
        self.btn_set_specific_range.setToolTip(tr("Apply the text in the box to the CURRENTLY SELECTED ROI only."))
        self.btn_set_specific_range.clicked.connect(self._set_range_for_selected_roi)
        self.btn_set_specific_range.setStyleSheet("background-color: #555; font-size: 10px; padding: 4px;")
        frame_layout.addWidget(self.btn_set_specific_range)
        batch_layout.addLayout(frame_layout)

        # Naming options
        naming_layout = QHBoxLayout()
        self.keep_index_check = QCheckBox(tr("Keep Original Frame Index"))
        self.keep_index_check.stateChanged.connect(lambda v: GlobalConfig.set("geo_keep_index", bool(v)))
        naming_layout.addWidget(self.keep_index_check)

        naming_layout.addWidget(QLabel(tr("Geo_Padding:")))
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(1, 12)
        self.padding_spin.setValue(5) 
        self.padding_spin.setSuffix(" digits")
        naming_layout.addWidget(self.padding_spin)
        naming_layout.addStretch()
        batch_layout.addLayout(naming_layout)
        
        self.force_square_check = QCheckBox(tr("Force Square Crops"))
        self.force_square_check.stateChanged.connect(lambda v: GlobalConfig.set("geo_force_square", bool(v)))
        batch_layout.addWidget(self.force_square_check)

        # Tools
        tools_layout = QHBoxLayout()
        self.start_batch_btn = QPushButton(f"✏️ {tr('Start Draw')}")
        self.start_batch_btn.clicked.connect(self._start_batch_mode)
        self.start_batch_btn.setStyleSheet("background-color: #444; font-weight: bold;")
        tools_layout.addWidget(self.start_batch_btn)

        self.adjust_batch_btn = QPushButton(f"🖐️ {tr('Adjust')}")
        self.adjust_batch_btn.clicked.connect(self._switch_to_select_mode)
        tools_layout.addWidget(self.adjust_batch_btn)
        batch_layout.addLayout(tools_layout)

        self.export_batch_btn = QPushButton(f"💾 {tr('Export Crops & Map')}")
        self.export_batch_btn.clicked.connect(self._export_batch_crops)
        self.export_batch_btn.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 6px;")
        batch_layout.addWidget(self.export_batch_btn)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        # Load settings
        self.suffix_edit.setText(GlobalConfig.get("geo_suffix"))
        self.suffix_edit.textChanged.connect(lambda t: GlobalConfig.set("geo_suffix", t))
        val_keep = GlobalConfig.get("geo_keep_index")
        is_checked_keep = (val_keep == 'true') if isinstance(val_keep, str) else bool(val_keep)
        self.keep_index_check.setChecked(is_checked_keep)
        self.keep_index_check.stateChanged.connect(lambda v: GlobalConfig.set("geo_keep_index", bool(v)))
        self.padding_spin.setValue(int(GlobalConfig.get("geo_padding")))
        self.padding_spin.valueChanged.connect(lambda v: GlobalConfig.set("geo_padding", v))
        val_sq = GlobalConfig.get("geo_force_square")
        is_checked_sq = (val_sq == 'true') if isinstance(val_sq, str) else bool(val_sq)
        self.force_square_check.setChecked(is_checked_sq)
        self.force_square_check.stateChanged.connect(lambda v: GlobalConfig.set("geo_force_square", bool(v)))
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
        self._sync_batch_layers()
        self._try_load_archived_substance()

    def _on_shortcut_apply(self, viewer):
        if "Batch_ROI" in self.viewer.layers and len(self.viewer.layers["Batch_ROI"].data) > 0:
            self._export_batch_crops()
            self.status_label.setText("⚡ Shortcut: Batch Export Triggered")
        elif "Crop_ROI" in self.viewer.layers and len(self.viewer.layers["Crop_ROI"].data) > 0:
            self._apply_crop()
            self.status_label.setText("⚡ Shortcut: Single Crop Triggered")

    def _on_shortcut_switch(self, viewer):
        if "Batch_ROI" in self.viewer.layers:
            layer = self.viewer.layers["Batch_ROI"]
            if layer.mode == 'add_rectangle':
                layer.mode = 'select'
                self.status_label.setText("⚡ Mode: Select/Adjust")
            else:
                layer.mode = 'add_rectangle'
                self.status_label.setText("⚡ Mode: Draw")

    def _refresh_layers(self, event=None):
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
            
            if current in layers:
                combo.setCurrentText(current)
            elif layers and combo in [self.rotate_layer_combo, self.simple_crop_combo]:
                active = self.viewer.layers.selection.active
                if active and active.name in layers:
                    combo.setCurrentText(active.name)
            
            combo.blockSignals(False)
        
        self.batch_data_combo.blockSignals(True)
        self.batch_view_combo.blockSignals(True)

        data_candidates = [l for l in layers if l.startswith("Cropped_Rotated")]
        if not data_candidates:
             data_candidates = [l for l in layers if "Rotated" in l and "Enh" not in l and "Contrast" not in l and "Burned" not in l]
        if data_candidates:
            self.batch_data_combo.setCurrentText(data_candidates[-1])

        view_candidates = [l for l in layers if l.startswith("Contrast_Enh")]
        if not view_candidates:
            view_candidates = [l for l in layers if l.startswith("Enh")]
        if view_candidates:
            self.batch_view_combo.setCurrentText(view_candidates[-1])
        elif self.batch_data_combo.currentText():
            self.batch_view_combo.setCurrentText(self.batch_data_combo.currentText())

        self.batch_data_combo.blockSignals(False)
        self.batch_view_combo.blockSignals(False)
        self._try_load_archived_substance()

    def _try_load_archived_substance(self):
        try:
            archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
            if not archive_path: return
            
            readme_path = Path(archive_path) / "readme.txt"
            if readme_path.exists():
                with open(readme_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith("Substance:"):
                            name = line.split(":", 1)[1].strip()
                            if name: self.sample_name_edit.setText(name)
                            break
            
            # [Req 6.1 Updated] 直接从共享配置获取日期，不再从文件夹名解析
            shared_date = QSettings("NapariUser", "Global").value("current_date", "")
            if shared_date:
                self.date_edit.setText(str(shared_date))
                
        except Exception:
            pass

    def _on_active_layer_changed(self, event=None):
        if self._force_view_active: return 
        active = self.viewer.layers.selection.active
        if active and hasattr(active, 'data') and isinstance(active.data, np.ndarray) and active.data.ndim == 3:
            name = active.name
            self.rotate_layer_combo.setCurrentText(name)
            self.simple_crop_combo.setCurrentText(name)

    def _sync_batch_layers(self):
        txt = self.batch_data_combo.currentText()
        if txt: self.batch_view_combo.setCurrentText(txt)

    def _enforce_view_visibility(self, event=None):
        if not self._force_view_active: return
        view_name = self.batch_view_combo.currentText()
        if not view_name or view_name not in self.viewer.layers: return
        try: self.viewer.layers.events.reordered.disconnect(self._enforce_view_visibility)
        except: pass
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                layer.visible = (layer.name == view_name)
        self.viewer.layers.events.reordered.connect(self._enforce_view_visibility)

    def _clear_residue(self, target_names):
        for name in target_names:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

    def _peek_data_layer_show(self):
        data_name = self.batch_data_combo.currentText()
        if not data_name or data_name not in self.viewer.layers: return
        self._force_view_active = False 
        for l in self.viewer.layers:
            if isinstance(l, napari.layers.Image):
                l.visible = (l.name == data_name)
                
    def _peek_data_layer_hide(self):
        if self.lock_view_check.isChecked():
            self._force_view_active = True
            self._enforce_view_visibility() 
        else:
            view_name = self.batch_view_combo.currentText()
            if view_name in self.viewer.layers:
                for l in self.viewer.layers:
                    if isinstance(l, napari.layers.Image):
                        l.visible = (l.name == view_name)

    # --- Rotation ---
    def _draw_rotation_line(self):
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

    def _apply_rotation(self):
        layer_name = self.rotate_layer_combo.currentText()
        if not layer_name: return
        angle = self.angle_spin.value()
        expand = self.enlarge_check.isChecked()
        image_stack = self.viewer.layers[layer_name].data
        self.status_label.setText("⏳ Rotating...")
        self.rot_progress = QProgressDialog(f"Rotating {angle:.1f}°...", "Cancel", 0, len(image_stack), self)
        self.rot_progress.setWindowModality(Qt.WindowModal)
        self.rot_progress.show()
        
        self.rot_thread = RotationThread(image_stack, angle, expand)
        self.rot_thread.progress.connect(lambda c, t: self.rot_progress.setValue(c))
        
        def on_finished(rotated):
            self.rot_progress.close()
            try:
                new_name = f"Rotated_{layer_name}"
                new_layer = self.viewer.add_image(rotated, name=new_name, colormap='gray', metadata={'source_layer': layer_name})
                if layer_name in self.viewer.layers:
                    self.viewer.layers[layer_name].visible = False
                self._clear_residue(["Rotation_Line"])
                self.simple_crop_combo.setCurrentText(new_name)
                self.viewer.layers.selection.active = new_layer
                undo_key = GlobalConfig.get_napari_shortcut("shortcut_undo_drift")
                @new_layer.bind_key(undo_key, overwrite=True)
                def undo_rotation(layer):
                    src = layer.metadata.get('source_layer')
                    if src and src in self.viewer.layers:
                        self.viewer.layers[src].visible = True
                        self.viewer.layers.selection.active = self.viewer.layers[src]
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
            self.rotate_layer_combo.setCurrentText(new_name)
            self.simple_crop_combo.setCurrentText(new_name)
            self.viewer.layers.selection.active = self.viewer.layers[new_name]
            self.status_label.setText(f"✅ Applied {direction} flip.")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")

    # --- Simple Crop ---
    def _draw_crop_rect(self):
        self._clear_residue(["Crop_ROI", "Rotation_Line", "Batch_ROI", "Interaction_Box", "Preview_Overlay", "Drift_ROI"])
        edge_col = GlobalConfig.get("style_crop_color")
        layer = self.viewer.add_shapes(
            name="Crop_ROI", shape_type='rectangle', 
            edge_color=edge_col, edge_width=3,
            face_color=[1, 1, 1, 0.01] 
        )
        layer.mode = 'add_rectangle'
        def on_data_change(event):
            if layer.mode == 'add_rectangle' and len(layer.data) > 0:
                layer.mode = 'select'
                self.viewer.layers.selection.active = layer
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
        self._clear_residue(["Crop_ROI"])
        self.viewer.layers.selection.active = new_layer

        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image) and layer.name != new_name:
                if "Preview" not in layer.name: 
                    layer.visible = False
        undo_key = GlobalConfig.get_napari_shortcut("shortcut_undo_drift") 
        
        @new_layer.bind_key(undo_key, overwrite=True)
        def undo_crop(layer):
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
            if target in self.viewer.layers:
                self.viewer.layers[target].visible = True
                self.viewer.layers.selection.active = self.viewer.layers[target]
            self._draw_crop_rect()
            self.status_label.setText("↩️ Crop Undone.")

        self.status_label.setText(f"✅ Crop applied. Press '{undo_key}' to Undo.")

    # --- Batch Crop Logic ---
    def _start_batch_mode(self):
        view_layer = self.batch_view_combo.currentText()
        if not view_layer: return
        
        self._clear_residue(["Batch_ROI", "Rotation_Line", "Crop_ROI", "Interaction_Box", "Preview_Overlay", "Drift_ROI"])
        
        if self.lock_view_check.isChecked():
            self._force_view_active = True
            self._enforce_view_visibility()
        else:
            self._force_view_active = False
            for l in self.viewer.layers:
                if isinstance(l, napari.layers.Image): l.visible = (l.name == view_layer)
        
        box_col = GlobalConfig.get("style_batch_box_color")
        width = int(GlobalConfig.get("style_batch_width"))
        txt_col = GlobalConfig.get("style_batch_text_color")
        font_size = int(GlobalConfig.get("style_batch_font_size"))

        roi_layer = self.viewer.add_shapes(
            name="Batch_ROI",
            shape_type='rectangle',
            edge_color=box_col, 
            face_color=[0, 1, 0, 0.05],
            edge_width=width,
            text={
                'string': '{label}\n{frame_info}', 
                'size': font_size, 
                'color': txt_col, 
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
        
        undo_key = GlobalConfig.get_napari_shortcut("shortcut_undo_drift")
        @roi_layer.bind_key(undo_key)
        def undo_batch_rect(layer):
            if layer.mode == 'add_rectangle' and len(layer.data) > 0:
                # 1. 获取当前数据和特征
                current_data = layer.data
                current_features = layer.features
                
                # 2. 只有当数据存在时才执行
                if len(current_data) > 0:
                    # 暂时断开事件监听，防止 _on_batch_data_change 在状态不稳时触发
                    self._is_updating = True 
                    try:
                        # 3. 同步切片：数据和特征都移除最后一个
                        new_data = current_data[:-1]
                        new_features = {k: v[:-1] for k, v in current_features.items()}
                        
                        # 4. 先清空选中，防止索引越界
                        layer.selected_data = set()
                        
                        # 5. 同时赋值（先赋特征，再赋数据，通常更稳妥）
                        layer.data = new_data
                        layer.features = new_features
                        
                        
                        self.status_label.setText("↩️ Last ROI removed.")
                    except Exception as e:
                        print(f"Undo Error: {e}")
                    finally:
                        self._is_updating = False
                        # 强制刷新一下图层以确保显示正确
                        layer.refresh()

        self.status_label.setText(f"✏️ Drawing on '{view_layer}'. (Ctrl+Z to Undo last)")

    def _set_range_for_selected_roi(self):
        if "Batch_ROI" not in self.viewer.layers: return
        layer = self.viewer.layers["Batch_ROI"]
        
        selected_idxs = list(layer.selected_data)
        if not selected_idxs:
            self.status_label.setText("⚠️ No ROI selected. Select a green box first.")
            return
        
        range_str = self.batch_frame_edit.text().strip()
        
        current_features = layer.features
        n_shapes = len(layer.data)
        
        labels = list(current_features.get('label', []))
        ranges = list(current_features.get('frame_range', []))
        infos = list(current_features.get('frame_info', []))
        
        while len(labels) < n_shapes: labels.append(str(len(labels)+1))
        while len(ranges) < n_shapes: ranges.append("")
        while len(infos) < n_shapes: infos.append("")
            
        for idx in selected_idxs:
            if idx < len(ranges): 
                if not range_str or range_str.lower() == "global":
                    ranges[idx] = "" 
                    infos[idx] = ""
                else:
                    ranges[idx] = str(range_str) 
                    infos[idx] = f"[{range_str}]"
        
        layer.features = {
            'label': labels,
            'frame_range': ranges,
            'frame_info': infos
        }
        
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
        
        img_layer = self.viewer.layers[view_layer_name]
        IMG_H, IMG_W = img_layer.data.shape[-2], img_layer.data.shape[-1]
        
        layer = self.viewer.layers["Batch_ROI"]
        if len(layer.data) == 0: return

        self._is_updating = True
        try:
            new_data_list = []
            modified = False
            
            for roi in layer.data:
                ys, xs = roi[:, 0], roi[:, 1]
                y1, y2 = np.min(ys), np.max(ys)
                x1, x2 = np.min(xs), np.max(xs)
                
                h, w = y2 - y1, x2 - x1
                needs_reshape = False
                
                if self.force_square_check.isChecked() and abs(w - h) > 1.0:
                    side = int(max(w, h))
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    ny1, ny2 = int(cy - side / 2), int(cy - side / 2) + side
                    nx1, nx2 = int(cx - side / 2), int(cx - side / 2) + side
                    needs_reshape = True
                else:
                    ny1, ny2, nx1, nx2 = y1, y2, x1, x2
                
                ny1 = max(0, min(ny1, IMG_H)); ny2 = max(0, min(ny2, IMG_H))
                nx1 = max(0, min(nx1, IMG_W)); nx2 = max(0, min(nx2, IMG_W))
                
                if needs_reshape or (ny1 != y1 or ny2 != y2 or nx1 != x1 or nx2 != x2):
                    new_rect = np.array([[ny1, nx1], [ny2, nx1], [ny2, nx2], [ny1, nx2]])
                    new_data_list.append(new_rect)
                    modified = True
                else:
                    new_data_list.append(roi)
            
            if modified:
                layer.data = new_data_list

            n_shapes = len(layer.data)
            labels = [str(i+1) for i in range(n_shapes)]
            current_ranges = layer.features.get('frame_range', [])
            current_infos = layer.features.get('frame_info', [])
            range_list = [str(x) for x in current_ranges]
            info_list = [str(x) for x in current_infos]
            while len(range_list) < n_shapes:
                range_list.append("")
                info_list.append("")
            range_list = range_list[:n_shapes]
            info_list = info_list[:n_shapes]
            
            layer.features = {
                'label': labels,
                'frame_range': range_list,
                'frame_info': info_list
            }
            
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

        # 2. 准备数据
        data_stack = self.viewer.layers[data_layer_name].data
        view_stack = self.viewer.layers[view_layer_name].data 
        
        layer = self.viewer.layers["Batch_ROI"]
        rois = layer.data 
        
        roi_features = layer.features
        frame_ranges_list = list(roi_features.get('frame_range', [""] * len(rois)))
        global_range_text = self.batch_frame_edit.text().strip()

        # 3. 路径与元数据提取
        # [Req 6.1] 优先从 UI 输入框获取日期，如果为空则使用当前日期
        date_str = self.date_edit.text().strip()
        if not date_str:
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            
        ds_id = QSettings("NapariUser", "Global").value("current_dataset_id", "ds1")
        sub_name = self.sample_name_edit.text().strip() or "Sample"
        
        archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
        if archive_path and Path(archive_path).exists():
            output_dir = Path(archive_path)
        else:
            last_import_folder = QSettings("NapariUser", "Importer").value("last_folder", "")
            if last_import_folder and Path(last_import_folder).exists():
                start_dir = str(Path(last_import_folder).parent)
            else:
                start_dir = str(Path.home())
            d = QFileDialog.getExistingDirectory(self, "Select Output Directory", start_dir)
            if not d: return
            output_dir = Path(d) / f"{date_str}_{sub_name}_Exports"
            output_dir.mkdir(parents=True, exist_ok=True)

        # 4. 打包参数
        params = {
            'data_stack': data_stack,
            'rois': rois,
            'frame_ranges_list': frame_ranges_list,
            'global_range_text': global_range_text,
            'output_dir': output_dir,
            
            # Naming Info
            'date_str': date_str,
            'dataset_id': ds_id,
            'sub_name': sub_name,
            
            # [Key Change] Pass configured suffixes
            'suffix_main': self.suffix_edit.text().strip(), 
            'aux_suffixes': {
                'lrtem': str(GlobalConfig.get("geo_suffix_lrtem")),
                'hrtem': str(GlobalConfig.get("geo_suffix_hrtem")),
                'mask': str(GlobalConfig.get("geo_suffix_mask")),
                'mask_new': str(GlobalConfig.get("geo_suffix_mask_new"))
            },
            
            # Flags
            'create_denoise': self.check_denoise.isChecked(),
            'create_refine': self.check_refine.isChecked(),
            
            'is_tiff': "TIFF" in self.batch_format_combo.currentText(),
            'keep_idx': self.keep_index_check.isChecked(),
            'pad': self.padding_spin.value(),
            'data_layer_name': data_layer_name 
        }

        # 5. UI 进度
        self.batch_progress = QProgressDialog("Exporting Crops...", "Cancel", 0, len(rois), self)
        self.batch_progress.setWindowModality(Qt.WindowModal)
        self.batch_progress.setMinimumDuration(0)
        self.batch_progress.canceled.connect(self._on_export_cancel)
        self.batch_progress.show()

        # 6. 线程启动
        self.export_thread = BatchExportThread(params)
        self.export_thread.progress.connect(self.batch_progress.setValue)
        
        # [Req 6] 获取当前帧数
        current_frame_idx = self.viewer.dims.current_step[0]
        
        self.export_thread.finished.connect(lambda c, path: self._on_export_finished(c, path, view_stack, rois, sub_name, output_dir, current_frame_idx))
        self.export_thread.error.connect(self._on_export_error)
        self.export_thread.start()
        
    def _on_export_cancel(self):
        if self.export_thread.isRunning():
            self.export_thread.requestInterruption()
            self.status_label.setText("⚠️ Export canceled.")

    def _on_export_finished(self, count, path_name, view_stack, rois, sub_name, output_dir, frame_idx):
        self.batch_progress.close()
        
        # [Req 6] 使用指定的 frame_idx
        self._create_overview_map(view_stack, rois, sub_name, output_dir, frame_idx)
        
        self.status_label.setText(f"✅ Exported {count} crops.")
        self._force_view_active = False
        QMessageBox.information(self, "Success", f"Exported {count} crops!\nSaved to: {path_name}")

    def _on_export_error(self, err):
        self.batch_progress.close()
        self.status_label.setText(f"❌ Error: {err}")
        QMessageBox.critical(self, "Export Error", str(err))

    def _create_overview_map(self, image_stack, rois, sample_name, output_dir, frame_idx):
        """保存 Overview Map"""
        if len(image_stack) == 0: return
        
        # 使用传入的 frame_idx，防止越界
        idx = max(0, min(frame_idx, len(image_stack)-1))
        bg_img = image_stack[idx]
        
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
            
        pil_img.save(output_dir / f"{sample_name}_Overview_Frame{idx}.png")
        
        # 记录 overview 使用的帧
        json_path = output_dir / "processing_log.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f: log_data = json.load(f)
                if "batch_crop" in log_data:
                    log_data["batch_crop"]["overview_frame"] = idx
                with open(json_path, 'w') as f: json.dump(log_data, f, indent=2)
            except: pass
    
    def _load_params_from_config(self):
        """热更新：从配置读取状态"""
        # Block signals
        widgets = [self.enlarge_check, self.keep_index_check, self.force_square_check, 
                   self.check_denoise, self.check_refine, self.suffix_edit, self.padding_spin]
        for w in widgets: w.blockSignals(True)

        self.enlarge_check.setChecked(bool(GlobalConfig.get("geo_enlarge")))
        self.keep_index_check.setChecked(bool(GlobalConfig.get("geo_keep_index")))
        self.force_square_check.setChecked(bool(GlobalConfig.get("geo_force_square")))
        self.check_denoise.setChecked(bool(GlobalConfig.get("geo_create_denoise")))
        self.check_refine.setChecked(bool(GlobalConfig.get("geo_create_refine")))
        self.suffix_edit.setText(str(GlobalConfig.get("geo_suffix")))
        self.padding_spin.setValue(int(GlobalConfig.get("geo_padding")))

        # Update Style Configs (Color etc.) for Draw Rect logic
        # 这里的样式参数会在 _draw_crop_rect 调用时实时读取 GlobalConfig.get()，无需刷新 UI 控件

        for w in widgets: w.blockSignals(False)
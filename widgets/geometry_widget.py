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
                            QMessageBox, QCheckBox, QProgressDialog, QSpinBox,QApplication)
from qtpy.QtCore import Qt, QTimer, QSettings, QThread, Signal
import numpy as np
from pathlib import Path
import cv2
from utils.ui_utils import setup_safe_scroll_all
from PIL import Image, ImageDraw, ImageFont
from core.geometry import (calculate_rotation_angle, rotate_image_stack, flip_image_stack,
                           crop_image_stack, validate_bbox)
from utils.video_export import export_to_tiff_stack
import napari
import json
import os
import datetime
from widgets.settings_widget import GlobalConfig, tr
from utils.session_logger import get_logger
from utils.utils import elide_text

# ==========================================
#  新增：后台图像读取线程 (防止界面卡死)
# ==========================================
class BatchImageLoaderThread(QThread):
    """
    后台批量读取图像数据 (TIFF 或 PNG序列)
    按顺序读取列表中的任务，每完成一个发射一次信号
    """
    # 信号定义: (data_array, layer_name, role)
    # role: 'data', 'view', or 'none' (用于后续自动设置下拉框)
    item_ready = Signal(object, str, str) 
    finished_all = Signal()
    error = Signal(str)

    def __init__(self, tasks):
        """
        tasks: list of dict {'path': Path, 'mode': 'tiff'/'stack', 'name': str, 'role': str}
        """
        super().__init__()
        self.tasks = tasks

    def run(self):
        try:
            for task in self.tasks:
                if self.isInterruptionRequested(): break
                
                path = task['path']
                mode = task['mode']
                name = task['name']
                role = task.get('role', 'none')
                
                data = None
                if mode == 'tiff':
                    import tifffile
                    data = tifffile.imread(str(path))
                elif mode == 'stack':
                    # 读取文件夹下的所有图像
                    files = sorted([f for f in path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']])
                    if not files:
                        self.error.emit(tr("Empty folder:") + f" {name}")
                        continue
                    
                    # 读取第一帧获取尺寸和类型
                    first = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
                    if first is None: 
                        self.error.emit(tr("Failed to read first frame of") + f" {name}")
                        continue
                    
                    # 预分配内存
                    count = len(files)
                    shape = (count, *first.shape)
                    dtype = first.dtype
                    data = np.zeros(shape, dtype=dtype)
                    
                    data[0] = first
                    for i in range(1, count):
                        img = cv2.imread(str(files[i]), cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            data[i] = img
                
                if data is not None:
                    self.item_ready.emit(data, name, role)
                
            self.finished_all.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

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
            dataset_id = self.p.get('dataset_id', '')
            sub_name = self.p['sub_name']
            
            # === View Layer Export Options ===
            export_view = self.p.get('export_view', False)
            view_stack = self.p.get('view_stack', None)
            view_suffix_main = self.p.get('view_suffix_main', '_contrasted')
            
            # === Suffixes ===
            suffix_main = self.p['suffix_main'] # 来自 UI 输入框 (e.g., "_origin")
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
                # 格式: 20251104_Dataset3_CRY2_NP1 或者 20251104_CRY2_NP1
                dataset_str = f"_{dataset_id}" if dataset_id else ""
                base_name = f"{date_str}{dataset_str}_{sub_name}_NP{i+1}"
                
                # 主数据文件夹 (加上 UI 输入的主后缀)
                # e.g. ..._NP1_origin
                folder_main = output_dir / f"{base_name}{suffix_main}"
                folder_main.mkdir(parents=True, exist_ok=True)
                
                # 同理准备视图层导出文件夹
                folder_view = None
                if export_view and view_stack is not None:
                    folder_view = output_dir / f"{base_name}{view_suffix_main}"
                    folder_view.mkdir(parents=True, exist_ok=True)
                    # 执行视图层的裁切
                    filtered_view_stack = np.asarray(view_stack[selected_indices])
                    crop_view = crop_image_stack(filtered_view_stack, bbox)
                
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
                    if export_view and folder_view:
                        export_to_tiff_stack(crop_view, str(folder_view / f"{base_name}.tiff"))
                else:
                    for k, img in enumerate(crop):
                        # 归一化数据层
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
                            
                        # 保存视图层
                        if export_view and folder_view:
                            img_v = crop_view[k]
                            if img_v.dtype != np.uint8:
                                mn_v, mx_v = img_v.min(), img_v.max()
                                if mx_v > mn_v: img_v = ((img_v - mn_v) / (mx_v - mn_v) * 255).astype(np.uint8)
                                else: img_v = img_v.astype(np.uint8)
                                
                            save_path_v = str(folder_view / file_name)
                            try:
                                is_success_v, im_buf_v = cv2.imencode(".png", img_v)
                                if is_success_v: im_buf_v.tofile(save_path_v)
                            except Exception as save_err:
                                print(f"View Layer Save Error: {save_err}")
                
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
        self._roi_history = []  # ROI 状态快照栈，用于撤销
        self._roi_history_max = int(GlobalConfig.get("geo_roi_history_max"))  # 历史记录上限
        self._prev_roi_state = None  # 追踪"变更前"的状态
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
        
        # [Dataset Ext/Input] - Added Feature
        name_layout.addWidget(QLabel(tr("Dataset:")))
        default_ds = QSettings("NapariUser", "Global").value("current_dataset_id", "ds1")
        self.dataset_edit = QLineEdit(default_ds)
        self.dataset_edit.setFixedWidth(50)
        self.dataset_edit.setToolTip(tr("Dataset ID (e.g. ds123). Mostly auto-extracted."))
        name_layout.addWidget(self.dataset_edit)

        name_layout.addWidget(QLabel(tr("Sub:")))
        self.sample_name_edit = QLineEdit("CRY2")
        self.sample_name_edit.setFixedWidth(60)
        name_layout.addWidget(self.sample_name_edit)

        # [Req 3] Suffix Input (Flexible)
        name_layout.addWidget(QLabel(tr("Suffix:")))
        self.suffix_edit = QLineEdit("_origin")
        self.suffix_edit.setPlaceholderText("e.g. _origin")
        self.suffix_edit.setMinimumWidth(80)
        name_layout.addWidget(self.suffix_edit, 1)
        batch_layout.addLayout(name_layout)
        
        view_export_layout = QHBoxLayout()
        self.export_view_check = QCheckBox(tr("Export View Layer"))
        self.export_view_check.setToolTip(tr("Additionally export the view layer (e.g. contrasted image)."))
        self.export_view_check.setChecked(bool(GlobalConfig.get("geo_export_view")))
        view_export_layout.addWidget(self.export_view_check)
        
        view_export_layout.addWidget(QLabel(tr("Suffix:")))
        self.export_view_suffix = QLineEdit("_contrasted")
        self.export_view_suffix.setPlaceholderText(tr("Suffix (e.g. _contrasted)"))
        view_export_layout.addWidget(self.export_view_suffix, 1)
        batch_layout.addLayout(view_export_layout)

        # [Req 4 & 5] Checkboxes for extra folders
        h_checks = QHBoxLayout()
        self.check_denoise = QCheckBox(tr("Gen Denoise Folders"))
        self.check_denoise.setToolTip(tr("Creates empty folders with Main Suffix + Configured Suffix (e.g. _contrasted_lrtem)"))
        

        self.check_refine = QCheckBox(tr("Gen Refine Folder"))
        self.check_refine.setToolTip(tr("Creates empty folder with Main Suffix + Configured Suffix (e.g. _contrasted_mask_new)"))
        

        h_checks.addWidget(self.check_denoise)
        h_checks.addWidget(self.check_refine)
        batch_layout.addLayout(h_checks)

        self.check_denoise.stateChanged.connect(lambda v: GlobalConfig.set("geo_create_denoise", bool(v)))
        self.check_refine.stateChanged.connect(lambda v: GlobalConfig.set("geo_create_refine", bool(v)))
        self.export_view_check.stateChanged.connect(lambda v: GlobalConfig.set("geo_export_view", bool(v)))
        self.suffix_edit.editingFinished.connect(lambda: GlobalConfig.set("geo_suffix", self.suffix_edit.text()))

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel(tr("Export Format:")))
        self.batch_format_combo = QComboBox()
        self.batch_format_combo.addItems([tr("PNG Sequence (Folder)"), tr("TIFF Stack (.tiff)")])
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
        
        self.clear_all_btn = QPushButton("🗑️")
        self.clear_all_btn.setToolTip(tr("Clear All Overlays (ROIs, Lines, Measures)"))
        self.clear_all_btn.setFixedWidth(40) 
        self.clear_all_btn.setStyleSheet("background-color: #555; color: #FF5252; font-weight: bold;")
        self.clear_all_btn.clicked.connect(self._clear_all_overlays) # 绑定新函数
        tools_layout.addWidget(self.clear_all_btn)
        
        batch_layout.addLayout(tools_layout)

        # === [新增] ROI 导入导出按钮行 ===
        h_roi_io = QHBoxLayout()
        btn_save_roi = QPushButton(f"💾 {tr('Save ROIs')}")
        btn_save_roi.clicked.connect(self._save_rois_to_json)
        btn_save_roi.setToolTip(tr("Save ROI coordinates + Reference Map"))
        
        btn_load_roi = QPushButton(f"📂 {tr('Load ROIs')}")
        btn_load_roi.clicked.connect(self._load_rois_from_json)
        btn_load_roi.setToolTip(tr("Load ROI JSON & Auto-load Image"))
        
        h_roi_io.addWidget(btn_save_roi)
        h_roi_io.addWidget(btn_load_roi)
        batch_layout.addLayout(h_roi_io)
        
        # === [新增] PNG/TIFF 快速导入按钮 ===
        h_quick_import = QHBoxLayout()
        btn_load_png = QPushButton(f"📂 {tr('Load PNG Seq')}")
        btn_load_png.setToolTip(tr("Quickly load a PNG sequence folder"))
        btn_load_png.clicked.connect(self._quick_load_png_sequence)
        
        btn_load_tiff = QPushButton(f"📂 {tr('Load TIFF')}")
        btn_load_tiff.setToolTip(tr("Quickly load a TIFF stack file"))
        btn_load_tiff.clicked.connect(self._quick_load_tiff_stack)
        
        h_quick_import.addWidget(btn_load_png)
        h_quick_import.addWidget(btn_load_tiff)
        batch_layout.addLayout(h_quick_import)

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

        # [Safety] Prevent accidental mouse wheel scroll
        setup_safe_scroll_all(
            self.rotate_layer_combo, self.simple_crop_combo,
            self.batch_data_combo, self.batch_view_combo, self.batch_format_combo,
            self.angle_spin, self.padding_spin
        )

    def _on_shortcut_apply(self, viewer):
        if "Batch_ROI" in self.viewer.layers and len(self.viewer.layers["Batch_ROI"].data) > 0:
            self._export_batch_crops()
            self.status_label.setText(f"⚡ {tr('Shortcut: Batch Export Triggered')}")
        elif "Crop_ROI" in self.viewer.layers and len(self.viewer.layers["Crop_ROI"].data) > 0:
            self._apply_crop()
            self.status_label.setText(f"⚡ {tr('Shortcut: Single Crop Triggered')}")

    def _on_shortcut_switch(self, viewer):
        if "Batch_ROI" in self.viewer.layers:
            layer = self.viewer.layers["Batch_ROI"]
            if layer.mode == 'add_rectangle':
                layer.mode = 'select'
                self.status_label.setText(f"⚡ {tr('Mode: Select/Adjust')}")
            else:
                layer.mode = 'add_rectangle'
                self.status_label.setText(f"⚡ {tr('Mode: Draw')}")

    def _refresh_layers(self, event=None):
        layers = [
            l.name for l in self.viewer.layers 
            if hasattr(l, 'data') and isinstance(l.data, np.ndarray) and l.data.ndim == 3
        ]
        
        for combo in [self.rotate_layer_combo, self.simple_crop_combo, 
                      self.batch_data_combo, self.batch_view_combo]:
            current_data = combo.currentData()
            current_text = combo.currentText()
            # 兼容：如果之前有 Data 则用 Data，否则用 Text (首次运行)
            stored_selection = current_data if current_data else current_text
            
            combo.blockSignals(True)
            combo.clear()
            for l_name in layers:
                short = elide_text(l_name, 25)
                combo.addItem(short, l_name)
                combo.setItemData(combo.count()-1, l_name, Qt.ToolTipRole)
            
            # 激活图层优先 (用于 rotate/crop 组合框)
            active_layer_applied = False
            if layers and combo in [self.rotate_layer_combo, self.simple_crop_combo]:
                active = self.viewer.layers.selection.active
                if active and active.name in layers:
                    idx = combo.findData(active.name)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                        active_layer_applied = True
            
            # 仅在未应用激活图层时使用保存的选择
            if not active_layer_applied and stored_selection:
                idx = combo.findData(stored_selection)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            
            combo.blockSignals(False)
        
        self.batch_data_combo.blockSignals(True)
        self.batch_view_combo.blockSignals(True)

        data_candidates = [l for l in layers if l.startswith("Cropped_Rotated")]
        if not data_candidates:
             data_candidates = [l for l in layers if "Rotated" in l and "Enh" not in l and "Contrast" not in l and "Burned" not in l]
        if data_candidates:
            idx = self.batch_data_combo.findData(data_candidates[-1])
            if idx >= 0: self.batch_data_combo.setCurrentIndex(idx)

        view_candidates = [l for l in layers if l.startswith("Contrast_Enh")]
        if not view_candidates:
            view_candidates = [l for l in layers if l.startswith("Enh")]
        if view_candidates:
            idx = self.batch_view_combo.findData(view_candidates[-1])
            if idx >= 0: self.batch_view_combo.setCurrentIndex(idx)
        elif self.batch_data_combo.currentData():
            idx = self.batch_view_combo.findData(self.batch_data_combo.currentData())
            if idx >= 0: self.batch_view_combo.setCurrentIndex(idx)

        self.batch_data_combo.blockSignals(False)
        self.batch_view_combo.blockSignals(False)
        self._try_load_archived_substance()
        
        # [Dataset Ext/Input] - Auto detect on refresh using current data layer
        if self.batch_data_combo.currentData():
            self._auto_detect_dataset(self.batch_data_combo.currentData())

    def _auto_detect_dataset(self, layer_name):
        """[Enhanced Feature] Auto-detect dataset ID from layer name using regex."""
        import re
        match_ds = re.search(r"dataset[-_]?.*?(\d+)", layer_name, re.IGNORECASE)
        # 也可以尝试匹配 ds 开头
        if not match_ds:
            match_ds = re.search(r"ds[-_]?.*?(\d+)", layer_name, re.IGNORECASE)
            
        if match_ds:
            ds_num = match_ds.group(1)
            # 只有当 UI 当前的值为空或者默认的 "ds1" 时才强行覆盖，或者总是覆盖以图层为主？
            # 为了流畅体验，这里选择只要匹配到就更新文本框
            self.dataset_edit.setText(f"ds{ds_num}")
            
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
            idx1 = self.rotate_layer_combo.findData(name)
            if idx1 >= 0: self.rotate_layer_combo.setCurrentIndex(idx1)
            idx2 = self.simple_crop_combo.findData(name)
            if idx2 >= 0: self.simple_crop_combo.setCurrentIndex(idx2)

    def _sync_batch_layers(self):
        # 将裁剪源(Data)同步为参考源(View)的选择
        txt = self.batch_view_combo.currentData()
        if txt:
            idx = self.batch_data_combo.findData(txt)
            if idx >= 0: self.batch_data_combo.setCurrentIndex(idx)

    def _enforce_view_visibility(self, event=None):
        if not self._force_view_active: return
        view_name = self.batch_view_combo.currentData()
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
        data_name = self.batch_data_combo.currentData()
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
            view_name = self.batch_view_combo.currentData()
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
        self.status_label.setText(f"✏️ {tr('Draw Horizon Line.')}")

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
        layer_name = self.rotate_layer_combo.currentData()
        if not layer_name: return
        angle = self.angle_spin.value()
        expand = self.enlarge_check.isChecked()
        image_stack = self.viewer.layers[layer_name].data
        self.status_label.setText(f"⏳ {tr('Rotating...')}")
        self.rot_progress = QProgressDialog(tr("Rotating %.1f°...") % angle, "Cancel", 0, len(image_stack), self)
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
                self._refresh_layers()  # 刷新后使用 findData
                idx = self.simple_crop_combo.findData(new_name)
                if idx >= 0: self.simple_crop_combo.setCurrentIndex(idx)
                self.viewer.layers.selection.active = new_layer
                undo_key = GlobalConfig.get_napari_shortcut("shortcut_undo_drift")
                
                # === [日志记录] 旋转操作 ===
                action_id = None
                try:
                    action_id = get_logger().log_action("geometry", "rotate", {
                        "source_layer": layer_name,
                        "angle": angle,
                        "expand": expand
                    })
                    new_layer.metadata['action_id'] = action_id
                except: pass
                
                @new_layer.bind_key(undo_key, overwrite=True)
                def undo_rotation(layer):
                    src = layer.metadata.get('source_layer')
                    if src and src in self.viewer.layers:
                        self.viewer.layers[src].visible = True
                        self.viewer.layers.selection.active = self.viewer.layers[src]
                    self.viewer.layers.remove(layer)
                    self.status_label.setText(f"↩️ {tr('Rotation Undone.')}")
                    # 记录撤回
                    aid = layer.metadata.get('action_id')
                    if aid:
                        try:
                            get_logger().log_undo(aid)
                        except: pass
                self.status_label.setText(f"✅ {tr('Rotated %.1f° (Expand=%s)') % (angle, expand)}")
            except Exception as e:
                self.status_label.setText(f"{tr('Error showing result:')} {e}")
        
        def on_error(err):
            self.rot_progress.close()
            self.status_label.setText(f"❌ {tr('Rotation Error:')} {err}")

        self.rot_thread.finished.connect(on_finished)
        self.rot_thread.error.connect(on_error)
        self.rot_thread.start()

    def _apply_flip(self, direction):
        layer_name = self.rotate_layer_combo.currentData()
        if not layer_name: return
        image_stack = self.viewer.layers[layer_name].data
        try:
            flipped = flip_image_stack(image_stack, direction)
            suffix = "FlipH" if direction == 'horizontal' else "FlipV"
            new_name = f"{suffix}_{layer_name}"
            self.viewer.add_image(flipped, name=new_name, colormap='gray')
            self._refresh_layers()
            idx1 = self.rotate_layer_combo.findData(new_name)
            if idx1 >= 0: self.rotate_layer_combo.setCurrentIndex(idx1)
            idx2 = self.simple_crop_combo.findData(new_name)
            if idx2 >= 0: self.simple_crop_combo.setCurrentIndex(idx2)
            self.viewer.layers.selection.active = self.viewer.layers[new_name]
            self.status_label.setText(f"✅ {tr('Applied %s flip.') % direction}")
            
            # === [日志记录] 翻转操作 ===
            try:
                get_logger().log_action("geometry", "flip", {
                    "source_layer": layer_name,
                    "direction": direction
                })
            except: pass
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
                self.status_label.setText(f"🖐️ {tr('Mode: Adjust Crop Rect (Drag corners to resize)')}")

        layer.events.data.connect(on_data_change)
        self.status_label.setText(f"✏️ {tr('Draw Single Crop Rect.')}")

    def _apply_crop(self):
        target = self.simple_crop_combo.currentData()
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
        
        # === [日志记录] 裁剪操作 ===
        action_id = None
        try:
            action_id = get_logger().log_action("geometry", "crop", {
                "source_layer": target,
                "bbox": list(bbox)  # [x1, y1, x2, y2]
            })
            new_layer.metadata['action_id'] = action_id
        except: pass
        
        @new_layer.bind_key(undo_key, overwrite=True)
        def undo_crop(layer):
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
            if target in self.viewer.layers:
                self.viewer.layers[target].visible = True
                self.viewer.layers.selection.active = self.viewer.layers[target]
            self._draw_crop_rect()
            self.status_label.setText(f"↩️ {tr('Crop Undone.')}")
            # 记录撤回
            aid = layer.metadata.get('action_id')
            if aid:
                try:
                    get_logger().log_undo(aid)
                except: pass

        self.status_label.setText(f"""✅ {tr("Crop applied. Press '%s' to Undo.") % undo_key}""")

    # --- ROI History for Undo ---
    def _init_roi_history_tracking(self):
        """初始化历史追踪（在开始绘制时调用）"""
        self._roi_history = []
        self._prev_roi_state = None  # 追踪"变更前"的状态
        # 保存初始空状态，允许撤销回空
        self._prev_roi_state = {'data': [], 'features': {'label': [], 'frame_range': [], 'frame_info': []}}
    
    def _save_prev_state(self):
        """保存当前状态作为"下一次变更前"的状态"""
        if "Batch_ROI" not in self.viewer.layers:
            return
        layer = self.viewer.layers["Batch_ROI"]
        
        import copy
        self._prev_roi_state = {
            'data': [roi.copy() for roi in layer.data] if len(layer.data) > 0 else [],
            'features': copy.deepcopy(dict(layer.features)) if len(layer.data) > 0 else {'label': [], 'frame_range': [], 'frame_info': []}
        }
    
    def _push_roi_history(self):
        """将"变更前的状态"推入历史栈"""
        if self._prev_roi_state is None:
            return
        
        # 推入的是变更前的状态，而非当前状态
        self._roi_history.append(self._prev_roi_state)
        
        # 限制历史记录数量
        if len(self._roi_history) > self._roi_history_max:
            self._roi_history.pop(0)

    def _pop_roi_history(self):
        """从历史栈恢复 ROI 状态"""
        if not self._roi_history:
            return False
        
        if "Batch_ROI" not in self.viewer.layers:
            return False
        
        layer = self.viewer.layers["Batch_ROI"]
        snapshot = self._roi_history.pop()
        
        self._is_updating = True
        try:
            layer.data = snapshot['data']
            layer.features = snapshot['features']
            layer.selected_data = set()
        finally:
            self._is_updating = False
            layer.refresh()
        
        self._last_shape_count = len(layer.data)
        
        # 恢复后，更新 _prev_roi_state 为当前状态
        self._save_prev_state()
        return True

    def _clear_roi_history(self):
        """清空历史栈（新建会话时调用）"""
        self._roi_history = []
        self._prev_roi_state = {'data': [], 'features': {'label': [], 'frame_range': [], 'frame_info': []}}


    # --- Batch Crop Logic ---
    def _start_batch_mode(self):
        view_layer = self.batch_view_combo.currentData()
        if not view_layer: return
        
        # 1. 如果图层已存在，进入恢复模式 (Resume)
        if "Batch_ROI" in self.viewer.layers:
            layer = self.viewer.layers["Batch_ROI"]
            layer.mode = 'add_rectangle'
            # === 【重要】重新绑定事件（防止丢失） ===
            try:
                layer.events.set_data.disconnect(self._on_selection_change)
            except:
                pass
            layer.events.set_data.connect(self._on_selection_change)

            self._bind_smart_mode_switch(layer)
            
            self.status_label.setText(f"""✏️ {tr("Resuming Draw on '%s'.") % view_layer}""")
            self._clear_residue(["Crop_ROI", "Rotation_Line", "Interaction_Box","Drift_ROI","Measurements"]) 
            if self.lock_view_check.isChecked():
                self._force_view_active = True
                self._enforce_view_visibility()
            return

        # 2. 如果图层不存在，初始化新图层 (Initialize)
        self._clear_residue(["Batch_ROI", "Rotation_Line", "Crop_ROI", "Interaction_Box", "Preview_Overlay", "Drift_ROI","Measurements"])
        
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
            features={'label': [], 'frame_range': [], 'frame_info': []}
        )
        
        roi_layer.events.data.connect(self._on_batch_data_change)
        # 绑定点击背景事件
        roi_layer.events.set_data.connect(self._on_selection_change) 

        roi_layer.mode = 'add_rectangle'

        self._bind_smart_mode_switch(roi_layer)
        
        # ===【增强】清空历史栈，开始新会话 ===
        self._clear_roi_history()
        
        # ===【增强】绑定撤销 (支持绘制模式和选择模式) ===
        undo_key = GlobalConfig.get_napari_shortcut("shortcut_undo_drift")
        @roi_layer.bind_key(undo_key, overwrite=True)
        def undo_batch_rect(layer):
            if self._pop_roi_history():
                self.status_label.setText(f"↩️ {tr('Undone. Adjust ROI and try again.')}")
            else:
                self.status_label.setText(f"⚠️ {tr('Nothing to undo.')}")
        
        switch_key = GlobalConfig.get_napari_shortcut("shortcut_switch_mode")
        self.status_label.setText(f"""✏️ {tr("Drawing on '%s'. (New Layer)") % view_layer}""")
        self._last_shape_count = 0
    
    def _bind_smart_mode_switch(self, layer):
        """
        使用 napari 的鼠标事件，更精确地处理交互
        """
        
        self._last_click_time = 0
        self._double_click_threshold = 0.3
        self._mouse_press_pos = None
        self._drag_threshold = 5
        
        # === 【方案1】使用 mouse_double_click 事件（推荐）===
        @layer.mouse_double_click_callbacks.append
        def on_double_click(layer, event):
            """双击事件专用处理"""
            if layer.mode != 'select':
                return
            
            # 检查是否双击在背景上
            clicked_on_shape = False
            click_pos = event.position
            
            for shape_data in layer.data:
                ys, xs = shape_data[:, 0], shape_data[:, 1]
                y_min, y_max = np.min(ys), np.max(ys)
                x_min, x_max = np.min(xs), np.max(xs)
                
                click_y = click_pos[-2] if len(click_pos) > 1 else click_pos[0]
                click_x = click_pos[-1]
                
                if y_min <= click_y <= y_max and x_min <= click_x <= x_max:
                    clicked_on_shape = True
                    break
            
            # 双击背景 → 切换到绘制模式
            if not clicked_on_shape:
                layer.mode = 'add_rectangle'
                layer.selected_data = set()
                self.status_label.setText(f"✏️ {tr('Draw Mode (double-click bg)')}")
        
        # === 【可选】单击背景取消选择（不切换模式）===
        self._is_dragging = False
        self._press_pos = None
        
        @layer.mouse_drag_callbacks.append  
        def track_drag(layer, event):
            """追踪拖拽，区分点击和拖拽"""
            if layer.mode != 'select':
                return
            
            if event.type == 'mouse_press':
                self._press_pos = np.array(event.position)
                self._is_dragging = False
                
            elif event.type == 'mouse_move' and self._press_pos is not None:
                current_pos = np.array(event.position)
                distance = np.linalg.norm(current_pos - self._press_pos)
                if distance > self._drag_threshold:
                    self._is_dragging = True
                    
            elif event.type == 'mouse_release':
                # 单击背景（非拖拽）→ 取消选择
                if not self._is_dragging and self._press_pos is not None:
                    clicked_on_shape = False
                    click_pos = event.position
                    
                    for shape_data in layer.data:
                        ys, xs = shape_data[:, 0], shape_data[:, 1]
                        y_min, y_max = np.min(ys), np.max(ys)
                        x_min, x_max = np.min(xs), np.max(xs)
                        click_y = click_pos[-2] if len(click_pos) > 1 else click_pos[0]
                        click_x = click_pos[-1]
                        
                        if y_min <= click_y <= y_max and x_min <= click_x <= x_max:
                            clicked_on_shape = True
                            break
                    
                    if not clicked_on_shape:
                        layer.selected_data = set()
                        self.status_label.setText(f"🖐️ {tr('Select Mode (double-click bg to draw)')}")
                
                self._press_pos = None
                self._is_dragging = False
    
    def _on_selection_change(self, event=None):
        pass
        # """当用户点击背景时，自动切回绘制模式"""
        # if "Batch_ROI" not in self.viewer.layers: 
        #     return
        # layer = self.viewer.layers["Batch_ROI"]
        
        # # 条件：
        # # 1. 当前是 select 模式
        # # 2. 没有选中任何ROI（selected_data 为空）
        # # 3. 至少已经画了一个ROI（避免初始化时误触发）
        # if (layer.mode == 'select' and 
        #     len(layer.selected_data) == 0 and 
        #     len(layer.data) > 0):
        #     layer.mode = 'add_rectangle'
        #     self.status_label.setText("✏️ Auto-switched to Draw Mode (clicked background)")

    # === [新增] 点击背景切回绘制 ===
    # === [核心逻辑] 点击背景 -> 自动切换回绘制模式 (相当于按 R) ===
    # def _on_click_background_switch(self, layer, event):
    #     """
    #     交互逻辑优化 V3 (QTimer 延迟):
    #     解决 Napari 原生逻辑覆盖问题。
    #     原理：让 Napari 先处理完所有的点击和状态刷新，
    #     然后我们通过 QTimer 延迟 50ms 强制执行模式切换。
    #     """
    #     # 只在选择模式下生效
    #     if layer.mode != 'select':
    #         return

    #     # 1. 记录按下位置
    #     start_pos = event.pos

    #     # 2. 等待释放
    #     yield
    #     while event.type == 'mouse_move':
    #         yield

    #     # --- 鼠标释放后 ---

    #     # 3. 判断是否为点击 (防抖)
    #     drag_distance = np.linalg.norm(np.array(event.pos) - np.array(start_pos))
    #     if drag_distance > 3:
    #         return # 是拖拽，不处理

    #     # 4. 判断是否点击了背景
    #     # 注意：这里我们不做过多的异常捕获，直接信任 get_value
    #     # 如果 Napari 版本更新导致 get_value 行为改变，这里可能需要调整
    #     val = None
    #     try:
    #         val = layer.get_value(
    #             event.position, 
    #             world=True, 
    #             view_direction=event.view_direction, 
    #             dims_displayed=event.dims_displayed
    #         )
    #     except:
    #         # 回退兼容
    #         try: val = layer.get_value(event.position, world=True)
    #         except: pass

    #     # 5. 如果是背景 (None)，则【延迟】切换
    #     if val is None:
            
    #         def do_switch():
    #             # 再次检查 (防止用户手速极快已经切走了)
    #             if layer.mode == 'select':
    #                 layer.selected_data = set() # 再次确保清空
    #                 layer.mode = 'add_rectangle'
    #                 self.status_label.setText("✏️ Drawing Mode (Auto-switch)")
    #                 layer.refresh()

    #         # 关键：延迟 20ms 执行，避开 Napari 内部事件循环的冲突
    #         QTimer.singleShot(20, do_switch)

    def _set_range_for_selected_roi(self):
        if "Batch_ROI" not in self.viewer.layers: return
        layer = self.viewer.layers["Batch_ROI"]
        
        selected_idxs = list(layer.selected_data)
        if not selected_idxs:
            self.status_label.setText(f"⚠️ {tr('No ROI selected. Select a green box first.')}")
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
        self.status_label.setText(f"""✅ {tr("Set range '%s' for %s ROI(s).") % (range_str, len(selected_idxs))}""")

    def _switch_to_select_mode(self):
        if "Batch_ROI" in self.viewer.layers:
            self.viewer.layers["Batch_ROI"].mode = 'select'
            self.status_label.setText(f"🖐️ {tr('Adjust Mode.')}")

    def _on_batch_data_change(self, event=None):
        if self._is_updating: return
        if "Batch_ROI" not in self.viewer.layers: return
        
        # ===【增强】修改前保存状态到历史栈 ===
        self._push_roi_history()
        
        layer = self.viewer.layers["Batch_ROI"]
        current_count = len(layer.data)

        # 🔄 先记录是否需要切换模式，但不立即执行
        should_switch_to_select = (
            hasattr(self, '_last_shape_count') and 
            current_count > self._last_shape_count and 
            layer.mode == 'add_rectangle'
        )

        # 交互优化：画完自动切换到 Select 模式
        # if hasattr(self, '_last_shape_count') and current_count > self._last_shape_count:
        #     if layer.mode == 'add_rectangle':
        #         layer.selected_data = {current_count - 1}
        #         layer.mode = 'select'
        #         self.status_label.setText("🖐️ Adjust Mode (Click bg to draw)")
        
        self._last_shape_count = current_count
        
        # 获取图像尺寸用于边界检查和强制正方形
        # 优先使用 combo box 选择的图层，否则使用任意可见图像图层
        view_layer_name = self.batch_view_combo.currentData()
        img_layer = None
        
        if view_layer_name and view_layer_name in self.viewer.layers:
            img_layer = self.viewer.layers[view_layer_name]
        else:
            # 回退：查找任意可见的图像图层
            for l in self.viewer.layers:
                if isinstance(l, napari.layers.Image) and l.visible:
                    img_layer = l
                    break
            # 如果没有可见的，查找任意图像图层
            if img_layer is None:
                for l in self.viewer.layers:
                    if isinstance(l, napari.layers.Image):
                        img_layer = l
                        break
        
        # 如果完全没有图像图层，只更新标签不做尺寸约束
        if img_layer is None:
            IMG_H, IMG_W = float('inf'), float('inf')  # 无限大，不做边界约束
        else:
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

        if should_switch_to_select:
            layer.selected_data = {current_count - 1}
            layer.mode = 'select'
            self.status_label.setText(f"🖐️ {tr('Adjust Mode (Click bg to draw)')}")
        
        # === [SessionLogger] 记录 ROI 快照用于恢复 ===
        try:
            view_layer_name = self.batch_view_combo.currentData()
            data_layer_name = self.batch_data_combo.currentData()
            
            rois_snapshot = []
            for i, poly in enumerate(layer.data):
                rois_snapshot.append({
                    "coordinates": poly.tolist(),
                    "label": str(layer.features['label'][i]) if i < len(layer.features['label']) else str(i),
                    "frame_range": str(layer.features['frame_range'][i]) if i < len(layer.features['frame_range']) else ""
                })
            
            get_logger().log_action("geometry", "update_batch_rois", {
                "view_layer": view_layer_name,
                "data_layer": data_layer_name,
                "roi_count": len(rois_snapshot),
                "rois": rois_snapshot
            })
        except: pass
        
        # ===【增强】处理完成后，保存当前状态作为下一次变更的"前状态" ===
        self._save_prev_state()

    
    # === [新增] 全面清理 (带弹窗保护) ===
    def _clear_all_overlays(self):
        targets = ["Batch_ROI", "Rotation_Line", "Crop_ROI", "Interaction_Box", 
                   "Preview_Overlay", "Drift_ROI", "Measurements"]
        
        has_residue = any(name in self.viewer.layers for name in targets)
        if not has_residue:
            self.status_label.setText(f"⚠️ {tr('Nothing to clear.')}")
            return

        if GlobalConfig.get("show_clear_warning"):
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle(tr("Clear All Overlays?"))
            msg_box.setText(tr("Clear ALL temporary drawings (ROIs, Lines, etc.)?"))
            msg_box.setInformativeText(tr("This action cannot be undone."))
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            
            cb_dont_ask = QCheckBox(tr("Do not ask again"))
            msg_box.setCheckBox(cb_dont_ask)
            
            if msg_box.exec_() == QMessageBox.No: return
            
            if cb_dont_ask.isChecked():
                GlobalConfig.set("show_clear_warning", False)

        self._clear_residue(targets)
        self.status_label.setText(f"🗑️ {tr('Canvas cleared.')}")
        if hasattr(self, '_last_shape_count'): self._last_shape_count = 0

    def _detect_image_source(self, base_path, layer_name):
        """
        辅助函数：根据 JSON 路径和图层名，嗅探是否存在 TIFF 或 PNG 序列
        返回: (Path, mode) or (None, None)
        """
        if not layer_name: return None, None

        names_to_check = [
            f"{layer_name}_ViewRef", 
            f"{layer_name}_DataRef", 
            layer_name
        ]
        
        # 候选路径策略
        candidates = []
        for name in names_to_check:
            # 1. 同级目录下的文件/文件夹
            candidates.append((base_path.with_name(f"{name}.tiff"), 'tiff'))
            candidates.append((base_path.with_name(f"{name}.tif"), 'tiff'))
            candidates.append((base_path.with_name(f"{name}_Seq"), 'stack'))
            
            # 2. 父目录下的文件/文件夹
            candidates.append((base_path.parent / f"{name}.tiff", 'tiff'))
            candidates.append((base_path.parent / f"{name}_Seq", 'stack'))
            candidates.append((base_path.parent / name, 'stack'))

        for p, mode in candidates:
            if p.exists():
                if mode == 'stack':
                    # 简单检查文件夹里是否有图
                    if any(p.glob("*.png")) or any(p.glob("*.jpg")): return p, mode
                else:
                    return p, mode
        return None, None

    # === [新增] 智能保存 ROI (含参考图和TIFF提示) ===
    # === [核心逻辑修改] 智能保存 ROI (处理 Same vs Different Layers) ===
    def _save_rois_to_json(self):
        if "Batch_ROI" not in self.viewer.layers: return
        roi_layer = self.viewer.layers["Batch_ROI"]
        if len(roi_layer.data) == 0: return
        
        view_layer_name = self.batch_view_combo.currentData()
        data_layer_name = self.batch_data_combo.currentData()
        
        if not view_layer_name or view_layer_name not in self.viewer.layers:
            self.status_label.setText(f"❌ {tr('Ref image missing.')}")
            return
        
        # 确定保存策略
        save_target_layers = [] # list of (layer_obj, suffix)
        format_ext = None # 初始化变量
        
        # 1. 检测是否需要保存参考图
        layers_are_same = (view_layer_name == data_layer_name)
        
        # 构建询问对话框
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(tr("Save Reference Images?"))
        
        text = tr("Saving ROI JSON.\nDo you also want to save the reference image(s)?")
        if not layers_are_same:
            text += "\n\n" + tr("Note: Data Layer and View Layer are DIFFERENT.\nData: %s\nView: %s") % (data_layer_name, view_layer_name)
        msg_box.setText(text)
        
        # 按钮设计
        btn_tiff = msg_box.addButton(tr("TIFF Stack"), QMessageBox.ActionRole)
        btn_png = msg_box.addButton(tr("PNG Sequence"), QMessageBox.ActionRole)
        btn_skip = msg_box.addButton(tr("Skip Images"), QMessageBox.RejectRole)
        
        msg_box.exec_()
        choice = msg_box.clickedButton()
        
        if choice == btn_skip:
            pass # Just save JSON
        else:
            format_ext = 'tiff' if choice == btn_tiff else 'png_seq'
            
            # 如果图层不同，询问保存哪一个
            if not layers_are_same:
                sub_box = QMessageBox(self)
                sub_box.setWindowTitle(tr("Select Layers"))
                sub_box.setText(tr("Which layer(s) should be saved as reference?"))
                btn_both = sub_box.addButton(tr("Save Both"), QMessageBox.ActionRole)
                btn_data = sub_box.addButton(tr("Data Only"), QMessageBox.ActionRole)
                btn_view = sub_box.addButton(tr("View Only"), QMessageBox.ActionRole)
                sub_box.exec_()
                
                sub_choice = sub_box.clickedButton()
                if sub_choice == btn_both:
                    save_target_layers.append((self.viewer.layers[data_layer_name], "_DataRef"))
                    save_target_layers.append((self.viewer.layers[view_layer_name], "_ViewRef"))
                elif sub_choice == btn_data:
                    save_target_layers.append((self.viewer.layers[data_layer_name], "_DataRef"))
                elif sub_choice == btn_view:
                    save_target_layers.append((self.viewer.layers[view_layer_name], "_ViewRef"))
            else:
                # 相同，直接保存一个
                save_target_layers.append((self.viewer.layers[view_layer_name], ""))

        # 开始文件操作
        start_dir = QSettings("NapariUser", "Global").value("archive_path", str(Path.home()))
        default_name = f"ROIs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path_str, _ = QFileDialog.getSaveFileName(self, "Save ROI JSON", str(Path(start_dir) / default_name), "JSON (*.json)")
        
        if not path_str: return
        json_path = Path(path_str)
        
        try:
            # 1. 保存 JSON
            rois_data = []
            feats = roi_layer.features
            for i, poly in enumerate(roi_layer.data):
                rois_data.append({
                    "id": i, "coordinates": poly.tolist(),
                    "label": str(feats['label'][i]) if i < len(feats['label']) else str(i),
                    "frame_range": str(feats['frame_range'][i]) if i < len(feats['frame_range']) else "",
                    "frame_info": str(feats['frame_info'][i]) if i < len(feats['frame_info']) else ""
                })
            
            env_info = {
                "source_layer_name": view_layer_name,
                "data_layer_name": data_layer_name,
                "rotation_applied": "Rotated" in view_layer_name
            }
            
            data_dump = {
                "version": "1.2", "type": "MagicImageJ_ROI",
                "timestamp": str(datetime.datetime.now()),
                "environment": env_info, "rois": rois_data
            }
            
            with open(json_path, 'w', encoding='utf-8') as f: json.dump(data_dump, f, indent=2)
            msg = f"✅ {tr('Saved JSON')}."

            # 2. 执行图像保存
            for layer_obj, suffix in save_target_layers:
                safe_name = f"{layer_obj.name}{suffix}"
                # 如果是PNG序列，建立文件夹
                if format_ext == 'png_seq':
                    seq_folder = json_path.parent / f"{safe_name}_Seq"
                    seq_folder.mkdir(parents=True, exist_ok=True)
                    
                    # 使用进度条防止卡死
                    prog = QProgressDialog(f"Saving {safe_name} Sequence...", "Cancel", 0, len(layer_obj.data), self)
                    prog.setWindowModality(Qt.WindowModal)
                    prog.show()
                    
                    for i, frame in enumerate(layer_obj.data):
                        if prog.wasCanceled(): break
                        save_path = seq_folder / f"{i:05d}.png"
                        # 简易归一化以确保 PNG 可视化
                        if frame.dtype != np.uint8:
                            mn, mx = frame.min(), frame.max()
                            if mx > mn: frame_out = ((frame - mn) / (mx - mn) * 255).astype(np.uint8)
                            else: frame_out = frame.astype(np.uint8)
                        else: frame_out = frame
                        cv2.imwrite(str(save_path), frame_out)
                        prog.setValue(i)
                    prog.close()
                    msg += f"\n{tr('Saved Seq')}: {seq_folder.name}"
                    
                # 如果是TIFF
                elif format_ext == 'tiff':
                    tiff_path = json_path.parent / f"{safe_name}.tiff"
                    if export_to_tiff_stack(layer_obj.data, str(tiff_path)):
                        msg += f"\n{tr('Saved TIFF:')} {tiff_path.name}"

            # 3. 总是保存一张 Ref Snapshot (PNG) 方便快速预览
            ref_png = json_path.with_name(json_path.stem + "_ref.png")
            self._save_reference_snapshot(self.viewer.layers[view_layer_name], roi_layer, ref_png)
            
            # === [SessionLogger] 记录 ROI 保存操作 ===
            try:
                get_logger().log_action("geometry", "save_rois", {
                    "json_path": str(json_path),
                    "roi_count": len(rois_data),
                    "view_layer": view_layer_name,
                    "data_layer": data_layer_name
                })
            except: pass
            
            self.status_label.setText(msg)
            QMessageBox.information(self, "Success", msg)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    
    def _save_reference_snapshot(self, img_layer, roi_layer, save_path):
        try:
            current_step = self.viewer.dims.current_step[0]
            idx = max(0, min(current_step, img_layer.data.shape[0] - 1))
            frame = img_layer.data[idx]
            
            if frame.dtype != np.uint8:
                mn, mx = frame.min(), frame.max()
                if mx > mn: frame = ((frame - mn) / (mx - mn) * 255).astype(np.uint8)
                else: frame = frame.astype(np.uint8)
            
            pil_img = Image.fromarray(frame).convert("RGB")
            draw = ImageDraw.Draw(pil_img)
            try: font = ImageFont.truetype("arial.ttf", 20)
            except: font = ImageFont.load_default()
            
            for i, poly in enumerate(roi_layer.data):
                ys, xs = poly[:, 0], poly[:, 1]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
                label = str(roi_layer.features['label'][i]) if i < len(roi_layer.features['label']) else str(i+1)
                draw.text((x1+5, y1+5), label, fill="#00FF00", font=font)
            
            draw.text((10, 10), f"Ref Frame: {idx}\nLayer: {img_layer.name}", fill="yellow", font=font)
            pil_img.save(save_path)
        except Exception as e: print(f"{tr('Ref snap failed:')} {e}")

    # =========================================================
    #  新增辅助函数：负责弹窗询问 + 文件选择 + 模式判断
    # =========================================================
    def _prompt_user_for_file(self, layer_name, title_prefix=""):
        """
        弹出对话框询问用户是否手动查找文件。
        返回: (path, mode) 或 (None, None)
        """
        reply = QMessageBox.question(
            self, 
            f"{title_prefix}{tr('Image Not Found')}", 
            tr("Could not auto-locate image for layer:\n\n'%s'\n\nBrowse for it manually?") % layer_name,
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return None, None

        # 打开文件选择器 (支持 TIFF 或 PNG/JPG 序列中的任意一张)
        start_dir = QSettings("NapariUser", "Global").value("archive_path", str(Path.home()))
        path_str, _ = QFileDialog.getOpenFileName(
            self, 
            tr("Select Image for '%s'") % layer_name, 
            start_dir, 
            "Images (*.tiff *.tif *.png *.jpg *.bmp)"
        )
        
        if not path_str:
            return None, None
            
        p = Path(path_str)
        # 智能判断：如果是 tiff 则为单文件模式，否则假设选中了序列中的一张，取其父文件夹
        mode = 'tiff' if p.suffix.lower() in ['.tiff', '.tif'] else 'stack'
        final_path = p if mode == 'tiff' else p.parent
        
        return final_path, mode

    # === [新增] 智能加载 ROI (含 Auto TIFF) ===
    # === [核心逻辑升级] 智能加载 ROI (探测报告 + 用户确认 + 手动回退) ===
    def _load_rois_from_json(self):
        start_dir = QSettings("NapariUser", "Global").value("archive_path", str(Path.home()))
        path_str, _ = QFileDialog.getOpenFileName(self, "Load ROI JSON", start_dir, "JSON (*.json)")
        if not path_str: return
        
        json_path = Path(path_str)
        try:
            with open(json_path, 'r', encoding='utf-8') as f: data_dump = json.load(f)
            if data_dump.get("type") != "MagicImageJ_ROI": raise ValueError("Invalid Format")
            
            env = data_dump.get("environment", {})
            view_name = env.get("source_layer_name", "Recovered_View")
            data_name = env.get("data_layer_name", view_name) 
            
            # 1. 自动探测阶段
            path_view, mode_view = self._detect_image_source(json_path, view_name)
            
            is_same_layer = (data_name == view_name)
            path_data, mode_data = (None, None)
            if not is_same_layer:
                path_data, mode_data = self._detect_image_source(json_path, data_name)
            
            # 2. 构建探测报告 (HTML 格式)
            msg_text = f"<b>{tr('Image Source Detection Report')}:</b><br><br>"
            
            # View Layer 报告
            msg_text += f"<b>{tr('View Layer')}:</b> {view_name}<br>"
            if path_view:
                msg_text += f"&nbsp;&nbsp;✅ {tr('Found')}: {path_view.name} ({mode_view})<br>"
            else:
                msg_text += f"&nbsp;&nbsp;❌ {tr('Not Found (Auto-detection failed)')}<br>"
                
            # Data Layer 报告
            if is_same_layer:
                 msg_text += f"<br><b>{tr('Data Layer')}:</b> ({tr('Same as View Layer')})<br>"
            else:
                msg_text += f"<br><b>{tr('Data Layer')}:</b> {data_name}<br>"
                if path_data:
                    msg_text += f"&nbsp;&nbsp;✅ {tr('Found')}: {path_data.name} ({mode_data})<br>"
                else:
                    msg_text += f"&nbsp;&nbsp;❌ {tr('Not Found (Auto-detection failed)')}<br>"
            
            msg_text += "<br>---------------------------------<br>"
            msg_text += tr("Do you want to load these images?")

            # 3. 弹窗询问用户
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle(tr("Confirm Import Sources"))
            msg_box.setTextFormat(Qt.RichText) # 启用 HTML 渲染
            msg_box.setText(msg_text)
            
            # 动态添加按钮
            # 只要探测到了至少一个文件，就允许 Auto Load
            has_auto_candidate = (path_view is not None) or (path_data is not None)
            
            btn_auto = None
            if has_auto_candidate:
                btn_auto = msg_box.addButton(f"✅ {tr('Auto Load Detected')}", QMessageBox.ActionRole)
            
            btn_manual = msg_box.addButton(f"🛠️ {tr('Manual Select')}", QMessageBox.ActionRole)
            btn_skip = msg_box.addButton(tr("Skip Images (ROIs Only)"), QMessageBox.RejectRole)
            
            msg_box.exec_()
            choice = msg_box.clickedButton()
            
            tasks_to_run = []
            
            # 4. 根据用户选择构建任务
            if choice == btn_skip:
                self._restore_rois_to_layer(data_dump)
                return

            elif choice == btn_auto:
                # 用户确认无误，使用探测到的路径
                if is_same_layer:
                    if path_view:
                        tasks_to_run.append({'path': path_view, 'mode': mode_view, 'name': view_name, 'role': 'both'})
                else:
                    if path_data:
                        tasks_to_run.append({'path': path_data, 'mode': mode_data, 'name': data_name, 'role': 'data'})
                    if path_view:
                        tasks_to_run.append({'path': path_view, 'mode': mode_view, 'name': view_name, 'role': 'view'})
            
            elif choice == btn_manual:
                # 用户觉得不对，进入手动选择流程
                if is_same_layer:
                    # View/Data 同层 -> 选一次
                    p, m = self._prompt_user_for_file(view_name, title_prefix="[View/Data] ")
                    if p: tasks_to_run.append({'path': p, 'mode': m, 'name': view_name, 'role': 'both'})
                else:
                    # View Layer -> 选一次
                    # 询问是否需要加载 View
                    if QMessageBox.question(self, tr("Load View Layer?"), tr("Load image for View Layer: '%s'?") % view_name, QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
                        p_v, m_v = self._prompt_user_for_file(view_name, title_prefix="[View Layer] ")
                        if p_v: tasks_to_run.append({'path': p_v, 'mode': m_v, 'name': view_name, 'role': 'view'})
                    
                    # Data Layer -> 选一次
                    if QMessageBox.question(self, tr("Load Data Layer?"), tr("Load image for Data Layer: '%s'?") % data_name, QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
                        p_d, m_d = self._prompt_user_for_file(data_name, title_prefix="[Data Layer] ")
                        if p_d: tasks_to_run.append({'path': p_d, 'mode': m_d, 'name': data_name, 'role': 'data'})

            # 5. 提交任务给后台线程
            if tasks_to_run:
                count = len(tasks_to_run)
                self.load_progress = QProgressDialog(tr("Loading %s image(s)...") % count, tr("Cancel"), 0, 0, self)
                self.load_progress.setWindowModality(Qt.WindowModal)
                self.load_progress.show()
                
                # 初始化线程并传入任务列表
                self.loader_thread = BatchImageLoaderThread(tasks_to_run)
                
                # 连接信号
                self.loader_thread.item_ready.connect(self._on_single_image_loaded)
                # 全部完成后，关闭进度条并加载 ROI
                self.loader_thread.finished_all.connect(lambda: (self.load_progress.close(), self._restore_rois_to_layer(data_dump)))
                self.loader_thread.error.connect(lambda e: self.status_label.setText(f"{tr('Load Error:')} {e}"))
                
                self.loader_thread.start()
            else:
                # 用户可能在手动选择时取消了所有操作，或者 auto 模式下没有有效路径
                # 此时直接恢复 ROI
                self._restore_rois_to_layer(data_dump)

        except Exception as e:
            QMessageBox.critical(self, "JSON Load Error", str(e))
    
    def _on_single_image_loaded(self, data, name, role):
        """
        后台线程加载完一张图片后触发此函数。
        data: 图片数据 (numpy array)
        name: 图层名称
        role: 'data' (仅设为数据层), 'view' (仅设为视图层), 'both' (同时设为两者)
        """
        if data is None: return
        
        try:
            # 1. 将数据添加到 Napari 视图中
            # colormap='gray' 是默认设置，您可以根据需要调整
            new_layer = self.viewer.add_image(data, name=name, colormap='gray')
            
            # 2. 根据 role 自动选中下拉框
            # 这样用户就不用手动去 ComboBox 里再选一次了
            self._refresh_layers()  # 先刷新图层列表
            if role == 'data':
                idx = self.batch_data_combo.findData(new_layer.name)
                if idx >= 0: self.batch_data_combo.setCurrentIndex(idx)
            elif role == 'view':
                idx = self.batch_view_combo.findData(new_layer.name)
                if idx >= 0: self.batch_view_combo.setCurrentIndex(idx)
            elif role == 'both':
                idx1 = self.batch_data_combo.findData(new_layer.name)
                if idx1 >= 0: self.batch_data_combo.setCurrentIndex(idx1)
                idx2 = self.batch_view_combo.findData(new_layer.name)
                if idx2 >= 0: self.batch_view_combo.setCurrentIndex(idx2)
                
            # 3. 强制刷新一下下拉框状态（有时候添加新图层后 Combo 不会自动刷新）
            # 注意：_refresh_layers 已经在 layer inserted 事件中绑定了，
            # 但为了确保 setCurrentText 生效，这里不做额外操作通常也可以。
            
        except Exception as e:
            print(f"Error adding layer '{name}' to viewer: {e}")
            # 如果出错，给用户一个非阻塞的提示（可选）
            self.status_label.setText(f"❌ Failed to add layer: {name}")

    def restore_from_processing_log(self, log_path):
        """
        从 processing_log.json 恢复 ROI (核心方法)
        此方法直接创建图层，不依赖 _start_batch_mode (避免 combo box 空的问题)
        """
        from pathlib import Path
        
        log_path = Path(log_path)
        if not log_path.exists():
            QMessageBox.critical(None, tr("Error"), tr("Log file not found: %s") % log_path)
            return False
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(None, tr("Error"), tr("Failed to load JSON: %s") % e)
            return False
        
        if "batch_crop" not in data or "rois" not in data["batch_crop"]:
            QMessageBox.warning(None, tr("Error"), tr("Invalid JSON format: missing 'batch_crop' or 'rois'"))
            return False
        
        rois_list = data["batch_crop"]["rois"]
        if not rois_list:
            QMessageBox.information(None, tr("Info"), tr("No ROIs found in log."))
            return False
        
        # 1. 先清除旧的 Batch_ROI 图层
        if "Batch_ROI" in self.viewer.layers:
            self.viewer.layers.remove("Batch_ROI")
        
        # 2. 解析数据
        new_data = []
        new_labels = []
        new_ranges = []
        new_infos = []
        
        for i, item in enumerate(rois_list):
            bbox = item.get("bbox")
            roi_id = item.get("id", i + 1)
            fr = str(item.get("frame_range_used", ""))
            
            # 处理帧范围显示
            if fr.lower() == "nan" or fr.strip() == "":
                fr_display = "All"  # nan 或空表示使用全部帧
                fr_store = ""  # 存储空值以保持后续导出一致
            else:
                fr_display = fr
                fr_store = fr
            
            if bbox and len(bbox) == 4:
                # processing_log 格式: [x1, y1, x2, y2]
                # Napari rectangle 格式: [[y1, x1], [y1, x2], [y2, x2], [y2, x1]]
                x1, y1, x2, y2 = bbox
                rect = np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]])
                new_data.append(rect)
                new_labels.append(str(roi_id))
                new_ranges.append(fr_store)  # 存储用于导出
                # frame_info 用于显示帧范围 (显示在ROI上)
                new_infos.append(f"[{fr_display}]")
        
        if not new_data:
            QMessageBox.information(None, tr("Info"), tr("No valid ROIs extracted from log."))
            return False
        
        # 3. 直接创建图层 (不调用 _start_batch_mode)
        box_col = GlobalConfig.get("style_batch_box_color")
        width = int(GlobalConfig.get("style_batch_width"))
        txt_col = GlobalConfig.get("style_batch_text_color")
        font_size = int(GlobalConfig.get("style_batch_font_size"))

        roi_layer = self.viewer.add_shapes(
            data=new_data,
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
                'label': new_labels, 
                'frame_range': new_ranges, 
                'frame_info': new_infos
            }
        )
        
        # 4. 绑定事件 (用于后续编辑)
        roi_layer.events.data.connect(self._on_batch_data_change)
        roi_layer.events.set_data.connect(self._on_selection_change)
        
        # 4.1 绑定智能模式切换 (双击背景切换绘制模式)
        self._bind_smart_mode_switch(roi_layer)
        
        # ===【增强】清空历史栈，开始新会话 ===
        self._clear_roi_history()
        
        # ===【增强】绑定撤销 (支持绘制模式和选择模式) ===
        undo_key = GlobalConfig.get_napari_shortcut("shortcut_undo_drift")
        @roi_layer.bind_key(undo_key, overwrite=True)
        def undo_batch_rect(layer):
            if self._pop_roi_history():
                self.status_label.setText(f"↩️ {tr('Undone. Adjust ROI and try again.')}")
            else:
                self.status_label.setText(f"⚠️ {tr('Nothing to undo.')}")
        
        # 6. 同步状态
        self._last_shape_count = len(new_data)
        
        # 7. 切换到选择模式
        roi_layer.mode = 'select'
        self.viewer.layers.selection.active = roi_layer
        
        # 8. 刷新下拉框，确保后续操作可以正常识别图层
        self._refresh_layers()
        
        # 9. 保存初始恢复状态，以便后续撤销
        self._save_prev_state()
        
        QMessageBox.information(None, tr("Success"), tr("Restored %s ROIs.") % len(new_data))
        return True

    def _restore_rois_to_layer(self, data_dump):
        """恢复 ROI - 不依赖 _start_batch_mode 以处理无图层的情况"""
        # 如果 Batch_ROI 图层不存在，直接创建（不要求必须有图层存在）
        if "Batch_ROI" not in self.viewer.layers:
            box_col = GlobalConfig.get("style_batch_box_color")
            width = int(GlobalConfig.get("style_batch_width"))
            txt_col = GlobalConfig.get("style_batch_text_color")
            font_size = int(GlobalConfig.get("style_batch_font_size"))
            
            roi_layer = self.viewer.add_shapes(
                name="Batch_ROI", shape_type='rectangle',
                edge_color=box_col, face_color=[0, 1, 0, 0.05], edge_width=width,
                text={'string': '{label}\n{frame_info}', 'size': font_size, 
                      'color': txt_col, 'anchor': 'upper_left', 'translation': [-5, -5]},
                features={'label': [], 'frame_range': [], 'frame_info': []}
            )
            roi_layer.events.data.connect(self._on_batch_data_change)
        
        layer = self.viewer.layers["Batch_ROI"]
        
        new_data, new_lbl, new_rng, new_inf = [], [], [], []
        for item in data_dump.get("rois", []):
            new_data.append(np.array(item["coordinates"]))
            new_lbl.append(item.get("label", ""))
            new_rng.append(item.get("frame_range", ""))
            new_inf.append(item.get("frame_info", ""))

        self._is_updating = True
        layer.data = new_data
        layer.features = {'label': new_lbl, 'frame_range': new_rng, 'frame_info': new_inf}
        self._is_updating = False
        layer.refresh()
        self._last_shape_count = len(new_data)
        
        if "Batch_ROI" in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers["Batch_ROI"]
            self.viewer.layers["Batch_ROI"].mode = 'select'

        self.status_label.setText(f"✅ Loaded {len(new_data)} ROIs.")

    def _export_batch_crops(self):
        # 1. 基础校验
        data_layer_name = self.batch_data_combo.currentData()
        view_layer_name = self.batch_view_combo.currentData()
        if "Batch_ROI" not in self.viewer.layers or not len(self.viewer.layers["Batch_ROI"].data):
            self.status_label.setText(f"❌ {tr('No ROIs defined.')}")
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
            
        ds_id = self.dataset_edit.text().strip()
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
            
            # 使用包含 dataset 编号的文件夹名称
            dataset_str = f"_{ds_id}" if ds_id else ""
            output_dir = Path(d) / f"{date_str}{dataset_str}_{sub_name}_Exports"
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
            
            # [View Export] - User Feature
            'export_view': self.export_view_check.isChecked(),
            'view_stack': view_stack,
            'view_suffix_main': self.export_view_suffix.text().strip() or "_contrasted",
            
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
        self.batch_progress = QProgressDialog(tr("Exporting Crops..."), tr("Cancel"), 0, len(rois), self)
        self.batch_progress.setWindowModality(Qt.WindowModal)
        self.batch_progress.setMinimumDuration(0)
        self.batch_progress.canceled.connect(self._on_export_cancel)
        self.batch_progress.show()

        # 6. 线程启动
        self.export_thread = BatchExportThread(params)
        self.export_thread.progress.connect(self.batch_progress.setValue)
        
        # [Req 6] 获取当前帧数
        current_frame_idx = self.viewer.dims.current_step[0]
        
        export_view_flag = self.export_view_check.isChecked()
        view_suffix = self.export_view_suffix.text().strip() or "_contrasted"
        data_suffix = self.suffix_edit.text().strip() or "_origin"
        
        self.export_thread.finished.connect(lambda c, path: self._on_export_finished(
            c, path, data_stack, view_stack, rois, sub_name, output_dir, current_frame_idx,
            export_view_flag, data_suffix, view_suffix
        ))
        self.export_thread.error.connect(self._on_export_error)
        self.export_thread.start()
        
    def _on_export_cancel(self):
        if self.export_thread.isRunning():
            self.export_thread.requestInterruption()
            self.status_label.setText(f"⚠️ {tr('Export canceled.')}")

    def _on_export_finished(self, count, path_name, data_stack, view_stack, rois, sub_name, output_dir, frame_idx, export_view_flag, data_suffix, view_suffix):
        self.batch_progress.close()
        
        # [Req 6] 使用指定的 frame_idx 分别导出数据层和视图层的概览图
        self._create_overview_map(data_stack, rois, sub_name, output_dir, frame_idx, suffix=data_suffix)
        if export_view_flag and view_stack is not None:
            self._create_overview_map(view_stack, rois, sub_name, output_dir, frame_idx, suffix=view_suffix)
        
        self.status_label.setText(f"✅ {tr('Exported %s crops.') % count}")
        self._force_view_active = False
        QMessageBox.information(self, tr("Success"), tr("Exported %s crops!\nSaved to: %s") % (count, path_name))

    def _on_export_error(self, err):
        self.batch_progress.close()
        self.status_label.setText(f"❌ {tr('Error:')} {err}")
        QMessageBox.critical(self, tr("Export Error"), str(err))

    def _create_overview_map(self, image_stack, rois, sample_name, output_dir, frame_idx, suffix=""):
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
            
        pil_img.save(output_dir / f"{sample_name}_Overview_Frame{idx}{suffix}.png")
        
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
                   self.check_denoise, self.check_refine, self.export_view_check, self.suffix_edit, self.padding_spin]
        for w in widgets: w.blockSignals(True)

        self.enlarge_check.setChecked(bool(GlobalConfig.get("geo_enlarge")))
        self.keep_index_check.setChecked(bool(GlobalConfig.get("geo_keep_index")))
        self.force_square_check.setChecked(bool(GlobalConfig.get("geo_force_square")))
        self.check_denoise.setChecked(bool(GlobalConfig.get("geo_create_denoise")))
        self.check_refine.setChecked(bool(GlobalConfig.get("geo_create_refine")))
        self.export_view_check.setChecked(bool(GlobalConfig.get("geo_export_view")))
        self.suffix_edit.setText(str(GlobalConfig.get("geo_suffix")))
        self.padding_spin.setValue(int(GlobalConfig.get("geo_padding")))

        # Update Style Configs (Color etc.) for Draw Rect logic
        # 这里的样式参数会在 _draw_crop_rect 调用时实时读取 GlobalConfig.get()，无需刷新 UI 控件

        for w in widgets: w.blockSignals(False)

    # === [新增] PNG 序列快速导入 ===
    def _quick_load_png_sequence(self):
        """快速加载 PNG 序列文件夹"""
        from qtpy.QtWidgets import QProgressDialog, QFileDialog
        from qtpy.QtCore import Qt, QSettings
        
        start_path = QSettings("NapariUser", "Global").value("archive_path", str(Path.home()))
        folder = QFileDialog.getExistingDirectory(self, tr("Select PNG Sequence Folder"), start_path)
        if not folder:
            return
        
        folder_path = Path(folder)
        png_files = sorted(list(folder_path.glob("*.png")))
        
        if not png_files:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(self, tr("No Images"), tr("No PNG files found in the selected folder."))
            return
        
        # 进度条
        progress = QProgressDialog(f"{tr('Loading')} {len(png_files)} PNG files...", tr("Cancel"), 0, len(png_files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(500)
        
        try:
            frames = []
            for i, f in enumerate(png_files):
                if progress.wasCanceled():
                    self.status_label.setText(tr("Loading canceled."))
                    return
                
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    frames.append(img)
                progress.setValue(i + 1)
            
            progress.close()
            
            if not frames:
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.warning(self, tr("Error"), tr("Could not read any valid images."))
                return
                
            stack = np.array(frames)
            name = f"PNG_{folder_path.name}"
            if len(name) > 30:
                name = name[:15] + "..." + name[-10:]
            
            new_layer = self.viewer.add_image(stack, name=name, colormap='gray')
            
            # 自动选中新加载的图层
            self._refresh_layers()
            idx1 = self.batch_view_combo.findData(new_layer.name)
            if idx1 >= 0: self.batch_view_combo.setCurrentIndex(idx1)
            idx2 = self.batch_data_combo.findData(new_layer.name)
            if idx2 >= 0: self.batch_data_combo.setCurrentIndex(idx2)
            self.status_label.setText(f"✅ {tr('Loaded')} {len(stack)} PNG frames.")
            
        except Exception as e:
            progress.close()
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.critical(self, tr("Error"), str(e))
            self.status_label.setText(f"❌ Error: {e}")

    # === [新增] TIFF Stack 快速导入 ===
    def _quick_load_tiff_stack(self):
        """快速加载 TIFF Stack 文件"""
        import tifffile
        from qtpy.QtWidgets import QFileDialog
        from qtpy.QtCore import QSettings
        
        start_path = QSettings("NapariUser", "Global").value("archive_path", str(Path.home()))
        file_path, _ = QFileDialog.getOpenFileName(
            self, tr("Select TIFF Stack File"), start_path, 
            "TIFF Files (*.tiff *.tif)"
        )
        if not file_path:
            return
        
        try:
            self.status_label.setText(tr("Loading TIFF..."))
            stack = tifffile.imread(file_path)
            
            # 确保是3D数组 (T, H, W)
            if stack.ndim == 2:
                stack = stack[np.newaxis, ...]
            elif stack.ndim == 4:
                stack = stack[..., 0]
            
            name = f"TIFF_{Path(file_path).stem}"
            if len(name) > 30:
                name = name[:15] + "..." + name[-10:]
            
            new_layer = self.viewer.add_image(stack, name=name, colormap='gray')
            
            # 自动选中新加载的图层
            self._refresh_layers()
            idx1 = self.batch_view_combo.findData(new_layer.name)
            if idx1 >= 0: self.batch_view_combo.setCurrentIndex(idx1)
            idx2 = self.batch_data_combo.findData(new_layer.name)
            if idx2 >= 0: self.batch_data_combo.setCurrentIndex(idx2)
            self.status_label.setText(f"✅ {tr('Loaded')} {len(stack)} TIFF frames.")
            
        except Exception as e:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.critical(self, tr("Error"), str(e))
            self.status_label.setText(f"❌ Error: {e}")
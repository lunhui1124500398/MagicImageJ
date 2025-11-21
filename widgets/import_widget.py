"""
数据导入控件 - 增强版
功能：
1. DM4序列加载
2. 路径记忆 (QSettings)
3. 电子剂量率自动计算 (基于 jamesra/dm4 库的精准 Tag 读取)
   - 支持读取曝光时间、亮度因子、像素单位
   - [NEW] 支持读取并显示采集日期和时间 (Acquisition Date/Time)
   - 自动换算为 e-·Å-2·s-1
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QFileDialog, QLabel, QSpinBox, QHBoxLayout,
                            QProgressBar, QComboBox, QGroupBox, QMessageBox,
                            QDoubleSpinBox)
from qtpy.QtCore import Signal, QThread, QSettings
import numpy as np
from pathlib import Path
import os
import struct

# 尝试导入 dm4 库 (https://github.com/jamesra/dm4)
try:
    import dm4
    DM4_LIB_AVAILABLE = True
except ImportError:
    DM4_LIB_AVAILABLE = False

# 保持 LoaderThread 使用 core.dm4_reader
from core.dm4_reader import read_dm4_sequence

class LoaderThread(QThread):
    """后台加载线程 (用于导入图层)"""
    progress = Signal(int, int)  # current, total
    finished = Signal(np.ndarray, dict)  # image_stack, metadata
    error = Signal(str)
    
    def __init__(self, folder_path, bit_depth, max_workers):
        super().__init__()
        self.folder_path = folder_path
        self.bit_depth = bit_depth
        self.max_workers = max_workers
    
    def run(self):
        try:
            def callback(current, total):
                self.progress.emit(current, total)
            
            image_stack, metadata = read_dm4_sequence(
                self.folder_path,
                bit_depth=self.bit_depth,
                max_workers=self.max_workers,
                progress_callback=callback
            )
            self.finished.emit(image_stack, metadata)
        except Exception as e:
            self.error.emit(str(e))


class DoseCalculationThread(QThread):
    """电子剂量率计算线程"""
    finished = Signal(float, str, dict) # dose_rate, filename, debug_info
    error = Signal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = Path(folder_path)

    def _decode_string(self, data):
        """解码 dm4 返回的字符串数据 (通常是 utf-16)"""
        if isinstance(data, str):
            return data
        if hasattr(data, 'tobytes'):
            b = data.tobytes()
        elif isinstance(data, (bytes, bytearray)):
            b = data
        elif isinstance(data, np.ndarray):
            b = data.tobytes()
        else:
            return str(data)
        
        try:
            # 尝试 utf-16 解码 (Gatan 标准)
            return b.decode('utf-16').rstrip('\x00')
        except:
            # 回退到 utf-8 或 ascii
            try:
                return b.decode('utf-8').rstrip('\x00')
            except:
                return str(data)

    def run(self):
        if not DM4_LIB_AVAILABLE:
            self.error.emit("Library 'dm4' not found. Please install it (e.g., pip install git+https://github.com/jamesra/dm4).")
            return

        try:
            # 1. 获取所有 DM4 文件 (递归查找)
            files = sorted(list(self.folder_path.rglob("*.dm4")))
            
            if not files:
                self.error.emit("No .dm4 files found in this folder (checked recursively).")
                return

            # 2. 智能选片策略
            total_files = len(files)
            mid_idx = total_files // 2
            search_radius = 5
            start_idx = max(0, mid_idx - search_radius)
            end_idx = min(total_files, mid_idx + search_radius + 1)
            
            candidate_files = files[start_idx:end_idx]
            if not candidate_files:
                 candidate_files = [files[mid_idx]]
            
            # 选体积最大的
            target_file = max(candidate_files, key=lambda p: p.stat().st_size)
            
            # 3. 使用 dm4 库读取
            with dm4.DM4File.open(str(target_file)) as dm4file:
                # 读取目录结构
                tags = dm4file.read_directory()
                
                # === 定位图像目录 ===
                image_list = tags.named_subdirs['ImageList']
                image_dir = image_list.unnamed_subdirs[0]
                
                # -------------------------------------------------------
                # 1. 获取 ImageData 和 ImageTags
                # -------------------------------------------------------
                image_tags_dir = image_dir.named_subdirs.get('ImageTags')
                image_data_dir = image_dir.named_subdirs.get('ImageData')
                
                if not image_tags_dir or not image_data_dir:
                    # 尝试 index 1
                    if len(image_list.unnamed_subdirs) > 1:
                        image_dir = image_list.unnamed_subdirs[1]
                        image_tags_dir = image_dir.named_subdirs.get('ImageTags')
                        image_data_dir = image_dir.named_subdirs.get('ImageData')
                
                if not image_data_dir:
                    raise ValueError("Could not find 'ImageData' directory in DM4 tags.")

                # -------------------------------------------------------
                # 2. 获取 曝光时间 & 采集时间 (新增)
                # -------------------------------------------------------
                exposure_time = 0.0
                acq_date = "Unknown"
                acq_time = ""

                if image_tags_dir and 'DataBar' in image_tags_dir.named_subdirs:
                    data_bar_dir = image_tags_dir.named_subdirs['DataBar']
                    # 曝光时间
                    if 'Exposure Time (s)' in data_bar_dir.named_tags:
                        exposure_time = dm4file.read_tag_data(data_bar_dir.named_tags['Exposure Time (s)'])
                    
                    # === 新增：获取采集日期 ===
                    if 'Acquisition Date' in data_bar_dir.named_tags:
                        raw_date = dm4file.read_tag_data(data_bar_dir.named_tags['Acquisition Date'])
                        acq_date = self._decode_string(raw_date)
                    
                    # === 新增：获取采集时间 ===
                    if 'Acquisition Time' in data_bar_dir.named_tags:
                        raw_time = dm4file.read_tag_data(data_bar_dir.named_tags['Acquisition Time'])
                        acq_time = self._decode_string(raw_time)
                
                # Fallback for Exposure Time
                if exposure_time == 0 and image_tags_dir and 'Acquisition' in image_tags_dir.named_subdirs:
                     try:
                         acq = image_tags_dir.named_subdirs['Acquisition'].named_subdirs['Parameters'].named_subdirs['High Level']
                         if 'Exposure' in acq.named_tags:
                             exposure_time = dm4file.read_tag_data(acq.named_tags['Exposure'])
                     except:
                         pass
                         
                if exposure_time <= 0:
                    raise ValueError(f"Invalid or missing Exposure Time: {exposure_time} s")

                # -------------------------------------------------------
                # 3. 获取 像素大小和单位
                # -------------------------------------------------------
                pixel_size = 1.0
                pixel_unit = ""
                
                calibrations_dir = image_data_dir.named_subdirs.get('Calibrations')
                if calibrations_dir and 'Dimension' in calibrations_dir.named_subdirs:
                    dims = calibrations_dir.named_subdirs['Dimension']
                    dim_x = dims.unnamed_subdirs[0]
                    if 'Scale' in dim_x.named_tags:
                        pixel_size = dm4file.read_tag_data(dim_x.named_tags['Scale'])
                    if 'Units' in dim_x.named_tags:
                        raw_unit = dm4file.read_tag_data(dim_x.named_tags['Units'])
                        pixel_unit = self._decode_string(raw_unit)
                
                # -------------------------------------------------------
                # 4. 获取 亮度转换因子
                # -------------------------------------------------------
                brightness_scale = 1.0
                if calibrations_dir and 'Brightness' in calibrations_dir.named_subdirs:
                    bright_dir = calibrations_dir.named_subdirs['Brightness']
                    if 'Scale' in bright_dir.named_tags:
                        brightness_scale = dm4file.read_tag_data(bright_dir.named_tags['Scale'])

                # -------------------------------------------------------
                # 5. 读取图像数据
                # -------------------------------------------------------
                if 'Data' in image_data_dir.named_tags:
                    raw_data = dm4file.read_tag_data(image_data_dir.named_tags['Data'])
                    image_array = np.array(raw_data)
                else:
                    raise ValueError("Image Data tag not found.")

            # -------------------------------------------------------
            # 6. 计算逻辑
            # -------------------------------------------------------
            
            # 单位换算
            pixel_size_A = pixel_size
            if 'nm' in pixel_unit:
                pixel_size_A = pixel_size * 10.0
            elif 'µm' in pixel_unit or 'um' in pixel_unit:
                pixel_size_A = pixel_size * 10000.0
            elif 'm' == pixel_unit:
                pixel_size_A = pixel_size * 1e10
            else:
                if pixel_size < 1e-6: pixel_size_A = pixel_size * 1e10

            pixel_area_A2 = pixel_size_A ** 2
            
            # 剂量率计算
            total_counts = np.sum(image_array)
            total_electrons = total_counts * brightness_scale
            total_area_A2 = image_array.size * pixel_area_A2
            
            dose_rate = total_electrons / (total_area_A2 * exposure_time)
            
            mean_intensity = np.mean(image_array)
            debug_info = {
                "file": target_file.name,
                "mean_counts": mean_intensity,
                "bright_scale": brightness_scale,
                "pixel_size_A": pixel_size_A,
                "unit_raw": pixel_unit,
                "exposure_s": exposure_time,
                "acq_date": acq_date, # 添加日期
                "acq_time": acq_time  # 添加时间
            }
            
            self.finished.emit(dose_rate, target_file.name, debug_info)
            
        except Exception as e:
            self.error.emit(str(e))


class ImportWidget(QWidget):
    """数据导入控件"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_folder = None
        self.loader_thread = None
        self.calc_thread = None
        
        self.settings = QSettings("NapariUser", "DM4Importer")
        
        self._setup_ui()
        
        last_folder = self.settings.value("last_folder", "")
        if last_folder and os.path.isdir(last_folder):
            self.current_folder = last_folder
            self.folder_label.setText(f"Selected: {last_folder}")
            self.load_btn.setEnabled(True)
            self.calc_dose_btn.setEnabled(True)
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("<h3>📂 Import DM4 Data</h3>")
        layout.addWidget(title)
        
        # 文件夹选择
        folder_group = QGroupBox("Data Source")
        folder_layout_box = QVBoxLayout()
        
        folder_btn_layout = QHBoxLayout()
        browse_btn = QPushButton("📂 Browse Folder...")
        browse_btn.clicked.connect(self._browse_folder)
        folder_btn_layout.addWidget(browse_btn)
        folder_layout_box.addLayout(folder_btn_layout)
        
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("color: gray; font-size: 11px;")
        folder_layout_box.addWidget(self.folder_label)
        
        folder_group.setLayout(folder_layout_box)
        layout.addWidget(folder_group)
        
        # 电子剂量率计算
        dose_group = QGroupBox("Electron Dose Rate")
        dose_layout = QVBoxLayout()
        
        calc_layout = QHBoxLayout()
        self.calc_dose_btn = QPushButton("🧮 Calculate Dose Rate")
        self.calc_dose_btn.setEnabled(False)
        self.calc_dose_btn.clicked.connect(self._calculate_dose_rate)
        calc_layout.addWidget(self.calc_dose_btn)
        
        calc_layout.addWidget(QLabel("Decimals:"))
        self.decimals_spin = QSpinBox()
        self.decimals_spin.setRange(0, 10)
        self.decimals_spin.setValue(2)
        self.decimals_spin.valueChanged.connect(self._update_dose_display)
        calc_layout.addWidget(self.decimals_spin)
        dose_layout.addLayout(calc_layout)
        
        self.dose_result_label = QLabel("Dose Rate: N/A")
        self.dose_result_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #2E8B57;")
        dose_layout.addWidget(self.dose_result_label)
        
        self.dose_info_label = QLabel("Select a folder to enable calculation.")
        self.dose_info_label.setWordWrap(True)
        self.dose_info_label.setStyleSheet("font-size: 10px; color: gray;")
        dose_layout.addWidget(self.dose_info_label)
        
        dose_group.setLayout(dose_layout)
        layout.addWidget(dose_group)
        
        # 加载参数
        load_group = QGroupBox("Loading Parameters")
        load_layout_box = QVBoxLayout()
        
        bit_layout = QHBoxLayout()
        bit_layout.addWidget(QLabel("Bit Depth:"))
        self.bit_depth_combo = QComboBox()
        self.bit_depth_combo.addItems(["8", "16", "32"])
        self.bit_depth_combo.setCurrentText("8")
        bit_layout.addWidget(self.bit_depth_combo)
        load_layout_box.addLayout(bit_layout)
        
        thread_layout = QHBoxLayout()
        thread_layout.addWidget(QLabel("Max Workers:"))
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 32)
        self.max_workers_spin.setValue(8)
        thread_layout.addWidget(self.max_workers_spin)
        load_layout_box.addLayout(thread_layout)
        
        load_group.setLayout(load_layout_box)
        layout.addWidget(load_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.load_btn = QPushButton("🚀 Load DM4 Sequence to Layer")
        self.load_btn.clicked.connect(self._load_data)
        self.load_btn.setEnabled(False)
        self.load_btn.setStyleSheet("padding: 8px; font-weight: bold;")
        layout.addWidget(self.load_btn)
        
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.last_dose_rate = None
    
    def _browse_folder(self):
        start_dir = self.settings.value("last_folder", str(Path.home()))
        
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing DM4 Files", start_dir
        )
        
        if folder:
            self.current_folder = folder
            self.folder_label.setText(f"Selected: {folder}")
            self.load_btn.setEnabled(True)
            self.calc_dose_btn.setEnabled(True)
            self.settings.setValue("last_folder", folder)
            
            self.dose_result_label.setText("Dose Rate: N/A")
            self.dose_info_label.setText("Ready to calculate.")
            self.last_dose_rate = None
    
    def _calculate_dose_rate(self):
        if not self.current_folder: return
        
        self.calc_dose_btn.setEnabled(False)
        self.dose_info_label.setText("⏳ Finding reference file and calculating...")
        self.dose_result_label.setText("Calculating...")
        
        self.calc_thread = DoseCalculationThread(self.current_folder)
        self.calc_thread.finished.connect(self._on_dose_calculated)
        self.calc_thread.error.connect(self._on_dose_error)
        self.calc_thread.start()

    def _on_dose_calculated(self, dose_rate, filename, debug_info):
        self.last_dose_rate = dose_rate
        self.calc_dose_btn.setEnabled(True)
        
        self._update_dose_display()
        
        # 显示详情，包含采集时间
        self.dose_info_label.setText(
            f"Ref File: {filename}\n"
            f"Date: {debug_info['acq_date']} {debug_info['acq_time']}\n"
            f"Time: {debug_info['exposure_s']}s, "
            f"Pixel: {debug_info['pixel_size_A']:.2f}Å ({debug_info['unit_raw']})\n"
            f"Mean Counts: {debug_info['mean_counts']:.1f}, "
            f"Factor: {debug_info['bright_scale']}"
        )

    def _update_dose_display(self):
        if self.last_dose_rate is None: return
        decimals = self.decimals_spin.value()
        fmt = f"{{:.{decimals}f}}"
        val_str = fmt.format(self.last_dose_rate)
        self.dose_result_label.setText(f"Dose Rate: {val_str} e⁻·Å⁻²·s⁻¹")

    def _on_dose_error(self, error_msg):
        self.calc_dose_btn.setEnabled(True)
        self.dose_result_label.setText("Error")
        self.dose_info_label.setText(f"❌ {error_msg}")
        if "dm4" in error_msg:
            QMessageBox.warning(self, "Missing Dependency", error_msg)

    def _load_data(self):
        if not self.current_folder: return
        
        self.load_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Loading...")
        
        bit_depth = int(self.bit_depth_combo.currentText())
        max_workers = self.max_workers_spin.value()
        
        self.loader_thread = LoaderThread(
            self.current_folder, bit_depth, max_workers
        )
        self.loader_thread.progress.connect(self._update_progress)
        self.loader_thread.finished.connect(self._on_load_finished)
        self.loader_thread.error.connect(self._on_load_error)
        self.loader_thread.start()
    
    def _update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
    
    def _on_load_finished(self, image_stack, metadata):
        layer_name = f"Original_{Path(metadata['source_folder']).name}"
        self.viewer.add_image(
            image_stack,
            name=layer_name,
            metadata=metadata,
            colormap='gray'
        )
        self.status_label.setText(
            f"✅ Loaded {metadata['num_frames']} frames\n"
            f"Shape: {metadata['shape']}\n"
            f"Dtype: {metadata['dtype']}"
        )
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)
    
    def _on_load_error(self, error_msg):
        self.status_label.setText(f"❌ Error: {error_msg}")
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)
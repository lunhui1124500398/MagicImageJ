"""
数据导入控件 - 增强版 (含智能数据归档)
功能：
1. DM4序列加载 (修复参数设置丢失问题)
2. 电子剂量率自动计算 (修复文件选择策略和日期解析)
3. 实验元数据录入 & 规范化命名
4. 智能归档：自动生成文件夹，复制数据，生成说明文档和参数Log。
   - [Fix] 修复 JSON 序列化报错 (Numpy float32 support)
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QFileDialog, QLabel, QSpinBox, QHBoxLayout,
                            QProgressBar, QComboBox, QGroupBox, QMessageBox,
                            QDoubleSpinBox, QScrollArea, QLineEdit, QTextEdit)
from qtpy.QtCore import Signal, QThread, QSettings, Qt
import numpy as np
from pathlib import Path
import os
import shutil
import json
import datetime

# 尝试导入 dm4 库
try:
    import dm4
    DM4_LIB_AVAILABLE = True
except ImportError:
    DM4_LIB_AVAILABLE = False

from core.dm4_reader import read_dm4_sequence

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class LoaderThread(QThread):
    """后台加载线程"""
    progress = Signal(int, int)
    finished = Signal(np.ndarray, dict)
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
                self.bit_depth,
                self.max_workers,
                callback
            )
            self.finished.emit(image_stack, metadata)
        except Exception as e:
            self.error.emit(str(e))

class DoseCalculationThread(QThread):
    """剂量率及元数据提取线程"""
    finished = Signal(float, str, dict)
    error = Signal(str)
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = Path(folder_path)

    def _decode_string(self, data):
        if isinstance(data, str): return data
        if hasattr(data, 'tobytes'): b = data.tobytes()
        elif isinstance(data, (bytes, bytearray)): b = data
        elif isinstance(data, np.ndarray): b = data.tobytes()
        else: return str(data)
        try: return b.decode('utf-16').rstrip('\x00')
        except:
            try: return b.decode('utf-8').rstrip('\x00')
            except: return str(data)

    def _parse_date(self, date_str):
        """将各种日期格式 (如 11/4/2025) 转换为 YYYYMMDD"""
        if not date_str: return datetime.datetime.now().strftime("%Y%m%d")
        
        formats = [
            "%m/%d/%Y", # 11/4/2025
            "%Y-%m-%d", # 2025-11-04
            "%d/%m/%Y", # 04/11/2025
            "%Y/%m/%d", # 2025/11/04
            "%m-%d-%Y"  # 11-04-2025
        ]
        
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(date_str, fmt)
                return dt.strftime("%Y%m%d")
            except ValueError:
                continue
        
        # Fallback: extract digits
        digits = ''.join(filter(str.isdigit, date_str))
        if len(digits) == 8: return digits
        return datetime.datetime.now().strftime("%Y%m%d")

    def run(self):
        if not DM4_LIB_AVAILABLE:
            self.error.emit("Library 'dm4' not found.")
            return
        try:
            files = sorted(list(self.folder_path.rglob("*.dm4")))
            if not files:
                self.error.emit("No .dm4 files found.")
                return
            
            # 智能文件选择策略
            mid = len(files) // 2
            start = max(0, mid - 5)
            end = min(len(files), mid + 5)
            candidates = files[start:end]
            if not candidates: candidates = files
            
            # 选择体积最大的文件作为代表
            candidate = max(candidates, key=lambda f: f.stat().st_size)
            
            with dm4.DM4File.open(str(candidate)) as dm4file:
                tags = dm4file.read_directory()
                image_list = tags.named_subdirs['ImageList']
                
                image_dir = None
                image_data_dir = None
                image_tags_dir = None
                
                for subdir in image_list.unnamed_subdirs:
                    if 'ImageData' in subdir.named_subdirs:
                        image_dir = subdir
                        image_data_dir = subdir.named_subdirs['ImageData']
                        image_tags_dir = subdir.named_subdirs.get('ImageTags')
                        break
                
                if not image_data_dir: raise ValueError("No ImageData found in DM4.")

                # --- Metadata Extraction ---
                exposure = 0.0
                acq_date_raw = ""
                mag = 0
                
                if image_tags_dir:
                    if 'DataBar' in image_tags_dir.named_subdirs:
                        db = image_tags_dir.named_subdirs['DataBar']
                        if 'Exposure Time (s)' in db.named_tags:
                            exposure = dm4file.read_tag_data(db.named_tags['Exposure Time (s)'])
                        if 'Acquisition Date' in db.named_tags:
                            acq_date_raw = self._decode_string(dm4file.read_tag_data(db.named_tags['Acquisition Date']))
                        if 'Magnification' in db.named_tags:
                             mag = dm4file.read_tag_data(db.named_tags['Magnification'])
                    
                    if mag == 0 and 'Microscope Info' in image_tags_dir.named_subdirs:
                        mi = image_tags_dir.named_subdirs['Microscope Info']
                        if 'Indicated Magnification' in mi.named_tags:
                            mag = dm4file.read_tag_data(mi.named_tags['Indicated Magnification'])
                    
                    if exposure == 0 and 'Acquisition' in image_tags_dir.named_subdirs:
                        acq = image_tags_dir.named_subdirs['Acquisition'].named_subdirs.get('Parameters', {}).named_subdirs.get('High Level', {})
                        if 'Exposure' in acq.named_tags:
                            exposure = dm4file.read_tag_data(acq.named_tags['Exposure'])

                acq_date = self._parse_date(acq_date_raw)

                # Pixel Size
                pixel_size = 1.0
                pixel_unit = "nm"
                calibrations = image_data_dir.named_subdirs.get('Calibrations')
                if calibrations and 'Dimension' in calibrations.named_subdirs:
                    dim_x = calibrations.named_subdirs['Dimension'].unnamed_subdirs[0]
                    if 'Scale' in dim_x.named_tags:
                        pixel_size = dm4file.read_tag_data(dim_x.named_tags['Scale'])
                    if 'Units' in dim_x.named_tags:
                        pixel_unit = self._decode_string(dm4file.read_tag_data(dim_x.named_tags['Units']))

                # Brightness Scale
                brightness = 1.0
                if calibrations and 'Brightness' in calibrations.named_subdirs:
                    br = calibrations.named_subdirs['Brightness']
                    if 'Scale' in br.named_tags:
                        brightness = dm4file.read_tag_data(br.named_tags['Scale'])

                # Data
                if 'Data' in image_data_dir.named_tags:
                    data = np.array(dm4file.read_tag_data(image_data_dir.named_tags['Data']))
                else: raise ValueError("No Image Data.")

                # Unit conversion to Angstrom
                pixel_A = pixel_size
                if 'nm' in pixel_unit: pixel_A *= 10.0
                elif 'um' in pixel_unit or 'µm' in pixel_unit: pixel_A *= 10000.0
                elif 'A' in pixel_unit or 'Å' in pixel_unit: pixel_A *= 1.0
                elif 'pm' in pixel_unit: pixel_A *= 0.01
                
                total_counts = np.sum(data, dtype=np.float64)
                total_electrons = total_counts * brightness
                area_A2 = data.size * (pixel_A**2)
                
                if exposure <= 0: exposure = 1.0
                
                dose_rate = total_electrons / (area_A2 * exposure)
                mean_int = np.mean(data)

                # Convert to native python types immediately to help JSON serialization later
                info = {
                    "file": str(candidate.name),
                    "date_raw": str(acq_date_raw),
                    "date_fmt": str(acq_date),
                    "exposure": float(exposure),
                    "pixel_A": float(pixel_A),
                    "pixel_unit_raw": str(pixel_unit),
                    "mag": float(mag),
                    "mean": float(mean_int)
                }
                self.finished.emit(float(dose_rate), candidate.name, info)

        except Exception as e:
            self.error.emit(str(e))

class ImportWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.settings = QSettings("NapariUser", "Importer")
        self._setup_ui()
        
        # State
        self.current_folder = None
        self.meta_cache = {} 
        self.archive_root = None

        # Restore last
        last = self.settings.value("last_folder", "")
        if last and os.path.isdir(last):
            self.current_folder = last
            self.folder_label.setText(last)
            self.load_btn.setEnabled(True)
            self.calc_dose_btn.setEnabled(True)

    def _setup_ui(self):
        main = QVBoxLayout()
        # [UI Update] Use minimal margins to save space
        main.setContentsMargins(2, 2, 2, 2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # [UI Update] Remove border to blend in
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        content = QWidget()
        layout = QVBoxLayout()
        # [UI Update] Tighter spacing between groups
        layout.setSpacing(8) 
        
        layout.addWidget(QLabel("<h3>📂 Import & Archive</h3>"))

        # --- 1. Data Source ---
        g_source = QGroupBox("1. Data Source")
        l_source = QVBoxLayout()
        # [UI Update] Compress layout inside groupbox
        l_source.setSpacing(4)
        l_source.setContentsMargins(8, 8, 8, 8) 
        
        h_brow = QHBoxLayout()
        btn_browse = QPushButton("📂 Browse Folder")
        btn_browse.clicked.connect(self._browse_folder)
        h_brow.addWidget(btn_browse)
        self.folder_label = QLabel("None")
        self.folder_label.setStyleSheet("color: gray; font-size: 11px;")
        l_source.addLayout(h_brow)
        l_source.addWidget(self.folder_label)
        g_source.setLayout(l_source)
        layout.addWidget(g_source)

        # --- 2. Metadata & Dose ---
        g_dose = QGroupBox("2. Scan Metadata")
        l_dose = QVBoxLayout()
        l_dose.setSpacing(4)
        l_dose.setContentsMargins(8, 8, 8, 8)
        
        h_calc = QHBoxLayout()
        self.calc_dose_btn = QPushButton("🧮 Scan & Calc Dose")
        self.calc_dose_btn.setEnabled(False)
        self.calc_dose_btn.clicked.connect(self._calc_dose)
        h_calc.addWidget(self.calc_dose_btn)
        self.dose_val_label = QLabel("Dose: N/A")
        self.dose_val_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        h_calc.addWidget(self.dose_val_label)
        l_dose.addLayout(h_calc)
        self.meta_info_label = QLabel("Select folder to extract date, mag, pixel size...")
        self.meta_info_label.setWordWrap(True)
        self.meta_info_label.setStyleSheet("font-size: 10px; color: gray;")
        l_dose.addWidget(self.meta_info_label)
        g_dose.setLayout(l_dose)
        layout.addWidget(g_dose)

        # --- 3. Experiment Info (For Naming) ---
        g_exp = QGroupBox("3. Archive Configuration")
        g_exp.setStyleSheet("QGroupBox { border: 1px solid #2196F3; margin-top: 6px; } QGroupBox::title { color: #2196F3; }")
        l_exp = QVBoxLayout()
        l_exp.setSpacing(4)
        l_exp.setContentsMargins(8, 12, 8, 8)
        
        # Row 1: Substance, Solvent, Dataset
        h1 = QHBoxLayout()
        self.substance_edit = QLineEdit(); self.substance_edit.setPlaceholderText("Sub (e.g. at)")
        self.solvent_edit = QLineEdit(); self.solvent_edit.setPlaceholderText("Solv (Default: Water)")
        self.dataset_edit = QLineEdit("ds1"); self.dataset_edit.setFixedWidth(50); self.dataset_edit.setPlaceholderText("ds#")
        h1.addWidget(QLabel("Sub:")); h1.addWidget(self.substance_edit)
        h1.addWidget(QLabel("Solv:")); h1.addWidget(self.solvent_edit)
        h1.addWidget(QLabel("ID:")); h1.addWidget(self.dataset_edit)
        l_exp.addLayout(h1)

        # Row 2: Aperture & Mag
        h2 = QHBoxLayout()
        self.aperture_combo = QComboBox()
        self.aperture_combo.addItems([str(i) for i in range(5)]) # 0-4
        self.mag_edit = QLineEdit("60K"); self.mag_edit.setPlaceholderText("Mag")
        h2.addWidget(QLabel("OL# (0-4):")); h2.addWidget(self.aperture_combo)
        h2.addWidget(QLabel("Mag:")); h2.addWidget(self.mag_edit)
        l_exp.addLayout(h2)

        # Row 3: Process Params (For Naming)
        h3 = QHBoxLayout()
        self.win_spin = QSpinBox(); self.win_spin.setValue(3); self.win_spin.setRange(1, 99)
        self.sigma_spin = QDoubleSpinBox(); self.sigma_spin.setValue(0.8); self.sigma_spin.setSingleStep(0.1)
        h3.addWidget(QLabel("Avg Win:")); h3.addWidget(self.win_spin)
        h3.addWidget(QLabel("Gaus σ:")); h3.addWidget(self.sigma_spin)
        l_exp.addLayout(h3)

        # Additional Info
        l_exp.addWidget(QLabel("Additional Info (Saved to txt):"))
        self.buffer_edit = QTextEdit()
        self.buffer_edit.setPlaceholderText("e.g. 50mM Tris, pH 7.5...")
        self.buffer_edit.setMaximumHeight(45) # Reduce height slightly
        l_exp.addWidget(self.buffer_edit)

        # Connect updates for preview
        for w in [self.substance_edit, self.solvent_edit, self.dataset_edit, self.mag_edit, 
                  self.aperture_combo, self.win_spin, self.sigma_spin]:
            if isinstance(w, (QLineEdit, QTextEdit)): w.textChanged.connect(self._update_preview)
            elif isinstance(w, (QComboBox)): w.currentTextChanged.connect(self._update_preview)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)): w.valueChanged.connect(self._update_preview)

        g_exp.setLayout(l_exp)
        layout.addWidget(g_exp)

        # --- 4. Create Archive ---
        g_arc = QGroupBox("4. Action")
        g_arc.setStyleSheet("QGroupBox { border: 1px solid #FF9800; margin-top: 6px; } QGroupBox::title { color: #FF9800; }")
        l_arc = QVBoxLayout()
        l_arc.setSpacing(6)
        l_arc.setContentsMargins(8, 12, 8, 8)
        
        l_arc.addWidget(QLabel("Preview Folder Name:"))
        self.preview_label = QLabel("...")
        self.preview_label.setWordWrap(True)
        # [UI Update] Improved font and style for readability
        self.preview_label.setStyleSheet("""
            font-family: "Segoe UI", sans-serif;
            font-size: 11px;
            color: #E0E0E0;
            background-color: #2D2D2D;
            padding: 8px;
            border: 1px solid #3E3E3E;
            border-radius: 4px;
        """)
        l_arc.addWidget(self.preview_label)

        self.create_archive_btn = QPushButton("📦 Create Archive Folder")
        self.create_archive_btn.clicked.connect(self._create_archive)
        self.create_archive_btn.setStyleSheet("background-color: #E65100; color: white; font-weight: bold; padding: 8px;")
        self.create_archive_btn.setEnabled(False)
        l_arc.addWidget(self.create_archive_btn)
        
        g_arc.setLayout(l_arc)
        layout.addWidget(g_arc)

        # --- 5. Loading ---
        g_load = QGroupBox("5. Load to Viewer")
        l_load = QVBoxLayout()
        l_load.setSpacing(4)
        l_load.setContentsMargins(8, 8, 8, 8)
        
        h_params = QHBoxLayout()
        h_params.addWidget(QLabel("Bit Depth:"))
        self.bit_depth_combo = QComboBox()
        self.bit_depth_combo.addItems(["8", "16", "32"])
        self.bit_depth_combo.setCurrentText("8")
        h_params.addWidget(self.bit_depth_combo)
        
        h_params.addWidget(QLabel("Workers:"))
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 32)
        self.max_workers_spin.setValue(8)
        h_params.addWidget(self.max_workers_spin)
        l_load.addLayout(h_params)

        self.load_btn = QPushButton("🚀 Load Images")
        self.load_btn.clicked.connect(self._load_data)
        self.load_btn.setEnabled(False)
        l_load.addWidget(self.load_btn)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        l_load.addWidget(self.progress)
        g_load.setLayout(l_load)
        layout.addWidget(g_load)

        self.status = QLabel("")
        layout.addWidget(self.status)

        content.setLayout(layout)
        scroll.setWidget(content)
        main.addWidget(scroll)
        self.setLayout(main)

    def _browse_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select Data Folder", self.settings.value("last_folder", ""))
        if f:
            self.current_folder = f
            self.settings.setValue("last_folder", f)
            self.folder_label.setText(f)
            self.load_btn.setEnabled(True)
            self.calc_dose_btn.setEnabled(True)
            self._update_preview()

    def _calc_dose(self):
        if not self.current_folder: return
        self.calc_dose_btn.setEnabled(False)
        self.status.setText("Scanning metadata...")
        self.thread_dose = DoseCalculationThread(self.current_folder)
        self.thread_dose.finished.connect(self._on_dose_done)
        self.thread_dose.error.connect(lambda e: (self.status.setText(e), self.calc_dose_btn.setEnabled(True)))
        self.thread_dose.start()

    def _on_dose_done(self, dose, fname, info):
        self.meta_cache = info
        self.meta_cache['dose'] = dose
        self.calc_dose_btn.setEnabled(True)
        self.dose_val_label.setText(f"{dose:.2f} e⁻/Å²/s")
        
        if info.get('mag', 0) > 0:
            m = info['mag']
            if m >= 1000: self.mag_edit.setText(f"{int(m/1000)}K")
            else: self.mag_edit.setText(str(int(m)))
            
        self.meta_info_label.setText(
            f"Ref: {fname}\n"
            f"Date: {info.get('date_fmt', 'N/A')}, Exp: {info.get('exposure', 0)}s\n"
            f"Pixel: {info.get('pixel_A', 0):.2f} Å"
        )
        self._update_preview()
        self.create_archive_btn.setEnabled(True)

    def _fmt_num(self, val):
        """Format number: 0.36 -> 0p36"""
        if val is None: return "0"
        s = f"{float(val):.4f}".rstrip('0').rstrip('.')
        return s.replace('.', 'p')

    def _generate_folder_name(self):
        date_str = self.meta_cache.get('date_fmt', datetime.datetime.now().strftime("%Y%m%d"))
        
        sub = self.substance_edit.text().strip() or "Sample"
        sol = self.solvent_edit.text().strip() or "Water"
        ds = self.dataset_edit.text().strip() or "ds1"
        
        win = self.win_spin.value()
        sigma = self._fmt_num(self.sigma_spin.value())
        
        dose = self._fmt_num(self.meta_cache.get('dose', 0))
        ol = self.aperture_combo.currentText()
        
        pix_nm = self.meta_cache.get('pixel_A', 0) / 10.0
        pix = self._fmt_num(pix_nm)
        
        exp = self._fmt_num(self.meta_cache.get('exposure', 0))
        mag = self.mag_edit.text().strip() or "0K"

        name = f"{date_str}_{sub}-{sol}-{ds}_average{win}to1_gaussian{sigma}-{dose}e-OL#{ol}-{pix}nm-{exp}s-{mag}"
        return name

    def _update_preview(self):
        name = self._generate_folder_name()
        self.preview_label.setText(name)

    def _create_archive(self):
        if not self.current_folder: return
        
        parent_dir = Path(self.current_folder).parent
        folder_name = self._generate_folder_name()
        archive_path = parent_dir / folder_name
        
        if archive_path.exists():
            if QMessageBox.warning(self, "Exists", f"Folder exists:\n{folder_name}\nOverwrite?", QMessageBox.Yes|QMessageBox.No) == QMessageBox.No: return
        
        try:
            self.status.setText("Archiving...")
            archive_path.mkdir(parents=True, exist_ok=True)
            
            raw_dest = archive_path / "Original_Dataset"
            if not raw_dest.exists():
                shutil.copytree(self.current_folder, raw_dest, dirs_exist_ok=True)
            
            with open(archive_path / "readme.txt", 'w', encoding='utf-8') as f:
                f.write(f"Archive: {folder_name}\n")
                f.write(f"Created: {datetime.datetime.now()}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Substance: {self.substance_edit.text()}\n")
                f.write(f"Solvent: {self.solvent_edit.text() or 'Water'}\n")
                f.write(f"Dataset: {self.dataset_edit.text()}\n")
                f.write(f"Buffer Info: {self.buffer_edit.toPlainText()}\n")
                f.write("-" * 30 + "\n")
                f.write("Original DM4 Metadata:\n")
                for k, v in self.meta_cache.items():
                    f.write(f"{k}: {v}\n")

            params = {
                "folder_name": folder_name,
                "import_meta": self.meta_cache,
                "planned_settings": {
                    "rolling_avg_window": self.win_spin.value(),
                    "gaussian_sigma": self.sigma_spin.value()
                }
            }
            # === Fix: Use NumpyEncoder ===
            with open(archive_path / "processing_log.json", 'w') as f:
                json.dump(params, f, indent=2, cls=NumpyEncoder)

            self.archive_root = str(archive_path)
            QSettings("NapariUser", "Global").setValue("archive_path", self.archive_root)
            
            self.status.setText(f"✅ Archived: {folder_name}")
            QMessageBox.information(self, "Success", f"Archive created!\n{archive_path}\n\nOther widgets will now auto-save to this folder.")
            
        except Exception as e:
            self.status.setText(f"Error: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def _load_data(self):
        if not self.current_folder: return
        self.load_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status.setText("Loading...")
        
        bit = int(self.bit_depth_combo.currentText())
        workers = self.max_workers_spin.value()
        
        self.thread_load = LoaderThread(self.current_folder, bit, workers)
        self.thread_load.progress.connect(lambda c, t: (self.progress.setMaximum(t), self.progress.setValue(c)))
        self.thread_load.finished.connect(self._on_loaded)
        self.thread_load.error.connect(lambda e: (self.status.setText(f"Error: {e}"), self.load_btn.setEnabled(True)))
        self.thread_load.start()

    def _on_loaded(self, stack, meta):
        self.progress.setVisible(False)
        self.load_btn.setEnabled(True)
        name = f"Original_{Path(self.current_folder).name}"
        self.viewer.add_image(stack, name=name, metadata=meta, colormap='gray')
        self.status.setText(f"Loaded {len(stack)} frames.")
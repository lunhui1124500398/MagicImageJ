"""
数据导入控件 - 增强版 (含智能数据归档 & 进度优化)
修复日志:
- [Critical Fix] ArchiveThread: 真正实装了 Windows 长路径支持 (添加 \\?\ 前缀并强制反斜杠)。
- [Critical Fix] ArchiveThread: 实装递归检测 (Recursion Guard)。
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

try:
    import dm4
    DM4_LIB_AVAILABLE = True
except ImportError:
    DM4_LIB_AVAILABLE = False

from core.dm4_reader import read_dm4_sequence

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)): return int(obj)
        elif isinstance(obj, (np.floating, float)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

class ArchiveThread(QThread):
    """归档复制线程"""
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, src, dst):
        super().__init__()
        self.src = src
        self.dst = dst
        
    def run(self):
        try:
            # 1. 路径标准化 (Resolve absolute paths)
            src_abs = os.path.abspath(self.src)
            dst_abs = os.path.abspath(self.dst)

            # 2. 递归检查 (Recursion Guard)
            # 如果 dst 是 src 的子目录，copytree 会死循环
            if dst_abs.startswith(src_abs):
                raise ValueError(f"Recursion Error: Destination is inside Source.\nSrc: {src_abs}\nDst: {dst_abs}")

            # 3. Windows 长路径支持 (Long Path Support)
            if os.name == 'nt':
                # 必须强制使用反斜杠，混合斜杠会导致 \\?\ 失效
                src_abs = src_abs.replace('/', '\\')
                dst_abs = dst_abs.replace('/', '\\')
                
                if not src_abs.startswith('\\\\?\\'): src_abs = '\\\\?\\' + src_abs
                if not dst_abs.startswith('\\\\?\\'): dst_abs = '\\\\?\\' + dst_abs
            
            # 4. 执行复制
            shutil.copytree(src_abs, dst_abs, dirs_exist_ok=True)
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))

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
                self.folder_path, self.bit_depth, self.max_workers, callback
            )
            self.finished.emit(image_stack, metadata)
        except Exception as e:
            self.error.emit(str(e))

class DoseCalculationThread(QThread):
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
        if not date_str: return datetime.datetime.now().strftime("%Y%m%d")
        formats = ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y"]
        for fmt in formats:
            try: return datetime.datetime.strptime(date_str, fmt).strftime("%Y%m%d")
            except ValueError: continue
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
            mid = len(files) // 2
            start = max(0, mid - 5); end = min(len(files), mid + 5)
            candidates = files[start:end] if files[start:end] else files
            candidate = max(candidates, key=lambda f: f.stat().st_size)
            
            with dm4.DM4File.open(str(candidate)) as dm4file:
                tags = dm4file.read_directory()
                image_list = tags.named_subdirs['ImageList']
                image_data_dir = None
                image_tags_dir = None
                for subdir in image_list.unnamed_subdirs:
                    if 'ImageData' in subdir.named_subdirs:
                        image_data_dir = subdir.named_subdirs['ImageData']
                        image_tags_dir = subdir.named_subdirs.get('ImageTags')
                        break
                if not image_data_dir: raise ValueError("No ImageData found.")

                exposure = 0.0; acq_date_raw = ""; mag = 0
                if image_tags_dir:
                    if 'DataBar' in image_tags_dir.named_subdirs:
                        db = image_tags_dir.named_subdirs['DataBar']
                        if 'Exposure Time (s)' in db.named_tags: exposure = dm4file.read_tag_data(db.named_tags['Exposure Time (s)'])
                        if 'Acquisition Date' in db.named_tags: acq_date_raw = self._decode_string(dm4file.read_tag_data(db.named_tags['Acquisition Date']))
                        if 'Magnification' in db.named_tags: mag = dm4file.read_tag_data(db.named_tags['Magnification'])
                    if mag == 0 and 'Microscope Info' in image_tags_dir.named_subdirs:
                        mi = image_tags_dir.named_subdirs['Microscope Info']
                        if 'Indicated Magnification' in mi.named_tags: mag = dm4file.read_tag_data(mi.named_tags['Indicated Magnification'])
                    if exposure == 0 and 'Acquisition' in image_tags_dir.named_subdirs:
                        acq = image_tags_dir.named_subdirs['Acquisition'].named_subdirs.get('Parameters', {}).named_subdirs.get('High Level', {})
                        if 'Exposure' in acq.named_tags: exposure = dm4file.read_tag_data(acq.named_tags['Exposure'])

                acq_date = self._parse_date(acq_date_raw)
                pixel_size = 1.0; pixel_unit = "nm"; brightness = 1.0
                
                calibrations = image_data_dir.named_subdirs.get('Calibrations')
                if calibrations and 'Dimension' in calibrations.named_subdirs:
                    dim_x = calibrations.named_subdirs['Dimension'].unnamed_subdirs[0]
                    if 'Scale' in dim_x.named_tags: pixel_size = dm4file.read_tag_data(dim_x.named_tags['Scale'])
                    if 'Units' in dim_x.named_tags: pixel_unit = self._decode_string(dm4file.read_tag_data(dim_x.named_tags['Units']))
                if calibrations and 'Brightness' in calibrations.named_subdirs:
                    br = calibrations.named_subdirs['Brightness']
                    if 'Scale' in br.named_tags: brightness = dm4file.read_tag_data(br.named_tags['Scale'])

                data = np.array(dm4file.read_tag_data(image_data_dir.named_tags['Data']))
                
                pixel_A = pixel_size
                if 'nm' in pixel_unit: pixel_A *= 10.0
                elif 'um' in pixel_unit: pixel_A *= 10000.0
                elif 'pm' in pixel_unit: pixel_A *= 0.01
                
                total_electrons = np.sum(data, dtype=np.float64) * brightness
                area_A2 = data.size * (pixel_A**2)
                if exposure <= 0: exposure = 1.0
                dose_rate = total_electrons / (area_A2 * exposure)
                mean_int = np.mean(data)

                info = {
                    "file": str(candidate.name), "date_raw": str(acq_date_raw), "date_fmt": str(acq_date),
                    "exposure": float(exposure), "pixel_A": float(pixel_A), "pixel_unit_raw": str(pixel_unit),
                    "mag": float(mag), "mean": float(mean_int)
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
        self.current_folder = None
        self.meta_cache = {} 
        self.archive_root = None
        self.archive_thread = None
        
        last = self.settings.value("last_folder", "")
        if last and os.path.isdir(last):
            self.current_folder = last
            self.folder_label.setText(last)
            self.load_btn.setEnabled(True)
            self.calc_dose_btn.setEnabled(True)

    def _setup_ui(self):
        main = QVBoxLayout(); main.setContentsMargins(2, 2, 2, 2)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        content = QWidget(); layout = QVBoxLayout(); layout.setSpacing(8)
        
        layout.addWidget(QLabel("<h3>📂 Import & Archive</h3>"))

        g_source = QGroupBox("1. Data Source"); l_source = QVBoxLayout(); l_source.setSpacing(4); l_source.setContentsMargins(8, 8, 8, 8)
        h_brow = QHBoxLayout(); btn_browse = QPushButton("📂 Browse Folder"); btn_browse.clicked.connect(self._browse_folder)
        h_brow.addWidget(btn_browse); self.folder_label = QLabel("None"); self.folder_label.setStyleSheet("color: gray; font-size: 11px;")
        l_source.addLayout(h_brow); l_source.addWidget(self.folder_label); g_source.setLayout(l_source); layout.addWidget(g_source)

        g_dose = QGroupBox("2. Scan Metadata"); l_dose = QVBoxLayout(); l_dose.setSpacing(4); l_dose.setContentsMargins(8, 8, 8, 8)
        h_calc = QHBoxLayout(); self.calc_dose_btn = QPushButton("🧮 Scan & Calc Dose"); self.calc_dose_btn.setEnabled(False); self.calc_dose_btn.clicked.connect(self._calc_dose)
        h_calc.addWidget(self.calc_dose_btn); self.dose_val_label = QLabel("Dose: N/A"); self.dose_val_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        h_calc.addWidget(self.dose_val_label); l_dose.addLayout(h_calc)
        self.meta_info_label = QLabel("Select folder to extract date, mag, pixel size..."); self.meta_info_label.setWordWrap(True); self.meta_info_label.setStyleSheet("font-size: 10px; color: gray;")
        l_dose.addWidget(self.meta_info_label); g_dose.setLayout(l_dose); layout.addWidget(g_dose)

        g_exp = QGroupBox("3. Archive Configuration"); g_exp.setStyleSheet("QGroupBox { border: 1px solid #2196F3; margin-top: 6px; } QGroupBox::title { color: #2196F3; }"); l_exp = QVBoxLayout(); l_exp.setSpacing(4); l_exp.setContentsMargins(8, 12, 8, 8)
        h1 = QHBoxLayout(); self.substance_edit = QComboBox(); self.substance_edit.setEditable(True); self.substance_edit.setPlaceholderText("Sub (e.g. CRY2)");self.substance_edit.setMinimumWidth(100);
        self._load_substance_history()
        self.solvent_edit = QLineEdit(); self.solvent_edit.setPlaceholderText("Solv (Default: Water)")
        self.dataset_edit = QLineEdit("ds1"); self.dataset_edit.setFixedWidth(50); self.dataset_edit.setPlaceholderText("ds#")
        h1.addWidget(QLabel("Sub:")); h1.addWidget(self.substance_edit); h1.addWidget(QLabel("Solv:")); h1.addWidget(self.solvent_edit); h1.addWidget(QLabel("ID:")); h1.addWidget(self.dataset_edit); l_exp.addLayout(h1)
        
        h2 = QHBoxLayout(); self.aperture_combo = QComboBox(); self.aperture_combo.addItems([str(i) for i in range(5)])
        self.mag_edit = QLineEdit("60K"); self.mag_edit.setPlaceholderText("Mag")
        h2.addWidget(QLabel("OL# (0-4):")); h2.addWidget(self.aperture_combo); h2.addWidget(QLabel("Mag:")); h2.addWidget(self.mag_edit); l_exp.addLayout(h2)
        
        h3 = QHBoxLayout(); self.win_spin = QSpinBox(); self.win_spin.setValue(3); self.win_spin.setRange(1, 99)
        self.sigma_spin = QDoubleSpinBox(); self.sigma_spin.setValue(0.8); self.sigma_spin.setSingleStep(0.1)
        h3.addWidget(QLabel("Avg Win:")); h3.addWidget(self.win_spin); h3.addWidget(QLabel("Gaus σ:")); h3.addWidget(self.sigma_spin); l_exp.addLayout(h3)
        
        l_exp.addWidget(QLabel("Additional Info (Saved to txt):")); self.buffer_edit = QTextEdit(); self.buffer_edit.setPlaceholderText("e.g. 50mM Tris, pH 7.5..."); self.buffer_edit.setMaximumHeight(45); l_exp.addWidget(self.buffer_edit)
        
        for w in [self.substance_edit, self.solvent_edit, self.dataset_edit, self.mag_edit, self.aperture_combo, self.win_spin, self.sigma_spin]:
            if isinstance(w, (QLineEdit, QTextEdit)): w.textChanged.connect(self._update_preview)
            elif isinstance(w, (QComboBox)): w.currentTextChanged.connect(self._update_preview)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)): w.valueChanged.connect(self._update_preview)
        g_exp.setLayout(l_exp); layout.addWidget(g_exp)

        g_arc = QGroupBox("4. Action"); g_arc.setStyleSheet("QGroupBox { border: 1px solid #FF9800; margin-top: 6px; } QGroupBox::title { color: #FF9800; }"); l_arc = QVBoxLayout(); l_arc.setSpacing(6); l_arc.setContentsMargins(8, 12, 8, 8)
        l_arc.addWidget(QLabel("Preview Folder Name:")); self.preview_label = QLabel("..."); self.preview_label.setWordWrap(True); self.preview_label.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-size: 11px; color: #E0E0E0; background-color: #2D2D2D; padding: 8px; border: 1px solid #3E3E3E; border-radius: 4px;")
        l_arc.addWidget(self.preview_label)
        self.create_archive_btn = QPushButton("📦 Create Archive Folder"); self.create_archive_btn.clicked.connect(self._create_archive); self.create_archive_btn.setStyleSheet("background-color: #E65100; color: white; font-weight: bold; padding: 8px;"); self.create_archive_btn.setEnabled(False)
        l_arc.addWidget(self.create_archive_btn)
        self.archive_progress = QProgressBar(); self.archive_progress.setVisible(False); self.archive_progress.setRange(0, 0) # Indeterminate
        l_arc.addWidget(self.archive_progress)
        g_arc.setLayout(l_arc); layout.addWidget(g_arc)

        g_load = QGroupBox("5. Load to Viewer"); l_load = QVBoxLayout(); l_load.setSpacing(4); l_load.setContentsMargins(8, 8, 8, 8)
        h_params = QHBoxLayout(); h_params.addWidget(QLabel("Bit Depth:")); self.bit_depth_combo = QComboBox(); self.bit_depth_combo.addItems(["8", "16", "32"]); self.bit_depth_combo.setCurrentText("8"); h_params.addWidget(self.bit_depth_combo)
        h_params.addWidget(QLabel("Workers:")); self.max_workers_spin = QSpinBox(); self.max_workers_spin.setRange(1, 32); self.max_workers_spin.setValue(8); h_params.addWidget(self.max_workers_spin); l_load.addLayout(h_params)
        self.load_btn = QPushButton("🚀 Load Images"); self.load_btn.clicked.connect(self._load_data); self.load_btn.setEnabled(False); l_load.addWidget(self.load_btn)
        self.progress = QProgressBar(); self.progress.setVisible(False); l_load.addWidget(self.progress)
        g_load.setLayout(l_load); layout.addWidget(g_load)

        self.status = QLabel(""); layout.addWidget(self.status)
        content.setLayout(layout); scroll.setWidget(content); main.addWidget(scroll); self.setLayout(main)

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
        self.meta_cache = info; self.meta_cache['dose'] = dose
        self.calc_dose_btn.setEnabled(True)
        self.dose_val_label.setText(f"{dose:.2f} e⁻/Å²/s")
        if info.get('mag', 0) > 0:
            m = info['mag']
            self.mag_edit.setText(f"{int(m/1000)}K" if m >= 1000 else str(int(m)))
        self.meta_info_label.setText(f"Ref: {fname}\nDate: {info.get('date_fmt', 'N/A')}, Exp: {info.get('exposure', 0)}s\nPixel: {info.get('pixel_A', 0):.2f} Å")
        self._update_preview()
        self.create_archive_btn.setEnabled(True)

    def _fmt_num(self, val):
        if val is None: return "0"
        s = f"{float(val):.4f}".rstrip('0').rstrip('.')
        return s.replace('.', 'p')

    def _generate_folder_name(self):
        date_str = self.meta_cache.get('date_fmt', datetime.datetime.now().strftime("%Y%m%d"))
        sub = self.substance_edit.currentText().strip() or "Sample"
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
        return f"{date_str}_{sub}-{sol}-{ds}_average{win}to1_gaussian{sigma}-{dose}e-OL#{ol}-{pix}nm-{exp}s-{mag}"

    def _update_preview(self):
        name = self._generate_folder_name()
        self.preview_label.setText(name)

    def _load_substance_history(self):
        history = self.settings.value("substance_history", [])
        if history:
            self.substance_edit.addItems(history)
            self.substance_edit.setCurrentIndex(0) # 默认选最近的一个

    def _save_substance_history(self):
        """在归档时调用此方法"""
        current_text = self.substance_edit.currentText().strip()
        if not current_text: return
        
        history = self.settings.value("substance_history", [])
        # 移除重复项，并将当前项插到最前
        if current_text in history:
            history.remove(current_text)
        history.insert(0, current_text)
        
        # 只保留最近5个
        history = history[:5]
        self.settings.setValue("substance_history", history)
        
        # 刷新UI
        self.substance_edit.blockSignals(True)
        self.substance_edit.clear()
        self.substance_edit.addItems(history)
        self.substance_edit.setCurrentText(current_text)
        self.substance_edit.blockSignals(False)

    def _create_archive(self):
        if not self.current_folder: return
        self._save_substance_history()
        parent_dir = Path(self.current_folder).parent
        folder_name = self._generate_folder_name()
        archive_path = parent_dir / folder_name
        
        if archive_path.exists():
            if QMessageBox.warning(self, "Exists", f"Folder exists:\n{folder_name}\nOverwrite?", QMessageBox.Yes|QMessageBox.No) == QMessageBox.No: return
        
        self.create_archive_btn.setEnabled(False)
        self.status.setText("Archiving... Do not close.")
        self.archive_progress.setVisible(True)
        
        # Create structure
        archive_path.mkdir(parents=True, exist_ok=True)
        self.archive_root = str(archive_path)
        QSettings("NapariUser", "Global").setValue("archive_path", self.archive_root)
        
        # Write info files
        with open(archive_path / "readme.txt", 'w', encoding='utf-8') as f:
            f.write(f"Archive: {folder_name}\nCreated: {datetime.datetime.now()}\n" + "-"*30 + "\n")
            f.write(f"Substance: {self.substance_edit.text()}\nSolvent: {self.solvent_edit.text()}\nDataset: {self.dataset_edit.text()}\n")
            f.write(f"Buffer Info: {self.buffer_edit.toPlainText()}\n" + "-"*30 + "\n")
            f.write("Original DM4 Metadata:\n")
            for k, v in self.meta_cache.items(): f.write(f"{k}: {v}\n")
            
        params = {"folder_name": folder_name, "import_meta": self.meta_cache, "planned_settings": {"rolling_avg_window": self.win_spin.value(), "gaussian_sigma": self.sigma_spin.value()}}
        with open(archive_path / "processing_log.json", 'w') as f: json.dump(params, f, indent=2, cls=NumpyEncoder)

        # Start Copy Thread
        # 传递字符串路径给线程，线程内部处理 \\?\ 转换
        raw_dest = archive_path / "Original_Dataset"
        
        self.archive_thread = ArchiveThread(str(self.current_folder), str(raw_dest))
        self.archive_thread.finished.connect(self._on_archive_done)
        self.archive_thread.error.connect(self._on_archive_error)
        self.archive_thread.start()

    def _on_archive_done(self):
        self.archive_progress.setVisible(False)
        self.create_archive_btn.setEnabled(True)
        self.status.setText(f"✅ Archived: {Path(self.archive_root).name}")
        QMessageBox.information(self, "Success", f"Archive created!\n{self.archive_root}\n\nWidgets linked to this archive.")

    def _on_archive_error(self, err):
        self.archive_progress.setVisible(False)
        self.create_archive_btn.setEnabled(True)
        self.status.setText(f"❌ Archive Error: {err}")
        QMessageBox.critical(self, "Error", str(err))

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
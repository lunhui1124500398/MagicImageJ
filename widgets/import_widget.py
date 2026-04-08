"""
数据导入控件 - 增强版 (含智能数据归档 & 进度优化)
修复日志:
- [Fix] 补全缺失的 _get_dir_size 方法。
- [Req 0] 自动正则匹配 Dataset ID。
- [Req 1] 原始数据文件夹按 {Date}_{Sub}_{OriginalDatasetN} 命名。
- [Req 2] >30GB 自动切换为移动模式，并弹窗提醒。
- [Info] 归档时保存 current_date 和 current_dataset_id 到全局配置。
- [Req New] 归档完成弹窗包含文件大小和模式信息，且可配置关闭。
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QFileDialog, QLabel, QSpinBox, QHBoxLayout,
                            QProgressBar, QComboBox, QGroupBox, QMessageBox,
                            QDoubleSpinBox, QScrollArea, QLineEdit, QTextEdit, QSizePolicy,
                            QCheckBox) # Added QCheckBox
from widgets.settings_widget import GlobalConfig
from qtpy.QtCore import Signal, QThread, QSettings, Qt
import numpy as np
from pathlib import Path
import os
import shutil
import json
import datetime
import gc
import re
import platform
import ctypes
from widgets.settings_widget import tr
from utils.session_logger import get_logger
from utils.utils import elide_text, natural_sort_key
from utils.ui_utils import setup_safe_scroll_all

# 尝试导入 psutil 获取更准确的内存信息，如果没有则使用 ctypes (Windows) 或 os (Linux)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import dm4
    DM4_LIB_AVAILABLE = True
except ImportError:
    DM4_LIB_AVAILABLE = False

from core.dm4_reader import read_dm4_sequence
from core.dm3_reader import read_dm3_sequence, read_dm3_metadata


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)): return int(obj)
        elif isinstance(obj, (np.floating, float)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

class ArchiveThread(QThread):
    """归档复制/移动线程"""
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, src, dst, move_mode=False):
        super().__init__()
        self.src = src
        self.dst = dst
        self.move_mode = move_mode 
        
    def run(self):
        try:
            # 1. 路径标准化
            src_abs = os.path.abspath(self.src)
            dst_abs = os.path.abspath(self.dst)

            # 2. 递归检查
            if dst_abs.startswith(src_abs):
                raise ValueError(f"Recursion Error: Destination is inside Source.\nSrc: {src_abs}\nDst: {dst_abs}")

            # 3. Windows 长路径支持
            if os.name == 'nt':
                src_abs = src_abs.replace('/', '\\')
                dst_abs = dst_abs.replace('/', '\\')
                if not src_abs.startswith('\\\\?\\'): src_abs = '\\\\?\\' + src_abs
                if not dst_abs.startswith('\\\\?\\'): dst_abs = '\\\\?\\' + dst_abs
            
            # 4. 执行复制或移动
            if self.move_mode:
                shutil.move(src_abs, dst_abs)
            else:
                shutil.copytree(src_abs, dst_abs, dirs_exist_ok=True)
                
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))

class LoaderThread(QThread):
    """后台加载线程"""
    progress = Signal(int, int)
    finished = Signal(np.ndarray, dict)
    error = Signal(str)
    def __init__(self, folder_path, bit_depth, max_workers, frame_indices=None):
        super().__init__()
        self.folder_path = folder_path
        self.bit_depth = bit_depth
        self.max_workers = max_workers
        self.frame_indices = frame_indices
    def run(self):
        try:
            def callback(current, total):
                self.progress.emit(current, total)
            
            # Auto-detect format
            has_dm4 = list(Path(self.folder_path).glob('**/*.dm4'))
            has_dm3 = list(Path(self.folder_path).glob('**/*.dm3'))
            
            if has_dm4:
                image_stack, metadata = read_dm4_sequence(
                    self.folder_path, self.bit_depth, self.max_workers, callback, self.frame_indices
                )
            elif has_dm3:
                image_stack, metadata = read_dm3_sequence(
                    self.folder_path, self.bit_depth, self.max_workers, callback, self.frame_indices
                )
            else:
                 raise ValueError("No DM4 or DM3 files found in folder.")
                 
            self.finished.emit(image_stack, metadata)
        except Exception as e:
            self.error.emit(str(e))


class DoseCalculationThread(QThread):
    """剂量计算线程"""
    finished = Signal(float, str, dict)
    error = Signal(str)
    
    def __init__(self, folder_path, frame_idx=-1):
        super().__init__()
        self.folder_path = Path(folder_path)
        self.frame_idx = frame_idx 

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
        # Allow DM3 even if DM4 lib missing (if ncempy is there)
        # if not DM4_LIB_AVAILABLE: ... (Skip check or make it smarter)
        
        try:
            p = self.folder_path
            # Mixed list? Usually one type. Prioritize user selection or mixed.
            files_dm4 = sorted(list(p.rglob("*.dm4")), key=natural_sort_key)
            files_dm3 = sorted(list(p.rglob("*.dm3")), key=natural_sort_key)
            files = files_dm4 + files_dm3
            
            if not files:
                self.error.emit("No .dm4 or .dm3 files found.")
                return
            
            total_files = len(files)
            target_idx = 0
            if self.frame_idx < 0:
                mid = total_files // 2
                start = max(0, mid - 5)
                end = min(total_files, mid + 5)
                candidates_list = files[start:end] if files[start:end] else files
                candidate = max(candidates_list, key=lambda f: f.stat().st_size)
                # Find index in full list
                if candidate in files:
                    target_idx = files.index(candidate)
            else:
                target_idx = max(0, min(self.frame_idx, total_files - 1))
                candidate = files[target_idx]
            
            candidate = files[target_idx]
            ext = candidate.suffix.lower()
            

            # === Metadata Caching Strategy ===
            # Key: (width, height, binning) -> {pixel_size, unit, brightness, mag, exposure}
            # We use this to fill in missing info for frames in the same sequence
            if not hasattr(self, '_seq_meta_cache'):
                self._seq_meta_cache = {}

            # Parse DM4 Metadata (Common Logic extraction)
            def parse_dm4_meta(dm4_path):
                with dm4.DM4File.open(str(dm4_path)) as f:
                    tags = f.read_directory()
                    image_list = tags.named_subdirs['ImageList']
                    image_data_dir = None
                    image_tags_dir = None
                    
                    # Logic to find correct ImageData (avoid thumbnail if possible, though mostly index 0 or 1)
                    # For simple files, usually index 0 is valid or index 1 is valid.
                    # We check for 'Dimensions' to be sure
                    target_subdir = None
                    for subdir in image_list.unnamed_subdirs:
                        if 'ImageData' in subdir.named_subdirs:
                            # Verify dims?
                            target_subdir = subdir
                            # If we see 'Calibrations' with non-1.0 scale, this is a winner.
                            # But we might just take what we find.
                            # Prefer the one with actual data?
                            # Often index 1 is the main image if index 0 is thumbnail.
                            pass
                            
                    # Re-iterate to pick best
                    # Heuristic: Pick the one with largest Data size matching file size roughly?
                    # Or just use the last one? Or ncempy logic?
                    # Current existing logic loop:
                    for subdir in image_list.unnamed_subdirs:
                        if 'ImageData' in subdir.named_subdirs:
                            image_data_dir = subdir.named_subdirs['ImageData']
                            image_tags_dir = subdir.named_subdirs.get('ImageTags')
                            # If this looks like a thumbnail (small dims), maybe skip?
                            # For now, keep existing behavior but maybe improve later.
                            break
                            
                    if not image_data_dir: raise ValueError("No ImageData found.")

                    # Read Basic Info
                    dims = [0, 0]
                    if 'Dimensions' in image_data_dir.named_subdirs:
                        d_tag = image_data_dir.named_subdirs['Dimensions']
                        if len(d_tag.unnamed_tags) >= 2:
                            dims[0] = f.read_tag_data(d_tag.unnamed_tags[0])
                            dims[1] = f.read_tag_data(d_tag.unnamed_tags[1])

                    exposure = 0.0; acq_date_raw = ""; mag = 0; binning = 1
                    
                    if image_tags_dir:
                        if 'DataBar' in image_tags_dir.named_subdirs:
                            db = image_tags_dir.named_subdirs['DataBar']
                            if 'Exposure Time (s)' in db.named_tags: exposure = f.read_tag_data(db.named_tags['Exposure Time (s)'])
                            if 'Acquisition Date' in db.named_tags: acq_date_raw = self._decode_string(f.read_tag_data(db.named_tags['Acquisition Date']))
                            if 'Magnification' in db.named_tags: mag = f.read_tag_data(db.named_tags['Magnification'])
                            if 'Binning' in db.named_tags: binning = f.read_tag_data(db.named_tags['Binning'])
                            
                        if mag == 0 and 'Microscope Info' in image_tags_dir.named_subdirs:
                            mi = image_tags_dir.named_subdirs['Microscope Info']
                            # Indicated Mag (60k) vs Actual Mag (83k). Prefer Indicated for UI.
                            if 'Indicated Magnification' in mi.named_tags: 
                                mag = f.read_tag_data(mi.named_tags['Indicated Magnification'])
                            if mag == 0 and 'Magnification' in mi.named_tags:
                                mag = f.read_tag_data(mi.named_tags['Magnification'])

                        if exposure == 0 and 'Acquisition' in image_tags_dir.named_subdirs:
                            acq = image_tags_dir.named_subdirs['Acquisition'].named_subdirs.get('Parameters', {}).named_subdirs.get('High Level', {})
                            if 'Exposure' in acq.named_tags: exposure = f.read_tag_data(acq.named_tags['Exposure'])

                    acq_date = self._parse_date(acq_date_raw)
                    pixel_size = 1.0; pixel_unit = "nm"; brightness = 1.0
                    
                    calibrations = image_data_dir.named_subdirs.get('Calibrations')
                    if calibrations and 'Dimension' in calibrations.named_subdirs:
                        dim_x = calibrations.named_subdirs['Dimension'].unnamed_subdirs[0]
                        if 'Scale' in dim_x.named_tags: pixel_size = f.read_tag_data(dim_x.named_tags['Scale'])
                        if 'Units' in dim_x.named_tags: pixel_unit = self._decode_string(f.read_tag_data(dim_x.named_tags['Units']))
                    if calibrations and 'Brightness' in calibrations.named_subdirs:
                        br = calibrations.named_subdirs['Brightness']
                        if 'Scale' in br.named_tags: brightness = f.read_tag_data(br.named_tags['Scale'])

                    data = np.array(f.read_tag_data(image_data_dir.named_tags['Data']))
                    
                    return {
                        'data': data,
                        'exposure': float(exposure),
                        'pixel_size': float(pixel_size),
                        'unit': str(pixel_unit),
                        'brightness': float(brightness),
                        'mag': float(mag),
                        'date_raw': str(acq_date_raw),
                        'date': str(acq_date),
                        'dims': tuple(dims),
                        'binning': int(binning)
                    }

            # --- Execution ---
            if ext == '.dm3':
                # DM3 path (Use core reader)
                from core.dm3_reader import read_dm3_metadata
                data, meta = read_dm3_metadata(str(candidate))
                if data is None: raise ValueError("Failed to read DM3")
                
                # Normalize keys to match internal parsed struct
                res = {
                    'data': data,
                    'exposure': meta.get('exposure', 0.1), # Avoid 0 div
                    'pixel_size': meta.get('pixel_size', 1.0),
                    'unit': meta.get('unit', 'nm'),
                    'brightness': meta.get('brightness_scale', 1.0),
                    'mag': meta.get('magnification', 0),
                    'date_raw': meta.get('raw_date', ''),
                    'date': meta.get('date', ''),
                    'dims': data.shape,
                    'binning': 1 # Unknown for DM3 usually
                }
            else:
                # DM4 Path
                if not DM4_LIB_AVAILABLE:
                    self.error.emit("Library 'dm4' not found.")
                    return
                res = parse_dm4_meta(candidate)

            # Metadata Inheritance / Caching Logic
            cache_key = (res['dims'], res['binning']) 
            
            # Check if current is "Valid" (has calibration)
            is_valid = (res['pixel_size'] != 1.0) and (res['brightness'] != 1.0)
            
            if is_valid:
                # Update Cache
                self._seq_meta_cache[cache_key] = {
                    'pixel_size': res['pixel_size'],
                    'unit': res['unit'],
                    'brightness': res['brightness'],
                    'mag': res['mag'],
                    'exposure': res['exposure'] if res['exposure'] > 0 else 1.0
                }
            elif cache_key in self._seq_meta_cache:
                # Inherit from Cache
                cached = self._seq_meta_cache[cache_key]
                print(f"Inheriting metadata for {candidate.name} from cache.")
                if res['pixel_size'] == 1.0: 
                    res['pixel_size'] = cached['pixel_size']
                    res['unit'] = cached['unit']
                if res['brightness'] == 1.0:
                    res['brightness'] = cached['brightness']
                if res['mag'] == 0:
                    res['mag'] = cached['mag']
                # Exposure might vary per frame, but if 0, maybe inherit? 
                # Usually exposure is constant in a stack.
                if res['exposure'] <= 0:
                     res['exposure'] = cached['exposure']

            # Final Calculations
            pixel_A = res['pixel_size']
            if 'nm' in res['unit']: pixel_A *= 10.0
            elif 'um' in res['unit']: pixel_A *= 10000.0
            elif 'pm' in res['unit']: pixel_A *= 0.01

            exposure = res['exposure'] if res['exposure'] > 0 else 1.0
            
            # Recalc Total Electrons with (potentially inherited) brightness
            total_electrons = np.sum(res['data'], dtype=np.float64) * res['brightness']
            
            area_A2 = res['data'].size * (pixel_A**2)
            dose_rate = total_electrons / (area_A2 * exposure)
            mean_int = np.mean(res['data'])

            info = {
                "file": str(candidate.name), 
                "date_raw": res['date_raw'], "date_fmt": res['date'],
                "exposure": float(exposure), "pixel_A": float(pixel_A), "pixel_unit_raw": res['unit'],
                "mag": float(res['mag']), "mean": float(mean_int)
            }
            self.finished.emit(float(dose_rate), str(candidate.resolve()), info)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class ImportWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.settings = QSettings("NapariUser", "Importer")
        
        # === 1. Detect System Specs ===
        self.total_ram_gb = self._get_total_memory_gb()
        self.cpu_count = os.cpu_count() or 4
        self.recommended_workers = self._calculate_optimal_workers()

        self._setup_ui()
        self.current_folder = None
        self.meta_cache = {} 
        self.archive_root = None
        self.archive_thread = None
        
        last = self.settings.value("last_folder", "")
        if last and os.path.isdir(last):
            self.current_folder = last
            self._update_folder_label(last)
            self.load_btn.setEnabled(True)
            self.calc_dose_btn.setEnabled(True)
            self.pick_file_btn.setEnabled(True)
            self._update_frame_range_hint()

    def _list_import_sequence_files(self):
        if not self.current_folder:
            return []

        folder = Path(self.current_folder)
        files_dm4 = sorted(list(folder.rglob("*.dm4")), key=natural_sort_key)
        if files_dm4:
            return files_dm4

        files_dm3 = sorted(list(folder.rglob("*.dm3")), key=natural_sort_key)
        return files_dm3

    def _compress_frame_indices(self, indices):
        if not indices:
            return []

        ranges = []
        start = indices[0]
        end = indices[0]
        for idx in indices[1:]:
            if idx == end + 1:
                end = idx
                continue
            ranges.append((start, end))
            start = idx
            end = idx
        ranges.append((start, end))
        return ranges

    def _format_frame_ranges(self, frame_ranges):
        if not frame_ranges:
            return "all"
        parts = []
        for start, end in frame_ranges:
            parts.append(str(start) if start == end else f"{start}-{end}")
        return ", ".join(parts)

    def _parse_frame_indices(self, text, total_frames):
        text = (text or "").strip()
        if not text:
            return None

        indices = set()
        try:
            parts = [p.strip() for p in text.split(',')]
            for part in parts:
                if not part:
                    continue
                if '-' in part:
                    start_text, end_text = [token.strip() for token in part.split('-', 1)]
                    if not start_text or not end_text:
                        return []
                    start = int(start_text)
                    end = int(end_text)
                    start = max(0, start)
                    end = min(total_frames - 1, end)
                    if start > end:
                        return []
                    indices.update(range(start, end + 1))
                else:
                    idx = int(part)
                    if 0 <= idx < total_frames:
                        indices.add(idx)
            return sorted(indices)
        except ValueError:
            return []

    def _update_frame_range_hint(self):
        if not hasattr(self, "frame_range_edit"):
            return

        files = self._list_import_sequence_files()
        total_frames = len(files)
        if total_frames <= 0:
            self.frame_range_edit.setPlaceholderText("All (Default) or e.g. 0-99")
            self.frame_range_edit.setToolTip("Supported syntax:\n- Range: 0-10\n- Single: 5\n- Mixed: 0-5, 8, 10-12")
            self.dose_idx_spin.setRange(-1, 99999)
            return

        max_frame = total_frames - 1
        self.frame_range_edit.setPlaceholderText(f"All (Default) or e.g. 0-{max_frame}")
        self.frame_range_edit.setToolTip(
            f"Valid frames: 0 to {max_frame}\n"
            "Supported syntax:\n- Range: 0-10\n- Single: 5\n- Mixed: 0-5, 8, 10-12"
        )
        self.dose_idx_spin.setRange(-1, max_frame)
    
    def elide_text(self, text, max_len=60):
        """[Fix] 缩短过长的文本，保留首尾"""
        if len(text) <= max_len: return text
        return text[:max_len//2-3] + "..." + text[-(max_len//2):]
    
    def _get_total_memory_gb(self):
        """获取系统物理内存 (GB)"""
        try:
            if PSUTIL_AVAILABLE:
                return psutil.virtual_memory().total / (1024**3)
            elif platform.system() == "Windows":
                # Windows fallback
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return stat.ullTotalPhys / (1024**3)
            else:
                return 16 # Default fallback
        except:
            return 16
    
    def _calculate_optimal_workers(self):
        """
        计算最佳 Worker 数量
        基于: RAM余量 (每线程~200MB) 和 CPU 核心数
        SSD 场景下，瓶颈通常是 RAM，而非 I/O
        """
        # 保留 4GB 给系统和 Napari 基础开销
        available_for_workers = max(1, self.total_ram_gb - 4)
        
        # 估算每个 Worker 满载时的峰值内存 (包含 DM4解析 + Numpy临时对象)
        # 经验值: 2048x2048 float32 加上 overhead 约为 250MB
        mem_limit_workers = int(available_for_workers * 1024 / 250) 
        
        # CPU 限制 (IO密集型可适当超频，但受限于 Python GIL 和 内存带宽)
        cpu_limit_workers = self.cpu_count * 2
        
        # 取交集，并设定硬限
        optimal = min(mem_limit_workers, cpu_limit_workers)
        optimal = max(2, min(optimal, 128)) # 至少2个，最多128
        
        return optimal

    def _setup_ui(self):
        main = QVBoxLayout(); main.setContentsMargins(2, 2, 2, 2)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        content = QWidget(); layout = QVBoxLayout(); layout.setSpacing(8)
        
        layout.addWidget(QLabel(f"<h3>📂 {tr('Import & Archive')}</h3>"))

        # Group 1: Data Source
        g_source = QGroupBox(tr("1. Data Source")); l_source = QVBoxLayout(); l_source.setSpacing(5); l_source.setContentsMargins(8, 15, 8, 8)
        h_brow = QHBoxLayout(); btn_browse = QPushButton(f"📂 {tr('Browse Folder')}"); btn_browse.clicked.connect(self._browse_folder)
        btn_browse.setToolTip(tr("Select DM4 folder for dose calculation and archive"))
        # 给按钮一个合理的固定宽度，防止它抢占过多空间
        btn_browse.setFixedWidth(160)
        
        self.folder_label = QLabel(tr("None")) 
        # 设置样式为灰色小字，保持原生风格
        self.folder_label.setStyleSheet("color: #AAAAAA; font-size: 11px;") 
        self.folder_label.setWordWrap(True) # 允许长路径换行
        # 设置SizePolicy，让它在水平方向尽可能伸展
        self.folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        h_brow.addWidget(btn_browse); h_brow.addWidget(self.folder_label)
        l_source.addLayout(h_brow); 
        
        # === [新增] PNG/TIFF 快速导入按钮 ===
        h_quick_import = QHBoxLayout()
        btn_load_png = QPushButton(f"📂 {tr('Load PNG Seq')}")
        btn_load_png.setToolTip(tr("Quickly load a PNG sequence folder to viewer"))
        btn_load_png.clicked.connect(self._load_png_sequence)
        
        btn_load_tiff = QPushButton(f"📂 {tr('Load TIFF')}")
        btn_load_tiff.setToolTip(tr("Quickly load a TIFF stack file to viewer"))
        btn_load_tiff.clicked.connect(self._load_tiff_stack)
        
        h_quick_import.addWidget(btn_load_png)
        h_quick_import.addWidget(btn_load_tiff)
        l_source.addLayout(h_quick_import)
        
        g_source.setLayout(l_source); layout.addWidget(g_source)

        # Group 2: Scan Metadata
        g_dose = QGroupBox(tr("2. Scan Metadata")); l_dose = QVBoxLayout(); l_dose.setSpacing(4); l_dose.setContentsMargins(8, 8, 8, 8)
        
        h_calc = QHBoxLayout()
        self.calc_dose_btn = QPushButton(f"🧮 {tr('Calc Dose')}")
        self.calc_dose_btn.setEnabled(False)
        self.calc_dose_btn.clicked.connect(self._calc_dose)
        h_calc.addWidget(self.calc_dose_btn)
        
        h_calc.addWidget(QLabel(tr("Img:")))
        self.dose_idx_spin = QSpinBox()
        self.dose_idx_spin.setRange(-1, 99999)
        self.dose_idx_spin.setValue(-1)
        self.dose_idx_spin.setSpecialValueText("Auto")
        self.dose_idx_spin.setToolTip(tr("Frame Index (-1 for Middle). Pick file to auto-set."))
        self.dose_idx_spin.setMinimumWidth(80)
        self.dose_idx_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_calc.addWidget(self.dose_idx_spin)

        self.pick_file_btn = QPushButton("📂")
        self.pick_file_btn.setToolTip(tr("Pick a specific .dm4 file from the folder to calculate dose"))
        self.pick_file_btn.setFixedWidth(50)
        self.pick_file_btn.setEnabled(False)
        self.pick_file_btn.clicked.connect(self._pick_single_file_for_dose)
        h_calc.addWidget(self.pick_file_btn)

        self.dose_val_label = QLabel("N/A")
        self.dose_val_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        h_calc.addWidget(self.dose_val_label)
        
        h_calc.addStretch()
        l_dose.addLayout(h_calc)
        
        # [Ref Display Fix] Use ReadOnly LineEdit for full path
        self.ref_path_edit = QLineEdit(); self.ref_path_edit.setPlaceholderText(tr("Ref Image Path")); self.ref_path_edit.setReadOnly(True); 
        self.ref_path_edit.setStyleSheet("QLineEdit { border: 1px solid #444; background: #222; color: #AAA; font-size: 10px; border-radius: 3px; padding: 2px; }")
        l_dose.addWidget(self.ref_path_edit)

        self.meta_info_label = QLabel(tr("Select folder to extract date, mag, pixel size...")); self.meta_info_label.setWordWrap(True); self.meta_info_label.setStyleSheet("font-size: 10px; color: gray;")
        l_dose.addWidget(self.meta_info_label); g_dose.setLayout(l_dose); layout.addWidget(g_dose)

        # Group 3: Archive Config
        g_exp = QGroupBox(tr("3. Archive Configuration")); g_exp.setStyleSheet("QGroupBox { border: 1px solid #2196F3; margin-top: 6px; } QGroupBox::title { color: #2196F3; }"); l_exp = QVBoxLayout(); l_exp.setSpacing(4); l_exp.setContentsMargins(8, 12, 8, 8)
        h1 = QHBoxLayout(); self.substance_edit = QComboBox(); self.substance_edit.setEditable(True); self.substance_edit.setPlaceholderText(tr("Sub (e.g. CRY2)"));self.substance_edit.setMinimumWidth(100);
        self._load_substance_history()
        self.solvent_edit = QLineEdit(); self.solvent_edit.setPlaceholderText(tr("Solv (Default: Water)"))
        self.dataset_edit = QLineEdit("ds1"); self.dataset_edit.setFixedWidth(50); self.dataset_edit.setPlaceholderText("ds#")
        h1.addWidget(QLabel(tr("Sub:"))); h1.addWidget(self.substance_edit); h1.addWidget(QLabel(tr("Solv:"))); h1.addWidget(self.solvent_edit); h1.addWidget(QLabel(tr("ID:"))); h1.addWidget(self.dataset_edit); l_exp.addLayout(h1)
        
        h2 = QHBoxLayout(); self.aperture_combo = QComboBox(); self.aperture_combo.addItems([str(i) for i in range(5)])
        self.mag_edit = QLineEdit("60K"); self.mag_edit.setPlaceholderText(tr("Mag"))
        h2.addWidget(QLabel(tr("OL# (0-4):"))); h2.addWidget(self.aperture_combo); h2.addWidget(QLabel(tr("Mag:"))); h2.addWidget(self.mag_edit); l_exp.addLayout(h2)
        
        h3 = QHBoxLayout(); self.win_spin = QSpinBox(); self.win_spin.setValue(3); self.win_spin.setRange(1, 99)
        self.sigma_spin = QDoubleSpinBox(); self.sigma_spin.setValue(0.8); self.sigma_spin.setSingleStep(0.1)
        h3.addWidget(QLabel(tr("Avg Win:"))); h3.addWidget(self.win_spin); h3.addWidget(QLabel(tr("Gaus σ:"))); h3.addWidget(self.sigma_spin); l_exp.addLayout(h3)
        
        l_exp.addWidget(QLabel(tr("Additional Info:"))); self.additional_info_edit = QTextEdit(); self.additional_info_edit.setPlaceholderText("e.g. 50mM Tris, pH 7.5..."); self.additional_info_edit.setMaximumHeight(45); l_exp.addWidget(self.additional_info_edit)
        
        for w in [self.substance_edit, self.solvent_edit, self.dataset_edit, self.mag_edit, self.aperture_combo, self.win_spin, self.sigma_spin]:
            if isinstance(w, (QLineEdit, QTextEdit)): w.textChanged.connect(self._update_preview)
            elif isinstance(w, (QComboBox)): w.currentTextChanged.connect(self._update_preview)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)): w.valueChanged.connect(self._update_preview)
        g_exp.setLayout(l_exp); layout.addWidget(g_exp)

        # Group 4: Action
        g_arc = QGroupBox(tr("4. Action")); g_arc.setStyleSheet("QGroupBox { border: 1px solid #FF9800; margin-top: 6px; } QGroupBox::title { color: #FF9800; }"); l_arc = QVBoxLayout(); l_arc.setSpacing(6); l_arc.setContentsMargins(8, 12, 8, 8)
        l_arc.addWidget(QLabel(tr("Preview Folder Name:"))); self.preview_label = QLabel("..."); self.preview_label.setWordWrap(True); self.preview_label.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-size: 11px; color: #E0E0E0; background-color: #2D2D2D; padding: 8px; border: 1px solid #3E3E3E; border-radius: 4px;")
        l_arc.addWidget(self.preview_label)
        self.create_archive_btn = QPushButton(f"📦 {tr('Create Archive Folder')}"); self.create_archive_btn.clicked.connect(self._create_archive); self.create_archive_btn.setStyleSheet("background-color: #E65100; color: white; font-weight: bold; padding: 8px;"); self.create_archive_btn.setEnabled(False)
        l_arc.addWidget(self.create_archive_btn)
        self.archive_progress = QProgressBar(); self.archive_progress.setVisible(False); self.archive_progress.setRange(0, 0) # Indeterminate
        l_arc.addWidget(self.archive_progress)
        
        # [New Config] Show Popup Checkbox
        self.check_show_popup = QCheckBox(tr("Show Result Popup"))
        # Load from QSettings, default True. type=bool ensures correct parsing
        self.check_show_popup.setChecked(self.settings.value("show_archive_popup", True, type=bool))
        self.check_show_popup.stateChanged.connect(lambda v: self.settings.setValue("show_archive_popup", bool(v)))
        self.check_show_popup.setToolTip(tr("Show a popup message with size and mode details after archiving."))
        l_arc.addWidget(self.check_show_popup)
        
        g_arc.setLayout(l_arc); layout.addWidget(g_arc)

        # Group 5: Load
        g_load = QGroupBox(tr("5. Load to Viewer")); l_load = QVBoxLayout(); l_load.setSpacing(4); l_load.setContentsMargins(8, 8, 8, 8)
        h_params = QHBoxLayout(); h_params.addWidget(QLabel(tr("Bit Depth:"))); self.bit_depth_combo = QComboBox(); self.bit_depth_combo.addItems(["8", "16", "32"]); self.bit_depth_combo.setCurrentText("8"); h_params.addWidget(self.bit_depth_combo)
        h_params.addWidget(QLabel(tr("Workers:"))); self.max_workers_spin = QSpinBox(); self.max_workers_spin.setRange(1, 128);  self.max_workers_spin.setValue(self.recommended_workers); 
        # Tooltip 显示系统信息
        ram_info = f"{self.total_ram_gb:.1f} GB"
        tip = (f"System RAM: {ram_info}\n"
               f"CPU Cores: {self.cpu_count}\n"
               f"Recommended: {self.recommended_workers} (Safe for your hardware)\n"
               f"Note: Higher is not always faster (IO/RAM bottlenecks).")
        self.max_workers_spin.setToolTip(tip)
        h_params.addWidget(self.max_workers_spin); l_load.addLayout(h_params)

        h_frame_range = QHBoxLayout()
        h_frame_range.addWidget(QLabel(tr("Frame Range")))
        self.frame_range_edit = QLineEdit()
        self.frame_range_edit.setPlaceholderText("All (Default) or e.g. 0-99")
        self.frame_range_edit.setToolTip("Supported syntax:\n- Range: 0-10\n- Single: 5\n- Mixed: 0-5, 8, 10-12")
        h_frame_range.addWidget(self.frame_range_edit)
        l_load.addLayout(h_frame_range)

        # Dynamic Warning Label
        self.lbl_memory_warning = QLabel("")
        self.lbl_memory_warning.setStyleSheet("color: #FF5252; font-size: 10px; font-weight: bold;")
        self.lbl_memory_warning.setVisible(False)
        l_load.addWidget(self.lbl_memory_warning)
        
        # Connect signal
        self.max_workers_spin.valueChanged.connect(self._check_worker_count)
        
        self.load_btn = QPushButton(f"🚀 {tr('Load Images')}"); self.load_btn.clicked.connect(self._load_data); self.load_btn.setEnabled(False); l_load.addWidget(self.load_btn)
        self.progress = QProgressBar(); self.progress.setVisible(False); l_load.addWidget(self.progress)
        g_load.setLayout(l_load); layout.addWidget(g_load)

        self.status = QLabel(""); self.status.setWordWrap(True); layout.addWidget(self.status)
        content.setLayout(layout); scroll.setWidget(content); main.addWidget(scroll); self.setLayout(main)
        
        # [新增] 防止滚轮误触更改数值
        setup_safe_scroll_all(
            self.dose_idx_spin, self.aperture_combo, self.win_spin, 
            self.sigma_spin, self.max_workers_spin, self.bit_depth_combo,
            self.substance_edit
        )
    
    def _check_worker_count(self, val):
        """动态显示内存警告"""
        # 计算当前设置预计消耗的内存
        # 基础: 4GB
        # 增量: 0.25 GB per worker
        est_usage = 4 + (val * 0.25)
        
        if val > self.recommended_workers:
            diff = val - self.recommended_workers
            if est_usage > self.total_ram_gb:
                self.lbl_memory_warning.setText(f"❌ DANGER! {val} workers may crash your PC (Est: {est_usage:.1f}GB > {self.total_ram_gb:.1f}GB)")
                self.lbl_memory_warning.setVisible(True)
            else:
                self.lbl_memory_warning.setText(f"⚠️ Warning: {val} exceeds recommended ({self.recommended_workers}). Watch RAM.")
                self.lbl_memory_warning.setVisible(True)
        else:
            self.lbl_memory_warning.setVisible(False)
    
    def _update_folder_label(self, path):
        """[Fix] 更新标签文本，自动缩短过长路径"""
        self.folder_label.setText(path)
        self.folder_label.setToolTip(path) # 鼠标悬停显示全名
        # self.folder_label.setCursorPosition(len(path))

    def _browse_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select Data Folder", self.settings.value("last_folder", ""))
        if f:
            self.current_folder = f
            self.settings.setValue("last_folder", f)
            self._update_folder_label(f)
            self.load_btn.setEnabled(True)
            self.calc_dose_btn.setEnabled(True)
            self.pick_file_btn.setEnabled(True)
            self._update_frame_range_hint()
            
            # === [Req 0] Auto-detect Dataset ID ===
            path_obj = Path(f)
            current_name = path_obj.name
            parent_name = path_obj.parent.name
            status_msgs = []
            # 匹配 dataset1, dataset-1, dataset_nonOL_1 等中的数字
            # match = re.search(r"dataset[-_]?.*?(\d+)", current_name, re.IGNORECASE)
            # 1. Dataset ID Detection
            match_ds = re.search(r"dataset[-_]?.*?(\d+)", current_name, re.IGNORECASE)
            if match_ds:
                ds_num = match_ds.group(1)
                self.dataset_edit.setText(f"ds{ds_num}")
                status_msgs.append(f"{tr('Auto-ID:')} ds{ds_num}")
            else:
                status_msgs.append(tr("No Dataset ID"))

            # 2. Aperture (OL) Detection
            match_ol = re.search(r"(?<!non)OL\s*[#\-_]?\s*(\d+)", current_name, re.IGNORECASE)
            source_level = "Current" # 用于调试日志
            if not match_ol:
                match_ol = re.search(r"(?<!non)OL\s*[#\-_]?\s*(\d+)", parent_name, re.IGNORECASE)
                source_level = "Parent"
            
            if match_ol:
                try:
                    raw_num = match_ol.group(1)
                    ol_num = str(int(raw_num)) # 去除前导零，例如 "02" -> "2"
                    
                    # 尝试在下拉框中查找该数值
                    idx = self.aperture_combo.findText(ol_num)
                    
                    if idx >= 0:
                        # 情况A: 列表中已有 (0-4)，直接选中
                        self.aperture_combo.setCurrentIndex(idx)
                        status_msgs.append(f"{tr('Auto-OL:')} {ol_num}")
                    else:
                        # 情况B: 列表中没有 (比如 OL5, OL7)，这是导致你Bug的核心原因！
                        # [Fix] 动态添加到下拉框并选中
                        self.aperture_combo.addItem(ol_num)
                        self.aperture_combo.setCurrentText(ol_num)
                        status_msgs.append(f"{tr('Auto-OL:')} {ol_num} {tr('(Auto-Added)')}")
                        
                except Exception as e:
                    print(f"OL Parse Error: {e}")
                    pass
            else:
                status_msgs.append(tr("OL Not Found"))
            
            self.status.setText(" | ".join(status_msgs))
            
            if not match_ds:
                 QMessageBox.information(self, tr("Check ID"), tr("Could not detect 'dataset' number.\nPlease check ID manually."))

            self._update_preview()

    def _pick_single_file_for_dose(self):
        if not self.current_folder: return
        f, _ = QFileDialog.getOpenFileName(self, tr("Select Image for Dose Calculation"), self.current_folder, "Microscopy Files (*.dm4 *.dm3)")
        if f:
            self.status.setText(tr("Locating file index..."))
            try:
                target_path = Path(f).resolve()
                all_files = self._list_import_sequence_files()
                found_idx = -1
                for i, p in enumerate(all_files):
                    if p.resolve() == target_path:
                        found_idx = i
                        break
                if found_idx >= 0:
                    self.dose_idx_spin.setValue(found_idx)
                    self.status.setText(f"{tr('Selected: %s (Index: %s)') % (target_path.name, found_idx)}")
                    self._calc_dose()
                else:
                    self.status.setText(f"❌ {tr('File not found in current structure match.')}")
            except Exception as e:
                self.status.setText(f"❌ {tr('Error picking file:')} {e}")

    def _calc_dose(self):
        if not self.current_folder: return
        self.calc_dose_btn.setEnabled(False)
        self.pick_file_btn.setEnabled(False)
        self.status.setText(tr("Scanning metadata..."))
        frame_idx = self.dose_idx_spin.value()
        self.thread_dose = DoseCalculationThread(self.current_folder, frame_idx=frame_idx)
        self.thread_dose.finished.connect(self._on_dose_done)
        self.thread_dose.error.connect(lambda e: (self.status.setText(e), self.calc_dose_btn.setEnabled(True), self.pick_file_btn.setEnabled(True)))
        self.thread_dose.start()

    def _on_dose_done(self, dose, fname, info):
        self.meta_cache = info; self.meta_cache['dose'] = dose
        self.calc_dose_btn.setEnabled(True)
        self.pick_file_btn.setEnabled(True)
        self.dose_val_label.setText(f"{dose:.2f} e⁻/Å²/s")
        if info.get('mag', 0) > 0:
            m = info['mag']
            self.mag_edit.setText(f"{int(m/1000)}K" if m >= 1000 else str(int(m)))
        
        # [Fix] Display full path in LineEdit
        self.ref_path_edit.setText(fname)
        self.ref_path_edit.setToolTip(fname)
        self.ref_path_edit.setCursorPosition(0) # Show start of path
        
        self.meta_info_label.setText(f"Date: {info.get('date_fmt', 'N/A')}, Exp: {info.get('exposure', 0)}s\nPixel: {info.get('pixel_A', 0):.2f} Å, Mean: {info.get('mean', 0):.1f}")
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
            self.substance_edit.setCurrentIndex(0)

    def _save_substance_history(self):
        current_text = self.substance_edit.currentText().strip()
        if not current_text: return
        history = self.settings.value("substance_history", [])
        if current_text in history: history.remove(current_text)
        history.insert(0, current_text)
        history = history[:5]
        self.settings.setValue("substance_history", history)
        self.substance_edit.blockSignals(True)
        self.substance_edit.clear()
        self.substance_edit.addItems(history)
        self.substance_edit.setCurrentText(current_text)
        self.substance_edit.blockSignals(False)

    def _get_dir_size(self, path):
        total = 0
        try:
            for entry in os.scandir(path):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += self._get_dir_size(entry.path)
        except Exception: pass
        return total

    def _create_archive(self):
        if not self.current_folder: return
        self._save_substance_history()
        
        # 1. 准备路径
        parent_dir = Path(self.current_folder).parent
        folder_name = self._generate_folder_name()
        archive_path = parent_dir / folder_name
        
        if archive_path.exists():
            if QMessageBox.warning(self, tr("Exists"), tr("Folder exists:\n%s\nOverwrite?") % folder_name, QMessageBox.Yes|QMessageBox.No) == QMessageBox.No: return
        
        self.create_archive_btn.setEnabled(False)
        self.archive_progress.setVisible(True)
        
        # 2. 创建归档根目录结构
        archive_path.mkdir(parents=True, exist_ok=True)
        self.archive_root = str(archive_path)
        QSettings("NapariUser", "Global").setValue("archive_path", self.archive_root)
        
        # 自动保存归档路径到持久化搜索列表（用于 Recovery 跨会话访问）
        from utils.session_logger import SessionLogger
        SessionLogger.add_search_path(self.archive_root)
        
        # === [信息共享] 保存关键信息供 GeometryWidget 使用 ===
        date_str = self.meta_cache.get('date_fmt', datetime.datetime.now().strftime("%Y%m%d"))
        ds_id = self.dataset_edit.text().strip() or "ds1"
        QSettings("NapariUser", "Global").setValue("current_date", date_str)
        QSettings("NapariUser", "Global").setValue("current_dataset_id", ds_id)
        # =======================================================

        with open(archive_path / "readme.txt", 'w', encoding='utf-8') as f:
            f.write(f"Archive: {folder_name}\nCreated: {datetime.datetime.now()}\n" + "-"*30 + "\n")
            self.sub_txt = self.substance_edit.currentText()
            f.write(f"Substance: {self.sub_txt}\nSolvent: {self.solvent_edit.text()}\nDataset: {self.dataset_edit.text()}\n")
            f.write(f"Additional Info: {self.additional_info_edit.toPlainText()}\n" + "-"*30 + "\n")
            f.write("Original DM4 Metadata:\n")
            for k, v in self.meta_cache.items(): f.write(f"{k}: {v}\n")
            
        params = {"folder_name": folder_name, "import_meta": self.meta_cache, "planned_settings": {"rolling_avg_window": self.win_spin.value(), "gaussian_sigma": self.sigma_spin.value()}}
        with open(archive_path / "processing_log.json", 'w') as f: json.dump(params, f, indent=2, cls=NumpyEncoder)
        
        # === [SessionLogger] 同步元数据到会话日志 ===
        try:
            logger = get_logger()
            logger.set_metadata(
                substance=self.substance_edit.currentText(),
                dataset_id=ds_id,
                archive_path=str(archive_path)
            )
            logger.log_action("import", "create_archive", {
                "folder_name": folder_name,
                "archive_path": str(archive_path),
                "bit_depth": int(self.bit_depth_combo.currentText())
            })
        except Exception as e:
            print(f"SessionLogger sync failed: {e}")

        # 4. [Req 2] 计算大小并决定 Move vs Copy
        self.status.setText(tr("Checking size..."))
        size_bytes = self._get_dir_size(str(self.current_folder))
        size_gb = size_bytes / (1024**3)
        move_mode = False
        
        # 阈值读取
        move_threshold = float(GlobalConfig.get("sys_move_threshold_gb"))

        if size_gb > move_threshold:
            move_mode = True
            # 注意：此处弹窗只是通知将要发生什么，不需要用户再次确认（因为已经在之前逻辑里确定了策略）
            # 或者，如果之前需求是自动切换并提醒，这里只是标记
            # 用户希望在 *完成后* 提醒，所以这里我们只记录状态
        
        # 5. [Req 1] 原始文件重命名逻辑
        # 提取 dataset ID 数字部分
        ds_num = re.sub(r'[^0-9]', '', ds_id) 
        if not ds_num: ds_num = "1"
        
        raw_folder_name = f"{date_str}_{self.sub_txt}_OriginalDataset{ds_num}"
        raw_dest = archive_path / raw_folder_name
        
        self.status.setText(tr("Moving raw data...") if move_mode else tr("Copying raw data..."))
        self.archive_thread = ArchiveThread(str(self.current_folder), str(raw_dest), move_mode=move_mode)
        # Pass move_mode and size_gb to callback
        self.archive_thread.finished.connect(lambda: self._on_archive_done(move_mode, size_gb, raw_dest))
        self.archive_thread.error.connect(self._on_archive_error)
        self.archive_thread.start()

    def _on_archive_done(self, was_moved, size_gb, new_path):
        self.archive_progress.setVisible(False)
        self.create_archive_btn.setEnabled(True)
        self.status.setText(f"✅ {tr('Archived:')} {Path(self.archive_root).name}")

        if was_moved:
            self.current_folder = str(new_path)
            self._update_folder_label(str(new_path))
            self.settings.setValue("last_folder", str(new_path))
            # 刷新一下 ID 提取 (可选，确保一致性)
            self.status.setText(f"ℹ️ {tr('Source path updated to archive location.')}")
        
        # [Req New] Popup with details
        if self.check_show_popup.isChecked():
            mode_str = tr("MOVE (Fast)") if was_moved else tr("COPY (Safe)")
            msg = tr("Archive created successfully!") + "\n\n" \
                  f"📂 Path: {elide_text(str(self.archive_root), 40)}\n" \
                  f"📦 Source Size: {size_gb:.2f} GB\n" \
                  f"⚙️ Mode: {mode_str}\n"
            
            if was_moved:
                msg += "\n" + tr("Large dataset detected (>100GB). Original folder was MOVED to archive to save time/space.")
            else:
                msg += "\n" + tr("Original folder was COPIED. Please delete the source manually if needed.")
                
            QMessageBox.information(self, tr("Archive Complete"), msg)

    def _on_archive_error(self, err):
        self.archive_progress.setVisible(False)
        self.create_archive_btn.setEnabled(True)
        self.status.setText(f"❌ {tr('Archive Error:')} {err}")
        QMessageBox.critical(self, tr("Error"), str(err))

    def _load_data(self):
        if not self.current_folder: return
        source_files = self._list_import_sequence_files()
        if not source_files:
            self.status.setText(f"❌ {tr('Error:')} No DM4 or DM3 files found in folder.")
            return

        frame_range_text = self.frame_range_edit.text().strip()
        selected_indices = self._parse_frame_indices(frame_range_text, len(source_files))
        if frame_range_text and not selected_indices:
            self.status.setText(f"❌ {tr('Invalid frame range syntax')}")
            return

        if len(self.viewer.layers) > 0:
            reply = QMessageBox.question(
                self, tr("Confirm Load"), tr("Loading new data will CLEAR ALL current layers.\nContinue?"),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.No: return
            print("Cleaning up existing layers...")
            self.viewer.layers.clear() 
            gc.collect()
        
        self.load_btn.setEnabled(False)
        self.progress.setVisible(True)
        if selected_indices is None:
            self.status.setText(tr("Loading..."))
        else:
            selected_ranges = self._format_frame_ranges(self._compress_frame_indices(selected_indices))
            self.status.setText(f"{tr('Loading...')} ({selected_ranges})")
        bit = int(self.bit_depth_combo.currentText())
        workers = self.max_workers_spin.value()
        self.thread_load = LoaderThread(self.current_folder, bit, workers, frame_indices=selected_indices)
        self.thread_load.progress.connect(lambda c, t: (self.progress.setMaximum(t), self.progress.setValue(c)))
        self.thread_load.finished.connect(self._on_loaded)
        self.thread_load.error.connect(lambda e: (self.status.setText(f"{tr('Error:')} {e}"), self.load_btn.setEnabled(True)))
        self.thread_load.start()

    def _on_loaded(self, stack, meta):
        self.progress.setVisible(False)
        self.load_btn.setEnabled(True)
        name = f"Original_{Path(self.current_folder).name}"
        if len(name) > 30: name = name[:15] + "..." + name[-10:]
        self.viewer.add_image(stack, name=name, metadata=meta, colormap='gray')
        source_total_frames = int(meta.get("source_num_frames", len(stack)))
        selected_ranges = meta.get("selected_frame_ranges", [])
        selection_text = self._format_frame_ranges(selected_ranges)
        if meta.get("frame_selection_applied"):
            self.status.setText(
                f"{tr('Loaded %s frames.') % len(stack)} "
                f"(source: {source_total_frames}, selected: {selection_text})"
            )
        else:
            self.status.setText(tr("Loaded %s frames.") % len(stack))
        
        # === [SessionLogger] 记录 DM4 导入操作 ===
        try:
            get_logger().log_action("import", "load_dm4_sequence", {
                "source_path": str(self.current_folder),
                "frame_count": len(stack),
                "source_total_frames": source_total_frames,
                "frame_selection": selection_text,
                "layer_name": name,
                "bit_depth": int(self.bit_depth_combo.currentText())
            })
        except: pass

    # === [新增] PNG 序列导入 ===
    def _load_png_sequence(self):
        """加载 PNG 序列文件夹"""
        from qtpy.QtWidgets import QProgressDialog
        from qtpy.QtCore import Qt
        import cv2
        
        start_path = self.settings.value("last_folder", str(Path.home()))
        folder = QFileDialog.getExistingDirectory(self, tr("Select PNG Sequence Folder"), start_path)
        if not folder:
            return
        
        folder_path = Path(folder)
        png_files = sorted(list(folder_path.glob("*.png")), key=natural_sort_key)
        
        if not png_files:
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
                    self.status.setText(tr("Loading canceled."))
                    return
                
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # 如果是彩色图，转为灰度
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    frames.append(img)
                progress.setValue(i + 1)
            
            progress.close()
            
            if not frames:
                QMessageBox.warning(self, tr("Error"), tr("Could not read any valid images."))
                return
                
            stack = np.array(frames)
            name = f"PNG_{folder_path.name}"
            if len(name) > 30:
                name = name[:15] + "..." + name[-10:]
            
            self.viewer.add_image(stack, name=name, colormap='gray')
            self.status.setText(f"✅ {tr('Loaded')} {len(stack)} PNG frames.")
            
            # 记录导入操作
            try:
                get_logger().log_action("import", "load_png_sequence", {
                    "source_path": str(folder_path),
                    "frame_count": len(stack),
                    "layer_name": name
                })
            except: pass
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, tr("Error"), str(e))
            self.status.setText(f"❌ {tr('Error:')} {e}")

    # === [新增] TIFF Stack 导入 ===
    def _load_tiff_stack(self):
        """加载 TIFF Stack 文件"""
        import tifffile
        
        start_path = self.settings.value("last_folder", str(Path.home()))
        file_path, _ = QFileDialog.getOpenFileName(
            self, tr("Select TIFF Stack File"), start_path, 
            "TIFF Files (*.tiff *.tif)"
        )
        if not file_path:
            return
        
        try:
            self.status.setText(tr("Loading TIFF..."))
            stack = tifffile.imread(file_path)
            
            # 确保是3D数组 (T, H, W)
            if stack.ndim == 2:
                stack = stack[np.newaxis, ...]  # 单帧转为 (1, H, W)
            elif stack.ndim == 4:
                # 可能是 (T, H, W, C)，取第一个通道
                stack = stack[..., 0]
            
            name = f"TIFF_{Path(file_path).stem}"
            if len(name) > 30:
                name = name[:15] + "..." + name[-10:]
            
            self.viewer.add_image(stack, name=name, colormap='gray')
            self.status.setText(f"✅ {tr('Loaded')} {len(stack)} TIFF frames.")
            
            # 记录导入操作
            try:
                get_logger().log_action("import", "load_tiff_stack", {
                    "source_path": file_path,
                    "frame_count": len(stack),
                    "layer_name": name
                })
            except: pass
            
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), str(e))
            self.status.setText(f"❌ {tr('Error:')} {e}")

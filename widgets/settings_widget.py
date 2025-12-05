"""
File: widgets/settings_widget.py
"""
from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QFormLayout, QTabWidget, 
                            QWidget, QKeySequenceEdit, QMessageBox, QSpinBox, 
                            QCheckBox, QGroupBox, QFileDialog, QDoubleSpinBox,
                            QColorDialog, QScrollArea)
from qtpy.QtGui import QKeySequence, QColor
from qtpy.QtCore import QObject, Signal
import json
import os
from pathlib import Path

class ConfigSignals(QObject):
    config_updated = Signal()

class GlobalConfig:
    """全局配置单例辅助类 (基于 JSON 文件)"""
    _config_path = Path.home() / ".napari_tem_config.json"

    signals = ConfigSignals()
    
    # === 1. 扩充默认配置 ===
    DEFAULTS = {
        # Shortcuts
        "shortcut_toggle_ui": "J",
        "shortcut_undo_drift": "Ctrl+Z",
        "shortcut_apply_crop": "Enter",
        "shortcut_switch_mode": "M",
        
        # Drift Defaults
        "drift_kernel": 11,
        "drift_workers": 8,
        "drift_auto_calc": True,  # [New] 是否自动计算

        # Cache
        "cache_dir": "", 
        
        # Geometry Defaults
        "geo_suffix": "_origin",
        "geo_padding": 5,
        "geo_keep_index": True,   # [New] 保持序号
        "geo_force_square": True, # [New] 强制正方形
        "geo_enlarge": True,      # [New] 扩大画布
        "geo_create_denoise": False, # [New] 创建去噪文件夹
        "geo_create_refine": False,  # [New] 创建Refine文件夹

        # Suffixes
        "geo_suffix_lrtem": "_lrtem",
        "geo_suffix_hrtem": "_hrtem",
        "geo_suffix_mask": "_mask",
        "geo_suffix_mask_new": "_mask_new",

        # Enhance Defaults
        "enh_use_gaussian": False, # [New]
        "enh_sigma": 0.8,          # [New]
        "enh_use_average": False,  # [New]
        "enh_window": 3,           # [New]
        "enh_workers": 8,

        # Visual Styles
        "style_measure_color": "#FFD700",
        "style_measure_width": 3,
        "style_measure_font_size": 11,
        "style_crop_color": "yellow",
        "style_batch_box_color": "#00FF00",
        "style_batch_width": 2,
        "style_batch_text_color": "#00FF00",
        "style_batch_font_size": 10,

        # System
        "sys_ram_threshold_gb": 4.0,
        "sys_disk_warn_gb": 10.0,
        "sys_move_threshold_gb": 30.0,
        "show_archive_popup": True
    }

    @classmethod
    def _load_config(cls):
        if not cls._config_path.exists():
            return {}
        try:
            with open(cls._config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    @classmethod
    def _save_config(cls, data):
        try:
            with open(cls._config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    @classmethod
    def get(cls, key):
        data = cls._load_config()
        # 如果 key 不存在，返回默认值
        val = data.get(key, cls.DEFAULTS.get(key))
        # 类型转换
        if isinstance(val, str):
            if val.lower() == 'true': return True
            if val.lower() == 'false': return False
        # 如果是 None (比如新增加的key在旧配置文件里没有)，回退到 DEFAULT
        if val is None:
            return cls.DEFAULTS.get(key)
        return val

    @classmethod
    def set(cls, key, value, emit_signal=True):
        """设置并立即保存配置，默认触发更新信号"""
        data = cls._load_config()
        data[key] = value
        cls._save_config(data)
        if emit_signal:
            cls.signals.config_updated.emit()
    
    @classmethod
    def get_napari_shortcut(cls, key):
        raw = str(cls.get(key))
        napari_key = raw.replace("Ctrl+", "Control-").replace("Shift+", "Shift-").replace("Alt+", "Alt-").replace("Meta+", "Meta-").replace("Enter", "Return")
        if "Ctrl " in napari_key: napari_key = napari_key.replace("Ctrl ", "Control-")
        return napari_key

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Preferences & Configuration")
        self.resize(750, 650)
        self.setStyleSheet("""
            QDialog { background-color: #262626; color: #E0E0E0; font-family: "Segoe UI"; font-size: 10pt; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #BBB; padding: 8px 12px; border: 1px solid #444; border-bottom: none; }
            QTabBar::tab:selected { background: #444; color: white; font-weight: bold; border-bottom: 2px solid #2196F3; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QKeySequenceEdit { background: #333; color: white; border: 1px solid #555; padding: 4px; border-radius: 3px; }
            QPushButton { background: #444; border: 1px solid #555; padding: 5px 10px; border-radius: 3px; color: white; }
            QPushButton:hover { background: #555; }
            QGroupBox { border: 1px solid #555; margin-top: 10px; padding-top: 15px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #2196F3; }
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        tabs = QTabWidget()

        # === Tab 1: General (Existing) ===
        tabs.addTab(self._create_general_tab(), "💾 General")
        
        # === Tab 2: Visual Styles (New) ===
        tabs.addTab(self._create_styles_tab(), "🎨 Visual Styles")

        # === Tab 3: System Thresholds (New) ===
        tabs.addTab(self._create_system_tab(), "⚙️ System Thresholds")

        # === Tab 4: Shortcuts (Existing) ===
        tabs.addTab(self._create_shortcuts_tab(), "⌨️ Shortcuts")

        layout.addWidget(tabs)

        # Buttons
        h_btn = QHBoxLayout()
        btn_reset = QPushButton("⚠️ Reset to Defaults")
        btn_reset.setStyleSheet("background-color: #8B0000;")
        btn_reset.clicked.connect(self._reset_defaults)
        
        btn_save = QPushButton("Save & Close")
        btn_save.setStyleSheet("background-color: #2E7D32; font-weight: bold;")
        btn_save.clicked.connect(self.accept)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        h_btn.addWidget(btn_reset)
        h_btn.addStretch()
        h_btn.addWidget(btn_cancel)
        h_btn.addWidget(btn_save)
        layout.addLayout(h_btn)
        self.setLayout(layout)

    def _create_general_tab(self):
        w = QWidget()
        l = QVBoxLayout()
        
        # Cache
        g_cache = QGroupBox("Cache Location")
        f_cache = QFormLayout()
        self.cache_dir_edit = QLineEdit(str(GlobalConfig.get("cache_dir")))
        btn_browse = QPushButton("📂")
        btn_browse.clicked.connect(lambda: self.cache_dir_edit.setText(QFileDialog.getExistingDirectory(self, "Cache Dir")))
        h = QHBoxLayout(); h.addWidget(self.cache_dir_edit); h.addWidget(btn_browse)
        f_cache.addRow("Path:", h)
        g_cache.setLayout(f_cache); l.addWidget(g_cache)

        # Algorithm
        g_algo = QGroupBox("Algorithm Defaults")
        f_algo = QFormLayout()
        self.drift_auto_check = QCheckBox("Auto-Calculate (Preview on ROI Draw)")
        self.drift_auto_check.setChecked(bool(GlobalConfig.get("drift_auto_calc")))
        self.drift_k_spin = QSpinBox(); self.drift_k_spin.setRange(3, 99); self.drift_k_spin.setValue(int(GlobalConfig.get("drift_kernel")))
        self.drift_w_spin = QSpinBox(); self.drift_w_spin.setRange(1, 64); self.drift_w_spin.setValue(int(GlobalConfig.get("drift_workers")))
        f_algo.addRow(self.drift_auto_check)
        f_algo.addRow("Drift Kernel:", self.drift_k_spin)
        f_algo.addRow("Max Workers:", self.drift_w_spin)
        g_algo.setLayout(f_algo); l.addWidget(g_algo)

        g_enh = QGroupBox("Image Enhancement Defaults")
        f_enh = QFormLayout()
        self.enh_gaus_check = QCheckBox("Enable Gaussian Blur by Default")
        self.enh_gaus_check.setChecked(bool(GlobalConfig.get("enh_use_gaussian")))
        
        self.enh_avg_check = QCheckBox("Enable Rolling Average by Default")
        self.enh_avg_check.setChecked(bool(GlobalConfig.get("enh_use_average")))
        # Sigma
        self.enh_sigma_spin = QDoubleSpinBox()
        self.enh_sigma_spin.setRange(0.1, 10.0)
        self.enh_sigma_spin.setSingleStep(0.1)
        self.enh_sigma_spin.setValue(float(GlobalConfig.get("enh_sigma")))
        
        # Window
        self.enh_win_spin = QSpinBox()
        self.enh_win_spin.setRange(1, 99)
        self.enh_win_spin.setSingleStep(2)
        self.enh_win_spin.setValue(int(GlobalConfig.get("enh_window")))
        
        # Workers
        self.enh_work_spin = QSpinBox()
        self.enh_work_spin.setRange(1, 64)
        self.enh_work_spin.setValue(int(GlobalConfig.get("enh_workers")))

        f_enh.addRow(self.enh_gaus_check)
        f_enh.addRow("Gaussian Sigma:", self.enh_sigma_spin)
        f_enh.addRow(self.enh_avg_check)
        f_enh.addRow("Roll Avg Window:", self.enh_win_spin)
        f_enh.addRow("Enhance Workers:", self.enh_work_spin)
        
        g_enh.setLayout(f_enh)
        l.addWidget(g_enh)

        # Geo Suffixes
        g_geo = QGroupBox("Generated Folder Suffixes")
        f_geo = QFormLayout()
        self.geo_enl_check = QCheckBox("Enlarge Canvas on Rotate")
        self.geo_enl_check.setChecked(bool(GlobalConfig.get("geo_enlarge")))
        self.geo_keep_idx_check = QCheckBox("Keep Original Frame Index")
        self.geo_keep_idx_check.setChecked(bool(GlobalConfig.get("geo_keep_index")))
        self.geo_sq_check = QCheckBox("Force Square Crops")
        self.geo_sq_check.setChecked(bool(GlobalConfig.get("geo_force_square")))
        self.geo_denoise_check = QCheckBox("Create Denoise Folders (LR/HR)")
        self.geo_denoise_check.setChecked(bool(GlobalConfig.get("geo_create_denoise")))
        self.geo_refine_check = QCheckBox("Create Refine Folders (Mask)")
        self.geo_refine_check.setChecked(bool(GlobalConfig.get("geo_create_refine")))
        f_geo.addRow("", self.geo_enl_check)
        f_geo.addRow("", self.geo_keep_idx_check)
        f_geo.addRow("", self.geo_sq_check)
        f_geo.addRow("", self.geo_denoise_check)
        f_geo.addRow("", self.geo_refine_check)
        f_geo.addRow(QLabel("<hr>")) # 分割线

        self.suff_main = QLineEdit(str(GlobalConfig.get("geo_suffix")))
        self.suff_lr = QLineEdit(str(GlobalConfig.get("geo_suffix_lrtem")))
        self.suff_hr = QLineEdit(str(GlobalConfig.get("geo_suffix_hrtem")))
        self.suff_mask = QLineEdit(str(GlobalConfig.get("geo_suffix_mask")))
        self.suff_new = QLineEdit(str(GlobalConfig.get("geo_suffix_mask_new")))
        f_geo.addRow("Main Suffix:", self.suff_main)
        f_geo.addRow("LR Suffix:", self.suff_lr)
        f_geo.addRow("HR Suffix:", self.suff_hr)
        f_geo.addRow("Mask Suffix:", self.suff_mask)
        f_geo.addRow("New Mask Suffix:", self.suff_new)
        g_geo.setLayout(f_geo); l.addWidget(g_geo)

        l.addStretch(); w.setLayout(l)
        
        # Wrap in ScrollArea in case height is too large
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setFrameShape(0) # No border
        return scroll

    def _create_styles_tab(self):
        w = QWidget()
        l = QVBoxLayout()
        
        # Helper to create color picker row
        def add_color_row(layout, label, config_key):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            curr_col = str(GlobalConfig.get(config_key))
            line = QLineEdit(curr_col)
            btn = QPushButton("🎨")
            btn.setFixedWidth(30)
            btn.setStyleSheet(f"background-color: {curr_col}; border: 1px solid #555;")
            
            def pick():
                c = QColorDialog.getColor(QColor(line.text()), self, "Select Color")
                if c.isValid():
                    hex_c = c.name()
                    line.setText(hex_c)
                    btn.setStyleSheet(f"background-color: {hex_c}; border: 1px solid #555;")
            
            btn.clicked.connect(pick)
            h.addWidget(line); h.addWidget(btn)
            layout.addLayout(h)
            return line

        # 1. Measure Tool
        g_meas = QGroupBox("📏 Measure Tool")
        l_meas = QVBoxLayout()
        self.style_meas_col = add_color_row(l_meas, "Line Color:", "style_measure_color")
        
        h_m = QHBoxLayout()
        self.style_meas_w = QSpinBox(); self.style_meas_w.setRange(1, 20); self.style_meas_w.setValue(int(GlobalConfig.get("style_measure_width")))
        self.style_meas_font = QSpinBox(); self.style_meas_font.setRange(5, 50); self.style_meas_font.setValue(int(GlobalConfig.get("style_measure_font_size")))
        h_m.addWidget(QLabel("Width:")); h_m.addWidget(self.style_meas_w)
        h_m.addWidget(QLabel("Font Size:")); h_m.addWidget(self.style_meas_font)
        l_meas.addLayout(h_m)
        g_meas.setLayout(l_meas); l.addWidget(g_meas)

        # 2. Simple Crop
        g_simp = QGroupBox("✂️ Simple Crop")
        l_simp = QVBoxLayout()
        self.style_crop_col = add_color_row(l_simp, "Box Color:", "style_crop_color")
        g_simp.setLayout(l_simp); l.addWidget(g_simp)

        # 3. Batch Crop
        g_batch = QGroupBox("📦 Batch Crop")
        l_batch = QVBoxLayout()
        self.style_batch_box_col = add_color_row(l_batch, "Box Color:", "style_batch_box_color")
        self.style_batch_txt_col = add_color_row(l_batch, "Text Color:", "style_batch_text_color")
        
        h_b = QHBoxLayout()
        self.style_batch_w = QSpinBox(); self.style_batch_w.setRange(1, 20); self.style_batch_w.setValue(int(GlobalConfig.get("style_batch_width")))
        self.style_batch_font = QSpinBox(); self.style_batch_font.setRange(5, 50); self.style_batch_font.setValue(int(GlobalConfig.get("style_batch_font_size")))
        h_b.addWidget(QLabel("Width:")); h_b.addWidget(self.style_batch_w)
        h_b.addWidget(QLabel("Font Size:")); h_b.addWidget(self.style_batch_font)
        l_batch.addLayout(h_b)
        g_batch.setLayout(l_batch); l.addWidget(g_batch)

        l.addStretch(); w.setLayout(l)
        return w

    def _create_system_tab(self):
        w = QWidget()
        l = QVBoxLayout()
        
        g_res = QGroupBox("Resource Thresholds")
        form = QFormLayout()
        
        self.sys_ram = QDoubleSpinBox(); self.sys_ram.setRange(0.1, 1024); self.sys_ram.setSuffix(" GB"); self.sys_ram.setValue(float(GlobalConfig.get("sys_ram_threshold_gb")))
        self.sys_disk = QDoubleSpinBox(); self.sys_disk.setRange(0.1, 10240); self.sys_disk.setSuffix(" GB"); self.sys_disk.setValue(float(GlobalConfig.get("sys_disk_warn_gb")))
        self.sys_move = QDoubleSpinBox(); self.sys_move.setRange(0.1, 10240); self.sys_move.setSuffix(" GB"); self.sys_move.setValue(float(GlobalConfig.get("sys_move_threshold_gb")))

        form.addRow("RAM vs Disk Limit:", self.sys_ram)
        l_hint1 = QLabel("If an array exceeds this size, create it on Disk (memmap).")
        l_hint1.setStyleSheet("color: gray; font-size: 9pt; margin-bottom: 10px;")
        form.addRow("", l_hint1)

        form.addRow("Disk Space Warning:", self.sys_disk)
        l_hint2 = QLabel("Warn if cache folder usage exceeds this size.")
        l_hint2.setStyleSheet("color: gray; font-size: 9pt; margin-bottom: 10px;")
        form.addRow("", l_hint2)

        form.addRow("Move vs Copy Limit:", self.sys_move)
        l_hint3 = QLabel("If raw data folder > this size, MOVE instead of COPY during archive.")
        l_hint3.setStyleSheet("color: gray; font-size: 9pt;")
        form.addRow("", l_hint3)

        g_res.setLayout(form); l.addWidget(g_res)
        l.addStretch(); w.setLayout(l)
        return w

    def _create_shortcuts_tab(self):
        w = QWidget()
        f = QFormLayout()
        self.key_edits = {}
        shortcuts_map = {
            "shortcut_toggle_ui": "Toggle Layer Controls",
            "shortcut_undo_drift": "Undo / Clear ROI",
            "shortcut_apply_crop": "Apply Crop / Export",
            "shortcut_switch_mode": "Switch Draw/Select Mode"
        }
        for key, label in shortcuts_map.items():
            val = str(GlobalConfig.get(key))
            edit = QKeySequenceEdit(QKeySequence(val))
            self.key_edits[key] = edit
            f.addRow(label, edit)
        w.setLayout(f)
        return w

    def accept(self):
        # 1. Save Cache
        GlobalConfig.set("cache_dir", self.cache_dir_edit.text(), emit_signal=False)
        
        # 2. Save Drift
        if hasattr(self, 'drift_auto_check'):
             GlobalConfig.set("drift_auto_calc", self.drift_auto_check.isChecked(), emit_signal=False)
        GlobalConfig.set("drift_auto_calc", self.drift_auto_check.isChecked(), emit_signal=False) # [New]
        GlobalConfig.set("drift_kernel", self.drift_k_spin.value(), emit_signal=False)
        GlobalConfig.set("drift_workers", self.drift_w_spin.value(), emit_signal=False)

        # 3. Save Enhancement
        GlobalConfig.set("enh_use_gaussian", self.enh_gaus_check.isChecked(), emit_signal=False) # [New]
        GlobalConfig.set("enh_use_average", self.enh_avg_check.isChecked(), emit_signal=False)   # [New]
        GlobalConfig.set("enh_sigma", self.enh_sigma_spin.value(), emit_signal=False)
        GlobalConfig.set("enh_window", self.enh_win_spin.value(), emit_signal=False)
        GlobalConfig.set("enh_workers", self.enh_work_spin.value(), emit_signal=False)

        # 4. Save Geometry Options [New]
        GlobalConfig.set("geo_enlarge", self.geo_enl_check.isChecked(), emit_signal=False)
        GlobalConfig.set("geo_keep_index", self.geo_keep_idx_check.isChecked(), emit_signal=False)
        GlobalConfig.set("geo_force_square", self.geo_sq_check.isChecked(), emit_signal=False)
        GlobalConfig.set("geo_create_denoise", self.geo_denoise_check.isChecked(), emit_signal=False)
        GlobalConfig.set("geo_create_refine", self.geo_refine_check.isChecked(), emit_signal=False)

        # 5. Save Suffixes
        GlobalConfig.set("geo_suffix", self.suff_main.text(), emit_signal=False)
        GlobalConfig.set("geo_suffix_lrtem", self.suff_lr.text(), emit_signal=False)
        GlobalConfig.set("geo_suffix_hrtem", self.suff_hr.text(), emit_signal=False)
        GlobalConfig.set("geo_suffix_mask", self.suff_mask.text(), emit_signal=False)
        GlobalConfig.set("geo_suffix_mask_new", self.suff_new.text(), emit_signal=False)

        # Save Styles
        GlobalConfig.set("style_measure_color", self.style_meas_col.text(), emit_signal=False)
        GlobalConfig.set("style_measure_width", self.style_meas_w.value(), emit_signal=False)
        GlobalConfig.set("style_measure_font_size", self.style_meas_font.value(), emit_signal=False)
        GlobalConfig.set("style_crop_color", self.style_crop_col.text(), emit_signal=False)
        GlobalConfig.set("style_batch_box_color", self.style_batch_box_col.text(), emit_signal=False)
        GlobalConfig.set("style_batch_text_color", self.style_batch_txt_col.text(), emit_signal=False)
        GlobalConfig.set("style_batch_width", self.style_batch_w.value(), emit_signal=False)
        GlobalConfig.set("style_batch_font_size", self.style_batch_font.value(), emit_signal=False)

        # Save System
        GlobalConfig.set("sys_ram_threshold_gb", self.sys_ram.value(), emit_signal=False)
        GlobalConfig.set("sys_disk_warn_gb", self.sys_disk.value(), emit_signal=False)
        GlobalConfig.set("sys_move_threshold_gb", self.sys_move.value(), emit_signal=False)

        # Save Shortcuts
        for key, edit in self.key_edits.items():
            seq = edit.keySequence().toString()
            if seq: GlobalConfig.set(key, seq, emit_signal=False)
        
        # 最后统一触发热更新
        GlobalConfig.signals.config_updated.emit()

        super().accept()
        QMessageBox.information(self, "Settings Saved", "Configuration updated successfully.\nSome changes may require restarting actions or the app.")

    def _reset_defaults(self):
        if QMessageBox.question(self, "Reset", "Reset ALL settings to defaults?") == QMessageBox.Yes:
            try:
                if GlobalConfig._config_path.exists():
                    os.remove(GlobalConfig._config_path)
                QMessageBox.information(self, "Reset", "Settings reset. Please close and reopen Settings.")
                self.reject()
            except Exception as e:
                print(e)
"""
File: widgets/settings_widget.py
配置管理与设置界面 (JSON持久化版)
修改日志:
- [Fix Persistence] 弃用 QSettings，改用 JSON 文件保存配置，解决设置丢失问题。
- [Fix Bug] 确保 _browse_cache_dir 方法存在。
"""
from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QFormLayout, QTabWidget, 
                            QWidget, QKeySequenceEdit, QMessageBox, QSpinBox, 
                            QCheckBox, QGroupBox, QFileDialog)
from qtpy.QtGui import QKeySequence
import json
import os
from pathlib import Path

class GlobalConfig:
    """全局配置单例辅助类 (基于 JSON 文件)"""
    # 配置文件路径: 用户主目录/.napari_tem_config.json
    _config_path = Path.home() / ".napari_tem_config.json"
    
    DEFAULTS = {
        "shortcut_toggle_ui": "J",
        "shortcut_undo_drift": "Ctrl+Z",
        "shortcut_apply_crop": "Enter",
        "shortcut_switch_mode": "M",
        "drift_kernel": 11,
        "drift_workers": 8,
        "cache_dir": "", 
        "geo_suffix": "_origin",
        "geo_suffix_lrtem": "_lrtem",
        "geo_suffix_hrtem": "_hrtem",
        "geo_suffix_mask": "_mask",
        "geo_suffix_mask_new": "_mask_new",
        "geo_padding": 5,
        "geo_keep_index": True,
        "geo_force_square": True,
        "geo_enlarge": True
    }

    @classmethod
    def _load_config(cls):
        """内部方法：读取所有配置"""
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
        """内部方法：保存配置"""
        try:
            with open(cls._config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    @classmethod
    def get(cls, key):
        """获取配置值，如果不存在则返回默认值"""
        data = cls._load_config()
        val = data.get(key, cls.DEFAULTS.get(key, ""))
        
        # 处理布尔值转换 (为了兼容某些特定逻辑)
        if isinstance(val, str):
            if val.lower() == 'true': return True
            if val.lower() == 'false': return False
        return val

    @classmethod
    def set(cls, key, value):
        """设置并立即保存配置"""
        data = cls._load_config()
        data[key] = value
        cls._save_config(data)
    
    @classmethod
    def get_napari_shortcut(cls, key):
        raw = str(cls.get(key))
        # Napari shortcut format conversion
        napari_key = raw.replace("Ctrl+", "Control-") \
                        .replace("Shift+", "Shift-") \
                        .replace("Alt+", "Alt-") \
                        .replace("Meta+", "Meta-") \
                        .replace("Enter", "Return")
        if "Ctrl " in napari_key:
            napari_key = napari_key.replace("Ctrl ", "Control-")
        return napari_key

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Preferences & Shortcuts")
        self.resize(650, 600) 
        
        self.setStyleSheet("""
            QDialog { background-color: #262626; color: #E0E0E0; font-family: "Segoe UI", sans-serif; font-size: 10pt; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #BBB; padding: 8px 12px; border: 1px solid #444; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #444; color: white; font-weight: bold; }
            QLabel { color: #E0E0E0; }
            QLineEdit, QSpinBox, QKeySequenceEdit { background: #333; color: white; border: 1px solid #555; padding: 4px; border-radius: 3px; }
            QPushButton { background: #444; border: 1px solid #555; padding: 6px 12px; border-radius: 3px; color: white; }
            QPushButton:hover { background: #555; border-color: #777; }
            QGroupBox { border: 1px solid #555; margin-top: 10px; padding-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
        """)
        
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        tabs = QTabWidget()
        
        # === Tab 1: Shortcuts ===
        tab_shortcuts = QWidget()
        form_short = QFormLayout()
        
        self.key_edits = {}
        shortcuts_map = {
            "shortcut_toggle_ui": "Toggle Layer Controls (J)",
            "shortcut_undo_drift": "Undo Drift / Clear ROI (Ctrl+Z)",
            "shortcut_apply_crop": "Apply Crop / Export (Enter)",
            "shortcut_switch_mode": "Switch Draw/Select Mode (M)"
        }
        
        for key, label in shortcuts_map.items():
            val = str(GlobalConfig.get(key))
            edit = QKeySequenceEdit(QKeySequence(val))
            self.key_edits[key] = edit
            form_short.addRow(label, edit)
            
        tab_shortcuts.setLayout(form_short)
        tabs.addTab(tab_shortcuts, "⌨️ Shortcuts")
        
        # === Tab 2: Default Params ===
        tab_params = QWidget()
        layout_params = QVBoxLayout()
        
        # --- Cache Group ---
        g_cache = QGroupBox("Performance & Cache (Important for Large Data)")
        f_cache = QFormLayout()
        
        self.cache_dir_edit = QLineEdit(str(GlobalConfig.get("cache_dir")))
        self.cache_dir_edit.setPlaceholderText("System Default (e.g. C:/Temp)")
        self.cache_dir_edit.setToolTip("Select a drive with at least 200GB free space for processing large datasets.")
        
        btn_browse_cache = QPushButton("📂")
        btn_browse_cache.setFixedWidth(40)
        btn_browse_cache.clicked.connect(self._browse_cache_dir)
        
        h_cache = QHBoxLayout()
        h_cache.addWidget(self.cache_dir_edit)
        h_cache.addWidget(btn_browse_cache)
        
        f_cache.addRow("Cache Directory:", h_cache)
        lbl_hint = QLabel("Note: For 60GB+ images, use an SSD with >200GB free space.")
        lbl_hint.setStyleSheet("color: #AAA; font-size: 9pt; font-style: italic;")
        f_cache.addRow("", lbl_hint)
        
        g_cache.setLayout(f_cache)
        layout_params.addWidget(g_cache)
        
        # Group: Drift
        g_drift = QGroupBox("Drift Correction Defaults")
        f_drift = QFormLayout()
        self.drift_kernel_spin = QSpinBox(); self.drift_kernel_spin.setRange(3, 99); self.drift_kernel_spin.setSingleStep(2)
        self.drift_kernel_spin.setValue(int(GlobalConfig.get("drift_kernel")))
        
        self.drift_workers_spin = QSpinBox(); self.drift_workers_spin.setRange(1, 64)
        self.drift_workers_spin.setValue(int(GlobalConfig.get("drift_workers")))
        
        f_drift.addRow("Kernel Size:", self.drift_kernel_spin)
        f_drift.addRow("Max Workers:", self.drift_workers_spin)
        g_drift.setLayout(f_drift)
        layout_params.addWidget(g_drift)
        
        # Group: Geometry
        g_geo = QGroupBox("Batch Crop Defaults")
        f_geo = QFormLayout()
        
        self.geo_suffix_edit = QLineEdit(str(GlobalConfig.get("geo_suffix")))
        self.geo_padding_spin = QSpinBox(); self.geo_padding_spin.setRange(1, 12); self.geo_padding_spin.setValue(int(GlobalConfig.get("geo_padding")))
        self.geo_keep_idx_check = QCheckBox("Keep Original Frame Index"); self.geo_keep_idx_check.setChecked(bool(GlobalConfig.get("geo_keep_index")))
        self.geo_sq_check = QCheckBox("Force Square Crops"); self.geo_sq_check.setChecked(bool(GlobalConfig.get("geo_force_square")))
        self.geo_enl_check = QCheckBox("Enlarge Canvas on Rotate"); self.geo_enl_check.setChecked(bool(GlobalConfig.get("geo_enlarge")))

        f_geo.addRow("Main Suffix (Default):", self.geo_suffix_edit)
        f_geo.addRow("Zero Padding:", self.geo_padding_spin)
        f_geo.addRow("", self.geo_keep_idx_check)
        f_geo.addRow("", self.geo_sq_check)
        f_geo.addRow("", self.geo_enl_check)
        g_geo.setLayout(f_geo)
        layout_params.addWidget(g_geo)
        
        # Group: Folder Suffixes (Auxiliary)
        g_aux = QGroupBox("Auxiliary Folder Suffixes")
        f_aux = QFormLayout()
        self.suff_lr_edit = QLineEdit(str(GlobalConfig.get("geo_suffix_lrtem")))
        self.suff_hr_edit = QLineEdit(str(GlobalConfig.get("geo_suffix_hrtem")))
        self.suff_mask_edit = QLineEdit(str(GlobalConfig.get("geo_suffix_mask")))
        self.suff_new_edit = QLineEdit(str(GlobalConfig.get("geo_suffix_mask_new")))
        
        f_aux.addRow("Low Res:", self.suff_lr_edit)
        f_aux.addRow("High Res:", self.suff_hr_edit)
        f_aux.addRow("Mask:", self.suff_mask_edit)
        f_aux.addRow("Mask (New):", self.suff_new_edit)
        g_aux.setLayout(f_aux)
        layout_params.addWidget(g_aux)

        layout_params.addStretch()
        tab_params.setLayout(layout_params)
        tabs.addTab(tab_params, "💾 Default Parameters")
        
        # === Tab 3: Reset ===
        tab_adv = QWidget()
        l_adv = QVBoxLayout()
        btn_reset = QPushButton("⚠️ Reset All Settings to Factory Defaults")
        btn_reset.setStyleSheet("background-color: #8B0000; color: white;")
        btn_reset.clicked.connect(self._reset_defaults)
        l_adv.addWidget(btn_reset)
        l_adv.addStretch()
        tab_adv.setLayout(l_adv)
        tabs.addTab(tab_adv, "🛠️ Advanced")

        layout.addWidget(tabs)
        
        # Buttons
        h_btn = QHBoxLayout()
        btn_save = QPushButton("Save & Close")
        btn_save.setStyleSheet("background-color: #2196F3; font-weight: bold;")
        btn_save.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        h_btn.addStretch()
        h_btn.addWidget(btn_cancel)
        h_btn.addWidget(btn_save)
        layout.addLayout(h_btn)
        
        self.setLayout(layout)

    def _browse_cache_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Cache Directory (SSD Recommended)")
        if d:
            self.cache_dir_edit.setText(d)

    def accept(self):
        # Save Shortcuts
        for key, edit in self.key_edits.items():
            seq = edit.keySequence().toString()
            if seq: GlobalConfig.set(key, seq)
            
        # Save Cache Dir
        GlobalConfig.set("cache_dir", self.cache_dir_edit.text())

        # Save Drift Params
        GlobalConfig.set("drift_kernel", self.drift_kernel_spin.value())
        GlobalConfig.set("drift_workers", self.drift_workers_spin.value())
        
        # Save Geometry Params
        GlobalConfig.set("geo_suffix", self.geo_suffix_edit.text())
        GlobalConfig.set("geo_padding", self.geo_padding_spin.value())
        GlobalConfig.set("geo_keep_index", self.geo_keep_idx_check.isChecked())
        GlobalConfig.set("geo_force_square", self.geo_sq_check.isChecked())
        GlobalConfig.set("geo_enlarge", self.geo_enl_check.isChecked())
        
        # Save Aux Suffixes
        GlobalConfig.set("geo_suffix_lrtem", self.suff_lr_edit.text())
        GlobalConfig.set("geo_suffix_hrtem", self.suff_hr_edit.text())
        GlobalConfig.set("geo_suffix_mask", self.suff_mask_edit.text())
        GlobalConfig.set("geo_suffix_mask_new", self.suff_new_edit.text())
        
        super().accept()
        # 提示用户
        QMessageBox.information(self, "Settings Saved", f"Configuration saved to:\n{GlobalConfig._config_path}")

    def _reset_defaults(self):
        if QMessageBox.question(self, "Confirm", "Reset ALL settings to factory defaults?") == QMessageBox.Yes:
            # 删除配置文件
            try:
                if GlobalConfig._config_path.exists():
                    os.remove(GlobalConfig._config_path)
            except Exception as e:
                print(f"Error resetting: {e}")
            self.reject()
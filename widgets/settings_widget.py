"""
File: widgets/settings_widget.py
配置管理与设置界面 (增强版)
修改日志:
- [Req 3] 新增后缀配置 (lrtem, hrtem, mask, mask_new)
"""
from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QFormLayout, QTabWidget, 
                            QWidget, QKeySequenceEdit, QMessageBox, QSpinBox, 
                            QCheckBox, QGroupBox)
from qtpy.QtCore import QSettings
from qtpy.QtGui import QKeySequence

class GlobalConfig:
    """全局配置单例辅助类"""
    _settings = QSettings("NapariUser", "GlobalConfig")
    
    DEFAULTS = {
        "shortcut_toggle_ui": "J",
        "shortcut_undo_drift": "Ctrl+Z",
        "shortcut_apply_crop": "Enter",
        "shortcut_switch_mode": "M",
        "drift_kernel": 11,
        "drift_workers": 8,
        # Geometry Defaults
        "geo_suffix": "_origin",      # 主后缀默认值
        "geo_suffix_lrtem": "_lrtem", # [New]
        "geo_suffix_hrtem": "_hrtem", # [New]
        "geo_suffix_mask": "_mask",   # [New]
        "geo_suffix_mask_new": "_mask_new", # [New]
        "geo_padding": 5,
        "geo_keep_index": True,
        "geo_force_square": True,
        "geo_enlarge": True
    }

    @classmethod
    def get(cls, key):
        val = cls._settings.value(key, cls.DEFAULTS.get(key, ""))
        if str(val).lower() == 'true': return True
        if str(val).lower() == 'false': return False
        return val

    @classmethod
    def set(cls, key, value):
        cls._settings.setValue(key, value)
    
    @classmethod
    def get_napari_shortcut(cls, key):
        raw = cls.get(key)
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
        self.resize(600, 550) 
        
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
            val = GlobalConfig.get(key)
            edit = QKeySequenceEdit(QKeySequence(val))
            self.key_edits[key] = edit
            form_short.addRow(label, edit)
            
        tab_shortcuts.setLayout(form_short)
        tabs.addTab(tab_shortcuts, "⌨️ Shortcuts")
        
        # === Tab 2: Default Params ===
        tab_params = QWidget()
        layout_params = QVBoxLayout()
        
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

    def accept(self):
        # Save Shortcuts
        for key, edit in self.key_edits.items():
            seq = edit.keySequence().toString()
            if seq: GlobalConfig.set(key, seq)
            
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
        QMessageBox.information(self, "Saved", "Settings saved successfully!")

    def _reset_defaults(self):
        if QMessageBox.question(self, "Confirm", "Reset ALL settings to factory defaults?") == QMessageBox.Yes:
            GlobalConfig._settings.clear()
            self.reject()
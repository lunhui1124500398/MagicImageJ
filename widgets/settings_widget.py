"""
File: widgets/settings_widget.py
"""
from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QFormLayout, QTabWidget, 
                            QWidget, QKeySequenceEdit, QMessageBox, QSpinBox, 
                            QCheckBox, QGroupBox, QFileDialog, QDoubleSpinBox,
                            QColorDialog, QScrollArea, QComboBox)
from qtpy.QtGui import QKeySequence, QColor
from qtpy.QtCore import QObject, Signal
import json
import os
from pathlib import Path

# 国际化 (i18n) 翻译字典
# =============================================================================
TRANS_CN = {
    # Main / Tabs
    "Workflow Tools": "工作流工具箱",
    "Measure": "测量",
    "Import": "导入数据",
    "Drift Correction": "漂移矫正",
    "Geometry": "几何变换",
    "Enhancement": "图像增强",
    "Annotation": "标注工具",
    "Export": "导出数据",
    "Settings": "设置",
    "Global Settings & Shortcuts": "全局设置与快捷键",
    "Settings Saved": "设置已保存",
    "General": "常规",
    "Visual Styles": "视觉样式",
    "System Thresholds": "系统阈值",
    
    # Common
    "Ready.": "就绪",
    "Ready": "就绪",
    "Cancel": "取消",
    "Apply": "应用",
    "Reset": "重置",
    "Workers:": "线程数:",
    "Target Layer:": "目标图层:",
    "Source Layer:": "源图层:",
    "Refresh Layers": "刷新图层",
    "Refresh All Layers": "刷新所有图层",
    "Browse": "浏览",
    "Save & Close": "保存并关闭",
    "Language": "语言 (Language)",
    "Restart required to apply language changes fully.": "语言更改需要重启部分界面才能完全生效。",
    
    # Import Widget
    "1. Data Source": "1. 数据源",
    "2. Scan Metadata": "2. 扫描元数据",
    "3. Archive Configuration": "3. 归档配置",
    "4. Action": "4. 执行操作",
    "5. Load to Viewer": "5. 加载至视图",
    "Browse Folder": "浏览文件夹",
    "Calc Dose": "计算剂量",
    "Create Archive Folder": "创建归档文件夹",
    "Load Images": "加载图像",
    "Show Result Popup": "显示结果弹窗",
    "Pick file to auto-set.": "选择文件以自动设置",
    'Img:': "图像:",
    "Frame Index (-1 for Middle). Pick file to auto-set.": "帧索引 (-1 表示中间帧)。选择文件以自动设置。",
    "Pick a specific .dm4 file from the folder to calculate dose": "从文件夹中选择特定的 .dm4 文件以计算剂量",
    "Select folder to extract date, mag, pixel size...": "选择文件夹以提取日期、放大倍数、像素大小...",
    "Solv (Default: Water)": "溶剂 (默认: Water)",
    "Sub (e.g. CRY2)": "样品 (例如 CRY2)",
    "Avg Win:": "平均窗口:",
    "Gaus σ:": "高斯 σ:",
    "Additional Info (Saved to txt):": "附加信息 (保存至 txt):",
    "Mag:": "放大倍数:",
    "OL# (0-4):": "物镜光阑 (0-4):",
    "ID:": "编号:",
    "Sub:": "样品:",
    "Solv:": "溶剂:",
    "Show a popup message with size and mode details after archiving.": "归档后显示包含大小和模式详情的弹窗消息。",
    "Bit Depth:": "位深:",
    "Import & Archive": "导入与归档",
    "Preview Folder Name:": "预览文件夹名称:",
    
    # Drift Widget
    "Step 1: Draw ROI": "第一步：绘制感兴趣区域 (ROI)",
    "Step 2: Preview Drift": "第二步：预览漂移",
    "Step 3: Apply Correction": "第三步：应用矫正",
    "Draw ROI (Add Shapes Layer)": "绘制 ROI (添加图形层)",
    "Auto Preview (Calc & Apply)": "自动预览 (计算并应用)",
    "Manual Recalc": "手动重算",
    "Apply / Commit": "应用 / 提交",
    "Reset / Clear ROI": "重置 / 清除 ROI",
    "Template Frame:": "模板帧:",
    "Calculate and show corrected result immediately after drawing ROI.": "绘制 ROI 后立即计算并显示矫正结果。",
    "Tip: Press 'Crtl + Z' on corrected layer to Undo/Retry.": "提示: 在矫正图层上按 'Crtl + Z' 撤销/重试。",
    "ROI cleared.": "ROI 已清除。",
    "Mode: Draw Rectangle (Auto-Preview ON)": "模式：绘制矩形 (自动预览已开启)",
    'ROI updated. Auto-calculating...': "ROI 已更新。正在自动计算...",
    'ROI updated. Click Recalculate.': "ROI 已更新。请点击重算。",
    'Draw ROI first.': "请先绘制 ROI。",
    'Invalid ROI (too small or out of bounds).': "无效的 ROI（过小或超出边界）。",
    'Calculating drift...': "正在计算漂移...",
    'Auto-applying correction...': "正在自动应用矫正...",
    'Calculated. Click Apply to see result.': "计算完成。点击应用以查看结果。",
    'Applying correction...': "正在应用矫正...",
    'Undone. Adjust ROI and try again.': "已撤销。请调整 ROI 并重试。",


    # Geometry Widget
    "Geometry": "几何变换",
    "Geometry & Batch Extraction (Multi-ROI)": "几何变换与批量提取",
    "1. Rotation (Horizon)": "1. 旋转 (水平校正)",
    "2. Simple Crop (Single)": "2. 简单裁剪 (单图)",
    "3. Batch Extraction (Multi-ROI)": "3. 批量提取 (多区域)",
    "Draw Horizon Line": "绘制水平线",
    "Apply Rotation": "应用旋转",
    "Enlarge Canvas (Fit All)": "扩大画布 (保留全图)",
    "Draw Rect": "绘制矩形",
    "Apply Crop (New Layer)": "应用裁剪 (新图层)",
    "Sync Select": "同步选择",
    "Peek Data (Hold)": "按住以查看数据",
    "Lock View Layer (Prevent auto-switching)": "锁定视图层 (防止自动切换)",
    "Gen Denoise Folders": "生成去噪文件夹",
    "Gen Refine Folder": "生成精修文件夹",
    "Force Square Crops": "强制正方形裁剪",
    "Start Draw": "开始绘制",
    "Adjust": "调整",
    "Export Crops & Map": "导出裁剪图与概览",
    "Keep Original Frame Index": "保留原始帧序号",
    "Set for Selected": "设置选中项",
    "Target:":"目标:",
    "Draw a line to define horizon:": "绘制一条线以定义水平线:",
    "Flip Horz": "水平翻转",
    "Flip Vert": "垂直翻转",
    "Angle:": "角度:",
    "Line-Calc": "水平角计算",
    "Draw a line to calculate angle": "绘制一条线以计算角度",
    "Expand image size to fit rotated content without cropping": "扩大图像尺寸以适应旋转内容而不裁剪",
    "Data Layer (Crop Source):": "数据层 (裁剪源):",
    "View Layer (Reference):": "视图层 (参考):",
    "Set View Layer same as Data Layer": "将视图层设置为与数据层相同",
    "Hold to temporarily show Data Layer to check alignment": "按住以临时显示数据层以检查对齐",
    "Date:": "日期:",
    "Date prefix (YYYYMMDD). Loaded from Archive or Today.": "日期前缀 (YYYYMMDD)。从归档或今天加载。",
    "Sub:": "样品:",
    "Suffix:": "后缀:",
    "Gen Denoise Folders": "生成去噪文件夹",
    "Creates empty folders with Main Suffix + Configured Suffix (e.g. _contrasted_lrtem)": "创建带有主后缀 + 配置后缀的空文件夹 (例如 _contrasted_lrtem)",
    "Gen Refine Folder": "生成精修文件夹",
    "Creates empty folder with Main Suffix + Configured Suffix (e.g. _contrasted_mask_new)": "创建带有主后缀 + 配置后缀的空文件夹 (例如 _contrasted_mask_new)",
    "Frame Filter:": "帧过滤器:",
    "All (Default) or 0-10, 15...": "所有 (默认) 或 0-10, 15...",
    "Leave empty for All frames.\nOr use: 0-10, 15, 20-25": "留空表示所有帧。\n或者使用: 0-10, 15, 20-25",
    "Apply the text in the box to the CURRENTLY SELECTED ROI only.": "仅将框中的文本应用于当前选定的 ROI。",
    "Geo_Padding:": "填充:",
    "Start Draw": "开始绘制",
    "Adjust": "调整",
    "Export Crops & Map": "导出裁剪图与概览",
    "Export Format:": "导出格式:",
    "Show Warning when Clearing Overlays": "清除覆盖层时显示警告",
    'Save ROIs':"保存ROIs",
    "Load ROIs":"加载ROIs",
    "Save ROI coordinates + Reference Map":"保存 ROI 坐标 + 参考图",
    "Load ROI JSON & Auto-load Image":"加载 ROI JSON & 自动加载图像",
    "Saving ROI JSON.\nDo you also want to save the reference image(s)?":"保存 ROI JSON。\n是否也要保存参考图像？",
    "Which layer(s) should be saved as reference?":"应保存为参考的图层是？",
    "TIFF Stack":"TIFF 堆栈",
    "PNG Sequence":"PNG 序列",
    "Skip Images":"跳过图像",
    "Save Reference Images?":"是否保存参考图像？",
    "Select Layers":"选择图层",
    "Save Both":"保存两者",
    "Data Only":"仅数据",
    "View Only":"仅视图",
    "Saved JSON":"成功保存到json文件",
    "Confirm Import Sources":"确认导入数据源",
    "Image Source Detection Report":"图像源自动检测报告",
    "View Layer":"参考层(看到的)",
    "Data Layer":"数据层(扣取的)",
    "Same as View Layer":"与参考层相同",
    "Do you want to load these images?": "您想要导入这些图片吗?",
    "Found":"找到",
    'Not Found (Auto-detection failed)':"未找到(自动探测失败)",
    "Auto Load Detected":"自动导入检测图层",
    "Manual Select":"人工选择",
    "Skip Images (ROIs Only)":"跳过图层(仅导入Rois)",

    
    # Enhance Widget
    "1. Filters": "1. 滤波器",
    "2. Contrast & Brightness (Post-Process)": "2. 对比度与亮度 (后处理)",
    "Gaussian Blur": "高斯模糊",
    "Roll Avg": "滚动平均",
    "Run Filters (Create Layer)": "运行滤波 (创建图层)",
    "Apply (Burn to New Layer)": "应用 (烧录到新图层)",
    'Sigma:': "标准差:",
    'win': "窗口数",
    'Kernel:': "核数目:",
    'Max:': "最大值:",
    'Min:': "最小值:",
    'Auto': "自动",
    'Reset': "重置",
    'Win:': "窗口数:",

    
    # Annotation Widget
    'Annotation (Lazy & Smart)': "智能懒加载标注工具",
    "Scale Bar": "比例尺",
    "Label": "标签",
    "Enable": "启用",
    "Appearance": "外观样式",
    "Initialize / Reset Preview": "初始化 / 重置预览",
    "Burn-in to New Layer": "烧录至新图层",
    "Clear All": "清除所有",
    "Auto BG Size (Smart)": "自动背景尺寸 (智能)",
    "Show Background": "显示背景",
    "Quick Snap": "快速定位",
    "Scale Ratio": "比例尺比例",
    "Appearance": "外观",
    "Bar Length:": "刻度长度:",
    "Auto BG Size (Smart)": "自动背景尺寸 (智能)",
    "Bar Thickness:": "刻度厚度:",
    "Font Size (pt):": "字体大小 (pt):",
    "Padding:": "内边距:",
    "BG Height (px):": "背景高度 (px):",
    "Text/Bar Color": "文字/刻度颜色",
    "BG Color": "背景颜色",
    "Show Background": "显示背景",
    "BG Opacity:": "背景不透明度:",
    "Enable Label": "启用标签",
    'Format': "标签格式",
    'Custom Format:': "自定义格式:",
    'Text Color': "文字颜色",
    "Start:": "起始值",
    "Step:": "步长",
    "Top-Left": "左上",
    "Top-Right": "右上",
    "Bottom-Left": "左下",
    "Bottom-Right": "右下",
    "Source:": "源图层:",
    
    # Export Widget
    "Frame Range": "帧范围",
    "All Frames": "所有帧",
    "Range": "指定范围",
    "Export Format": "导出格式",
    "Video Options": "视频选项",
    "Start Export": "开始导出",
    "Overlay Annotations": "叠加标注",
    "Video (.mp4, .avi)": "视频 (.mp4, .avi)",
    "TIFF Stack (.tiff)": "TIFF 堆栈 (.tiff)",
    "Image Sequence (Folder)": "图像序列 (文件夹)",
    'FPS:': "帧率 (FPS):",
    'Codec:': "编码器:",
    'Quality (0-100):': "质量 (0-100):",
    "Sequence Options": "序列选项",
    'Format:': "格式:",
    'Pattern:': "命名模式:",
    "Scale Bar": "比例尺",
    "Timestamp": "时间戳",
    "No path selected": "未选择路径",
    "Styles loaded from Annotation Tab": "样式从标注标签加载",
    
    # Settings
    "Preferences & Configuration": "偏好设置",
    "Reset to Defaults": "重置为默认值",
    "General": "常规",
    "Visual Styles": "视觉样式",
    "System Thresholds": "系统阈值",
    "Shortcuts": "快捷键",
    "Language": "语言 (Language)",
    "Algorithm Defaults": "算法默认值",
    "Image Enhancement Defaults": "图像增强默认值",
    "Generated Folder Suffixes": "生成文件夹后缀",
    "Cache Location": "缓存位置",
    "Restart required to apply language changes fully.": "语言更改需要重启部分界面才能完全生效。",

    # Others
    "Algorithm Defaults": "算法默认值",
    "Auto-Calculate (Preview on ROI Draw)": "自动计算 (绘制 ROI 时预览)",
    "Enable Gaussian Blur by Default": "默认启用高斯模糊",
    "Enable Rolling Average by Default": "默认启用滚动平均",
    "Gaussian Sigma:": "高斯标准差:",
    "Roll Avg Window:": "滚动平均窗口:",
    "Enhance Workers:": "增强线程数:",
    "Enlarge Canvas on Rotate": "旋转时扩大画布",
    "Create Denoise Folders (LR/HR)": "创建去噪文件夹 (LR/HR)",
    "Create Refine Folders (Mask)": "创建精修文件夹 (Mask)",
    "Main Suffix:": "主后缀:",
    "LR Suffix:": "LR 后缀:",
    "HR Suffix:": "HR 后缀:",
    "Mask Suffix:": "Mask 后缀:",
    "New Mask Suffix:": "新 Mask 后缀:",
    "RAM vs Disk Limit:": "RAM 与磁盘限制:",
    "Disk Space Warning:": "磁盘空间警告:",
    "Move vs Copy Limit:": "移动与复制限制:",
    "Generated Folder Suffixes": "生成文件夹后缀",
    "Cache Location": "缓存位置",
    "Resource Thresholds": "资源阈值",
    "Measure Tool": "测量工具",
    "Measure": "测量",
    "Batch Crop": "批量裁剪",
    "Simple Crop": "简单裁剪",
    "Line Color:": "线条颜色:",
    "Box Color:": "边框颜色:",
    "Text Color:": "文字颜色:",
    "Width:": "宽度:",
    "Font Size:": "字体大小:",
    "Path:": "路径:",
    "Configuration updated successfully.\nSome changes may require restarting actions or the app.": "配置更新成功。\n某些更改可能需要重新启动操作或应用程序。",
    "Reset": "重置",
    "Reset ALL settings to defaults?": "是否将所有设置重置为默认值？",
    "Settings reset. Please close and reopen Settings.": "设置已重置。请关闭并重新打开设置。",
    "Drift Kernel:": "漂移核数目:",
    "Max Workers:": "最大线程数:",
    "Toggle Layer Controls": "切换图层控件",
    "Undo / Clear ROI": "撤销/清除 ROI",
    "Apply Crop / Export": "应用裁剪/导出",
    "Switch Draw/Select Mode": "切换绘制/选择模式",
    "If an array exceeds this size, create it on Disk (memmap).": "如果数组超过此大小，则在磁盘上创建它(memmap)",
    "Warn if cache folder usage exceeds this size.": "如果缓存文件夹使用量超过此大小则发出警告",
    "If raw data folder > this size, MOVE instead of COPY during archive.": "如果原始数据文件夹大于此大小，则在归档过程中移动而不是复制",

    # === 新增翻译 ===
    "Memory Warning": "内存警告",
    "Pagefile Warning Body": "Allocating {0:.1f} GB. Available RAM: {1:.1f} GB.\n\nSince the data hasn't been flushed to disk yet, this operation exceeds physical RAM and may force Windows to expand 'pagefile.sys', occupying C: drive space.\n\nContinue?",
    # 注意：为了让中文显示更友好，我们在代码里用 tr() 获取这个key时，会返回下面的中文
    # 但由于字典key是英文原文，我们需要把中文翻译写在value里。
    # 这里有点特殊，因为正文带有格式化参数 {0} {1}，建议直接用英文做Key，中文做Value
    
    "Allocating {0:.1f} GB. Available RAM: {1:.1f} GB.\n\nThis operation exceeds physical RAM and may cause Windows to expand 'pagefile.sys' on C: drive.\n\nContinue?": 
    "即将分配 {0:.1f} GB。当前可用内存: {1:.1f} GB。\n\n此操作超出物理内存，且临时文件尚未完全写入磁盘，可能导致 Windows 强制扩大 C 盘的虚拟内存文件 (pagefile.sys)。\n\n是否继续？",

    "Memmap Warning Threshold:": "内存分配预警阈值:",
    "Show warning if single allocation exceeds this size (Check RAM vs C: drive).": "若单次分配超过此大小则预警 (可能物理内存撑不住，殃及C盘)。",
}

def tr(text):
    """Translation helper function"""
    lang = GlobalConfig.get("language")
    if lang == "zh_CN":
        return TRANS_CN.get(text, text)
    return text

class ConfigSignals(QObject):
    config_updated = Signal()

class GlobalConfig:
    """全局配置单例辅助类 (基于 JSON 文件)"""
    _config_path = Path.home() / ".napari_tem_config.json"

    signals = ConfigSignals()
    
    # === 1. 扩充默认配置 ===
    DEFAULTS = {
        "language": "en",
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
        "show_clear_warning": True,
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
        "sys_mem_warn_gb":4.0,
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
        self.setWindowTitle(f"⚙️ {tr('Preferences & Configuration')}")
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
        tabs.addTab(self._create_general_tab(), f"💾 {tr('General')}")
        
        # === Tab 2: Visual Styles (New) ===
        tabs.addTab(self._create_styles_tab(), f"🎨 {tr('Visual Styles')}")

        # === Tab 3: System Thresholds (New) ===
        tabs.addTab(self._create_system_tab(), f"⚙️ {tr('System Thresholds')}")

        # === Tab 4: Shortcuts (Existing) ===
        tabs.addTab(self._create_shortcuts_tab(), f"⌨️ {tr('Shortcuts')}")

        layout.addWidget(tabs)

        # Buttons
        h_btn = QHBoxLayout()
        btn_reset = QPushButton(f"⚠️ {tr('Reset to Defaults')}")
        btn_reset.setStyleSheet("background-color: #8B0000;")
        btn_reset.clicked.connect(self._reset_defaults)
        
        btn_save = QPushButton(tr("Save & Close"))
        btn_save.setStyleSheet("background-color: #2E7D32; font-weight: bold;")
        btn_save.clicked.connect(self.accept)
        
        btn_cancel = QPushButton(tr("Cancel"))
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

        # Language Selection
        g_lang = QGroupBox(tr("Language"))
        f_lang = QFormLayout()
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("简体中文", "zh_CN")
        # Set current selection
        curr_lang = GlobalConfig.get("language")
        idx = self.lang_combo.findData(curr_lang)
        if idx >= 0: self.lang_combo.setCurrentIndex(idx)
        
        f_lang.addRow(tr("Language") + ":", self.lang_combo)
        l_hint = QLabel(tr("Restart required to apply language changes fully."))
        l_hint.setStyleSheet("color: #FFC107; font-style: italic;")
        f_lang.addRow("", l_hint)
        g_lang.setLayout(f_lang)
        l.addWidget(g_lang)
        
        # Cache
        g_cache = QGroupBox(tr("Cache Location"))
        f_cache = QFormLayout()
        self.cache_dir_edit = QLineEdit(str(GlobalConfig.get("cache_dir")))
        btn_browse = QPushButton("📂")
        btn_browse.clicked.connect(lambda: self.cache_dir_edit.setText(QFileDialog.getExistingDirectory(self, "Cache Dir")))
        h = QHBoxLayout(); h.addWidget(self.cache_dir_edit); h.addWidget(btn_browse)
        f_cache.addRow(tr("Path:"), h)
        g_cache.setLayout(f_cache); l.addWidget(g_cache)

        # Algorithm
        g_algo = QGroupBox(tr("Algorithm Defaults"))
        f_algo = QFormLayout()
        self.drift_auto_check = QCheckBox(tr("Auto-Calculate (Preview on ROI Draw)"))
        self.drift_auto_check.setChecked(bool(GlobalConfig.get("drift_auto_calc")))
        self.drift_k_spin = QSpinBox(); self.drift_k_spin.setRange(3, 99); self.drift_k_spin.setValue(int(GlobalConfig.get("drift_kernel")))
        self.drift_w_spin = QSpinBox(); self.drift_w_spin.setRange(1, 64); self.drift_w_spin.setValue(int(GlobalConfig.get("drift_workers")))
        f_algo.addRow(self.drift_auto_check)
        f_algo.addRow(tr("Drift Kernel:"), self.drift_k_spin)
        f_algo.addRow(tr("Max Workers:"), self.drift_w_spin)
        g_algo.setLayout(f_algo); l.addWidget(g_algo)

        g_ui = QGroupBox(tr("Interaction Settings"))
        f_ui = QFormLayout()
        self.warn_clear_check = QCheckBox(tr("Show Warning when Clearing Overlays"))
        self.warn_clear_check.setChecked(bool(GlobalConfig.get("show_clear_warning")))
        f_ui.addRow(self.warn_clear_check)
        g_ui.setLayout(f_ui)
        l.addWidget(g_ui)

        g_enh = QGroupBox(tr("Image Enhancement Defaults"))
        f_enh = QFormLayout()
        self.enh_gaus_check = QCheckBox(tr("Enable Gaussian Blur by Default"))
        self.enh_gaus_check.setChecked(bool(GlobalConfig.get("enh_use_gaussian")))
        
        self.enh_avg_check = QCheckBox(tr("Enable Rolling Average by Default"))
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
        f_enh.addRow(tr("Gaussian Sigma:"), self.enh_sigma_spin)
        f_enh.addRow(self.enh_avg_check)
        f_enh.addRow(tr("Roll Avg Window:"), self.enh_win_spin)
        f_enh.addRow(tr("Enhance Workers:"), self.enh_work_spin)
        
        g_enh.setLayout(f_enh)
        l.addWidget(g_enh)

        # Geo Suffixes
        g_geo = QGroupBox(tr("Generated Folder Suffixes"))
        f_geo = QFormLayout()
        self.geo_enl_check = QCheckBox(tr("Enlarge Canvas on Rotate"))
        self.geo_enl_check.setChecked(bool(GlobalConfig.get("geo_enlarge")))
        self.geo_keep_idx_check = QCheckBox(tr("Keep Original Frame Index"))
        self.geo_keep_idx_check.setChecked(bool(GlobalConfig.get("geo_keep_index")))
        self.geo_sq_check = QCheckBox(tr("Force Square Crops"))
        self.geo_sq_check.setChecked(bool(GlobalConfig.get("geo_force_square")))
        self.geo_denoise_check = QCheckBox(tr("Create Denoise Folders (LR/HR)"))
        self.geo_denoise_check.setChecked(bool(GlobalConfig.get("geo_create_denoise")))
        self.geo_refine_check = QCheckBox(tr("Create Refine Folders (Mask)"))
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
        f_geo.addRow(tr("Main Suffix:"), self.suff_main)
        f_geo.addRow(tr("LR Suffix:"), self.suff_lr)
        f_geo.addRow(tr("HR Suffix:"), self.suff_hr)
        f_geo.addRow(tr("Mask Suffix:"), self.suff_mask)
        f_geo.addRow(tr("New Mask Suffix:"), self.suff_new)
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
        g_meas = QGroupBox(f"📏 {tr('Measure Tool')}")
        l_meas = QVBoxLayout()
        self.style_meas_col = add_color_row(l_meas, tr("Line Color:"), "style_measure_color")
        
        h_m = QHBoxLayout()
        self.style_meas_w = QSpinBox(); self.style_meas_w.setRange(1, 20); self.style_meas_w.setValue(int(GlobalConfig.get("style_measure_width")))
        self.style_meas_font = QSpinBox(); self.style_meas_font.setRange(5, 50); self.style_meas_font.setValue(int(GlobalConfig.get("style_measure_font_size")))
        h_m.addWidget(QLabel(tr("Width:"))); h_m.addWidget(self.style_meas_w)
        h_m.addWidget(QLabel(tr("Font Size:"))); h_m.addWidget(self.style_meas_font)
        l_meas.addLayout(h_m)
        g_meas.setLayout(l_meas); l.addWidget(g_meas)

        # 2. Simple Crop
        g_simp = QGroupBox(f"✂️ {tr('Simple Crop')}")
        l_simp = QVBoxLayout()
        self.style_crop_col = add_color_row(l_simp, tr("Box Color:"), "style_crop_color")
        g_simp.setLayout(l_simp); l.addWidget(g_simp)

        # 3. Batch Crop
        g_batch = QGroupBox(f"📦 {tr('Batch Crop')}")
        l_batch = QVBoxLayout()
        self.style_batch_box_col = add_color_row(l_batch, tr("Box Color:"), "style_batch_box_color")
        self.style_batch_txt_col = add_color_row(l_batch, tr("Text Color:"), "style_batch_text_color")
        
        h_b = QHBoxLayout()
        self.style_batch_w = QSpinBox(); self.style_batch_w.setRange(1, 20); self.style_batch_w.setValue(int(GlobalConfig.get("style_batch_width")))
        self.style_batch_font = QSpinBox(); self.style_batch_font.setRange(5, 50); self.style_batch_font.setValue(int(GlobalConfig.get("style_batch_font_size")))
        h_b.addWidget(QLabel(tr("Width:"))); h_b.addWidget(self.style_batch_w)
        h_b.addWidget(QLabel(tr("Font Size:"))); h_b.addWidget(self.style_batch_font)
        l_batch.addLayout(h_b)
        g_batch.setLayout(l_batch); l.addWidget(g_batch)

        l.addStretch(); w.setLayout(l)
        return w

    def _create_system_tab(self):
        w = QWidget()
        l = QVBoxLayout()
        
        g_res = QGroupBox(tr("Resource Thresholds"))
        form = QFormLayout()
        
        self.sys_ram = QDoubleSpinBox(); self.sys_ram.setRange(0.1, 1024); self.sys_ram.setSuffix(" GB"); self.sys_ram.setValue(float(GlobalConfig.get("sys_ram_threshold_gb")))
        self.sys_disk = QDoubleSpinBox(); self.sys_disk.setRange(0.1, 10240); self.sys_disk.setSuffix(" GB"); self.sys_disk.setValue(float(GlobalConfig.get("sys_disk_warn_gb")))
        self.sys_move = QDoubleSpinBox(); self.sys_move.setRange(0.1, 10240); self.sys_move.setSuffix(" GB"); self.sys_move.setValue(float(GlobalConfig.get("sys_move_threshold_gb")))
        self.sys_mem_warn = QDoubleSpinBox()
        self.sys_mem_warn.setRange(0, 1024)
        self.sys_mem_warn.setSuffix(" GB")
        self.sys_mem_warn.setSingleStep(1)
        self.sys_mem_warn.setValue(float(GlobalConfig.get("sys_mem_warn_gb")))

        form.addRow(tr("RAM vs Disk Limit:"), self.sys_ram)
        l_hint1 = QLabel(tr("If an array exceeds this size, create it on Disk (memmap)."))
        l_hint1.setStyleSheet("color: gray; font-size: 9pt; margin-bottom: 10px;")
        form.addRow("", l_hint1)

        form.addRow(tr("Memmap Warning Threshold:"), self.sys_mem_warn)
        l_hint_new = QLabel(tr("Show warning if single allocation exceeds this size (Check RAM vs C: drive)."))
        l_hint_new.setStyleSheet("color: gray; font-size: 9pt; margin-bottom: 10px;")
        form.addRow("", l_hint_new)

        form.addRow(tr("Disk Space Warning:"), self.sys_disk)
        l_hint2 = QLabel(tr("Warn if cache folder usage exceeds this size."))
        l_hint2.setStyleSheet("color: gray; font-size: 9pt; margin-bottom: 10px;")
        form.addRow("", l_hint2)

        form.addRow(tr("Move vs Copy Limit:"), self.sys_move)
        l_hint3 = QLabel(tr("If raw data folder > this size, MOVE instead of COPY during archive."))
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
            "shortcut_toggle_ui": tr("Toggle Layer Controls"),
            "shortcut_undo_drift": tr("Undo / Clear ROI"),
            "shortcut_apply_crop": tr("Apply Crop / Export"),
            "shortcut_switch_mode": tr("Switch Draw/Select Mode")
        }
        for key, label in shortcuts_map.items():
            val = str(GlobalConfig.get(key))
            edit = QKeySequenceEdit(QKeySequence(val))
            self.key_edits[key] = edit
            f.addRow(label, edit)
        w.setLayout(f)
        return w

    def accept(self):
        if hasattr(self, 'lang_combo'):
            # 获取当前选中的数据 (例如 "zh_CN")
            selected_lang = self.lang_combo.currentData()
            # 强制保存到配置文件
            GlobalConfig.set("language", selected_lang, emit_signal=False)

        # 1. Save Cache
        GlobalConfig.set("cache_dir", self.cache_dir_edit.text(), emit_signal=False)
        
        # 2. Save Drift
        if hasattr(self, 'drift_auto_check'):
             GlobalConfig.set("drift_auto_calc", self.drift_auto_check.isChecked(), emit_signal=False)
        GlobalConfig.set("drift_auto_calc", self.drift_auto_check.isChecked(), emit_signal=False) # [New]
        GlobalConfig.set("drift_kernel", self.drift_k_spin.value(), emit_signal=False)
        GlobalConfig.set("drift_workers", self.drift_w_spin.value(), emit_signal=False)

        if hasattr(self, 'warn_clear_check'):
            GlobalConfig.set("show_clear_warning", self.warn_clear_check.isChecked(), emit_signal=False)

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
        GlobalConfig.set("sys_mem_warn_gb", self.sys_mem_warn.value(), emit_signal=False)

        # Save Shortcuts
        for key, edit in self.key_edits.items():
            seq = edit.keySequence().toString()
            if seq: GlobalConfig.set(key, seq, emit_signal=False)
        
        # 最后统一触发热更新
        GlobalConfig.signals.config_updated.emit()

        super().accept()
        QMessageBox.information(self, tr("Settings Saved"), tr("Configuration updated successfully.\nSome changes may require restarting actions or the app."))

    def _reset_defaults(self):
        if QMessageBox.question(self, tr("Reset"), tr("Reset ALL settings to defaults?")) == QMessageBox.Yes:
            try:
                if GlobalConfig._config_path.exists():
                    os.remove(GlobalConfig._config_path)
                QMessageBox.information(self, tr("Reset"), tr("Settings reset. Please close and reopen Settings."))
                self.reject()
            except Exception as e:
                print(e)
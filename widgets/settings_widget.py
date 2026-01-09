"""
File: widgets/settings_widget.py
"""
from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QFormLayout, QTabWidget, 
                            QWidget, QKeySequenceEdit, QMessageBox, QSpinBox, 
                            QCheckBox, QGroupBox, QFileDialog, QDoubleSpinBox,
                            QColorDialog, QScrollArea, QComboBox, QListWidget,
                            QListWidgetItem)
from qtpy.QtGui import QKeySequence, QColor
from qtpy.QtCore import QObject, Signal, Qt
import json
import os
from pathlib import Path
from utils.ui_utils import setup_safe_scroll_all

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
    "Preferences & Configuration": "首选项与配置",
    "Reset to Defaults": "恢复默认设置",
    "Cache Location": "缓存位置",
    "Path:": "路径:",
    "Settings Saved": "设置已保存",
    "Configuration updated successfully.\nSome changes may require restarting actions or the app.": "配置已更新。\n部分更改可能需要重启操作或应用才能生效。",
    "Error saving config:": "保存配置时出错:",
    
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
    
    # === PNG/TIFF 快速导入 ===
    "Load PNG Seq": "加载PNG序列",
    "Load TIFF": "加载TIFF",
    "Quickly load a PNG sequence folder": "快速加载PNG序列文件夹",
    "Quickly load a TIFF stack file": "快速加载TIFF堆栈文件",
    "Quickly load a PNG sequence folder to viewer": "快速加载PNG序列到视图",
    "Quickly load a TIFF stack file to viewer": "快速加载TIFF堆栈到视图",
    "Select PNG Sequence Folder": "选择PNG序列文件夹",
    "Select TIFF Stack File": "选择TIFF堆栈文件",
    "Select DM4 folder for dose calculation and archive": "选择DM4文件夹以计算剂量和归档",
    "No Images": "无图像",
    "No PNG files found in the selected folder.": "所选文件夹中没有找到PNG文件。",
    "Loading": "加载中",
    "Loading canceled.": "加载已取消。",
    "Could not read any valid images.": "无法读取任何有效图像。",
    "Loaded": "已加载",
    "Loading TIFF...": "正在加载TIFF...",
    "Error": "错误",
    
    # === Session & Recovery ===
    "Session & Recovery": "会话与恢复",
    "Session Log Settings": "会话日志设置",
    "Default Log Directory:": "默认日志目录:",
    "Select Log Directory": "选择日志目录",
    "Used when no archive path is set. Leave empty to use system default.": "归档路径未设置时使用。留空则使用系统默认路径。",
    "Default Substance:": "默认样品名:",
    "Default Dataset ID:": "默认数据集编号:",
    "Recovery Settings": "恢复设置",
    "Ask before recovering session": "恢复会话前询问",
    "If unchecked, will attempt automatic recovery without asking.": "取消勾选则自动恢复，不再询问。",
    
    # === Recovery Dialog ===
    "Session Recovery": "会话恢复",
    "Incomplete Session Detected": "检测到未完成的会话",
    "Session ID": "会话ID",
    "Created": "创建时间",
    "Substance": "样品名",
    "Dataset": "数据集",
    "Warning: Log file may have been modified": "警告: 日志文件可能已被修改",
    "Recoverable Actions": "可恢复的操作",
    "Select All": "全选",
    "Deselect All": "取消全选",
    "Only selected actions will be displayed. Undone actions are automatically excluded.": "仅显示选中的操作。已撤回的操作已自动排除。",
    "Skip Recovery": "跳过恢复",
    "Recover Selected": "恢复选中项",
    "Required Data Sources": "需要的数据源",
    "Important: You must load these files FIRST before recovery can work!": "重要: 恢复前必须先加载这些文件!",
    "Please use Import tab to load the data, then close this dialog.": "请使用导入标签页加载数据，然后关闭此对话框。",
    "Auto-import data sources on recovery": "恢复时自动导入数据源",
    "If checked, will try to auto-load data files. Otherwise manual selection only.": "勾选后自动加载数据文件，否则仅手动选择。",
    "Data source found": "已找到数据源",
    "Data source NOT found (may have been moved)": "未找到数据源 (可能已移动)",
    "Auto Import": "自动导入",
    "Manual Select": "手动选择",
    "Skip": "跳过",
    "Data Source Import": "数据源导入",
    "Session Log Location": "会话日志位置",
    "System Default Path": "系统默认路径",
    "Open Log Folder": "打开日志文件夹",
    "Manual Recovery": "手动恢复",
    "Recovery complete": "恢复完成",
    "actions recovered": "个操作已恢复",
    "No incomplete sessions found, or recovery was skipped.": "未找到未完成的会话，或已跳过恢复。",
    "Continue": "继续",
    "Skip This": "跳过此项",
    "Cancel All": "取消全部",
    "Continue Without Import": "不导入继续",
    "Archived:": "已归档：",
    "Source path updated to archive location.": "源路径已更新至归档位置。",
    "Checking size...": "正在检查大小...",
    "Archive Only": "仅归档",
    "Starred Only": "仅收藏",
    
    # === Recovery Widget (独立组件) ===
    "Recovery": "会话恢复",
    "Session List": "会话列表",
    "Refresh Sessions": "刷新会话列表",
    "No incomplete sessions": "没有未完成的会话",
    "Click Refresh to check again": "点击刷新重新检查",
    "Session Details": "会话详情",
    "Abandon Session": "放弃此会话",
    "Session abandoned": "会话已放弃",
    "Are you sure you want to abandon this session? This cannot be undone.": "确定要放弃此会话吗？此操作无法撤销。",
    "No actions selected for recovery.": "未选择任何操作进行恢复。",
    "Select Log File": "选择日志文件",
    "Recent Sessions": "历史会话",
    "No sessions found": "未找到会话记录",
    "Start Recovery": "开始恢复",
    "Please load data sources first": "请先加载数据源",
    "Or use Manual Mode to select a log file": "或者使用手动模式选择日志文件",
    "Select Data Source Folder": "选择数据源文件夹",
    "Select TIFF File": "选择TIFF文件",
    "completed": "已完成",
    "in_progress": "进行中",
    "crashed": "崩溃",
    "recovered": "已恢复",
    "abandoned": "已放弃",
    "Manual Mode": "手动模式",
    "Current": "当前",
    "Not selected": "未选择",
    "Max Sessions to Keep:": "最大保留会话数:",
    "Older sessions will be automatically cleaned up.": "超出的旧会话将自动清理。",
    "Recovery Mode:": "恢复模式:",
    "Auto (fully automatic)": "自动 (完全自动执行)",
    "Review (confirm each step)": "审查 (每步确认)",
    "Replaying drift correction...": "正在重放漂移矫正...",
    "Drift correction replayed successfully": "漂移矫正重放成功",
    "Continue with this result?": "继续使用此结果吗?",
    "Data Source Detection": "数据源检测",
    "Auto-detected data source": "自动检测到数据源",
    "Do you want to import this?": "是否要导入此数据?",
    "Import Detected": "导入检测到的",
    "Manual Select": "手动选择",
    "Skip Import": "跳过导入",
    "Data source not found": "未找到数据源",
    "Export Confirmation": "导出确认",
    "The session contains export operations. Do you want to re-export?": "会话包含导出操作。是否要重新导出?",
    "PNG Sequence": "PNG 序列",
    "TIFF Stack": "TIFF 堆栈",
    "Video (MP4)": "视频 (MP4)",
    "Skip Export": "跳过导出",

    # === Session Protection (新增) ===
    "Session Management": "会话管理",
    "Starred": "已收藏",
    "Star Session": "收藏会话",
    "Unstar Session": "取消收藏",
    "Edit Label": "编辑标签",
    "Delete Session": "删除会话",
    "Refresh": "刷新",
    
    # === Settings UI Specific ===
    "Algorithm Defaults": "算法默认值",
    "Auto-Calculate (Preview on ROI Draw)": "自动计算 (绘制 ROI 时预览)",
    "Drift Kernel:": "漂移核大小:",
    "Max Workers:": "最大线程数:",
    "Interaction Settings": "交互设置",
    "Show Warning when Clearing Overlays": "清除覆盖层时显示警告",
    
    "Image Enhancement Defaults": "图像增强默认值",
    "Enable Gaussian Blur by Default": "默认启用高斯模糊",
    "Enable Rolling Average by Default": "默认启用滚动平均",
    "Gaussian Sigma:": "高斯 Sigma:",
    "Roll Avg Window:": "滚动平均窗口:",
    "Enhance Workers:": "增强处理线程数:",
    
    "Generated Folder Suffixes": "生成文件夹后缀",
    "Enlarge Canvas on Rotate": "旋时扩大画布",
    "Create Denoise Folders (LR/HR)": "创建去噪文件夹 (LR/HR)",
    "Create Refine Folders (Mask)": "创建精修文件夹 (Mask)",
    
    # === Drift Widget ===
    "Max Shift:": "最大漂移:",
    "Previewing: %s. Press 'Crtl+Z' to Undo.": "正在预览: %s。按 'Ctrl+Z' 撤销。",
    "ROI cleared.": "ROI 已清除。",
    "Mode: Draw Rectangle (Auto-Preview ON)": "模式: 绘制矩形 (自动预览开启)",
    "ROI updated. Auto-calculating...": "ROI 已更新。正在自动计算...",
    "ROI updated. Click Recalculate.": "ROI 已更新。请点击“重新计算”。",
    "Draw ROI first.": "请先绘制 ROI。",
    "Invalid ROI (too small or out of bounds).": "无效 ROI (太小或超出边界)。",
    "Calculating drift...": "正在计算漂移...",
    "Auto-applying correction...": "正在自动应用矫正...",
    "Calculated. Click Apply to see result.": "计算完成。点击“应用”查看结果。",
    "Applying correction...": "正在应用矫正...",
    "Apply Error:": "应用出错:",
    "Undone. Adjust ROI and try again.": "已撤销。调整 ROI 并重试。",
    "Rotation Undone.": "旋转已撤销。",
    "Crop Undone.": "裁剪已撤销。",
    "Enhancement Undone.": "增强已撤销。",
    "Contrast Undo.": "对比度调整已撤销。",
    
    # === Import Widget ===
    "No Dataset ID": "无数据集 ID",
    "Auto-OL:": "自动识别 OL:",
    "(Auto-Added)": "(自动添加)",
    "OL Not Found": "未找到 OL 信息",
    "Check ID": "检查 ID",
    "Could not detect 'dataset' number.\nPlease check ID manually.": "无法检测到 'dataset' 编号。\n请手动检查 ID。",
    "Select DM4 Image for Dose Calculation": "选择用于剂量计算的 DM4 图像",
    "Locating file index...": "正在定位文件索引...",
    "Selected: %s (Index: %s)": "已选: %s (索引: %s)",
    "File not found in current structure match.": "在当前结构中未找到文件。",
    "Error picking file:": "选择文件出错:",
    "Scanning metadata...": "正在扫描元数据...",
    "Folder exists:\n%s\nOverwrite?": "文件夹已存在:\n%s\n覆盖?",
    "Checking size...": "正在检查大小...",
    "Moving raw data...": "正在移动原始数据...",
    "Copying raw data...": "正在复制原始数据...",
    "Archived: %s": "已归档: %s",
    "Source path updated to archive location.": "源路径已更新为归档位置。",
    "Archive created successfully!": "归档创建成功!",
    "Large dataset detected (>100GB). Original folder was MOVED to archive to save time/space.": "检测到大数据集 (>100GB)。原始文件夹已**移动**到归档以节省时间/空间。",
    "Original folder was COPIED. Please delete the source manually if needed.": "原始文件夹已**复制**。如有需要请手动删除源文件。",
    "Archive Complete": "归档完成",
    "Archive Error:": "归档错误:",
    "Confirm Load": "确认加载",
    "Loading new data will CLEAR ALL current layers.\nContinue?": "加载新数据将清除所有当前图层。\n继续?",
    "Loading...": "正在加载...",
    "Loaded %s frames.": "加载了 %s 帧。",
    
    # === Geometry Widget ===
    "Empty folder:": "空文件夹:",
    "Failed to read first frame of": "读取第一帧失败:",
    "Shortcut: Batch Export Triggered": "快捷键: 触发批量导出",
    "Shortcut: Single Crop Triggered": "快捷键: 触发单次裁剪",
    "Mode: Select/Adjust": "模式: 选择/调整",
    "Mode: Draw": "模式: 绘制",
    "Draw Horizon Line.": "绘制水平线。",
    "Rotating...": "正在旋转...",
    "Rotating %.1f°...": "正在旋转 %.1f°...",
    "Rotated %.1f° (Expand=%s)": "已旋转 %.1f° (扩大画布=%s)",
    "Error showing result:": "显示结果出错:",
    "Rotation Error:": "旋转错误:",
    "Applied %s flip.": "已应用 %s 翻转。",
    "Mode: Adjust Crop Rect (Drag corners to resize)": "模式: 调整裁剪框 (拖动角调整大小)",
    "Draw Single Crop Rect.": "绘制单次裁剪框。",
    "Crop applied. Press '%s' to Undo.": "裁剪已应用。按 '%s' 撤销。",
    "Resuming Draw on '%s'.": "恢复在 '%s' 上绘制。",
    "Last ROI removed.": "撤销了上一个 ROI。",
    "Drawing on '%s'. (New Layer)": "在 '%s' 上绘制。(新图层)",
    "Draw Mode (double-click bg)": "绘制模式 (双击背景)",
    "Select Mode (double-click bg to draw)": "选择模式 (双击背景以绘制)",
    "No ROI selected. Select a green box first.": "未选择 ROI。请先选择一个绿框。",
    "Set range '%s' for %s ROI(s).": "已为 %s 个 ROI 设置范围 '%s'。",
    "Adjust Mode.": "调整模式。",
    "Adjust Mode (Click bg to draw)": "调整模式 (点击背景以绘制)",
    "Nothing to clear.": "没有可清除的内容。",
    "Clear All Overlays?": "清除所有覆盖层?",
    "Clear ALL temporary drawings (ROIs, Lines, etc.)?": "清除所有临时绘制 (ROI, 线条 等)?",
    "This action cannot be undone.": "此操作无法撤销。",
    "Do not ask again": "不再询问",
    "Canvas cleared.": "画布已清除。",
    "Ref image missing.": "参考图像缺失。",
    "Save Reference Images?": "保存参考图像?",
    "Saving ROI JSON.\nDo you also want to save the reference image(s)?": "正在保存 ROI JSON。\n您是否也想保存参考图像?",
    "Note: Data Layer and View Layer are DIFFERENT.\nData: %s\nView: %s": "注意: 数据图层和视图图层不同。\n数据: %s\n视图: %s",
    "Skip Images": "跳过图像",
    "Select Layers": "选择图层",
    "Which layer(s) should be saved as reference?": "应将哪些图层保存为参考?",
    "Save Both": "保存两者",
    "Data Only": "仅数据",
    "View Only": "仅视图",
    "Load View Layer?": "加载视图图层?",
    "Load image for View Layer: '%s'?": "为视图图层加载图像: '%s'?",
    "Load Data Layer?": "加载数据图层?",
    "Load image for Data Layer: '%s'?": "为数据图层加载图像: '%s'?",
    "Loading %s image(s)...": "正在加载 %s 张图像...",
    "Load Error:": "加载错误:",
    "No ROIs defined.": "未定义 ROI。",
    "Exporting Crops...": "正在导出裁剪...",
    "Export canceled.": "导出已取消。",
    "Exported %s crops.": "已导出 %s 个裁剪。",
    "Exported %s crops!\nSaved to: %s": "已导出 %s 个裁剪!\n保存至: %s",
    "Export Error": "导出错误",
    "Error": "错误",
    "Error:": "错误:",
    "Exists": "文件夹已存在",
    "MOVE (Fast)": "移动 (快速)",
    "COPY (Safe)": "复制 (安全)",
    "Saved JSON": "已保存 JSON",
    "Saved Seq": "已保存序列",
    "Saved TIFF:": "已保存 TIFF:",
    "Ref snap failed:": "参考快照失败:",
    "Image Not Found": "未找到图像",
    "Could not auto-locate image for layer:\n\n'%s'\n\nBrowse for it manually?": "无法自动定位图层图像:\n\n'%s'\n\n手动浏览?",
    "Select Image for '%s'": "为 '%s' 选择图像",
    "Image Source Detection Report": "图像源检测报告",
    "View Layer": "视图图层",
    "Found": "已找到",
    
    "DM4 Archive": "DM4 归档",
    "PNG Sequence": "PNG 序列",
    "TIFF Stack": "TIFF 堆栈",
    "DM4 Archive path detected.\nPlease use Import tab to load the images.\n\n%s": "检测到 DM4 归档路径。\n请使用导入标签页加载图像。\n\n%s",
    "✅ %s loaded successfully!": "✅ %s 加载成功!",
    "Import failed: %s": "导入失败: %s",
    "✅ PNG Sequence loaded!": "✅ PNG 序列已加载!",
    "✅ TIFF Stack loaded!": "✅ TIFF 堆栈已加载!",
    "DM4 loading requires the full Import workflow.\nPlease use Import tab.": "DM4 加载需要完整的导入流程。\n请使用导入标签页。",
    "DM4 Sequence": "DM4 序列",
    "Toggle Measurement Tool (Draw lines to measure distance)": "切换测量工具 (画线测量距离)",
    "Last measurement removed.": "已移除最近的测量。",
    "Measurement Mode: Draw lines to measure.(Ctrl+Z to Undo)": "测量模式: 画线测量。(Ctrl+Z 撤销)",
    "End Measurement.": "结束测量。",
    "Session recovery: %s actions loaded": "会话恢复: 已加载 %s 个操作",
    "Hidden": "已隐藏",
    "Shown": "已显示",
    "layer controls": "图层控制",
    
    # === Enhance Widget ===
    "Output: %s frames (Loss: %s)": "输出: %s 帧 (损失: %s)",
    "Select at least one filter.": "请至少选择一个滤波器。",
    "Running filters...": "正在运行滤波器...",
    "Done. Layer: %s.": "完成。图层: %s。",
    "Filter Error:": "滤波器错误:",
    "No image selected.": "未选择图像。",
    "Applied. New layer: %s": "已应用。新图层: %s",
    
    # === Export Widget ===
    "Export function returned False.": "导出函数返回失败。",
    "Higher is better, but larger filesize.": "越高越好，但文件更大。",
    "No path selected": "未选择路径",
    "Ready": "就绪",
    "Check inputs": "检查输入",
    "Missing Annotations": "缺少标注",
    "You are exporting a video WITHOUT Scale Bar or Timestamp.\n\nAre you sure?": "您正在导出没有比例尺或时间戳的视频。\n\n确定吗?",
    "Exporting...": "正在导出...",
    "Invalid frame range syntax": "无效的帧范围语法",
    "Done: %s": "完成: %s",
    
    # === Annotation Widget ===
    "Ready.": "就绪。",
    "1 px =": "1 px =",
    "Preview Active.": "预览已激活。",
    "Initializing Preview Layer...": "正在初始化预览图层...",
    "Burning annotations...": "正在烧录标注...",
    "Done.": "完成。",
    "Cleared.": "已清除。",
    
    "Main Suffix:": "主后缀:",
    "LR Suffix:": "LR 后缀:",
    "HR Suffix:": "HR 后缀:",
    "Mask Suffix:": "Mask 后缀:",
    "New Mask Suffix:": "新 Mask 后缀:",
    
    "Measure Tool": "测量工具",
    "Line Color:": "线条颜色:",
    "Width:": "宽度:",
    "Font Size:": "字体大小:",
    "Simple Crop": "简单裁剪",
    "Box Color:": "边框颜色:",
    "Batch Crop": "批量裁剪",
    
    "Resource Thresholds": "资源阈值",
    "RAM vs Disk Limit:": "RAM vs 磁盘限制:",
    "If an array exceeds this size, create it on Disk (memmap).": "如果数组超过此大小，则在磁盘上创建 (memmap)。",
    "Memmap Warning Threshold:": "Memmap 警告阈值:",
    "Show warning if single allocation exceeds this size (Check RAM vs C: drive).": "如果单次分配超过此大小则显示警告 (检查 RAM vs C盘)。",
    "Disk Space Warning:": "磁盘空间警告:",
    "Warn if cache folder usage exceeds this size.": "如果缓存文件夹占用超过此大小则警告。",
    "Move vs Copy Limit:": "移动 vs 复制限制:",
    "If raw data folder > this size, MOVE instead of COPY during archive.": "如果原始数据文件夹 > 此大小，归档时使用剪切而非复制。",
    
    "starred sessions": "个收藏会话",
    "protected sessions": "个受保护会话",
    "normal sessions": "个普通会话",
    "Archive Protected": "归档保护",
    "Total": "总计",
    "Enter label for this session:": "请输入此会话的标签:",
    "Session Label": "会话标签",
    "Are you sure you want to delete this session?": "确定要删除此会话吗?",
    "Session deleted": "会话已删除",
    "Many Starred Sessions": "收藏会话较多",
    "You have {0} starred sessions. Consider cleaning up unused ones.": "您有 {0} 个收藏会话。建议清理不再需要的会话。",
    "Don't remind me for {0} days": "{0} 天内不再提醒",
    "Never remind me": "永不提醒",
    "Remind me later": "稍后提醒",
    "Open Management": "打开管理",
    "Star this session?": "是否收藏此会话?",
    "Session recovered successfully. Would you like to star it for future reference?": "会话恢复成功。是否收藏以便将来参考?",
    "Starred Reminder Threshold:": "收藏提醒阈值:",
    "Show reminder when starred sessions exceed this count.": "收藏会话超过此数量时显示提醒。",
    "Reminder Cooldown (days):": "提醒冷却期 (天):",
    "0 = never remind": "0 = 永不提醒",
    "Please select a session first.": "请先选择一个会话。",
    "Star / Unstar Session": "收藏/取消收藏会话",
    "Edit Session Label": "编辑会话标签",
    "actions": "个操作",
    

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
    
    # === 路径管理与 Everything 集成 ===
    "Manage Paths": "管理搜索路径",
    "Manage Search Paths": "管理搜索路径",
    "Saved Search Paths": "已保存的搜索路径",
    "Sessions from these folders will be shown in the list.": "这些文件夹中的会话将显示在列表中。",
    "Add Path": "添加路径",
    "Select Archive Folder": "选择归档文件夹",
    "Locate": "定位",
    "Path Valid": "路径有效",
    "This path is valid.": "此路径有效。",
    "Path Located": "路径已定位",
    "Found at": "已找到于",
    "Manual Select Folder": "手动选择文件夹",
    "Remove": "移除",
    "Cleanup Invalid": "清理无效路径",
    "Cleanup Complete": "清理完成",
    "Removed": "已移除",
    "invalid paths": "个无效路径",
    "No invalid paths found.": "未发现无效路径。",
    "search available": "搜索可用",
    "not available": "不可用",
    "Download": "下载",
    "Close": "关闭",
    "Tip": "提示",
    "Everything search engine not detected.": "未检测到 Everything 搜索引擎。",
    "Installing it enables": "安装后可获得以下功能",
    "Auto-locate moved archive folders": "自动定位移动后的归档文件夹",
    "Millisecond full-disk search": "毫秒级全盘搜索",
    
    # === 对比度应用进度条 ===
    "Applying contrast adjustment...": "正在应用对比度调整...",
    "Converting data type...": "正在转换数据类型...",
    "Normalizing...": "正在归一化...",
    "Clipping values...": "正在裁剪数值...",
    "Mapping to uint8...": "正在映射到 uint8...",
    "Rendering result...": "正在渲染结果...",
    
    # === 路径管理器错误提示 ===
    "Please select a path first.": "请先选择一个路径。",
    "Invalid path data.": "无效的路径数据。",
    "Not Found": "未找到",
    "Could not auto-locate. Browse manually?": "自动定位失败。是否手动浏览？",
    
    # === 右键菜单与会话管理 ===
    "Open in Explorer": "在文件浏览器中打开",
    "Open folder in file explorer?": "是否在文件浏览器中打开此文件夹？",
    "Don't ask again": "不再询问",
    "Delete the selected session from disk": "从磁盘删除选中的会话",
    "Session has been deleted.": "会话已删除。",
    "Failed to delete session": "删除会话失败",
    
    # === 导入 Session 文件 ===
    "Import Session File": "导入 Session 文件",
    "Select Session Log File": "选择 Session 日志文件",
    "Import Success": "导入成功",
    "Session imported successfully!": "Session 导入成功！",
    "Path added": "已添加路径",
    
    # === DM4 自动导入改进 ===
    "DM4 Auto-Import": "DM4 自动导入",
    "DM4 Archive detected. Auto-set path and switch to Import tab?": "检测到 DM4 归档。是否自动设置路径并切换到导入标签页？",
    "Path Set": "路径已设置",
    "DM4 folder path set. Click Load Images to proceed.": "DM4 文件夹路径已设置。点击'加载图像'继续。",
    "Select DM4 Folder": "选择 DM4 文件夹",
    
    # === Annotation Recovery (New) ===
    "Restore annotation parameters": "恢复标注参数",
    "This will update scale bar and timestamp settings.": "这将更新比例尺和时间戳设置。",
    "Annotation parameters restored": "标注参数已恢复",
    "You can now preview and adjust the settings.": "您现在可以预览并调整设置。",
    "This is a burn-in operation": "这是一个烧录操作",
    "How would you like to recover?": "您希望如何恢复？",
    "Restore Parameters Only": "仅恢复参数",
    "Editable": "可编辑",
    "Execute Burn-in": "执行烧录",
    "New Layer": "新图层",
    "Annotation parameters restored (editable mode)": "标注参数已恢复 (可编辑模式)",
    "This action was recovered from another session": "此操作是从另一个会话恢复的",
    "What would you like to do?": "您希望做什么？",
    "Load Source Session": "加载源会话",
    "Source session file not found": "找不到源会话文件",
    "The original session may have been moved or deleted.": "原始会话可能已被移动或删除。",
    "Source session loaded": "源会话已加载",
    "Please select the actions you want to recover from this session.": "请选择您想从此会话恢复的操作。",
    
    # === Manual Import (Recovery) ===
    "Manual Data Import": "手动数据导入",
    "No image layers detected": "未检测到图像层",
    "Detected paths from session log": "从会话日志检测到的路径",
    "No source path found in session log.": "会话日志中未找到源路径。",
    "Please select how to import data": "请选择导入数据的方式",
    "Manual Import": "手动导入",
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
        "shortcut_delete_session": "Delete",  # 删除会话快捷键
        
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
        "show_archive_popup": True,
        
        # Session & Recovery
        "session_log_dir": "",
        "session_substance_default": "Sample",
        "session_dataset_default": "ds1",
        "session_ask_on_recovery": True,
        "session_auto_import_data": True,  # 恢复时是否尝试自动导入数据
        "session_max_keep": 20,  # 最大保留会话数
        "session_recovery_mode": "review",  # 恢复模式: "auto"(全自动) 或 "review"(每步确认)
        "session_auto_detect_source": True,  # 恢复前自动检测数据源
        "session_confirm_export": True,  # 导出操作前询问用户
        
        # Session Protection (新增)
        "session_starred_reminder_threshold": 20,  # 收藏超过此数量时显示提醒
        "session_starred_reminder_cooldown_days": 30,  # 提醒冷却期 (天), 0=永不提醒
        "session_last_starred_reminder": "",  # 上次提醒时间 (ISO格式)
        
        # Session Shortcuts (新增)
        "shortcut_session_star": "S",  # 收藏快捷键
        "shortcut_session_label": "L"  # 编辑标签快捷键
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
        
        # === Tab 4: Session & Recovery (New) ===
        tabs.addTab(self._create_session_tab(), f"📋 {tr('Session & Recovery')}")

        # === Tab 5: Shortcuts (Existing) ===
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
        
        # [Safety] Prevent accidental mouse wheel scroll
        setup_safe_scroll_all(
            self.lang_combo,
            self.drift_k_spin, self.drift_w_spin,
            self.enh_sigma_spin, self.enh_win_spin, self.enh_work_spin,
            self.style_meas_w, self.style_meas_font,
            self.style_batch_w, self.style_batch_font,
            self.sys_ram, self.sys_disk, self.sys_move, self.sys_mem_warn,
            self.session_max_keep, self.session_starred_threshold, self.session_reminder_cooldown
        )
        
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
    
    def _create_session_tab(self):
        """Session & Recovery 标签页 (含会话管理面板)"""
        w = QWidget()
        main_layout = QVBoxLayout()
        
        # 使用 Splitter 分割上下两部分
        from qtpy.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Vertical)
        
        # === 上半部分: 设置 ===
        settings_widget = QWidget()
        l = QVBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        
        # === Group 1: Log Directory ===
        g_log = QGroupBox(tr("Session Log Settings"))
        f_log = QFormLayout()
        
        # 默认日志目录
        self.session_log_dir = QLineEdit(str(GlobalConfig.get("session_log_dir")))
        btn_browse_log = QPushButton("📂")
        btn_browse_log.setFixedWidth(40)
        btn_browse_log.clicked.connect(lambda: self.session_log_dir.setText(
            QFileDialog.getExistingDirectory(self, tr("Select Log Directory")) or self.session_log_dir.text()
        ))
        h_log = QHBoxLayout()
        h_log.addWidget(self.session_log_dir)
        h_log.addWidget(btn_browse_log)
        f_log.addRow(tr("Default Log Directory:"), h_log)
        
        l_hint_log = QLabel(tr("Used when no archive path is set. Leave empty to use system default."))
        l_hint_log.setStyleSheet("color: gray; font-size: 9pt;")
        f_log.addRow("", l_hint_log)
        
        # 默认物质名
        self.session_substance = QLineEdit(str(GlobalConfig.get("session_substance_default")))
        self.session_substance.setPlaceholderText("e.g. CRY2, BSA")
        f_log.addRow(tr("Default Substance:"), self.session_substance)
        
        # 默认 Dataset ID
        self.session_dataset = QLineEdit(str(GlobalConfig.get("session_dataset_default")))
        self.session_dataset.setPlaceholderText("e.g. ds1, ds2")
        f_log.addRow(tr("Default Dataset ID:"), self.session_dataset)
        
        g_log.setLayout(f_log)
        l.addWidget(g_log)
        
        # === Group 2: Recovery Settings ===
        g_rec = QGroupBox(tr("Recovery Settings"))
        f_rec = QFormLayout()
        
        self.session_ask_recovery = QCheckBox(tr("Ask before recovering session"))
        self.session_ask_recovery.setChecked(bool(GlobalConfig.get("session_ask_on_recovery")))
        self.session_ask_recovery.setToolTip(tr("If unchecked, will attempt automatic recovery without asking."))
        f_rec.addRow(self.session_ask_recovery)
        
        self.session_auto_import = QCheckBox(tr("Auto-import data sources on recovery"))
        self.session_auto_import.setChecked(bool(GlobalConfig.get("session_auto_import_data")))
        self.session_auto_import.setToolTip(tr("If checked, will try to auto-load data files. Otherwise manual selection only."))
        f_rec.addRow(self.session_auto_import)
        
        # 最大保留会话数
        self.session_max_keep = QSpinBox()
        self.session_max_keep.setRange(5, 200)
        self.session_max_keep.setValue(int(GlobalConfig.get("session_max_keep") or 20))
        self.session_max_keep.setToolTip(tr("Older sessions will be automatically cleaned up."))
        f_rec.addRow(tr("Max Sessions to Keep:"), self.session_max_keep)
        
        # 收藏提醒阈值
        self.session_starred_threshold = QSpinBox()
        self.session_starred_threshold.setRange(5, 100)
        self.session_starred_threshold.setValue(int(GlobalConfig.get("session_starred_reminder_threshold") or 20))
        self.session_starred_threshold.setToolTip(tr("Show reminder when starred sessions exceed this count."))
        f_rec.addRow(tr("Starred Reminder Threshold:"), self.session_starred_threshold)
        
        # 提醒冷却期 (注意: 0 是有效值，不能用 or)
        self.session_reminder_cooldown = QSpinBox()
        self.session_reminder_cooldown.setRange(0, 365)
        cooldown_val = GlobalConfig.get("session_starred_reminder_cooldown_days")
        self.session_reminder_cooldown.setValue(int(cooldown_val) if cooldown_val is not None else 30)
        self.session_reminder_cooldown.setToolTip(tr("0 = never remind"))
        self.session_reminder_cooldown.setSuffix(" days")
        f_rec.addRow(tr("Reminder Cooldown (days):"), self.session_reminder_cooldown)
        
        g_rec.setLayout(f_rec)
        l.addWidget(g_rec)
        
        settings_widget.setLayout(l)
        splitter.addWidget(settings_widget)
        
        # === 下半部分: 会话管理面板 ===
        g_mgmt = QGroupBox(f"📋 {tr('Session Management')}")
        mgmt_layout = QVBoxLayout()
        
        # 会话列表
        self.session_list = QListWidget()
        self.session_list.setSelectionMode(QListWidget.SingleSelection)
        self.session_list.itemDoubleClicked.connect(self._edit_session_label)  # 双击编辑标签
        self.session_list.setStyleSheet("""
            QListWidget::item { padding: 6px; border-bottom: 1px solid #444; }
            QListWidget::item:selected { background-color: #2196F3; }
        """)
        mgmt_layout.addWidget(self.session_list)
        
        # 设置快捷键
        from qtpy.QtWidgets import QShortcut
        from qtpy.QtGui import QKeySequence
        star_key = str(GlobalConfig.get("shortcut_session_star") or "S")
        label_key = str(GlobalConfig.get("shortcut_session_label") or "L")
        QShortcut(QKeySequence(star_key), self, self._toggle_session_star)
        QShortcut(QKeySequence(label_key), self, self._edit_session_label)
        
        # 操作按钮行
        h_actions = QHBoxLayout()
        
        btn_star = QPushButton(f"⭐ {tr('Star Session')}")
        btn_star.clicked.connect(self._toggle_session_star)
        h_actions.addWidget(btn_star)
        self._btn_star = btn_star
        
        btn_label = QPushButton(f"🏷️ {tr('Edit Label')}")
        btn_label.clicked.connect(self._edit_session_label)
        h_actions.addWidget(btn_label)
        
        btn_delete = QPushButton(f"🗑️ {tr('Delete Session')}")
        btn_delete.clicked.connect(self._delete_session)
        btn_delete.setStyleSheet("background-color: #8B0000;")
        h_actions.addWidget(btn_delete)
        
        btn_refresh = QPushButton(f"🔄 {tr('Refresh')}")
        btn_refresh.clicked.connect(self._refresh_session_list)
        h_actions.addWidget(btn_refresh)
        
        mgmt_layout.addLayout(h_actions)
        
        # 统计标签
        self.session_stats_label = QLabel("")
        self.session_stats_label.setStyleSheet("color: #888; font-size: 9pt; padding: 4px;")
        mgmt_layout.addWidget(self.session_stats_label)
        
        g_mgmt.setLayout(mgmt_layout)
        splitter.addWidget(g_mgmt)
        
        # 设置 splitter 比例
        splitter.setSizes([250, 350])
        
        main_layout.addWidget(splitter)
        
        # 底部: 日志位置和快捷操作
        h_bottom = QHBoxLayout()
        
        from pathlib import Path
        default_path = Path.home() / ".napari_tem" / "sessions"
        lbl_path = QLabel(f"📁 {default_path}")
        lbl_path.setStyleSheet("color: #666; font-size: 9pt;")
        h_bottom.addWidget(lbl_path)
        
        h_bottom.addStretch()
        
        btn_open_folder = QPushButton(f"📂 {tr('Open Log Folder')}")
        btn_open_folder.clicked.connect(self._open_log_folder)
        h_bottom.addWidget(btn_open_folder)
        
        btn_manual_recovery = QPushButton(f"🔄 {tr('Manual Recovery')}")
        btn_manual_recovery.clicked.connect(self._trigger_manual_recovery)
        btn_manual_recovery.setStyleSheet("background-color: #2196F3;")
        h_bottom.addWidget(btn_manual_recovery)
        
        main_layout.addLayout(h_bottom)
        
        w.setLayout(main_layout)
        
        # 初始加载会话列表
        self._refresh_session_list()
        
        return w
    
    def _refresh_session_list(self):
        """刷新会话列表"""
        from utils.session_logger import SessionLogger
        from pathlib import Path
        import datetime
        
        self.session_list.clear()
        self._session_data = []  # 存储会话数据供后续操作
        
        # 获取所有会话
        sessions = SessionLogger.find_all_sessions(limit=100)
        
        # 统计
        starred_count = 0
        archive_protected_count = 0
        
        # 获取归档路径
        from qtpy.QtCore import QSettings
        archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
        archive_dir = Path(archive_path) if archive_path else None
        
        for session_path in sessions:
            summary = SessionLogger.get_session_summary(session_path)
            if not summary:
                continue
            
            self._session_data.append(summary)
            
            # 判断保护状态
            is_archive_protected = False
            if archive_dir and archive_dir.exists():
                try:
                    session_path.resolve().relative_to(archive_dir.resolve())
                    is_archive_protected = True
                    archive_protected_count += 1
                except ValueError:
                    pass
            
            is_starred = summary.get("starred", False)
            if is_starred:
                starred_count += 1
            
            # 构建显示文本
            star_icon = "⭐ " if is_starred else "   "
            archive_icon = "🔒" if is_archive_protected else ""
            label = summary.get("label", "")
            label_text = f"[{label}] " if label else ""
            
            # 时间格式化
            created_at = summary.get("created_at", "")
            try:
                dt = datetime.datetime.fromisoformat(created_at)
                time_str = dt.strftime("%m-%d %H:%M")
            except:
                time_str = created_at[:16] if created_at else ""
            
            status = summary.get("status", "unknown")
            status_icon = {
                "completed": "✅",
                "in_progress": "🔄",
                "crashed": "💥",
                "recovered": "♻️",
                "abandoned": "❌"
            }.get(status, "❓")
            
            meta = summary.get("metadata", {})
            substance = meta.get("substance", "")
            actions_count = summary.get("actions_count", 0)
            
            # 显示格式: ⭐ 🔒 [标签] 物质名 ✅ 时间 (N个操作)
            display_text = f"{star_icon}{archive_icon}{label_text}{substance or summary['session_id']} {status_icon} {time_str} ({actions_count}{tr('actions')})"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, len(self._session_data) - 1)  # 存储索引
            self.session_list.addItem(item)
        
        # 更新统计
        normal_count = len(self._session_data) - starred_count - archive_protected_count
        self.session_stats_label.setText(
            f"{tr('Total')}: {len(self._session_data)} | "
            f"⭐ {starred_count} {tr('starred sessions')} | "
            f"🔒 {archive_protected_count} {tr('protected sessions')} | "
            f"📋 {normal_count} {tr('normal sessions')}"
        )
    
    def _get_selected_session(self):
        """获取当前选中的会话数据"""
        items = self.session_list.selectedItems()
        if not items:
            return None
        idx = items[0].data(Qt.UserRole)
        if idx is not None and idx < len(self._session_data):
            return self._session_data[idx]
        return None
    
    def _toggle_session_star(self):
        """切换收藏状态"""
        from utils.session_logger import SessionLogger
        
        session = self._get_selected_session()
        if not session:
            QMessageBox.warning(self, tr("Warning"), tr("Please select a session first."))
            return
        
        path = session.get("path")
        current_starred = session.get("starred", False)
        new_starred = not current_starred
        
        if SessionLogger.update_session_file(path, starred=new_starred):
            self._refresh_session_list()
    
    def _edit_session_label(self):
        """编辑会话标签"""
        from utils.session_logger import SessionLogger
        from qtpy.QtWidgets import QInputDialog
        
        session = self._get_selected_session()
        if not session:
            QMessageBox.warning(self, tr("Warning"), tr("Please select a session first."))
            return
        
        path = session.get("path")
        current_label = session.get("label", "")
        
        new_label, ok = QInputDialog.getText(
            self, tr("Session Label"), 
            tr("Enter label for this session:"),
            text=current_label
        )
        
        if ok:
            if SessionLogger.update_session_file(path, label=new_label):
                self._refresh_session_list()
    
    def _delete_session(self):
        """删除会话"""
        session = self._get_selected_session()
        if not session:
            QMessageBox.warning(self, tr("Warning"), tr("Please select a session first."))
            return
        
        path = session.get("path")
        
        reply = QMessageBox.question(
            self, tr("Delete Session"),
            tr("Are you sure you want to delete this session?"),
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                path.unlink()
                QMessageBox.information(self, tr("Delete Session"), tr("Session deleted"))
                self._refresh_session_list()
            except Exception as e:
                QMessageBox.critical(self, tr("Error"), f"Failed to delete: {e}")

    def _create_shortcuts_tab(self):
        w = QWidget()
        f = QFormLayout()
        self.key_edits = {}
        shortcuts_map = {
            "shortcut_toggle_ui": tr("Toggle Layer Controls"),
            "shortcut_undo_drift": tr("Undo / Clear ROI"),
            "shortcut_apply_crop": tr("Apply Crop / Export"),
            "shortcut_switch_mode": tr("Switch Draw/Select Mode"),
            "shortcut_session_star": tr("Star / Unstar Session"),
            "shortcut_session_label": tr("Edit Session Label"),
            "shortcut_delete_session": tr("Delete Session")
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
        
        # Save Session & Recovery
        if hasattr(self, 'session_log_dir'):
            GlobalConfig.set("session_log_dir", self.session_log_dir.text(), emit_signal=False)
            GlobalConfig.set("session_substance_default", self.session_substance.text(), emit_signal=False)
            GlobalConfig.set("session_dataset_default", self.session_dataset.text(), emit_signal=False)
            GlobalConfig.set("session_ask_on_recovery", self.session_ask_recovery.isChecked(), emit_signal=False)
            GlobalConfig.set("session_auto_import_data", self.session_auto_import.isChecked(), emit_signal=False)
            GlobalConfig.set("session_max_keep", self.session_max_keep.value(), emit_signal=False)
            GlobalConfig.set("session_starred_reminder_threshold", self.session_starred_threshold.value(), emit_signal=False)
            GlobalConfig.set("session_starred_reminder_cooldown_days", self.session_reminder_cooldown.value(), emit_signal=False)

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
    
    def _open_log_folder(self):
        """打开日志文件夹"""
        from pathlib import Path
        import subprocess
        import platform
        
        # 确定日志目录
        user_dir = GlobalConfig.get("session_log_dir")
        if user_dir and Path(user_dir).exists():
            log_dir = Path(user_dir)
        else:
            log_dir = Path.home() / ".napari_tem" / "sessions"
        
        # 确保目录存在
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 打开文件夹
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", str(log_dir)])
            elif platform.system() == "Darwin":
                subprocess.run(["open", str(log_dir)])
            else:
                subprocess.run(["xdg-open", str(log_dir)])
        except Exception as e:
            QMessageBox.warning(self, tr("Error"), f"Could not open folder: {e}")
    
    def _trigger_manual_recovery(self):
        """手动触发恢复对话框"""
        try:
            from widgets.recovery_dialog import check_and_show_recovery
            recovered = check_and_show_recovery(self)
            if recovered:
                QMessageBox.information(self, tr("Manual Recovery"), 
                    f"✅ {tr('Recovery complete')}: {len(recovered)} {tr('actions recovered')}")
            else:
                QMessageBox.information(self, tr("Manual Recovery"), 
                    tr("No incomplete sessions found, or recovery was skipped."))
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"Recovery failed: {e}")
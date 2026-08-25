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
    # Recovery memory retention (F2) — 恢复时中间层保留策略
    "Memory Retention:": "内存保留:",
    "Lean (evict intermediates)": "省内存 (驱逐中间层)",
    "Full (keep all, spill to disk)": "全内存 (全部保留·写盘)",
    "Custom (choose steps to keep)": "自定义 (选择保留步骤)",
    "Select intermediate steps to keep": "选择要保留的中间步骤",
    "kept %d, spilled %d to disk, removed %d intermediates": "保留 %d 层, 写盘 %d 层, 移除 %d 个中间层",
    # Full-layer export (F3) — 独立整层导出
    "Export Full Layer (Data+View) as PNG": "导出整层(数据+视图) PNG",
    "Export the whole uncropped data + contrasted view as a PNG sequence for fast re-import next session": "把未裁切的完整数据层+对比度视图层导出成 PNG 序列, 下次可直接导入而非巨大 dm4",
    "No raw data layer found — import origin first.": "未找到原始数据层 —— 请先导入原始数据(origin)。",
    "Select Output Folder": "选择输出文件夹",
    "Format": "格式",
    "Exported full layer(s): ": "已导出整层: ",
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
    "Max Font Size:": "最大字号:",
    "Upper limit for adaptive font scaling when zoomed out.": "缩小视图时自适应字体的大小上限。",
    "Label Spacing:": "标签间距:",
    "Line spacing between ROI label and frame info.": "数字标号和帧数信息之间的换行间距。",
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
    "Disabled until 'Calc Dose' succeeds: the folder name needs date/dose/pixel size from the DM4 metadata.":
        "需先成功执行「计算剂量」才可用: 归档文件夹名要用到 DM4 元数据里的日期/剂量/像素尺寸。",
    "Run 2. Scan Metadata (Calc Dose) first to enable archiving.":
        "请先在「2. 扫描元数据」点「计算剂量」, 归档按钮才会启用。",
    "Metadata scan failed, archiving stays disabled:": "元数据扫描失败, 归档按钮保持禁用:",
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
    "Geo Export options...": "几何导出选项...",
    "Clone/Drag Modifier:": "克隆/正方形修饰键:",
    "Stamp Modifier:": "全图盖章修饰键:",
    "Press Key...": "按下按键...",
    "⚠ (Not recommended to use 'Alt')": "⚠ (强烈不建议使用 'Alt')",
    
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
    "Log file not found: %s": "未找到日志文件: %s",
    "Failed to load JSON: %s": "加载 JSON 失败: %s",
    "Invalid JSON format: missing 'batch_crop' or 'rois'": "无效的 JSON 格式: 缺少 'batch_crop' 或 'rois'",
    "No ROIs found in log.": "日志中未找到 ROI。",
    "No valid ROIs extracted from log.": "未能从日志中提取出有效的 ROI。",
    "Restored %s ROIs.": "成功恢复了 %s 个 ROI。",
    "Info": "提示",
    "Success": "成功",
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
    "Dataset:": "数据集:",
    "Dataset ID (e.g. ds123). Mostly auto-extracted.": "数据集 ID (例如 ds123)。大多数会自动提取。",
    "Date:": "日期:",
    "Date prefix (YYYYMMDD). Loaded from Archive or Today.": "日期前缀 (YYYYMMDD)。从归档或今天加载。",
    "Sub:": "样品:",
    "Suffix:": "后缀:",
    "Suffix (e.g. _contrasted)": "后缀 (例如 _contrasted)",
    "Export View Layer": "导出视图层",
    "Export View Layer by Default": "默认导出视图层",
    "Additionally export the view layer (e.g. contrasted image).": "同时导出视图层 (如调整过对比度的图像)。",
    "PNG Sequence (Folder)": "PNG 序列 (文件夹)",
    "TIFF Stack (.tiff)": "TIFF 图像堆栈 (.tiff)",
    "Gen Denoise Folders": "生成去噪文件夹",
    "Creates empty folders with Main Suffix + Configured Suffix (e.g. _contrasted_lrtem)": "创建带有主后缀 + 配置后缀的空文件夹 (例如 _contrasted_lrtem)",
    "Gen Refine Folder": "生成精修文件夹",
    "Creates empty mask + refined-mask folders (e.g. _contrasted_mask + _contrasted_mask_refined)": "创建空的掩膜 + 精修掩膜文件夹 (例如 _contrasted_mask + _contrasted_mask_refined)",
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
    # 几何 tab 按钮分组小标题
    "ROI Files:": "ROI 文件:",
    "ROI View:": "ROI 查看:",
    "Auto Detect:": "自动识别:",
    "Liquid Mask:": "液池蒙版:",
    # 完整液池层导出 / 缺 view 提醒 (Fix 3)
    "Frames:": "帧:",
    "Also export full liquid-cell layers (full data + view, all frames)": "也导出完整液池层 (整帧 data + view)",
    "Writes the WHOLE data + view layers (not cropped) to new '{layer}__{timestamp}' folders — the full liquid-cell movie for archive/viewing.": "把完整的 data 层和 view 层(不裁切)整帧导到新的 '{图层名}__{时间戳}' 文件夹 —— 整张液池影片, 便于留档/查看。",
    "All (empty) or 0-100, 120": "全部帧(留空) 或 0-100, 120",
    "Don't warn again": "不再提醒",
    "No View (contrasted) layer selected — '_contrasted' will NOT be exported. Large liquid cells usually need both origin + contrasted.": "未选 View(contrasted) 层 —— 不会导出 _contrasted。大液池通常 origin + contrasted 都需要。",
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
    "Nothing to undo.": "没有可撤回的操作。",
    "ROI Undo History:": "ROI 撤销历史:",
    "Maximum ROI undo history. Larger values use more memory.": "ROI 撤销历史记录上限。数值越大占用内存越多。",
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
    "Overview Map Settings": "概览图设置",
    "Frame:": "帧 (Frame):",
    "Empty=Current": "空=当前帧",
    "Leave empty to use the current viewer frame.": "留空以使用当前视图的帧。",
    "Font Size:": "字体大小 (Font Size):",
    "Box Color:": "边框颜色 (Box Color):",
    "Text Color:": "文字颜色 (Text Color):",
    "Preview Overview": "预览概览图",
    "Text Pos:": "文本位置 (Text Pos):",
    "Top": "顶层 (Top)",
    "Bottom": "底层 (Bottom)",
    "Left": "左侧 (Left)",
    "Right": "右侧 (Right)",
    "Style Presets": "主题预设 (Style Presets)",
    "Load": "加载 (Load)",
    "Save Current...": "保存当前... (Save Current...)",
    "Preset Name": "预设名称",
    "Enter a name for the current visual style preset:": "为当前视觉样式预设输入一个名称:",
    
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
    # -- v2: FPS / Duration / Sampling --
    "Manual FPS": "手动帧率",
    "Target Duration": "目标时长",
    "Duration (sec):": "时长 (秒):",
    "Frame Sampling": "帧抽样",
    "FPS too high for most players": "帧率过高, 多数播放器无法正常播放",
    "Sampling: export %d of %d frames @ %dfps": "抽帧: 导出 %d / %d 帧 @ %d fps",
    "Every %d frames, duration %.1fs": "每 %d 帧取 1 帧, 时长 %.1f 秒",
    "frames": "帧",
    "Playback duration": "播放时长",
    # -- v2: Encoding --
    "Encoder:": "编码器:",
    "Quality (CRF):": "画质 (CRF):",
    "18=best quality  28=smaller file": "18=最佳画质  28=较小文件",
    "Encoding Speed:": "编码速度:",
    "Slower = better compression": "越慢 = 压缩率越高",
    "FFmpeg available": "FFmpeg 可用",
    "FFmpeg not available, using OpenCV": "FFmpeg 不可用, 使用 OpenCV 编码",
    # -- v2: Smart hints --
    "Recommended: H.264 + Target Duration": "推荐: H.264 视频 + 目标时长模式",
    "Recommended: Video (H.264)": "推荐: 视频 (H.264)",
    "Small data, GIF or Video both work": "数据量较小, GIF 或视频均可",
    # -- v2: Post-export compression --
    "Large File Detected": "检测到大文件",
    "Exported file is %.1f MB.\n\nCompress with FFmpeg H.264?\n(Typically 50-80%% smaller)": "导出文件为 %.1f MB。\n\n是否使用 FFmpeg H.264 压缩？\n(通常可缩小 50-80%%)",
    "Compressing...": "正在压缩...",
    "Compression failed": "压缩失败",
    "Compression Complete": "压缩完成",
    "Compressed: %.1f MB → %.1f MB (%.0f%% saved)\n\nReplace original file?": "已压缩: %.1f MB → %.1f MB (节省 %.0f%%)\n\n是否替换原文件？",
    "Compressed: %.1f MB → %.1f MB": "已压缩: %.1f MB → %.1f MB",
    "Replace failed:": "替换失败:",
    "Compressed saved as:": "压缩文件已保存为:",
    
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
    "Refined Mask Suffix:": "精修 Mask 后缀:",
    
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
    "Auto-create subfolder": "自动创建子文件夹",
    "Automatically create a named subfolder (e.g. LayerName_20260129_143022)": "自动创建一个命名的子文件夹 (例如 图层名_20260129_143022)",
    "Select Parent Folder for Sequence": "选择序列输出的父文件夹",
    "Select a folder first": "请先选择文件夹",
    "Will create": "将创建",
    
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
    "Refined Mask Suffix:": "精修 Mask 后缀:",
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
    
    # === Script Runner ===
    "Scripts": "脚本",
    "Script Runner": "脚本运行器",
    "Available Scripts": "可用脚本",
    "Open Scripts Folder": "打开脚本目录",
    "Description": "描述",
    "Select a script to view its description.": "选择脚本以查看其描述。",
    "Run Script": "运行脚本",
    "Stop": "停止",
    "Console Output": "控制台输出",
    "Clear Console": "清空控制台",
    "No scripts found. Place .py/.bat/.ps1 files in the 'scripts' folder.": "未找到脚本。请将 .py/.bat/.ps1 文件放入 'scripts' 文件夹。",
    "No description available.": "暂无描述。",
    "Could not read script.": "无法读取脚本。",
    "Script file not found.": "脚本文件未找到。",
    "Running": "正在运行",
    "Path": "路径",
    "Script stopped by user.": "脚本已被用户停止。",
    "Script finished successfully.": "脚本执行成功。",
    "Script finished with exit code": "脚本执行完毕，退出码",
    
    # === GIF Export ===
    "GIF Animation (.gif)": "GIF 动图 (.gif)",

    # === Frame Filter Widget ===
    "Frame Filter": "帧筛选",
    "1. Select Source Layer": "1. 选择源图层",
    "2. Specify Frames to Remove": "2. 指定要删除的帧",
    "3. Apply Filter": "3. 应用筛选",
    "Enter frame indices to DROP:": "输入要删除的帧索引:",
    "Supported formats:": "支持格式:",
    "Range": "范围",
    "Single": "单帧",
    "Mixed": "混合",
    "Chinese commas (，) are also supported.": "支持中文逗号 (，)。",
    "No layer selected": "未选择图层",
    "Apply Filter": "应用筛选",
    "Frames: %s | Has original indices": "帧数: %s | 保留原始索引",
    "Frames: %s | Original data": "帧数: %s | 原始数据",
    "No frames will be dropped.": "不会删除任何帧。",
    "⚠️ Invalid input or no valid frames to drop.": "⚠️ 输入无效或没有要删除的帧。",
    "❌ Cannot drop ALL frames!": "❌ 不能删除所有帧！",
    "Drop %s frames, Keep %s frames.": "删除 %s 帧，保留 %s 帧。",
    "Cannot drop ALL frames!": "不能删除所有帧！",
    "Info": "提示",
    "No frames dropped. Nothing to do.": "未删除任何帧。无需操作。",
    "Created: %s (%s frames)": "已创建: %s (%s 帧)",
    "Ref Image Path": "参考图像路径",
    # === Geometry Enhancements translations ===
    "Set current frame as range start": "设置当前帧为范围起点",
    "Set current frame as range end": "设置当前帧为范围终点",
    "Lock": "锁定",
    "Lock slider to the set In/Out range": "将滑条锁定在设定的起止范围",
    "Clear slider range lock": "清除滑条范围锁定",
    "Grid": "网格",
    "Enable partition grid for local zoom": "启用分区网格以进行局部放大",
    "Rows:": "行:",
    "Cols:": "列:",
    "Regenerate grid with current rows/cols": "使用当前行列数重新生成网格",
    "Clear": "清除",
    "Refresh": "刷新",
    
    # === ROI & Overview Toggle ===
    "Hide ROI Labels": "隐藏 ROI 标注",
    "Overview Uses View Layer": "概览图使用视图层数据",
    "Set Frame Range (Selected ROI)": "设置帧范围(选中 ROI)",

    # === Missing Translations Checked (2026-03-07) ===
    "0 = infinite loop": "0 = 无限循环",
    "Additional Info:": "附加信息:",
    "Applying contrast...": "正在应用对比度...",
    "Auto-ID:": "自动编号:",
    "Box Color": "边框颜色",
    "Clear All Overlays (ROIs, Lines, Measures)": "清除所有标注图层 (ROI、线条、测量)",
    "Colors (2-256):": "颜色数 (2-256):",
    "Could not auto-locate image for layer:\n\n'%s'\n\nBrowse for it manually?": "无法自动定位图层的图像:\n\n'%s'\n\n是否手动浏览？",
    "Could not detect 'dataset' number.\nPlease check ID manually.": "无法检测到 'dataset' 编号。\n请手动检查 ID。",
    "DM4 Folder": "DM4 文件夹",
    "Exported %s crops!\nSaved to: %s": "已导出 %s 个裁剪图像！\n保存至: %s",
    "Fewer colors = smaller file size, but lower quality": "颜色越少 = 文件越小，但图像质量越低",
    "Folder exists:\n%s\nOverwrite?": "文件夹已存在:\n%s\n是否覆盖？",
    "GIF Options": "GIF 选项",
    "Leave empty for All frames.\nOr use: 0-10, 15, 20-25": "留空表示所有帧。\n或使用格式: 0-10, 15, 20-25",
    "Loading DM4 sequence...": "正在加载 DM4 序列...",
    "Loading new data will CLEAR ALL current layers.\nContinue?": "加载新数据将清除当前所有的图层。\n是否继续？",
    "Loop (0=infinite):": "循环次数 (0=无限):",
    "Mag": "放大倍率",
    "No ROIs to preview.": "没有需要预览的 ROI。",
    "No image data found for preview.": "未找到用于预览的图像数据。",
    "None": "无",
    "Note: Data Layer and View Layer are DIFFERENT.\nData: %s\nView: %s": "注意：数据图层和视图图层不一致。\n数据层: %s\n视图层: %s",
    "Output: %s frames (Padding: Edge)": "输出: %s 帧 (边缘填充)",
    "Overview Preview": "概略图预览",
    "Preview Folder Name:": "预览文件夹名称:",
    "Replaying enhancement...": "正在重放图像增强...",
    "Replaying rotation...": "正在重放旋转操作...",
    "Saving ROI JSON.\nDo you also want to save the reference image(s)?": "正在保存 ROI JSON。\n是否还要保存参考图像？",
    "Script executed successfully.": "脚本执行成功。",
    "Script failed.": "脚本执行失败。",
    "Select Image for Dose Calculation": "选择用于剂量计算的图像",
    "Select data source type:": "选择数据源类型:",
    "Warning": "警告",
    "You are exporting a video WITHOUT Scale Bar or Timestamp.\n\nAre you sure?": "您正在导出一个缺少比例尺或时间戳的视频。\n\n是否确定？",

    # --- Session 3 additions (2026-05-13) ---
    "Success": "成功",
    "Error": "错误",
    "Select Output Directory": "选择输出目录",
    "Select Data Folder": "选择数据文件夹",
    "Cache Dir": "缓存目录",
    "Select Color": "选择颜色",
    "Hold to temporarily show Data Layer to check alignment": "按住临时显示数据层以检查对齐",
    "e.g. _origin": "例如 _origin",
    "Save ROI JSON": "保存 ROI JSON",
    "Load ROI JSON": "加载 ROI JSON",
    "JSON Load Error": "JSON 加载错误",
    "Export already in progress.": "导出任务进行中。",
    "No ROIs to preview.": "没有需要预览的 ROI。",
    "No previous ROI size for stamp.": "没有上一个 ROI 尺寸可用于标记。",
    "Select a view layer first.": "请先选择视图图层。",
    "1×1 grid has no lines.": "1×1 网格没有线条。",
    "Crop applied. Press '%s' to Undo.": "裁剪已应用。按 '%s' 撤销。",
    "Loaded %d ROIs.": "已加载 %d 个 ROI。",
    "Stamped ROI #%d (%s)": "标记 ROI #%d (%s)",
    "Cloned ROI #%d → #%d": "克隆 ROI #%d → #%d",
    "Cloned & Dragging ROI #%d": "克隆并拖动 ROI #%d",
    "Grid %d×%d enabled. Right-click cell to zoom.": "网格 %d×%d 已启用。右键点击单元格缩放。",
    "Zoomed to cell. Press Esc to zoom out.": "已缩放至单元格。按 Esc 缩小。",
    "Restored global view.": "已恢复全局视图。",
    "Failed to add layer: %s": "无法添加图层: %s",
    "Date: %s, Exp: %ss": "日期: %s, 曝光: %ss",
    "Pixel: %.2f Å, Mean: %.1f": "像素: %.2f Å, 均值: %.1f",
    "DANGER! %d workers may crash your PC": "危险！%d 个线程可能导致崩溃",
    "Warning: %d exceeds recommended (%d). Watch RAM.": "警告: %d 超出推荐值 (%d)。请关注内存。",
    "Supported formats:": "支持的格式:",
    "Range: 0-10": "范围: 0-10",
    "Single: 5": "单帧: 5",
    "Mixed: 0-5, 8, 10-12": "混合: 0-5, 8, 10-12",
    "e.g. 0-10, 15, 20-25": "例如 0-10, 15, 20-25",
    "e.g. CRY2, BSA": "例如 CRY2, BSA",
    "e.g. ds1, ds2": "例如 ds1, ds2",
    "All (Default) or e.g. 0-99": "全部 (默认) 或例如 0-99",
    # Liquid Cell Mask
    "Auto-detect Mask": "自动检测液层边界",
    "Detect liquid cell boundary from temporal variance. Used by Auto-suggest and YOLO.": "根据时间维度方差自动检测液层边界。Auto-suggest 和 YOLO 共用。",
    "Edit Mask": "编辑 Mask",
    "Switch to mask layer for manual editing (paint=1, erase=0)": "切换到 Mask 图层手动编辑（绘制=1, 擦除=0）",
    "Clear Mask": "清除 Mask",
    "Mask": "液层蒙版",
    "Mask detected.": "液层边界已检测。",
    "Edit Mask to refine.": "可编辑 Mask 微调。",
    "Please auto-detect or load a data layer first.": "请先自动检测或加载数据图层。",
    "Painting mask. Use paint (label=1) and erase (label=0).": "正在绘制 Mask。绘制(label=1), 擦除(label=0)。",
    "Mask cleared.": "Mask 已清除。",
    "Detecting liquid cell boundary...": "正在检测液层边界...",
    # YOLO
    "Auto-detect (YOLO)": "自动检测 (YOLO)",
    "Run trained YOLO model to detect particles within the mask": "在液层蒙版内运行 YOLO 模型检测粒子",
    "YOLO model not yet trained. Accumulate 300+ ROIs first.": "YOLO 模型尚未训练。请先积累 300+ 个 ROI。",
    "Running YOLO detection...": "正在运行 YOLO 检测...",
    "candidates detected": "个候选粒子检出",
    # ROI Preview
    "Preview ROI": "预览 ROI",
    "Open cropped animation of selected ROI with auto-contrast": "打开选中 ROI 的裁剪动画（自动增强对比度）",
    "Preview": "预览",
    "Please select an ROI first (click on a rectangle).": "请先选中一个 ROI（点击矩形框）。",
    "Selected ROI has zero area.": "所选 ROI 面积为零。",
    "Preview opened:": "预览已打开:",
    "Export current layer": "导出当前图层",
    "Select export folder": "选择导出文件夹",
    "Export Enhanced (CLAHE)": "导出增强版 (CLAHE)",
    "Export CLAHE-enhanced crops alongside originals (local contrast adjustment)": "在原始裁剪旁导出 CLAHE 局部增强版（方便对比和后续处理）",
    # ROI Video Export (Session 13)
    "Export ROI Videos": "导出 ROI 视频",
    "Export each ROI as a small region video with per-ROI auto contrast": "为每个 ROI 单独生成局域对比度小视频",
    "No ROIs defined. Please draw or load ROIs first.": "尚未定义 ROI。请先绘制或载入 ROI。",
    "Please select a Data Layer first.": "请先选择数据图层。",
    "Source": "源数据",
    "Data Layer (Crop Source):": "数据图层（裁剪源）：",
    "View Layer (info only):": "视图图层（仅参考）：",
    "Refresh ROI List": "刷新 ROI 列表",
    "ROIs (uncheck to skip)": "ROI 列表（取消勾选跳过）",
    "Export": "导出",
    "Label": "名称",
    "Contrast Min": "对比度下限",
    "Contrast Max": "对比度上限",
    "Frame Range": "帧范围",
    "Per-ROI Actions": "单 ROI 操作",
    "Select All": "全选",
    "Select None": "全不选",
    "Only Napari-Selected": "仅 Napari 选中",
    "Match the ROIs currently selected in the napari Batch_ROI layer": "匹配 napari Batch_ROI 图层当前选中的 ROI",
    "Auto Contrast All Enabled": "对所有已勾选 ROI 重算对比度",
    "Auto": "自动",
    "Recompute auto contrast from this ROI's data": "从当前 ROI 的数据重算自动对比度",
    "Open this ROI in a new napari viewer with current contrast": "在新的 napari 窗口中预览当前 ROI（应用当前对比度）",
    "Video Parameters": "视频参数",
    "Target Duration (s):": "目标时长（秒）：",
    "Auto-compute FPS from frame count and target duration": "根据帧数与目标时长自动计算 FPS",
    "Codec:": "编码器：",
    "Quality (CRF, lower=better):": "画质（CRF，越低越好）：",
    "18=best, 23=default, 28=smaller (FFmpeg only)": "18=最佳，23=默认，28=较小（仅 FFmpeg 支持）",
    "Encoding Preset:": "编码速度：",
    "Output": "输出",
    "Output Directory:": "输出目录：",
    "Browse...": "选择...",
    "Computed FPS:": "已计算 FPS：",
    "frames": "帧",
    "Computing auto-contrast for each ROI...": "正在为每个 ROI 计算自动对比度...",
    "FFmpeg not detected — will use OpenCV fallback.": "未检测到 FFmpeg — 将使用 OpenCV 回退方案。",
    "No Batch_ROI layer found. Please draw ROIs first.": "未找到 Batch_ROI 图层。请先绘制 ROI。",
    "Data layer not found in viewer.": "在查看器中未找到数据图层。",
    "No ROIs drawn.": "尚未绘制 ROI。",
    "No codec available.": "没有可用的编码器。",
    "Selection": "选择",
    "No ROI is currently selected in napari.": "napari 中当前未选中任何 ROI。",
    "Auto contrast recomputed for all enabled ROIs.": "已为所有勾选 ROI 重算对比度。",
    "Nothing to export": "无可导出项",
    "No ROI is enabled for export.": "没有勾选任何 ROI。",
    "Please choose an output directory.": "请选择输出目录。",
    "Invalid contrast": "对比度无效",
    "Codec": "编码器",
    "Cancelling...": "正在取消...",
    "Confirm": "确认",
    "Export in progress. Cancel and close?": "正在导出。取消并关闭吗？",
    "Export Videos": "导出视频",
    "Export complete": "导出完成",
    "All ROI videos exported.": "全部 ROI 视频已导出。",
    "All exports failed.": "全部导出失败。",
    "Cancelled.": "已取消。",
    "succeeded": "成功",
    "failed": "失败",
    "Output:": "输出：",
    "Export error": "导出错误",
    "all": "全部",
    # Phase 0 (2026-05-29): missing keys from ROI video export dialog
    "Actions": "操作",
    "ROIs (uncheck to skip, click row to preview)": "ROI 列表（取消勾选跳过，点击行预览）",
    "Click a row to preview": "点击某一行以查看预览",
    "Preview frame:": "预览帧：",
    "FPS:": "FPS：",
    "CRF:": "CRF：",
    "Preset:": "速度预设：",
    "Directory:": "输出目录：",
    "Export Mode": "导出模式",
    "Adjusted only (one file per ROI)": "仅调整后（每个 ROI 一个文件）",
    "Original auto-contrast only": "仅原始自动对比度",
    "Both (adjusted + original)": "两者都导出（调整后 + 原始）",
    "Adjusted: use the values you tuned per ROI.\nOriginal: use the initial auto-contrast.\nBoth: emit two files per ROI (_adjusted.mp4 + _original.mp4).": "调整后：使用每个 ROI 手动调节的值。\n原始：使用初始自动对比度。\n两者都：每个 ROI 生成两个文件（_adjusted.mp4 + _original.mp4）。",
    "(empty or invalid ROI)": "（ROI 为空或无效）",
    "Select Output Directory": "选择输出目录",
    "Exporting": "正在导出",
    "file(s)": "个文件",
    "Reset to the initial auto-contrast value": "重置为初始自动对比度值",
    "ROIs loaded.": "个 ROI 已载入。",
    "Crop error:": "裁剪错误：",
    "frame": "帧",
    "contrast adjusted=": "调整后对比度=",
    "original=": "原始=",
    "Auto contrast failed:": "自动对比度失败：",
    "reset to original": "已重置为原始值",

    # Phase 1 (2026-05-29): batch export confirm dialog
    "Confirm Batch Export": "确认批量导出",
    "Select ROIs to export": "选择要导出的 ROI",
    "BBox (y1,x1,y2,x2)": "BBox (y1,x1,y2,x2)",
    "Frames": "帧范围",
    "Magnifier Preview": "Magnifier 预览",
    "Loading preview...": "加载预览中...",
    "Preview (napari current frame)": "预览（napari 当前帧）",
    "(none — uses napari default)": "（无 — 使用 napari 默认对比度）",
    "Apply Magnifier preview contrast (ROIs that have it)": "应用 Magnifier preview 对比度（仅对已设置的 ROI 生效）",
    "TIFF output: contrast is burned in (grayscale), LUT is dropped (TIFF stack stays single-channel).":
        "TIFF 输出：灰度对比度直接烧入，LUT 被丢弃（TIFF 堆栈保持单通道）。",
    "PNG output: contrast burned in; non-Gray LUT produces RGB PNG.":
        "PNG 输出：对比度烧入；非 Gray LUT 生成 RGB PNG。",
    "No ROI has Magnifier-saved preview values. Open Magnifier, tune contrast and Apply first.":
        "没有 ROI 保存了 Magnifier preview 值。请先打开 Magnifier 调整对比度并 Apply。",
    "Also generate overview map (overview.png)": "同时生成概览图 (overview.png)",
    "Overview map uses napari's current display contrast (not Magnifier preview).":
        "概览图使用 napari 当前显示对比度（不是 Magnifier preview）。",
    "Invert": "反选",
    "Options": "选项",
    "No ROI selected.": "未选择任何 ROI。",
    "No ROI selected for export.": "未选择要导出的 ROI。",
    "ROI(s)": "个 ROI",
    "frames": "帧",
    "files, ~": "个文件, ~",
    "estimate": "估算",
    "Frame": "帧",
    "source": "源",
    "ROIs marked for export": "已标记导出",
    "Preview unavailable:": "预览不可用：",
    "all": "全部",

    # Phase 2 (2026-05-29): Recovery chapter UI
    "Chapters (toggle to include historical loads)": "章节（勾选以包含历史载入）",
    "Each load_dm4 / load_png / load_tiff starts a new chapter. By default only the current chapter is replayed.":
        "每次 load_dm4 / load_png / load_tiff 都会开启新章节。默认只回放当前章节。",
    "current": "当前",
    "historical": "历史",
    "actions": "个操作",
    "(No actions in selected chapters)": "（所选章节中无操作）",

    # Phase 5 (2026-05-29): trial/committed UI
    "Commit Selected Trials": "提交所选试探",
    "Promote selected trial actions to committed (persists to session JSON). Committed actions are always replayed.":
        "把所选 trial 操作升级为 committed (写入 session JSON)。Committed 总是回放。",
    "Commit Trials": "提交试探",
    "Session log path missing.": "Session 日志路径缺失。",
    "Session file not found.": "Session 文件未找到。",
    "No trial actions selected. Select trial rows in the list first.":
        "未选择任何 trial 操作。请先在列表中选中 🧪 行。",
    "Nothing to commit.": "没有需要提交的操作。",
    "Committed %d trial action(s).": "已提交 %d 个 trial 操作。",
    # 2026-07-16: Commit 语义澄清 + 恢复正确性提示
    "Bookkeeping only: marks the selected trial as the CHOSEN version for future replay (writes state in the session JSON). It does NOT re-apply anything to the image — the result was already produced live when you clicked Apply. Nothing on the canvas changes.":
        "仅记账：把所选 trial 标记为将来【回放采用】的版本（写入 session JSON 的 state）。"
        "它不会把任何东西重新应用到图像——效果在你当初点 Apply 时就已实时产生。画布不会有任何变化。",
    "This only records which version replay should use; it does not re-apply anything to the image (the canvas is unchanged).":
        "这只是记录回放应采用哪个版本；不会把任何东西重新应用到图像（画布保持不变）。",
    "All selected actions were superseded or undone — nothing to replay. Select the effective (non-grayed) rows.":
        "所选操作全部已被取代或撤销——没有可回放的内容。请选中【有效（未变灰）】的行。",

    # Phase 6 (2026-05-29): edit_records right-click menu
    "Enable (clear disable/delete)": "启用（清除禁用/删除）",
    "Disable": "禁用",
    "Delete": "删除",
    "Edit Record": "编辑记录",
    "View / Edit Details": "查看 / 编辑详情",
    "Reset to original state": "重置到原始状态",

    # Issue #5 (2026-05-29): action detail / param-edit dialog
    "Action Details": "操作详情",
    "Overview (read-only)": "概要（只读）",
    "Widget:": "组件:",
    "Action:": "操作:",
    "State:": "状态:",
    "Action ID:": "操作 ID:",
    "Timestamp:": "时间:",
    "Edits applied:": "已应用的编辑:",
    "Result:": "结果:",
    "Parameters (editable — saved as a non-destructive edit)": "参数（可编辑 — 以非破坏性编辑保存）",
    "Key": "键",
    "Value": "值",
    "(This action has no editable parameters.)": "（此操作没有可编辑的参数。）",
    "Parameter Edit": "参数编辑",
    "Saved %d change(s): %s\n\nThe action now shows ✏️ in the list; "
    "recovery will replay with the new parameters.":
        "已保存 %d 处改动: %s\n\n该操作现在列表中显示 ✏️ 标记; 恢复时将以新参数回放。",
    "No parameter changes detected.": "未检测到参数改动。",
    "Parameters edited (double-click to view/edit).": "参数已被编辑（双击查看/修改）。",

    # Param-edit validation (2026-05-29): read-only reasons + Plan C messages
    "Gray rows are read-only (derived / measured / structured / "
    "layer refs). Edit an editable value then click OK; values "
    "are parsed as JSON (e.g. 12, 1.5, true, [1,2], \"text\").":
        "灰色行为只读（派生 / 测量 / 结构化 / 图层引用）。编辑可编辑的值后点 OK; "
        "值按 JSON 解析（如 12, 1.5, true, [1,2], \"text\"）。",
    "Derived from other fields (e.g. roi_count = len(rois)); "
    "editing it has no effect on recovery.":
        "派生自其它字段（如 roi_count = len(rois)）; 单独改它对恢复无影响。",
    "A measured output; recovery re-measures this value, "
    "so editing it has no effect.":
        "这是测量输出; 恢复时会重新测量该值, 改它无影响。",
    "Layer reference; renaming it here can make recovery "
    "fail to find the source layer.":
        "图层引用; 在此改名可能导致恢复时找不到源图层。",
    "Structured data — edit it in its own panel, "
    "not as raw JSON here.":
        "结构化数据 — 请在其对应面板里编辑, 不要在此改原始 JSON。",
    "Invalid Parameter Edit": "参数编辑无效",
    "These edits were NOT saved. Fix the values and click OK again:":
        "以下修改未保存。请修正后再次点 OK:",
    "%s: must be a non-negative integer frame index (got %r)":
        "%s: 必须是非负整数帧号（得到 %r）",
    "%s: frame %s is out of range [0, %s]":
        "%s: 帧号 %s 超出范围 [0, %s]",
    "%s: must be 4 numbers [x1, y1, x2, y2] (got %r)":
        "%s: 必须是 4 个数 [x1, y1, x2, y2]（得到 %r）",
    "%s: requires x2 > x1 and y2 > y1 (got %s)":
        "%s: 需满足 x2 > x1 且 y2 > y1（得到 %s）",
    "%s: must be a positive finite number (got %r)":
        "%s: 必须是正的有限数值（得到 %r）",
    "%s: must be a positive integer (got %r)":
        "%s: 必须是正整数（得到 %r）",
    "%s: must be a finite number (got %r)":
        "%s: 必须是有限数值（得到 %r）",

    # Recovery (2026-05-30): reverted (Ctrl+Z) drift trials kept as history
    "Reverted with Ctrl+Z (kept as history); "
    "not replayed unless you select it.":
        "已用 Ctrl+Z 撤销（保留为历史记录）; 除非手动勾选, 否则不会回放。",
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
        "shortcut_hide_roi": "K",      # [New] 隐藏 ROI 标注
        "shortcut_toggle_overview": "V",  # [New] 切换概览图独立源
        "shortcut_set_range": "R",     # [New] 为选中 ROI 设置帧范围
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
        "geo_export_view": False,    # [New] 默认导出视图层
        "geo_hide_roi_labels": False, # [New] 隐藏 ROI 标注
        "geo_overview_use_view": False, # [New] 概览图使用视图层数据
        "geo_roi_history_max": 128,   # [New] ROI 撤销历史记录上限

        # ROI Clone & Stamp (快速复制)
        "geo_clone_modifier": "Shift",        # 克隆ROI的修饰键 (Shift+拖拽)
        "geo_stamp_modifier": "Alt",      # 盖章ROI的修饰键 (Alt+点击)
        "geo_roi_stamp_center": True,       # 盖章时以点击位置为中心 (False=左上角)

        # Grid Partition (分区网格)
        "geo_grid_rows": 3,      # 默认网格行数
        "geo_grid_cols": 3,      # 默认网格列数

        # Suffixes
        "geo_suffix_lrtem": "_lrtem",
        "geo_suffix_hrtem": "_hrtem",
        "geo_suffix_mask": "_mask",
        "geo_suffix_mask_refined": "_mask_refined",  # 精修掩膜后缀 (与 Finetuning 标准化输出一致; 旧默认 _mask_new 已标准化)

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
        "style_batch_font_max_size": 24,  # [Fix 3] 字体自适应上限
        "style_batch_font_spacing": 0,    # [Fix 5] 字体行间距
        
        "geo_overview_frame": "",
        "geo_overview_font_size": 24,
        "geo_overview_box_color": "#FFFF00",
        "geo_overview_text_color": "#FFFF00",
        "geo_overview_text_pos": "Top",
        
        "style_presets": {},

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
        "session_recovery_retention": "lean",  # 中间层保留: lean(省内存驱逐) / full(全保留·写盘) / custom(自选)
        "session_auto_detect_source": True,  # 恢复前自动检测数据源
        "session_confirm_export": True,  # 导出操作前询问用户
        "session_auto_continue_after_recovery": True,  # recovery 后自动续写
        
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


def set_font_for_widget(widget: QWidget, is_bold: bool = False, size: int = None):
    font = widget.font()
    if size: font.setPointSize(size)
    if is_bold: font.setBold(True)
    widget.setFont(font)




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
            self.style_batch_w, self.style_batch_font, self.style_batch_font_max, self.style_batch_font_spacing,
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
        btn_browse.clicked.connect(lambda: self.cache_dir_edit.setText(QFileDialog.getExistingDirectory(self, tr("Cache Dir"))))
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
        
        # Modifier Keys Configuration
        h_clone = QHBoxLayout()
        self.clone_mod_edit = QComboBox()
        self.clone_mod_edit.addItems(["Shift", "Control", "Alt"])
        curr_clone = str(GlobalConfig.get("geo_clone_modifier"))
        if curr_clone in ["Shift", "Control", "Alt"]:
            self.clone_mod_edit.setCurrentText(curr_clone)
            
        lbl_warn = QLabel(tr("⚠ (Not recommended to use 'Alt')"))
        lbl_warn.setStyleSheet("color: orange; font-style: italic; font-size: 10px;")
        h_clone.addWidget(self.clone_mod_edit)
        h_clone.addWidget(lbl_warn)
        h_clone.addStretch()
        f_ui.addRow(tr("Clone/Drag Modifier:"), h_clone)
        
        self.stamp_mod_edit = QComboBox()
        self.stamp_mod_edit.addItems(["Shift", "Control", "Alt"])
        curr_stamp = str(GlobalConfig.get("geo_stamp_modifier"))
        if curr_stamp in ["Shift", "Control", "Alt"]:
            self.stamp_mod_edit.setCurrentText(curr_stamp)
            
        f_ui.addRow(tr("Stamp Modifier:"), self.stamp_mod_edit)
        
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
        self.geo_export_view_check = QCheckBox(tr("Export View Layer by Default"))
        self.geo_export_view_check.setChecked(bool(GlobalConfig.get("geo_export_view")))
        
        self.geo_hide_roi_labels_check = QCheckBox(tr("Hide ROI Labels"))
        self.geo_hide_roi_labels_check.setChecked(bool(GlobalConfig.get("geo_hide_roi_labels")))
        self.geo_overview_use_view_check = QCheckBox(tr("Overview Uses View Layer"))
        self.geo_overview_use_view_check.setChecked(bool(GlobalConfig.get("geo_overview_use_view")))
        
        f_geo.addRow("", self.geo_enl_check)
        f_geo.addRow("", self.geo_keep_idx_check)
        f_geo.addRow("", self.geo_sq_check)
        f_geo.addRow("", self.geo_denoise_check)
        f_geo.addRow("", self.geo_refine_check)
        f_geo.addRow("", self.geo_export_view_check)
        f_geo.addRow("", self.geo_hide_roi_labels_check)
        f_geo.addRow("", self.geo_overview_use_view_check)
        
        # ROI 撤销历史记录上限
        self.geo_history_max_spin = QSpinBox()
        self.geo_history_max_spin.setRange(10, 1000)
        self.geo_history_max_spin.setValue(int(GlobalConfig.get("geo_roi_history_max")))
        self.geo_history_max_spin.setToolTip(tr("Maximum ROI undo history. Larger values use more memory."))
        f_geo.addRow(tr("ROI Undo History:"), self.geo_history_max_spin)
        
        f_geo.addRow(QLabel("<hr>")) # 分割线

        self.suff_main = QLineEdit(str(GlobalConfig.get("geo_suffix")))
        self.suff_lr = QLineEdit(str(GlobalConfig.get("geo_suffix_lrtem")))
        self.suff_hr = QLineEdit(str(GlobalConfig.get("geo_suffix_hrtem")))
        self.suff_mask = QLineEdit(str(GlobalConfig.get("geo_suffix_mask")))
        self.suff_refined = QLineEdit(str(GlobalConfig.get("geo_suffix_mask_refined")))
        f_geo.addRow(tr("Main Suffix:"), self.suff_main)
        f_geo.addRow(tr("LR Suffix:"), self.suff_lr)
        f_geo.addRow(tr("HR Suffix:"), self.suff_hr)
        f_geo.addRow(tr("Mask Suffix:"), self.suff_mask)
        f_geo.addRow(tr("Refined Mask Suffix:"), self.suff_refined)
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
                c = QColorDialog.getColor(QColor(line.text()), self, tr("Select Color"))
                if c.isValid():
                    hex_c = c.name()
                    line.setText(hex_c)
                    btn.setStyleSheet(f"background-color: {hex_c}; border: 1px solid #555;")
            
            btn.clicked.connect(pick)
            h.addWidget(line); h.addWidget(btn)
            layout.addLayout(h)
            return line

        # === 0. Style Presets [New] ===
        g_presets = QGroupBox(f"🎭 {tr('Style Presets')}")
        l_presets = QVBoxLayout()
        h_presets = QHBoxLayout()
        
        self.preset_combo = QComboBox()
        self._refresh_preset_combo()
        h_presets.addWidget(self.preset_combo, stretch=1)
        
        btn_load_preset = QPushButton(f"🔄 {tr('Load')}")
        btn_load_preset.clicked.connect(self._load_style_preset)
        h_presets.addWidget(btn_load_preset)
        
        btn_save_preset = QPushButton(f"💾 {tr('Save Current...')}")
        btn_save_preset.clicked.connect(self._save_style_preset)
        h_presets.addWidget(btn_save_preset)
        
        btn_del_preset = QPushButton("🗑️")
        btn_del_preset.setFixedWidth(40)
        btn_del_preset.clicked.connect(self._delete_style_preset)
        h_presets.addWidget(btn_del_preset)
        
        l_presets.addLayout(h_presets)
        g_presets.setLayout(l_presets)
        l.addWidget(g_presets)

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
        
        # [Fix 3] Max font size SpinBox
        h_b2 = QHBoxLayout()
        self.style_batch_font_max = QSpinBox(); self.style_batch_font_max.setRange(10, 100); self.style_batch_font_max.setValue(int(GlobalConfig.get("style_batch_font_max_size")))
        self.style_batch_font_max.setToolTip(tr("Upper limit for adaptive font scaling when zoomed out."))
        h_b2.addWidget(QLabel(tr("Max Font Size:"))); h_b2.addWidget(self.style_batch_font_max)
        
        # [Fix 5] Line spacing SpinBox
        self.style_batch_font_spacing = QSpinBox(); self.style_batch_font_spacing.setRange(0, 10); self.style_batch_font_spacing.setValue(int(GlobalConfig.get("style_batch_font_spacing")))
        self.style_batch_font_spacing.setToolTip(tr("Line spacing between ROI label and frame info."))
        h_b2.addWidget(QLabel(tr("Label Spacing:"))); h_b2.addWidget(self.style_batch_font_spacing)
        
        l_batch.addLayout(h_b2)
        
        g_batch.setLayout(l_batch); l.addWidget(g_batch)

        # 4. Overview Map [New]
        g_overview = QGroupBox(f"🖼️ {tr('Overview Map Settings')}")
        l_overview = QVBoxLayout()
        
        self.geo_ov_box_c = add_color_row(l_overview, tr("Box Color:"), "geo_overview_box_color")
        self.geo_ov_txt_c = add_color_row(l_overview, tr("Text Color:"), "geo_overview_text_color")
        self.geo_ov_box_c.textChanged.connect(lambda t: GlobalConfig.set("geo_overview_box_color", t))
        self.geo_ov_txt_c.textChanged.connect(lambda t: GlobalConfig.set("geo_overview_text_color", t))
        
        h_ov = QHBoxLayout()
        h_ov.addWidget(QLabel(tr("Font Size:")))
        self.geo_ov_font = QSpinBox(); self.geo_ov_font.setRange(10, 100); self.geo_ov_font.setValue(int(GlobalConfig.get("geo_overview_font_size")))
        self.geo_ov_font.valueChanged.connect(lambda v: GlobalConfig.set("geo_overview_font_size", v))
        h_ov.addWidget(self.geo_ov_font)
        
        h_ov.addWidget(QLabel(tr("Text Pos:")))
        self.geo_ov_pos = QComboBox()
        self.geo_ov_pos.addItems(["Top", "Bottom", "Left", "Right"])
        self.geo_ov_pos.setCurrentText(str(GlobalConfig.get("geo_overview_text_pos")))
        self.geo_ov_pos.currentTextChanged.connect(lambda text: GlobalConfig.set("geo_overview_text_pos", text))
        h_ov.addWidget(self.geo_ov_pos)
        
        l_overview.addLayout(h_ov)
        g_overview.setLayout(l_overview); l.addWidget(g_overview)

        l.addStretch(); w.setLayout(l)
        return w

    def _refresh_preset_combo(self):
        self.preset_combo.clear()
        presets: dict = GlobalConfig.get("style_presets") or {}
        for name in presets.keys():
            self.preset_combo.addItem(name)
            
    def _save_style_preset(self):
        from qtpy.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, tr("Preset Name"), tr("Enter a name for the current visual style preset:"))
        if ok and name.strip():
            presets: dict = GlobalConfig.get("style_presets") or {}
            
            # Pack current panel styles
            current_style = {
                "style_measure_color": self.style_meas_col.text(),
                "style_measure_width": self.style_meas_w.value(),
                "style_measure_font_size": self.style_meas_font.value(),
                "style_crop_color": self.style_crop_col.text(),
                "style_batch_box_color": self.style_batch_box_col.text(),
                "style_batch_text_color": self.style_batch_txt_col.text(),
                "style_batch_width": self.style_batch_w.value(),
                "style_batch_font_size": self.style_batch_font.value(),
                "geo_overview_box_color": self.geo_ov_box_c.text(),
                "geo_overview_text_color": self.geo_ov_txt_c.text(),
                "geo_overview_font_size": self.geo_ov_font.value(),
                "geo_overview_text_pos": self.geo_ov_pos.currentText(),
            }
            
            presets[name.strip()] = current_style
            GlobalConfig.set("style_presets", presets)
            self._refresh_preset_combo()
            self.preset_combo.setCurrentText(name.strip())
            
    def _load_style_preset(self):
        name = self.preset_combo.currentText()
        if not name: return
        
        presets: dict = GlobalConfig.get("style_presets") or {}
        if name in presets:
            st = presets[name]
            # Load back into visual styles UI components 
            # Note: We must also trigger the color button updates manually to keep it in sync.
            def _apply_color(line_widget, color_val):
                line_widget.setText(color_val)
                # Find sibling button and update its stylesheet
                parent_layout = line_widget.parentWidget().layout()
                if parent_layout:
                    # Very hacky way to find the button inside the layout created by add_color_row
                    for i in range(parent_layout.count()):
                        item = parent_layout.itemAt(i)
                        if item and item.layout():
                            sub_layout = item.layout()
                            for j in range(sub_layout.count()):
                                widget = sub_layout.itemAt(j).widget()
                                if widget == line_widget:
                                    # The button is exactly next to it
                                    btn = sub_layout.itemAt(j+1).widget()
                                    if btn and isinstance(btn, QPushButton):
                                        btn.setStyleSheet(f"background-color: {color_val}; border: 1px solid #555;")
            
            if "style_measure_color" in st: _apply_color(self.style_meas_col, st["style_measure_color"])
            if "style_measure_width" in st: self.style_meas_w.setValue(st["style_measure_width"])
            if "style_measure_font_size" in st: self.style_meas_font.setValue(st["style_measure_font_size"])
            
            if "style_crop_color" in st: _apply_color(self.style_crop_col, st["style_crop_color"])
            
            if "style_batch_box_color" in st: _apply_color(self.style_batch_box_col, st["style_batch_box_color"])
            if "style_batch_text_color" in st: _apply_color(self.style_batch_txt_col, st["style_batch_text_color"])
            if "style_batch_width" in st: self.style_batch_w.setValue(st["style_batch_width"])
            if "style_batch_font_size" in st: self.style_batch_font.setValue(st["style_batch_font_size"])
            
            if "geo_overview_box_color" in st: _apply_color(self.geo_ov_box_c, st["geo_overview_box_color"])
            if "geo_overview_text_color" in st: _apply_color(self.geo_ov_txt_c, st["geo_overview_text_color"])
            if "geo_overview_font_size" in st: self.geo_ov_font.setValue(st["geo_overview_font_size"])
            if "geo_overview_text_pos" in st: self.geo_ov_pos.setCurrentText(st["geo_overview_text_pos"])

    def _delete_style_preset(self):
        name = self.preset_combo.currentText()
        if not name: return
        presets: dict = GlobalConfig.get("style_presets") or {}
        if name in presets:
            del presets[name]
            GlobalConfig.set("style_presets", presets)
            self._refresh_preset_combo()

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
        self.session_substance.setPlaceholderText(tr("e.g. CRY2, BSA"))
        f_log.addRow(tr("Default Substance:"), self.session_substance)
        
        # 默认 Dataset ID
        self.session_dataset = QLineEdit(str(GlobalConfig.get("session_dataset_default")))
        self.session_dataset.setPlaceholderText(tr("e.g. ds1, ds2"))
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
            "shortcut_hide_roi": tr("Hide ROI Labels"),
            "shortcut_toggle_overview": tr("Overview Uses View Layer"),
            "shortcut_set_range": tr("Set Frame Range (Selected ROI)"),
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

        # 4. Save Geometry / Interaction Options [New]
        GlobalConfig.set("geo_clone_modifier", self.clone_mod_edit.currentText(), emit_signal=False)
        GlobalConfig.set("geo_stamp_modifier", self.stamp_mod_edit.currentText(), emit_signal=False)
        GlobalConfig.set("geo_enlarge", self.geo_enl_check.isChecked(), emit_signal=False)
        GlobalConfig.set("geo_keep_index", self.geo_keep_idx_check.isChecked(), emit_signal=False)
        GlobalConfig.set("geo_force_square", self.geo_sq_check.isChecked(), emit_signal=False)
        GlobalConfig.set("geo_create_denoise", self.geo_denoise_check.isChecked(), emit_signal=False)
        GlobalConfig.set("geo_create_refine", self.geo_refine_check.isChecked(), emit_signal=False)
        GlobalConfig.set("geo_export_view", self.geo_export_view_check.isChecked(), emit_signal=False)
        GlobalConfig.set("geo_hide_roi_labels", self.geo_hide_roi_labels_check.isChecked(), emit_signal=False)
        GlobalConfig.set("geo_overview_use_view", self.geo_overview_use_view_check.isChecked(), emit_signal=False)
        GlobalConfig.set("geo_roi_history_max", self.geo_history_max_spin.value(), emit_signal=False)

        # 5. Save Suffixes
        GlobalConfig.set("geo_suffix", self.suff_main.text(), emit_signal=False)
        GlobalConfig.set("geo_suffix_lrtem", self.suff_lr.text(), emit_signal=False)
        GlobalConfig.set("geo_suffix_hrtem", self.suff_hr.text(), emit_signal=False)
        GlobalConfig.set("geo_suffix_mask", self.suff_mask.text(), emit_signal=False)
        GlobalConfig.set("geo_suffix_mask_refined", self.suff_refined.text(), emit_signal=False)

        # Save Styles
        GlobalConfig.set("style_measure_color", self.style_meas_col.text(), emit_signal=False)
        GlobalConfig.set("style_measure_width", self.style_meas_w.value(), emit_signal=False)
        GlobalConfig.set("style_measure_font_size", self.style_meas_font.value(), emit_signal=False)
        GlobalConfig.set("style_crop_color", self.style_crop_col.text(), emit_signal=False)
        GlobalConfig.set("style_batch_box_color", self.style_batch_box_col.text(), emit_signal=False)
        GlobalConfig.set("style_batch_text_color", self.style_batch_txt_col.text(), emit_signal=False)
        GlobalConfig.set("style_batch_width", self.style_batch_w.value(), emit_signal=False)
        GlobalConfig.set("style_batch_font_size", self.style_batch_font.value(), emit_signal=False)
        GlobalConfig.set("style_batch_font_max_size", self.style_batch_font_max.value(), emit_signal=False)
        GlobalConfig.set("style_batch_font_spacing", self.style_batch_font_spacing.value(), emit_signal=False)

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
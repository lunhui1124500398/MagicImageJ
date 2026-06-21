"""
TEM Data Processing Workflow - Napari GUI
统一的TEM数据处理平台
[Fix] 增加了 resize 初始化，防止界面在启动时过大。
"""
import sys
import os

# =========================================================================
# 【关键修复】标准输出重定向
# 解决 PyInstaller 打包无控制台模式下，tqdm 和 print 导致的崩溃问题
# 原理：给 tqdm 一个“假的” write 方法，让它把文字写到虚空里，
# 这样程序就不会崩，而你的 GUI 进度条 (QProgressBar) 依然可以正常接收信号！
# =========================================================================
class NullWriter:
    def write(self, text):
        pass # 什么都不做，假装写成功了
    def flush(self):
        pass
    def isatty(self):
        return False

# 只有在打包后的环境 (frozen) 且没有控制台的情况下才劫持
if getattr(sys, 'frozen', False):
    if sys.stdout is None:
        sys.stdout = NullWriter()
    if sys.stderr is None:
        sys.stderr = NullWriter()

import napari
from qtpy.QtWidgets import QTabWidget, QDockWidget, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QApplication
from qtpy.QtCore import Qt, QTimer, QSettings
from qtpy.QtGui import QFont, QIcon
from widgets.import_widget import ImportWidget
from widgets.drift_widget import DriftCorrectionWidget
from widgets.filter_widget import FilterWidget  # [NEW] 帧筛选控件
from widgets.geometry_widget import GeometryWidget
from widgets.enhance_widget import EnhanceWidget
from widgets.annotation_widget import AnnotationWidget
from widgets.export_widget import ExportWidget
from widgets.script_widget import ScriptWidget
from widgets.recovery_widget import RecoveryWidget
from widgets.settings_widget import SettingsDialog, GlobalConfig, tr
import numpy as np
import math
import ctypes
from utils.utils import resource_path
from napari.qt import get_qapp


class TEMWorkflow:
    def __init__(self):
        if os.name == 'nt':
            myappid = 'WHKTZ.YSImageJ.TEMWORKFLOW.V1'  # 任意唯一的字符串
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        self.viewer = napari.Viewer(title="TEM Data Processing Workflow-YSImageJ")
        QSettings("NapariUser", "Global").remove("archive_path")
        # === [新增代码] 启动时清除 Dataset ID，防止不归档时出现上次的 ID ===
        QSettings("NapariUser", "Global").remove("current_dataset_id")
        # === Fix: Set a reasonable default size to prevent layout overflow ===
        self.viewer.window.resize(1200, 800)

        icon_path = resource_path(os.path.join("assets", "app_icon.ico"))
        print(f"DEBUG: Loading icon from: {icon_path}")
        
        if os.path.exists(icon_path):
            app_icon = QIcon(icon_path)
            # 获取 Napari 的底层 Qt 窗口对象 (QMainWindow)
            qt_window = self.viewer.window._qt_window
            # 设置窗口图标
            qt_window.setWindowIcon(app_icon)
            app = get_qapp()
            app.setWindowIcon(app_icon)
            self.viewer.window._qt_window.setWindowIcon(app_icon)
        else:
            print(f"Warning: Icon file not found at {icon_path}")

        # font_family = ' "Segoe UI", "Microsoft YaHei", "PingFang SC", "Helvetica Neue", sans-serif'
        # 针对 Napari 的 Qt 主窗口应用样式
        app = QApplication.instance()
        if app:
            # 优先使用微软雅黑/苹方，无衬线字体作为回退
            font = QFont("Microsoft YaHei")
            font.setStyleHint(QFont.SansSerif)
            font.setPointSize(10) # 设置字号 10pt
            app.setFont(font)
        
        self._setup_widgets()
        # 使用 QTimer.singleShot 确保在 Napari 界面完全加载后执行初始化隐藏
        QTimer.singleShot(100, self._setup_hotkeys_and_init_view)
        # 监听热更新信号，用于刷新 Measure 样式
        GlobalConfig.signals.config_updated.connect(self._refresh_measure_style)

    def _setup_widgets(self):
        # === 1. 创建一个主容器 (Wrapper) ===
        # 用它来包裹 "顶部工具栏" 和 "Tab组件"
        main_container = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0) # 紧凑布局

        # === 2. 创建顶部工具栏 (Toolbar) ===
        toolbar = QWidget()
        # toolbar.setStyleSheet("background-color: #262626; border-bottom: 1px solid #333;") # 可选：加个底色区分
        toolbar.setStyleSheet("background: transparent;")
        h_bar = QHBoxLayout()
        h_bar.setContentsMargins(4, 0, 4, 0) 
        h_bar.setSpacing(4)
        common_font = '"Segoe UI", "Microsoft YaHei", "San Francisco", "Helvetica Neue", sans-serif'

        # (可选) 左侧加一个小标题，显得不那么空
        lbl_title = QLabel(tr("Workflow Tools"))
        lbl_title.setStyleSheet(f"""
            color: #777; 
            font-weight: bold; 
            font-size: 10px; 
            font-family: {common_font};
            margin-top: 2px;
        """)
        h_bar.addWidget(lbl_title)
        
        h_bar.addStretch() # 弹簧：把右边的按钮顶过去

        self.btn_ruler = QPushButton(f"📏 {tr('Measure')}")
        self.btn_ruler.setCheckable(True) # 这是一个开关按钮
        self.btn_ruler.setToolTip(tr("Toggle Measurement Tool (Draw lines to measure distance)"))
        self.btn_ruler.setStyleSheet("""
            QPushButton { border: 1px solid #444; background: #333; color: #DDD; border-radius: 3px; padding: 2px 8px; font-size: 11px; }
            QPushButton:checked { background: #2196F3; color: white; border: 1px solid #2196F3; }
            QPushButton:hover { border-color: #666; }
        """)
        self.btn_ruler.clicked.connect(self._toggle_measurement_tool)
        h_bar.addWidget(self.btn_ruler)
        h_bar.addStretch()

        # === 设置按钮 (移到这里) ===
        btn_settings = QPushButton("⚙️")
        btn_settings.setToolTip(tr("Global Settings & Shortcuts"))
        btn_settings.setCursor(Qt.PointingHandCursor)
        # 字体大小可以放心设大一点，因为现在高度不受限制了
        btn_settings.setStyleSheet("""
            QPushButton { 
                border: none; 
                background: transparent; 
                font-size: 16px; /* 稍微减小图标尺寸 */
                padding: 2px;    /* 减小内边距 */
                margin: 0px;
            } 
            QPushButton:hover { 
                color: #2196F3; 
                background: #383838; /* 鼠标悬停时给一个淡淡的背景，提升交互感 */
                border-radius: 3px;
            }
        """)
        btn_settings.clicked.connect(self._open_settings)
        h_bar.addWidget(btn_settings)
        
        toolbar.setLayout(h_bar)
        main_layout.addWidget(toolbar)

        """设置所有控制面板"""
        self.tab_widget = QTabWidget()
        # self.tab_widget.setCornerWidget(btn_settings, Qt.TopRightCorner)

        # === 界面美化：全局样式表 (增加留白与现代感) ===
        self.tab_widget.setStyleSheet("""
            QWidget {
                font-family: "Segoe UI", "Microsoft YaHei", "San Francisco", "Helvetica Neue", sans-serif;
                font-size: 10pt;
                color: #E0E0E0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 15px;
                padding-bottom: 8px;
                padding-left: 8px;
                padding-right: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 12px;
                background-color: #333;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #444;
                border-color: #777;
            }
            QPushButton:pressed {
                background-color: #222;
            }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                padding: 4px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #222;
                min-height: 22px;
                selection-background-color: #2196F3;
                margin: 2px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QLabel {
                margin-right: 4px;
                margin-left: 2px;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                min-height: 12px;
            }
        """)

        # 1. 数据导入
        self.import_widget = ImportWidget(self.viewer)
        self.tab_widget.addTab(self.import_widget, f"📂 {tr('Import')}")

        # 2. 漂移矫正
        self.drift_widget = DriftCorrectionWidget(self.viewer)
        self.tab_widget.addTab(self.drift_widget, f"🔧 {tr('Drift Correction')}")

        # 4. 几何变换
        self.geometry_widget = GeometryWidget(self.viewer)
        self.tab_widget.addTab(self.geometry_widget, f"📐 {tr('Geometry')}")

        # 4. 图像增强 (Original 4, actually 5 now)
        self.enhance_widget = EnhanceWidget(self.viewer)
        self.tab_widget.addTab(self.enhance_widget, f"✨ {tr('Enhancement')}")

        # 5. [NEW] 帧筛选
        self.filter_widget = FilterWidget(self.viewer)
        self.tab_widget.addTab(self.filter_widget, f"🔍 {tr('Frame Filter')}")

        # 6. 标注工具
        self.annotation_widget = AnnotationWidget(self.viewer)
        self.tab_widget.addTab(self.annotation_widget, f"📝 {tr('Annotation')}")

        # 6. 导出
        self.export_widget = ExportWidget(self.viewer)
        self.tab_widget.addTab(self.export_widget, f"💾 {tr('Export')}")

        # 7. 脚本运行器
        self.script_widget = ScriptWidget(self.viewer)
        self.tab_widget.addTab(self.script_widget, f"📜 {tr('Scripts')}")

        # 8. 会话恢复
        self.recovery_widget = RecoveryWidget(self.viewer)
        self.tab_widget.addTab(self.recovery_widget, f"🔄 {tr('Recovery')}")
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # === 4. 将 Tab 加入主布局 ===
        main_layout.addWidget(self.tab_widget)
        main_container.setLayout(main_layout)

        # === 5. 添加 Dock Widget (注意：这里放入的是 main_container) ===
        dock = self.viewer.window.add_dock_widget(
            main_container, 
            area='right', 
            name='TEM Workflow'
        )

        dock.setStyleSheet(f"""
            QDockWidget {{
                font-family: {common_font};
                font-size: 10pt;
            }}
            QDockWidget::title {{
                font-family: {common_font};
                background: #262626; /* 可选：让标题栏背景也融入暗色主题 */
                padding-left: 5px;
            }}
        """)

    def _open_settings(self):
        dlg = SettingsDialog(self.viewer.window._qt_window)
        dlg.exec_()

    def _setup_hotkeys_and_init_view(self):
        """设置快捷键并初始化视图状态 (隐藏不必要的面板)"""
        # 定义需要管理的面板标题关键词
        left_target = ['layer list', 'layer controls']
        bottom_target = ['console'] 

        def get_docks(keywords):
            """辅助函数：根据标题关键词查找 DockWidget"""
            qt_window = self.viewer.window._qt_window
            all_docks = qt_window.findChildren(QDockWidget)
            targets = []
            for dock in all_docks:
                title = dock.windowTitle().lower()
                # 确保只匹配目标，且不误伤我们的插件 ('TEM Workflow')
                if any(k in title for k in keywords) and 'tem workflow' not in title:
                    targets.append(dock)
            return targets

        toggle_key = GlobalConfig.get_napari_shortcut("shortcut_toggle_ui")
        @self.viewer.bind_key(toggle_key)
        def toggle_left_view(viewer):
            """
            J 键逻辑：
            1. 切换【左侧面板】的显示/隐藏。
            2. 强制【隐藏控制台】，防止布局挤压导致时间轴消失。
            """
            # 1. 处理左侧面板
            left_docks = get_docks(left_target)
            if left_docks:
                # 判断当前状态（只要有一个是显示的，就认为需要隐藏）
                should_hide = any(dock.isVisible() for dock in left_docks)
                for dock in left_docks:
                    dock.setVisible(not should_hide)
                status = tr("Hidden") if should_hide else tr("Shown")
                viewer.status = f"{status} {tr('layer controls')}"

            # 2. 强制处理底部控制台 (Bug修复：防止控制台占用时间轴空间)
            # 无论左侧是显示还是隐藏，我们都希望控制台保持隐藏，除非用户手动打开
            # 这里的逻辑是：如果在切换左侧时，控制台"意外"弹出了（Qt布局特性），我们把它按回去
            console_docks = get_docks(bottom_target)
            for dock in console_docks:
                if dock.isVisible():
                    dock.hide()

        # === 初始化执行：隐藏不需要的面板 ===
        # 1. 隐藏左侧 (Layer List & Controls)
        for dock in get_docks(left_target):
            dock.hide()
        # 2. 隐藏底部 (Console)
        for dock in get_docks(bottom_target):
            dock.hide()

    def _on_tab_changed(self, index):
        current_widget = self.tab_widget.widget(index)
        if hasattr(current_widget, '_refresh_layers'):
            current_widget._refresh_layers()
    
    def _refresh_measure_style(self):
        """热更新：如果测量图层存在，立即更新其颜色和字体"""
        if "Measurements" in self.viewer.layers:
            layer = self.viewer.layers["Measurements"]
            
            # 读取新配置
            new_color = GlobalConfig.get("style_measure_color")
            new_width = int(GlobalConfig.get("style_measure_width"))
            new_font_size = int(GlobalConfig.get("style_measure_font_size"))
            
            # 更新图层属性
            layer.edge_color = new_color
            layer.edge_width = new_width
            layer.text.color = 'white' # 保持白色文字，或者也配置
            layer.text.size = new_font_size
            layer.refresh()

    def _toggle_measurement_tool(self, checked):
        layer_name = "Measurements"
        
        if checked:
            # 开启测量模式
            if layer_name not in self.viewer.layers:
                color = GlobalConfig.get("style_measure_color")
                width = int(GlobalConfig.get("style_measure_width"))
                font_size = int(GlobalConfig.get("style_measure_font_size"))
                # 创建 Shapes 图层
                self.measure_layer = self.viewer.add_shapes(
                    name=layer_name,
                    shape_type='line',
                    edge_color=color,
                    edge_width=width,
                    face_color=[0,0,0,0],
                    # 设置文本显示属性
                    text={
                        'string': '{length}', 
                        'size': font_size, 
                        'color': 'white', 
                        'anchor': 'center', 
                        'translation': [0, -15]
                    },
                    features={'length': []} # 初始化 features
                )
                self.measure_layer.mode = 'add_line'
                self.measure_layer.events.data.connect(self._on_measure_data_change)
                undo_key = GlobalConfig.get_napari_shortcut("shortcut_undo_drift") # 通常是 Control-Z
                @self.measure_layer.bind_key(undo_key)
                def undo_last_line(layer):
                    if len(layer.data) > 0:
                        layer.data = layer.data[:-1] # 移除最后一个数据
                        # 注意：features 会由 _on_measure_data_change 自动重新计算，无需手动 pop
                        self.viewer.status = f"↩️ {tr('Last measurement removed.')}"
            else:
                self.measure_layer = self.viewer.layers[layer_name]
                self.measure_layer.visible = True
                self.measure_layer.mode = 'add_line'
            
            self.viewer.layers.selection.active = self.measure_layer
            self.viewer.status = f"📏 {tr('Measurement Mode: Draw lines to measure.(Ctrl+Z to Undo)')}"
            
        else:
            # 关闭测量模式 (但不删除图层，只是切换回选择模式或隐藏)
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].mode = 'pan_zoom'
                # 可选：是否隐藏图层？通常用户可能想保留测量结果，所以这里不隐藏
            # 隐藏图层
            self.viewer.status = tr("End Measurement.")

    def _on_measure_data_change(self, event=None):
        """计算线段长度并更新标签"""
        if "Measurements" not in self.viewer.layers: return
        layer = self.viewer.layers["Measurements"]
        if len(layer.data) == 0: 
            layer.features = {'length': []}
            return
        
        # 获取像素尺寸 (尝试从当前选中的 Image 图层获取)
        pixel_size = 1.0
        unit = "px"
        
        # 遍历图层寻找底图的 metadata
        for l in self.viewer.layers:
            if isinstance(l, napari.layers.Image) and l.visible:
                meta = l.metadata
                if meta and 'pixel_A' in meta:
                    pixel_size = meta['pixel_A'] / 10.0 # 转换为 nm
                    unit = "nm"
                elif meta and 'pixel_unit' in meta and 'pixel_size' in meta:
                     pixel_size = meta['pixel_size']
                     unit = meta['pixel_unit']
                break
        
        new_lengths = []
        for shape_data in layer.data:
            # Line data shape: (2, 2) -> [[y1, x1], [y2, x2]]
            p1 = shape_data[0]
            p2 = shape_data[1]
            # 欧几里得距离
            dist_px = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            dist_phys = dist_px * pixel_size
            
            if unit == "px":
                lbl = f"{dist_phys:.1f} px"
            else:
                lbl = f"{dist_phys:.2f} {unit}"
            new_lengths.append(lbl)
            
        # 更新 features 以刷新显示
        layer.features = {'length': new_lengths}
        layer.refresh()

    def run(self):
        napari.run()
        # 正常退出时标记会话完成
        try:
            from utils.session_logger import get_logger
            get_logger().mark_completed()
        except:
            pass

def main():
    # 初始化 SessionLogger
    try:
        from utils.session_logger import get_logger
        get_logger()  # 确保单例被创建
    except Exception as e:
        print(f"SessionLogger init failed: {e}")
    
    app = TEMWorkflow()
    
    # 延迟执行恢复检查 (等待 UI 完全加载)
    def check_recovery():
        try:
            from widgets.recovery_dialog import check_and_show_recovery
            recovered_actions = check_and_show_recovery(app.viewer.window._qt_window)
            if recovered_actions:
                print(f"[Recovery] Recovered {len(recovered_actions)} actions ")
                # 这里可以根据 recovered_actions 执行实际恢复逻辑
                # 目前仅显示信息，实际重放需要更复杂的逻辑
                app.viewer.status = f"✅ {tr('Session recovery: %s actions loaded') % len(recovered_actions)}"
        except Exception as e:
            print(f"Recovery check failed: {e}")
    
    QTimer.singleShot(500, check_recovery)

    app.run()

if __name__ == '__main__':
    main()
"""
TEM Data Processing Workflow - Napari GUI
统一的TEM数据处理平台
[Fix] 增加了 resize 初始化，防止界面在启动时过大。
"""
import napari
from qtpy.QtWidgets import QTabWidget, QDockWidget, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from qtpy.QtCore import Qt, QTimer, QSettings
from widgets.import_widget import ImportWidget
from widgets.drift_widget import DriftCorrectionWidget
from widgets.geometry_widget import GeometryWidget
from widgets.enhance_widget import EnhanceWidget
from widgets.annotation_widget import AnnotationWidget
from widgets.export_widget import ExportWidget
from widgets.settings_widget import SettingsDialog, GlobalConfig

class TEMWorkflow:
    def __init__(self):
        self.viewer = napari.Viewer(title="TEM Data Processing Workflow-YSImageJ")
        QSettings("NapariUser", "Global").remove("archive_path")
        # === Fix: Set a reasonable default size to prevent layout overflow ===
        self.viewer.window.resize(1200, 800)
        
        self._setup_widgets()
        # 使用 QTimer.singleShot 确保在 Napari 界面完全加载后执行初始化隐藏
        QTimer.singleShot(100, self._setup_hotkeys_and_init_view)

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
        h_bar.setSpacing(0)
        common_font = '"Segoe UI", "Microsoft YaHei", "San Francisco", "Helvetica Neue", sans-serif'

        # (可选) 左侧加一个小标题，显得不那么空
        lbl_title = QLabel("Workflow Tools")
        lbl_title.setStyleSheet(f"""
            color: #777; 
            font-weight: bold; 
            font-size: 10px; 
            font-family: {common_font};
            margin-top: 2px;
        """)
        h_bar.addWidget(lbl_title)
        
        h_bar.addStretch() # 弹簧：把右边的按钮顶过去

        # === 设置按钮 (移到这里) ===
        btn_settings = QPushButton("⚙️")
        btn_settings.setToolTip("Global Settings & Shortcuts")
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
        self.tab_widget.addTab(self.import_widget, "📂 Import")

        # 2. 漂移矫正
        self.drift_widget = DriftCorrectionWidget(self.viewer)
        self.tab_widget.addTab(self.drift_widget, "🔧 Drift Correction")

        # 3. 几何变换
        self.geometry_widget = GeometryWidget(self.viewer)
        self.tab_widget.addTab(self.geometry_widget, "📐 Geometry")

        # 4. 图像增强
        self.enhance_widget = EnhanceWidget(self.viewer)
        self.tab_widget.addTab(self.enhance_widget, "✨ Enhancement")

        # 5. 标注工具
        self.annotation_widget = AnnotationWidget(self.viewer)
        self.tab_widget.addTab(self.annotation_widget, "📝 Annotation")

        # 6. 导出
        self.export_widget = ExportWidget(self.viewer)
        self.tab_widget.addTab(self.export_widget, "💾 Export")

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
                status = "Hidden" if should_hide else "Shown"
                viewer.status = f"{status} layer controls"

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

    def run(self):
        napari.run()

def main():
    app = TEMWorkflow()
    app.run()

if __name__ == '__main__':
    main()
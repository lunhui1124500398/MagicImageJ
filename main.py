"""
TEM Data Processing Workflow - Napari GUI
统一的TEM数据处理平台
"""
import napari
from qtpy.QtWidgets import QTabWidget, QDockWidget
from qtpy.QtCore import Qt, QTimer
from widgets.import_widget import ImportWidget
from widgets.drift_widget import DriftCorrectionWidget
from widgets.geometry_widget import GeometryWidget
from widgets.enhance_widget import EnhanceWidget
from widgets.annotation_widget import AnnotationWidget
from widgets.export_widget import ExportWidget

class TEMWorkflow:
    def __init__(self):
        self.viewer = napari.Viewer(title="TEM Data Processing Workflow")
        self._setup_widgets()
        # 使用 QTimer.singleShot 确保在 Napari 界面完全加载后执行初始化隐藏
        QTimer.singleShot(100, self._setup_hotkeys_and_init_view)

    def _setup_widgets(self):
        """设置所有控制面板"""
        self.tab_widget = QTabWidget()
        
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

        self.viewer.window.add_dock_widget(
            self.tab_widget, 
            area='right', 
            name='TEM Workflow'
        )

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

        @self.viewer.bind_key('j')
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
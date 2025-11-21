"""
TEM Data Processing Workflow - Napari GUI
统一的TEM数据处理平台
"""
import napari
from qtpy.QtWidgets import QTabWidget
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

    def _setup_widgets(self):
        """设置所有控制面板"""
        # === 关键修改 1：使用 self.tab_widget 而不是 tab_widget ===
        self.tab_widget = QTabWidget()
        
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
        
        # === 关键修改 2：监听切换事件 ===
        # 这里的 self.tab_widget 必须在上面定义过
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # 将标签页添加到napari
        self.viewer.window.add_dock_widget(
            self.tab_widget, 
            area='right', 
            name='TEM Workflow'
        )

    def _on_tab_changed(self, index):
        """当标签页切换时，自动刷新当前页面的图层列表"""
        current_widget = self.tab_widget.widget(index)
        
        # 检查组件是否有刷新方法
        if hasattr(current_widget, '_refresh_layers'):
            # 调用组件的刷新方法，这会重新读取图层列表并尝试选中当前活跃图层
            current_widget._refresh_layers()

    def run(self):
        """启动应用"""
        napari.run()

def main():
    app = TEMWorkflow()
    app.run()

if __name__ == '__main__':
    main()
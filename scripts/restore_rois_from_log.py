"""
从 processing_log.json 恢复 ROI (v3.0)

修复: 不再创建新的 GeometryWidget 实例，而是查找已存在的实例
这避免了 keybinding 冲突错误
"""
from pathlib import Path
from qtpy.QtWidgets import QFileDialog, QMessageBox, QDockWidget


def find_geometry_widget(viewer):
    """
    在 Napari 窗口中查找已存在的 GeometryWidget 实例
    返回: GeometryWidget 或 None
    """
    try:
        qt_window = viewer.window._qt_window
        all_docks = qt_window.findChildren(QDockWidget)
        
        for dock in all_docks:
            # 获取 dock 内的 widget
            widget = dock.widget()
            if widget is None:
                continue
            
            # 递归查找子 widget
            from widgets.geometry_widget import GeometryWidget
            
            # 检查 widget 本身或其子 widget
            if isinstance(widget, GeometryWidget):
                return widget
            
            # 检查 Tab 组件内部
            from qtpy.QtWidgets import QTabWidget
            for child in widget.findChildren(GeometryWidget):
                return child
                
            # 检查 QTabWidget 的 tabs
            for tab in widget.findChildren(QTabWidget):
                for i in range(tab.count()):
                    tab_widget = tab.widget(i)
                    if isinstance(tab_widget, GeometryWidget):
                        return tab_widget
        
        return None
    except Exception as e:
        print(f"[find_geometry_widget] Error: {e}")
        return None


def main(viewer):
    # 1. 选择 Log 文件
    base_dir = Path(__file__).parent.parent
    default_dir = base_dir / "problems"
    
    path_str, _ = QFileDialog.getOpenFileName(
        None, 
        "Select processing_log.json", 
        str(default_dir) if default_dir.exists() else str(base_dir), 
        "JSON (*.json)"
    )
    
    if not path_str:
        return
    
    # 2. 查找已存在的 GeometryWidget
    geo_widget = find_geometry_widget(viewer)
    
    if geo_widget is None:
        # 找不到现有 widget，提示用户
        QMessageBox.warning(
            None, "Widget Not Found", 
            "Could not find GeometryWidget in the application.\n"
            "Please ensure you are running this script from within the full application."
        )
        return
    
    # 3. 调用恢复方法
    try:
        success = geo_widget.restore_from_processing_log(path_str)
        if success:
            # 确保切换到 Geometry tab (可选)
            viewer.status = f"✅ ROIs restored from {Path(path_str).name}"
    except Exception as e:
        QMessageBox.critical(
            None, "Error", 
            f"Restoration failed:\n{e}"
        )


if __name__ == "__main__":
    pass

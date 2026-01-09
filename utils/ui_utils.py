"""
UI工具模块 - 提供UI辅助功能
包含：
- setup_safe_scroll: 防止鼠标滚轮误触更改控件值
"""
from qtpy.QtCore import QObject, QEvent, Qt


class _ScrollFilter(QObject):
    """
    事件过滤器：只有当控件拥有焦点时，才允许滚轮事件改变其值。
    否则，忽略事件（让它传递给父控件处理，比如滚动页面）。
    """
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if not obj.hasFocus():
                # 忽略此事件，让父控件处理（如滚动页面）
                event.ignore()
                return True  # 阻止事件传递给控件本身
        return super().eventFilter(obj, event)


# 全局缓存，避免重复创建 Filter 实例
_filter_instance = None


def setup_safe_scroll(widget):
    """
    为 QSpinBox, QDoubleSpinBox, QComboBox 等控件设置安全滚轮。
    只有当控件获得焦点（如用户点击）后，滚轮才能更改值。
    
    用法:
        from utils.ui_utils import setup_safe_scroll
        
        self.my_spinbox = QSpinBox()
        setup_safe_scroll(self.my_spinbox)
    
    参数:
        widget: 需要保护的控件
    """
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = _ScrollFilter()
    
    # 确保控件可以获得焦点（通过点击）
    widget.setFocusPolicy(Qt.StrongFocus)
    # 安装事件过滤器
    widget.installEventFilter(_filter_instance)


def setup_safe_scroll_all(*widgets):
    """
    批量为多个控件设置安全滚轮。
    
    用法:
        setup_safe_scroll_all(spin1, spin2, combo1, spin3)
    """
    for w in widgets:
        if w is not None:
            setup_safe_scroll(w)

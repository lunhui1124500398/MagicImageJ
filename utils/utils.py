import sys
import os

def resource_path(relative_path):
    """ 获取资源的绝对路径，适配 Dev 环境和 PyInstaller 打包后的环境 """
    try:
        # PyInstaller 创建临时文件夹，路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        # 如果是正常运行 python main.py，则使用当前目录
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def elide_text(text: str, max_len: int = 25) -> str:
    """如果文本过长，只显示头尾，中间用...代替"""
    if not text: return ""
    if len(text) <= max_len:
        return text
    half = (max_len - 3) // 2
    return f"{text[:half]}...{text[-half:]}"
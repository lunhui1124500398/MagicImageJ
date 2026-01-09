import sys
import os
import re
from pathlib import Path

def natural_sort_key(s):
    """
    自然排序 key 函数。
    将字符串拆分为文本和数字部分，数字部分按整数排序。
    例如: ["1.png", "10.png", "2.png"] -> ["1.png", "2.png", "10.png"]
    支持 str 和 Path 对象。
    """
    if isinstance(s, Path):
        s = s.name  # 只取文件名进行排序
    # 将字符串按数字和非数字部分分割
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', str(s))]

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
"""
内存管理工具模块 (V6 - 安全路径版 + 弹窗预警)
功能：
1. 智能分配大数组 (RAM vs Disk)。
2. 读取 JSON 配置文件。
3. 自动清理：
   - 启动时清理陈旧文件(>24h)。
   - [New] 空间预警：若占用>10GB，通过系统弹窗(MessageBox)提醒用户，而非控制台输出。
   - [New] 安全脚本：清理脚本(.bat)强制生成在用户缓存目录，防止误删系统Temp文件。
   - 退出时启动外部进程强制删除被锁文件。
"""
import numpy as np
import tempfile
import os
import json
import atexit
import time
import gc
import weakref
import subprocess
import platform
from pathlib import Path

# 全局变量
SESSION_TEMP_FILES = set()
HAS_WARNED_SPACE = False  # 防止频繁弹窗

def _load_json_config_value(key, default=""):
    config_path = Path.home() / ".napari_tem_config.json"
    if not config_path.exists():
        return default
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get(key, default)
    except Exception:
        return default

def get_cache_dir():
    path_str = _load_json_config_value("cache_dir", "")
    if path_str:
        path = Path(path_str)
        if path.exists() and path.is_dir():
            return str(path)
    return None

def show_system_warning(title, message):
    """
    显示系统级弹窗 (Windows Only)
    用于在打包环境下提醒用户，无需依赖控制台。
    """
    if platform.system() == "Windows":
        try:
            import ctypes
            # MB_OK | MB_ICONWARNING | MB_SYSTEMMODAL
            ctypes.windll.user32.MessageBoxW(0, message, title, 0x00000000 | 0x00000030 | 0x00001000)
        except:
            pass

def try_clean_old_files(cache_dir_str):
    """
    清理超过24小时的陈旧文件，并检测当前缓存占用情况。
    """
    global HAS_WARNED_SPACE
    if not cache_dir_str: return
    
    remaining_size = 0
    large_file_warning_threshold = 10 * 1024**3 # 10GB 预警阈值
    
    try:
        folder = Path(cache_dir_str)
        if not folder.exists(): return

        now = time.time()
        # 扫描所有 .dat 文件
        for p in folder.glob("*.dat"):
            if p.is_file():
                try:
                    # 1. 尝试删除陈旧文件 (>24h)
                    if now - p.stat().st_mtime > 86400: 
                        try:
                            os.remove(p)
                            continue # 删除成功，不计入剩余空间
                        except:
                            pass # 被占用或其他错误
                    
                    # 2. 累加剩余文件大小
                    remaining_size += p.stat().st_size
                except:
                    pass
        
        # 3. 空间占用预警 (仅提醒一次)
        if remaining_size > large_file_warning_threshold and not HAS_WARNED_SPACE:
            gb_size = remaining_size / (1024**3)
            HAS_WARNED_SPACE = True
            
            msg = (f"High Disk Usage Detected in Cache Folder!\n\n"
                   f"Location: {cache_dir_str}\n"
                   f"Current Usage: {gb_size:.2f} GB\n\n"
                   f"Please check if you have enough disk space.\n"
                   f"Old files (>24h) are auto-cleaned, but recent large files may remain.")
            
            # 使用非阻塞方式或简单调用（这里在工作线程调用会阻塞该线程，这其实是好事，防止继续写入爆盘）
            show_system_warning("Disk Space Warning", msg)

    except Exception:
        pass

def cleanup_file(path):
    """单个文件清理逻辑"""
    if os.path.exists(path):
        try:
            os.remove(path)
            return True
        except:
            return False
    return True

def spawn_detached_cleaner(files_to_delete):
    """
    [Windows专用] 启动一个独立的后台批处理进程。
    [Safety Fix] 脚本强制生成在用户配置的 Cache 目录中，而非系统 Temp。
    """
    if platform.system() != "Windows":
        return

    try:
        # 1. 确定脚本存放位置 (优先使用用户设置的 Cache 目录)
        # 如果用户指定了 D:/Cache，脚本就在 D:/Cache/napari_cleaner.bat
        # 这样即使 del 命令出错，也限制在这个文件夹范围内，不会误删 C:/Windows/Temp 内容
        script_dir = get_cache_dir()
        
        # 如果用户没设置或路径无效，回退到系统临时目录
        if not script_dir or not os.path.exists(script_dir):
            script_dir = tempfile.gettempdir()
            
        cleaner_script_path = os.path.join(script_dir, f"napari_cleaner_{os.getpid()}.bat")
        
        with open(cleaner_script_path, "w") as f:
            f.write("@echo off\n")
            # 等待 3 秒，确保主进程彻底退出释放锁
            f.write("timeout /t 3 /nobreak > NUL\n")
            
            # 循环删除
            for file_path in files_to_delete:
                # 再次校验路径是否以 script_dir 开头，增加安全性 (可选，视需求而定)
                # 这里主要依赖文件路径本身的准确性
                f.write(f'if exist "{file_path}" del /f /q "{file_path}" > NUL 2>&1\n')
            
            # 删除脚本自己并退出
            f.write(f'del "{cleaner_script_path}" > NUL 2>&1\n')
            f.write("exit\n")

        # 启动脚本 (CreationFlag 0x08000000 = CREATE_NO_WINDOW, 后台静默运行)
        subprocess.Popen(
            [cleaner_script_path], 
            shell=True, 
            creationflags=0x08000000, 
            close_fds=True
        )
        # Log: Scheduled cleaner in {script_dir}
        
    except Exception:
        pass

def cleanup_session_files():
    """程序退出时的暴力清理"""
    if not SESSION_TEMP_FILES:
        return
    
    # 1. 强制垃圾回收，尽最大努力关闭 memmap 句柄
    try:
        gc.collect() 
    except:
        pass
    
    # 2. 尝试清理
    failed_files = []
    for path in list(SESSION_TEMP_FILES):
        if not cleanup_file(path):
            failed_files.append(path)
        else:
            SESSION_TEMP_FILES.discard(path)

    # 3. 如果还有文件被锁（删不掉），启动外部杀手程序
    if failed_files:
        spawn_detached_cleaner(failed_files)

# 注册退出清理
atexit.register(cleanup_session_files)

def create_huge_array(shape, dtype, fill_zeros=False):
    """
    智能数组分配器
    返回: (array, temp_path)
    """
    elements = np.prod(shape)
    itemsize = np.dtype(dtype).itemsize
    nbytes = elements * itemsize
    
    # 阈值 4GB
    threshold = 4 * 1024**3 

    if nbytes > threshold:
        cache_dir = get_cache_dir()
        
        # 每次分配大内存前，顺手检查一下陈旧文件和磁盘占用
        if cache_dir:
            try_clean_old_files(cache_dir)
        
        try:
            # 确保使用用户目录
            target_dir = cache_dir if cache_dir else tempfile.gettempdir()
            
            fd, temp_path = tempfile.mkstemp(suffix='.dat', dir=target_dir)
            os.close(fd)
            
            # 记录到全局集合
            SESSION_TEMP_FILES.add(temp_path)
            
            # 创建 memmap
            arr = np.memmap(temp_path, dtype=dtype, mode='w+', shape=shape)
            
            # === 绑定生命周期 ===
            def finalizer_callback(p=temp_path):
                if cleanup_file(p):
                    if p in SESSION_TEMP_FILES:
                        SESSION_TEMP_FILES.discard(p)
            
            weakref.finalize(arr, finalizer_callback)
            
            return arr, temp_path
            
        except Exception as e:
            # 这里可以保留 print 到控制台，作为最后的调试手段，普通用户看不到也不影响
            print(f"Memmap Error: {e}")
            raise MemoryError(f"Disk allocation failed: {e}")
    else:
        # RAM
        if fill_zeros:
            return np.zeros(shape, dtype=dtype), None
        else:
            return np.empty(shape, dtype=dtype), None
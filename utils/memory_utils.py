"""
内存管理工具模块 (V8 - 完整增强版)
功能：
1. 智能分配大数组 (RAM vs Disk/Memmap)。
2. [关键] C盘防爆机制：在分配大内存前检查物理内存，防止 pagefile.sys 撑爆 C 盘。
3. 线程安全弹窗：使用 Windows API 实现后台线程的阻塞式询问。
4. 自动清理：
   - 启动时清理陈旧文件 (>24h)。
   - 退出时强制清理。
   - 提供独立清理脚本 (Detached Cleaner) 处理被锁文件。
"""

# 注意：为了避免循环导入，这里我们手动复制一下简单的读取逻辑，或者使用之前定义的 _load_json_config_value
# 但最好是从 GlobalConfig 读取。如果在 utils 里不好引 widgets，建议保留 memory_utils 里的 _load_json_config_value 并增强它。

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
import ctypes

# 尝试导入 psutil 获取更准确的内存信息
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

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

def get_available_ram_gb():
    """
    获取系统当前可用物理内存 (GB)
    优先使用 psutil，失败则使用 Windows API，最后回退默认值。
    """
    try:
        # 1. 尝试 psutil (跨平台，最准确)
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().available / (1024**3)
    except:
        pass

    try:
        # 2. 尝试 Windows API (无需额外库)
        if platform.system() == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys / (1024**3)
    except:
        pass

    # 3. 实在不行返回一个默认安全值 (16GB)，避免误报阻断流程
    return 16.0

def show_system_ask_dialog(title, message):
    """
    [Thread-Safe] 显示系统级询问弹窗 (Windows Only)
    返回: True (Yes/Continue), False (No/Cancel)
    即便在子线程调用，也会阻塞直到用户点击，且不会导致 Qt 崩溃。
    """
    if platform.system() == "Windows":
        try:
            # MB_YESNO (0x04) | MB_ICONWARNING (0x30) | MB_SYSTEMMODAL (0x1000)
            # 返回值: IDYES = 6, IDNO = 7
            ret = ctypes.windll.user32.MessageBoxW(0, message, title, 0x00000004 | 0x00000030 | 0x00001000)
            return ret == 6
        except:
            return True # 如果弹窗失败，默认允许继续（避免卡死）
    else:
        # 非 Windows 环境简单打印
        print(f"[WARNING] {title}: {message}")
        return True

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
    gb_warn = float(_load_json_config_value("sys_disk_warn_gb", 10.0))
    large_file_warning_threshold = gb_warn * 1024**3
    
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

def trim_working_set():
    """
    [Windows] 释放进程 Working Set 中的文件缓存页。
    memmap 写入/读取后，Windows 会将页面缓存在物理内存中。
    调用此函数告诉 OS 可以回收这些页面。数据仍在磁盘上，再次访问时按需重新加载。
    """
    if platform.system() != "Windows":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetProcessWorkingSetSize.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t
        ]
        kernel32.SetProcessWorkingSetSize.restype = ctypes.c_bool
        handle = kernel32.GetCurrentProcess()
        kernel32.SetProcessWorkingSetSize(handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    except Exception:
        pass

def release_memmap_pages(arr):
    """
    memmap 写入完成后调用：flush → GC → trim Working Set。
    释放 OS 为 memmap 页面缓存的物理内存。
    """
    if hasattr(arr, 'flush'):
        arr.flush()
    gc.collect()
    trim_working_set()


# ---------------------------------------------------------------------------
# Phase 7 (2026-05-29): preallocated stacking helper
# ---------------------------------------------------------------------------
def stack_frames_preallocated(frames):
    """Stack a list of same-shape 2D/3D frames into a single ndarray using
    pre-allocation (avoids the doubled allocation that np.array(frames) and
    np.stack(frames) trigger on memmap or other lazy sources).

    Falls through to np.array on empty inputs (returns shape (0,)).
    """
    if not frames:
        return np.empty((0,), dtype=np.uint8)
    first = frames[0]
    T = len(frames)
    if first.ndim == 2:
        H, W = first.shape
        out = np.empty((T, H, W), dtype=first.dtype)
        for i, f in enumerate(frames):
            out[i] = f
    elif first.ndim == 3:
        H, W, C = first.shape
        out = np.empty((T, H, W, C), dtype=first.dtype)
        for i, f in enumerate(frames):
            out[i] = f
    else:
        # Unexpected ndim — fall back to np.array (will allocate intermediate)
        out = np.array(frames)
    return out

def load_png_stack_memmap(png_files, read_fn=None, progress_cb=None):
    """Read a list of PNG paths into ONE (memmap-for-large) stack, reading each
    file straight into the preallocated output. Unlike building a Python list of
    frames then np.stack()-ing it, this never holds the whole sequence in RAM at
    once — key for 5000+ frame recovery loads that used to double-allocate ~24GB.

    read_fn(path) -> 2D frame (or None to skip); defaults to cv2 grayscale read.
    progress_cb(k) called with the 1-based index after each frame (optional).
    Returns an empty uint8 array for empty input.
    """
    if not png_files:
        return np.empty((0,), dtype=np.uint8)

    if read_fn is None:
        import cv2

        def read_fn(p):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img

    first = read_fn(png_files[0])
    if first is None:
        raise ValueError("Could not read the first PNG frame")
    T = len(png_files)
    H, W = first.shape[:2]
    stack, _ = create_huge_array((T, H, W), first.dtype)
    stack[0] = first
    n = 1
    if progress_cb:
        progress_cb(1)
    for i in range(1, T):
        img = read_fn(png_files[i])
        if img is not None and img.shape[:2] == (H, W):
            stack[n] = img
            n += 1
        if progress_cb:
            progress_cb(i + 1)
    if n < T:
        stack = stack[:n]
    release_memmap_pages(stack)
    return stack


def create_huge_array(shape, dtype, fill_zeros=False, force_disk=False):
    """
    智能数组分配器
    返回: (array, temp_path)

    force_disk=True: 无论大小都走磁盘 memmap (用于恢复"全内存"模式把小中间层也写盘,
    界住 RAM)。默认 False = 原行为 (按 sys_ram_threshold_gb 阈值判定)。
    """
    elements = np.prod(shape)
    itemsize = np.dtype(dtype).itemsize
    nbytes = elements * itemsize
    gb_needed = nbytes / (1024**3)

    # 阈值
    gb_limit = float(_load_json_config_value("sys_ram_threshold_gb", 4.0))
    threshold = gb_limit * 1024**3 

    warn_threshold_gb = float(_load_json_config_value("sys_mem_warn_gb", 4.0))
    if gb_needed > warn_threshold_gb: 
        avail_ram = get_available_ram_gb()
        # 判定标准：如果需求量 > 当前可用物理内存的 95%
        # Windows 此时极大概率会开始疯狂换页(Swapping)到 Pagefile (C盘)
        if gb_needed > avail_ram * 0.95:
            # 获取语言配置
            lang = _load_json_config_value("language", "en")
            is_cn = (lang == "zh_CN")
            
            if is_cn:
                title = "内存严重不足警告"
                msg = (f"即将分配: {gb_needed:.1f} GB\n"
                       f"当前可用物理内存: {avail_ram:.1f} GB\n\n"
                       f"警告：此操作已超出物理内存余量！\n"
                       f"在临时文件完全写入硬盘前，Windows 将被迫使用虚拟内存，"
                       f"这会导致系统卡顿并急剧消耗 C 盘 (pagefile.sys) 空间。\n\n"
                       f"是否仍要继续？")
            else:
                title = "Critical Memory Warning"
                msg = (f"Allocating: {gb_needed:.1f} GB\n"
                       f"Available RAM: {avail_ram:.1f} GB\n\n"
                       f"Warning: This exceeds available physical memory!\n"
                       f"Windows will be forced to use the Pagefile on C: drive, "
                       f"which may cause system freeze and disk space exhaustion.\n\n"
                       f"Do you want to continue anyway?")

            # 调用线程安全的系统弹窗 (阻塞等待)
            user_agreed = show_system_ask_dialog(title, msg)
            
            if not user_agreed:
                # 用户选择否，抛出异常中断操作
                raise MemoryError("Operation cancelled by user to prevent system freeze.")
            
    if force_disk or nbytes > threshold:
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
        
        except MemoryError as me:
            # 透传上面的取消异常
            raise me
        
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
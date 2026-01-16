"""
图像增强模块
更新日志:
- 增加 progress_callback 支持
- 修复 OOM 问题，引入 create_huge_array
"""
import numpy as np
import cv2
import concurrent.futures
from utils.memory_utils import create_huge_array
import os

def gaussian_blur_stack(image_stack: np.ndarray,
                        ksize: int = 3,
                        sigma: float = None,
                        progress_callback=None) -> np.ndarray:
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    
    # 预分配 (Disk/RAM)
    blurred, temp_file = create_huge_array(image_stack.shape, image_stack.dtype, fill_zeros=False)
    
    total = len(image_stack)
    
    try:
        for i, frame in enumerate(image_stack):
            blurred[i] = cv2.GaussianBlur(frame, (ksize, ksize), sigma)
            if progress_callback and i % 5 == 0: # 减少回调频率
                progress_callback(i + 1, total)
                
        if progress_callback: progress_callback(total, total)
        if hasattr(blurred, 'flush'): blurred.flush()
        return blurred
        
    except Exception as e:
        if temp_file and os.path.exists(temp_file):
            try: os.remove(temp_file)
            except: pass
        raise e

def rolling_average(image_stack: np.ndarray,
                    window_size: int = 3,
                    num_threads: int = 8,
                    progress_callback=None) -> np.ndarray:
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    
    T, H, W = image_stack.shape
    if T < window_size:
        raise ValueError(f"Not enough frames for window {window_size}")
    
    # output_frames = T (preserved via padding)
    
    # === 预分配 (Disk/RAM) ===
    # 结果大小与输入完全一致 (T, H, W)
    avg_stack, temp_file = create_huge_array((T, H, W), np.uint8, fill_zeros=False)
    
    chunk_size = max(1, (H + num_threads - 1) // num_threads)
    starts = list(range(0, H, chunk_size))
    ends = [min(s + chunk_size, H) for s in starts]
    
    pad_width = window_size // 2

    try:
        def process_chunk(y_start, y_end):
            # 对这一块 Y 区域，处理所有时间帧
            
            # 1. 提取原有数据 (T, chunk_h, W)
            chunk_data_raw = image_stack[:, y_start:y_end, :]
            
            # 2. 在时间轴 (axis=0) 进行边缘填充 (Replicate/Edge Padding)
            # pad_width 个首帧, pad_width 个尾帧
            chunk_padded = np.pad(chunk_data_raw, ((pad_width, pad_width), (0, 0), (0, 0)), mode='edge').astype(np.float32)
            
            # 3. 计算 Cumsum
            # Padded shape: (T + 2*pad, ...)
            chunk_cum = np.zeros((chunk_padded.shape[0] + 1, y_end - y_start, W), dtype=np.float32)
            np.cumsum(chunk_padded, axis=0, out=chunk_cum[1:])
            
            # 4. 计算平均
            # Window sum at valid positions
            # The result should have length T.
            # Start index: window_size (which is 2*pad + 1). 
            # We want T outputs.
            chunk_avg_float = (chunk_cum[window_size:] - chunk_cum[:-window_size]) / window_size
            
            # 释放大数组
            del chunk_data_raw
            del chunk_padded
            del chunk_cum
            
            return y_start, y_end, np.clip(chunk_avg_float, 0, 255).astype(np.uint8)
        
        # 强制增加线程数/切片数以减小单块内存
        # 如果图像很大，强制切成更多条带
        safe_threads = max(num_threads, 32)
        
        # Safer chunking calculation
        # Use ceil division-like logic or just simple steps
        chunk_size = max(1, (H + safe_threads - 1) // safe_threads)
        starts = list(range(0, H, chunk_size))
        ends = [min(s + chunk_size, H) for s in starts]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor: # 实际并发线程仍受参数控制
            futures = [executor.submit(process_chunk, y_start, y_end) 
                       for y_start, y_end in zip(starts, ends)]
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                y_start, y_end, chunk_avg = future.result()
                avg_stack[:, y_start:y_end, :] = chunk_avg
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(futures))

        if hasattr(avg_stack, 'flush'): avg_stack.flush()
        return avg_stack

    except Exception as e:
        print(f"Rolling Avg Error: {e}")
        if temp_file and os.path.exists(temp_file):
            try: os.remove(temp_file)
            except: pass
        raise e

def enhance_image_stack(image_stack: np.ndarray,
                        use_gaussian: bool = False,
                        ksize: int = 3,
                        sigma: float = None,
                        use_average: bool = False,
                        average_window: int = 3,
                        num_threads: int = 8,
                        progress_callback=None) -> np.ndarray:
    """
    综合图像增强 - 带进度条回调
    """
    # 注意：这里我们尽量避免 result = image_stack.copy()，因为这会立即复制 60GB
    # 我们让第一个滤镜直接生成新数组，第二个滤镜在第一个基础上处理
    
    current_data = image_stack
    
    steps = 0
    if use_gaussian: steps += 1
    if use_average: steps += 1
    
    current_step = 0
    
    def sub_progress(c, t):
        if progress_callback:
            base = (current_step / steps) * 100
            fraction = (c / t) * (100 / steps)
            progress_callback(min(100, int(base + fraction)))

    if use_gaussian:
        # gaussian_blur_stack 内部会 create_huge_array，不会就地修改
        current_data = gaussian_blur_stack(current_data, ksize, sigma, sub_progress)
        current_step += 1
    
    if use_average:
        # rolling_average 内部也会 create_huge_array
        current_data = rolling_average(current_data, average_window, num_threads, sub_progress)
        current_step += 1
    
    if progress_callback: progress_callback(100)
    
    return current_data
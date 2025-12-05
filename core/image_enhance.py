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
    
    output_frames = T - window_size + 1
    
    # === 预分配 (Disk/RAM) ===
    # 结果可能很大，必须使用 safe alloc
    avg_stack, temp_file = create_huge_array((output_frames, H, W), np.uint8, fill_zeros=False)
    
    chunk_size = max(1, H // num_threads)
    starts = [i * chunk_size for i in range(num_threads)]
    ends = [start + chunk_size for start in starts]
    ends[-1] = H
    
    try:
        def process_chunk(y_start, y_end):
            # 对这一块 Y 区域，处理所有时间帧
            # 注意：np.cumsum 可能会产生非常大的临时数组 (float32)，这里需要小心
            # 如果 H 很大，chunk_size 应该尽量小，或者这里不一次性处理所有 T
            
            # 优化：为了防止内部 cumsum 爆内存，我们可以分批次处理时间轴？
            # 但 rolling average 需要时间连续。目前的瓶颈是 chunk_data 的大小。
            # chunk_data shape: (T, chunk_h, W). float32 是 uint8 的4倍。
            # 如果 T=15000, chunk_h=200, W=2000 -> 15000*200*2000*4 = 24GB !!! 依然会爆。
            
            # === 二级优化：在 chunk 内部再分块处理时间轴吗？ ===
            # Cumsum 必须全时间轴。所以对于超大 T，Rolling Average 很难并行化而不爆内存。
            # 妥协方案：减小 num_threads (增加 chunk_size) 是反向操作。
            # 应该减小 chunk_size。让 chunk_h 变小。
            
            # 即使 chunk_h = 1 行: 15000 * 1 * 2048 * 4 bytes = 122 MB. 
            # 只要切得够细，内存是可以控制的。
            
            chunk_data = image_stack[:, y_start:y_end, :].astype(np.float32)
            chunk_cum = np.zeros((chunk_data.shape[0] + 1, y_end - y_start, W), dtype=np.float32)
            np.cumsum(chunk_data, axis=0, out=chunk_cum[1:])
            
            # 计算平均
            chunk_avg_float = (chunk_cum[window_size:] - chunk_cum[:-window_size]) / window_size
            
            # 释放大数组
            del chunk_data
            del chunk_cum
            
            return y_start, y_end, np.clip(chunk_avg_float, 0, 255).astype(np.uint8)
        
        # 强制增加线程数/切片数以减小单块内存
        # 如果图像很大，强制切成更多条带
        safe_threads = max(num_threads, 32) 
        chunk_size = max(1, H // safe_threads)
        starts = [i * chunk_size for i in range(safe_threads)]
        ends = [start + chunk_size for start in starts]
        ends[-1] = H

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
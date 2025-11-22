"""
图像增强模块
更新日志:
- 增加 progress_callback 支持，移除 tqdm 依赖（逻辑移交UI）
"""
import numpy as np
import cv2
import concurrent.futures

def gaussian_blur_stack(image_stack: np.ndarray,
                        ksize: int = 3,
                        sigma: float = None,
                        progress_callback=None) -> np.ndarray:
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    
    blurred = np.zeros_like(image_stack)
    total = len(image_stack)
    
    for i, frame in enumerate(image_stack):
        blurred[i] = cv2.GaussianBlur(frame, (ksize, ksize), sigma)
        if progress_callback and i % 5 == 0: # 减少回调频率
            progress_callback(i + 1, total)
            
    if progress_callback: progress_callback(total, total)
    return np.ascontiguousarray(blurred)

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
    avg_stack = np.zeros((output_frames, H, W), dtype=np.uint8)
    
    chunk_size = max(1, H // num_threads)
    starts = [i * chunk_size for i in range(num_threads)]
    ends = [start + chunk_size for start in starts]
    ends[-1] = H
    
    def process_chunk(y_start, y_end):
        chunk_data = image_stack[:, y_start:y_end, :].astype(np.float32)
        chunk_cum = np.zeros((chunk_data.shape[0] + 1, y_end - y_start, W), dtype=np.float32)
        np.cumsum(chunk_data, axis=0, out=chunk_cum[1:])
        chunk_avg = (chunk_cum[window_size:] - chunk_cum[:-window_size]) / window_size
        del chunk_data
        del chunk_cum
        return y_start, y_end, np.clip(chunk_avg, 0, 255).astype(np.uint8)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_chunk, y_start, y_end)
                   for y_start, y_end in zip(starts, ends)]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            y_start, y_end, chunk_avg = future.result()
            avg_stack[:, y_start:y_end, :] = chunk_avg
            completed += 1
            if progress_callback:
                progress_callback(completed, len(futures))

    return avg_stack

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
    progress_callback: func(current_step, total_steps) 
    注意：这里稍微复杂，因为有两个步骤。我们将进度拆分。
    """
    result = image_stack.copy()
    
    steps = 0
    if use_gaussian: steps += 1
    if use_average: steps += 1
    
    current_step = 0
    
    # 简单的进度包装器
    def sub_progress(c, t):
        if progress_callback:
            # 这里的逻辑比较粗略，假设两步耗时差不多
            base = (current_step / steps) * 100
            fraction = (c / t) * (100 / steps)
            progress_callback(min(100, int(base + fraction)))

    if use_gaussian:
        result = gaussian_blur_stack(result, ksize, sigma, sub_progress)
        current_step += 1
    
    if use_average:
        result = rolling_average(result, average_window, num_threads, sub_progress)
        current_step += 1
    
    if progress_callback: progress_callback(100)
    
    return result
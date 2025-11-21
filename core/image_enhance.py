"""
图像增强模块
简化版 - 只保留滚动平均
"""
import numpy as np
import cv2
from tqdm import tqdm
import concurrent.futures


def gaussian_blur_stack(image_stack: np.ndarray,
                       ksize: int = 3,
                       sigma: float = None) -> np.ndarray:
    """
    对图像栈应用高斯模糊
    
    Parameters:
    -----------
    image_stack : np.ndarray
        形状为 (T, Y, X) 的图像栈
    ksize : int
        高斯核大小（必须为奇数）
    sigma : float
        高斯标准差，None时自动计算
        
    Returns:
    --------
    blurred_stack : np.ndarray
        模糊后的图像栈
    """
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    
    blurred = np.array([
        cv2.GaussianBlur(frame, (ksize, ksize), sigma)
        for frame in tqdm(image_stack, desc="Gaussian blur")
    ])
    
    # 修复：强制连续性
    return np.ascontiguousarray(blurred)


# def rolling_average(image_stack: np.ndarray,
#                    window_size: int = 3,
#                    num_threads: int = 8) -> np.ndarray:
#     """
#     滚动平均 - 边界会被丢弃
    
#     Parameters:
#     -----------
#     image_stack : np.ndarray
#         形状为 (T, Y, X) 的图像栈
#     window_size : int
#         窗口大小（必须为奇数）
#     num_threads : int
#         线程数
        
#     Returns:
#     --------
#     averaged_stack : np.ndarray
#         平均后的图像栈，形状为 (T - window_size + 1, Y, X)
#     """
#     if window_size % 2 == 0:
#         raise ValueError("window_size must be odd")
    
#     T, H, W = image_stack.shape
    
#     # 检查是否有足够的帧
#     if T < window_size:
#         raise ValueError(
#             f"Image stack has {T} frames but window_size is {window_size}. "
#             f"Need at least {window_size} frames."
#         )
    
#     gray_stack = image_stack.astype(np.float32)
    
#     # 计算输出帧数
#     output_frames = T - window_size + 1
    
#     # 分块并行处理
#     avg_stack = np.zeros((output_frames, H, W), dtype=np.uint8)
#     chunk_size = max(1, H // num_threads)
#     starts = [i * chunk_size for i in range(num_threads)]
#     ends = [start + chunk_size for start in starts]
#     ends[-1] = H
    
#     def process_chunk(y_start, y_end):
#         chunk_data = gray_stack[:, y_start:y_end, :]
#         chunk_cum = np.zeros((chunk_data.shape[0] + 1, y_end - y_start, W), dtype=np.float32)
#         np.cumsum(chunk_data, axis=0, out=chunk_cum[1:])
#         chunk_avg = (chunk_cum[window_size:] - chunk_cum[:-window_size]) / window_size
#         return y_start, y_end, chunk_avg.astype(np.uint8)
    
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
#         futures = [executor.submit(process_chunk, y_start, y_end)
#                    for y_start, y_end in zip(starts, ends)]
        
#         for future in tqdm(concurrent.futures.as_completed(futures),
#                           total=len(futures),
#                           desc="Rolling average"):
#             y_start, y_end, chunk_avg = future.result()
#             avg_stack[:, y_start:y_end, :] = chunk_avg
    
#     return avg_stack
def rolling_average(image_stack: np.ndarray,
                    window_size: int = 3,
                    num_threads: int = 8) -> np.ndarray:
    """
    滚动平均 - 内存优化版
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    
    T, H, W = image_stack.shape
    
    if T < window_size:
        raise ValueError(
            f"Image stack has {T} frames but window_size is {window_size}. "
            f"Need at least {window_size} frames."
        )
    
    # 计算输出帧数
    output_frames = T - window_size + 1
    
    # 预分配输出数组 (uint8)
    avg_stack = np.zeros((output_frames, H, W), dtype=np.uint8)
    
    # 分块配置
    chunk_size = max(1, H // num_threads)
    starts = [i * chunk_size for i in range(num_threads)]
    ends = [start + chunk_size for start in starts]
    ends[-1] = H
    
    def process_chunk(y_start, y_end):
        # 优化点：只将当前处理的 chunk 转为 float32，大大降低内存峰值
        # 注意：这里直接从原始 image_stack (uint8) 切片，然后转换
        chunk_data = image_stack[:, y_start:y_end, :].astype(np.float32)
        
        # 计算累积和
        chunk_cum = np.zeros((chunk_data.shape[0] + 1, y_end - y_start, W), dtype=np.float32)
        np.cumsum(chunk_data, axis=0, out=chunk_cum[1:])
        
        # 利用累积和计算平均值：(Sum[i+w] - Sum[i]) / w
        chunk_avg = (chunk_cum[window_size:] - chunk_cum[:-window_size]) / window_size
        
        # 显式清理临时大数组，辅助 GC
        del chunk_data
        del chunk_cum
        
        # 转回 uint8 并返回
        return y_start, y_end, np.clip(chunk_avg, 0, 255).astype(np.uint8)
    
    # 并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_chunk, y_start, y_end)
                   for y_start, y_end in zip(starts, ends)]
        
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Rolling average"):
            try:
                y_start, y_end, chunk_avg = future.result()
                avg_stack[:, y_start:y_end, :] = chunk_avg
            except Exception as e:
                print(f"Error in chunk processing: {e}")
                raise e

    return avg_stack


def enhance_image_stack(image_stack: np.ndarray,
                       use_gaussian: bool = False,
                       ksize: int = 3,
                       sigma: float = None,
                       use_average: bool = False,
                       average_window: int = 3,
                       num_threads: int = 8) -> np.ndarray:
    """
    综合图像增强 - 简化版
    
    Parameters:
    -----------
    image_stack : np.ndarray
        输入图像栈
    use_gaussian : bool
        是否使用高斯模糊
    ksize : int
        高斯核大小
    sigma : float
        高斯标准差
    use_average : bool
        是否使用滚动平均
    average_window : int
        滚动平均窗口大小
    num_threads : int
        线程数
        
    Returns:
    --------
    enhanced_stack : np.ndarray
        增强后的图像栈（注意：滚动平均会减少帧数）
    """
    result = image_stack.copy()
    
    # 高斯模糊
    if use_gaussian:
        result = gaussian_blur_stack(result, ksize, sigma)
    
    # 滚动平均
    if use_average:
        result = rolling_average(result, average_window, num_threads)
    
    return result

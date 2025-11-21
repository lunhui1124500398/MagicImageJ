"""
漂移矫正模块
提供交互式ROI选择和漂移计算
"""
import numpy as np
import cv2
from scipy.signal import medfilt
from typing import Tuple, Optional
import concurrent.futures
from tqdm import tqdm


def calculate_drift_single_frame(frame: np.ndarray, 
                                 roi: np.ndarray,
                                 base_loc: Tuple[int, int]) -> np.ndarray:
    """
    计算单帧的漂移量
    
    Parameters:
    -----------
    frame : np.ndarray
        当前帧
    roi : np.ndarray
        参考ROI
    base_loc : tuple
        参考位置 (x, y)
        
    Returns:
    --------
    drift : np.ndarray
        漂移量 [dx, dy]
    """
    try:
        # 模板匹配
        res = cv2.matchTemplate(frame, roi, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        
        # 计算漂移
        drift = np.array(base_loc) - np.array(max_loc)
        return drift
    except Exception as e:
        print(f"Error in drift calculation: {e}")
        return np.array([0, 0])


def calculate_drift_curve(image_stack: np.ndarray,
                         roi_bbox: Tuple[int, int, int, int],
                         template_frame_idx: int,
                         max_workers: int = 32,
                         kernel_size: int = 11) -> np.ndarray:
    """
    计算整个序列的漂移曲线
    
    Parameters:
    -----------
    image_stack : np.ndarray
        形状为 (T, Y, X) 的图像栈
    roi_bbox : tuple
        ROI边界框 (x1, y1, x2, y2)
    template_frame_idx : int
        模板帧索引
    max_workers : int
        并行线程数
    kernel_size : int
        中值滤波核大小（必须为奇数）
        
    Returns:
    --------
    drifts : np.ndarray
        形状为 (T, 2) 的漂移数组，经过中值滤波
    """
    x1, y1, x2, y2 = roi_bbox
    
    # 提取模板ROI
    template_frame = image_stack[template_frame_idx]
    roi = template_frame[y1:y2, x1:x2]
    base_loc = (x1, y1)
    
    # 并行计算所有帧的漂移
    drifts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(calculate_drift_single_frame, frame, roi, base_loc)
                   for frame in image_stack]
        
        for future in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures),
                          desc="Calculating drift"):
            drifts.append(future.result())
    
    drifts = np.array(drifts)
    
    # 中值滤波平滑
    drifts = np.apply_along_axis(lambda x: medfilt(x, kernel_size), 0, drifts)
    
    return drifts


def apply_drift_correction(image_stack: np.ndarray,
                           drifts: np.ndarray,
                           max_workers: int = 8) -> np.ndarray:
    """
    应用漂移矫正
    
    Parameters:
    -----------
    image_stack : np.ndarray
        形状为 (T, Y, X) 的图像栈
    drifts : np.ndarray
        形状为 (T, 2) 的漂移数组
    max_workers : int
        并行线程数
        
    Returns:
    --------
    corrected_stack : np.ndarray
        矫正后的图像栈
    """
    def translate_frame(args):
        frame, drift = args
        rows, cols = frame.shape[:2]
        M = np.float32([[1, 0, drift[0]], [0, 1, drift[1]]])
        return cv2.warpAffine(frame, M, (cols, rows))
    
    # 并行平移所有帧
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        corrected_frames = list(
            tqdm(
                executor.map(translate_frame, zip(image_stack, drifts)),
                total=len(image_stack),
                desc="Applying correction"
            )
        )
    # stack 之后，内存可能不连续，强制转换
    result = np.stack(corrected_frames, axis=0)
    return np.ascontiguousarray(result)


def validate_roi(image_shape: Tuple[int, int],
                roi_bbox: Tuple[int, int, int, int]) -> bool:
    """
    验证ROI是否有效
    
    Parameters:
    -----------
    image_shape : tuple
        图像形状 (height, width)
    roi_bbox : tuple
        ROI边界框 (x1, y1, x2, y2)
        
    Returns:
    --------
    valid : bool
        ROI是否有效
    """
    h, w = image_shape
    x1, y1, x2, y2 = roi_bbox
    
    # 检查边界
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    
    # 检查大小
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return False
    
    return True

"""
漂移矫正模块
更新日志:
- calculate_drift_curve: 增加 progress_callback 支持
- apply_drift_correction: 增加 progress_callback 支持
"""
import numpy as np
import cv2
from scipy.signal import medfilt
from typing import Tuple, Optional
import concurrent.futures

def calculate_drift_single_frame(frame: np.ndarray, 
                                 roi: np.ndarray,
                                 base_loc: Tuple[int, int]) -> np.ndarray:
    try:
        res = cv2.matchTemplate(frame, roi, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        drift = np.array(base_loc) - np.array(max_loc)
        return drift
    except Exception as e:
        print(f"Error in drift calculation: {e}")
        return np.array([0, 0])

def calculate_drift_curve(image_stack: np.ndarray,
                          roi_bbox: Tuple[int, int, int, int],
                          template_frame_idx: int,
                          max_workers: int = 32,
                          kernel_size: int = 11,
                          progress_callback=None) -> np.ndarray:
    """
    计算整个序列的漂移曲线
    progress_callback: func(current, total)
    """
    x1, y1, x2, y2 = roi_bbox
    template_frame = image_stack[template_frame_idx]
    roi = template_frame[y1:y2, x1:x2]
    base_loc = (x1, y1)
    
    drifts = [None] * len(image_stack) # 预分配保持顺序
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 保持 future 和 index 的映射
        future_to_idx = {
            executor.submit(calculate_drift_single_frame, frame, roi, base_loc): i 
            for i, frame in enumerate(image_stack)
        }
        
        completed_count = 0
        total_count = len(image_stack)
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                drifts[idx] = future.result()
            except Exception:
                drifts[idx] = np.array([0, 0])
            
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total_count)
    
    drifts = np.array(drifts)
    drifts = np.apply_along_axis(lambda x: medfilt(x, kernel_size), 0, drifts)
    return drifts

def apply_drift_correction(image_stack: np.ndarray,
                           drifts: np.ndarray,
                           max_workers: int = 8,
                           progress_callback=None) -> np.ndarray:
    """
    应用漂移矫正
    progress_callback: func(current, total)
    """
    def translate_frame(args):
        frame, drift = args
        rows, cols = frame.shape[:2]
        M = np.float32([[1, 0, drift[0]], [0, 1, drift[1]]])
        return cv2.warpAffine(frame, M, (cols, rows))
    
    corrected_frames = [None] * len(image_stack)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(translate_frame, (image_stack[i], drifts[i])): i
            for i in range(len(image_stack))
        }
        
        completed_count = 0
        total_count = len(image_stack)
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            corrected_frames[idx] = future.result()
            
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total_count)
                
    result = np.stack(corrected_frames, axis=0)
    return np.ascontiguousarray(result)

def validate_roi(image_shape: Tuple[int, int],
                 roi_bbox: Tuple[int, int, int, int]) -> bool:
    h, w = image_shape
    x1, y1, x2, y2 = roi_bbox
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h: return False
    if (x2 - x1) < 10 or (y2 - y1) < 10: return False
    return True
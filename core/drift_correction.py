"""
漂移矫正模块
更新日志:
- calculate_drift_curve: 增加 progress_callback 支持
- apply_drift_correction: 修复内存爆炸问题，引入 memmap 支持
"""
import numpy as np
import cv2
from typing import Tuple, Optional
import concurrent.futures
from utils.memory_utils import create_huge_array, release_memmap_pages
import os

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
    return drifts

def apply_drift_correction(image_stack: np.ndarray,
                           drifts: np.ndarray,
                           max_workers: int = 8,
                           progress_callback=None) -> np.ndarray:
    """
    应用漂移矫正 (内存优化版)
    使用 create_huge_array 预分配内存/硬盘空间，避免 OOM。
    """
    T, H, W = image_stack.shape
    
    # === 使用内存工具分配结果数组 (可能在 RAM，也可能在 Disk) ===
    result, temp_filename = create_huge_array(image_stack.shape, image_stack.dtype, fill_zeros=False)
    
    try:
        def process_batch(batch_indices):
            # 处理一批索引，直接写入 result
            for i in batch_indices:
                frame = image_stack[i]
                drift = drifts[i]
                M = np.float32([[1, 0, drift[0]], [0, 1, drift[1]]])
                # 直接写入结果数组
                result[i] = cv2.warpAffine(frame, M, (W, H))
            return len(batch_indices)
        
        # 分块处理 (避免 Future 对象过多消耗内存)
        chunk_size = 100
        indices = list(range(T))
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
        
        completed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_batch, chunk): chunk for chunk in chunks}
            
            for future in concurrent.futures.as_completed(futures):
                count = future.result()
                completed_count += count
                if progress_callback:
                    progress_callback(completed_count, T)
        
        release_memmap_pages(result)
        return result

    except Exception as e:
        print(f"Error in apply_drift_correction: {e}")
        # 尝试清理临时文件（如果创建了）
        if temp_filename and os.path.exists(temp_filename):
            try:
                # 尽量清理，注意：如果是 memmap，因为被 result 引用，可能无法立即删除 (Windows)
                # 通常需要 del result 之后才能删除。
                # 这里我们重新抛出异常，让上层处理，或者依靠系统清理
                pass
            except:
                pass
        raise e

def validate_roi(image_shape: Tuple[int, int],
                 roi_bbox: Tuple[int, int, int, int]) -> bool:
    h, w = image_shape
    x1, y1, x2, y2 = roi_bbox
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h: return False
    if (x2 - x1) < 10 or (y2 - y1) < 10: return False
    return True
"""
几何变换模块
提供旋转和裁剪功能
更新日志:
- rotate_image_stack: 增加 expand 参数，支持旋转时扩大画布保留全图 (仿ImageJ Enlarge)
"""
import numpy as np
import cv2
from typing import Tuple, Optional

def calculate_rotation_angle(line_points: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    """根据画线计算旋转角度"""
    (x1, y1), (x2, y2) = line_points
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return 90.0 if dy > 0 else -90.0
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    return angle_deg 

def rotate_image_stack(image_stack: np.ndarray,
                       angle: float,
                       center: Optional[Tuple[int, int]] = None,
                       expand: bool = False) -> np.ndarray:
    """
    旋转图像栈
    
    Parameters:
    -----------
    image_stack : np.ndarray
        形状为 (T, Y, X) 的图像栈
    angle : float
        旋转角度
    center : tuple, optional
        旋转中心
    expand : bool
        是否扩大画布以包含所有旋转后的像素 (ImageJ Enlarge模式)
        
    Returns:
    --------
    rotated_stack : np.ndarray
    """
    T, H, W = image_stack.shape
    
    if center is None:
        center = (W // 2, H // 2)
    
    # 获取基础旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    dest_w, dest_h = W, H
    
    if expand:
        # 计算旋转后的新边界框尺寸
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_w = int((H * sin) + (W * cos))
        new_h = int((H * cos) + (W * sin))
        
        # 调整旋转矩阵的平移分量，确保图像居中
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        dest_w, dest_h = new_w, new_h

    # 旋转所有帧
    rotated_frames = []
    for frame in image_stack:
        # 使用 borderMode=cv2.BORDER_CONSTANT (黑色填充)
        rotated = cv2.warpAffine(frame, M, (dest_w, dest_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_frames.append(rotated)
    
    return np.ascontiguousarray(np.stack(rotated_frames, axis=0))

def crop_image_stack(image_stack: np.ndarray,
                     bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """裁剪图像栈"""
    x1, y1, x2, y2 = bbox
    T, H, W = image_stack.shape
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))
    return np.ascontiguousarray(image_stack[:, y1:y2, x1:x2])

def crop_multiple_rois(image_stack: np.ndarray, bboxes: list) -> list:
    """批量裁剪"""
    return [crop_image_stack(image_stack, bbox) for bbox in bboxes]

def validate_bbox(image_shape: Tuple[int, int],
                  bbox: Tuple[int, int, int, int]) -> bool:
    """验证边界框"""
    h, w = image_shape
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h: return False
    if x2 <= x1 or y2 <= y1: return False
    return True
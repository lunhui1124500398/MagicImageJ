"""
几何变换模块
提供旋转和裁剪功能
"""
import numpy as np
import cv2
from typing import Tuple, Optional


def calculate_rotation_angle(line_points: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    """
    根据画线计算旋转角度
    
    Parameters:
    -----------
    line_points : tuple
        线段的两个端点 ((x1, y1), (x2, y2))
        
    Returns:
    --------
    angle : float
        需要旋转的角度（度数），使线段水平
    """
    (x1, y1), (x2, y2) = line_points
    
    # 计算角度
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0:
        return 90.0 if dy > 0 else -90.0
    
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg 


def rotate_image_stack(image_stack: np.ndarray,
                      angle: float,
                      center: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    旋转图像栈
    
    Parameters:
    -----------
    image_stack : np.ndarray
        形状为 (T, Y, X) 的图像栈
    angle : float
        旋转角度（度数，逆时针为正）
    center : tuple, optional
        旋转中心，默认为图像中心
        
    Returns:
    --------
    rotated_stack : np.ndarray
        旋转后的图像栈
    """
    T, H, W = image_stack.shape
    
    if center is None:
        center = (W // 2, H // 2)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 旋转所有帧
    rotated_frames = []
    for frame in image_stack:
        rotated = cv2.warpAffine(frame, M, (W, H))
        rotated_frames.append(rotated)
    
    return np.ascontiguousarray(np.stack(rotated_frames, axis=0))


def crop_image_stack(image_stack: np.ndarray,
                    bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    裁剪图像栈
    
    Parameters:
    -----------
    image_stack : np.ndarray
        形状为 (T, Y, X) 的图像栈
    bbox : tuple
        边界框 (x1, y1, x2, y2)
        
    Returns:
    --------
    cropped_stack : np.ndarray
        裁剪后的图像栈
    """
    x1, y1, x2, y2 = bbox
    
    # 确保坐标在有效范围内
    T, H, W = image_stack.shape
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))
    
    # 修复：NumPy 切片返回的是 View (非连续)，必须 copy 为连续数组
    # 否则 Vispy 会在渲染时崩溃
    return np.ascontiguousarray(image_stack[:, y1:y2, x1:x2])


def crop_multiple_rois(image_stack: np.ndarray,
                      bboxes: list) -> list:
    """
    批量裁剪多个ROI
    
    Parameters:
    -----------
    image_stack : np.ndarray
        形状为 (T, Y, X) 的图像栈
    bboxes : list
        边界框列表，每个元素为 (x1, y1, x2, y2)
        
    Returns:
    --------
    cropped_stacks : list
        裁剪后的图像栈列表
    """
    return [crop_image_stack(image_stack, bbox) for bbox in bboxes]


def validate_bbox(image_shape: Tuple[int, int],
                 bbox: Tuple[int, int, int, int]) -> bool:
    """
    验证边界框是否有效
    
    Parameters:
    -----------
    image_shape : tuple
        图像形状 (height, width)
    bbox : tuple
        边界框 (x1, y1, x2, y2)
        
    Returns:
    --------
    valid : bool
        边界框是否有效
    """
    h, w = image_shape
    x1, y1, x2, y2 = bbox
    
    # 检查边界
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    
    # 检查大小
    if x2 <= x1 or y2 <= y1:
        return False
    
    return True

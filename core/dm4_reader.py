"""
DM4文件读取模块
包含安全归一化算法，修复黑帧Bug
"""
import numpy as np
import dm4
from pathlib import Path
from typing import Optional, Tuple
import concurrent.futures
from tqdm import tqdm


def safe_normalize(data: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """
    安全归一化算法 - 修复黑帧Bug
    使用百分位数裁剪异常值
    """
    # 转为float32避免溢出
    data = data.astype(np.float32)
    
    # 使用0.1和99.9百分位数裁剪异常值
    q001 = np.percentile(data, 0.1)
    q999 = np.percentile(data, 99.9)
    rng = q999 - q001
    
    # 处理范围过小的情况（防止除零）
    if rng < 1e-6:
        glob_min = np.min(data)
        glob_max = np.max(data)
        rng = glob_max - glob_min
        
        if rng > 0:
            data = (data - glob_min) / rng * (2**bit_depth - 1)
        else:
            # 全图相同值，返回全零
            data = np.zeros_like(data)
    else:
        # 正常归一化
        data = np.clip(data, q001, q999)
        data = (data - q001) / rng * (2**bit_depth - 1)
    
    # 处理NaN值
    data = np.nan_to_num(data, nan=0)
    
    # 转换为目标位深度
    if bit_depth == 8:
        return data.astype(np.uint8)
    elif bit_depth == 16:
        return data.astype(np.uint16)
    else:
        return data.astype(np.float32)


def read_single_dm4(filepath: str, bit_depth: int = 8) -> Optional[np.ndarray]:
    """
    读取单个DM4文件
    
    Parameters:
    -----------
    filepath : str
        DM4文件路径
    bit_depth : int
        输出位深度 (8/16/32)
        
    Returns:
    --------
    image : np.ndarray or None
        2D图像数组，失败返回None
    """
    try:
        with dm4.DM4File.open(filepath) as dm4data:
            tags = dm4data.read_directory()
            
            # 尝试获取ImageData标签
            try:
                image_data_tag = tags.named_subdirs['ImageList'].unnamed_subdirs[1].named_subdirs['ImageData']
            except (IndexError, KeyError):
                try:
                    image_data_tag = tags.named_subdirs['ImageList'].unnamed_subdirs[0].named_subdirs['ImageData']
                except Exception:
                    return None
            
            # 读取图像数据
            image_tag = image_data_tag.named_tags['Data']
            XDim = dm4data.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[0])
            YDim = dm4data.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[1])
            
            # 转换为numpy数组
            np_array = np.array(dm4data.read_tag_data(image_tag), dtype=np.float32)
            np_array = np.reshape(np_array, (YDim, XDim))
            
            # 安全归一化
            return safe_normalize(np_array, bit_depth)
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def read_dm4_sequence(folder_path: str, 
                     bit_depth: int = 8,
                     max_workers: int = 8,
                     progress_callback=None) -> Tuple[np.ndarray, dict]:
    """
    读取文件夹中的DM4序列
    
    Parameters:
    -----------
    folder_path : str
        包含DM4文件的文件夹路径
    bit_depth : int
        输出位深度
    max_workers : int
        并行线程数
    progress_callback : callable
        进度回调函数
        
    Returns:
    --------
    image_stack : np.ndarray
        形状为 (T, Y, X) 的图像栈
    metadata : dict
        元数据字典
    """
    folder = Path(folder_path)
    dm4_files = sorted(folder.glob('**/*.dm4'))
    
    if not dm4_files:
        raise ValueError(f"No DM4 files found in {folder_path}")
    
    print(f"Found {len(dm4_files)} DM4 files")
    
    # 并行读取
    images = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_single_dm4, str(f), bit_depth): i 
                   for i, f in enumerate(dm4_files)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="Loading DM4 files"):
            idx = futures[future]
            img = future.result()
            if img is not None:
                images.append((idx, img))
            
            if progress_callback:
                progress_callback(len(images), len(dm4_files))
    
    # 按索引排序并堆叠
    images.sort(key=lambda x: x[0])
    # 修复：stack 后强制连续
    image_stack = np.ascontiguousarray(np.stack([img for _, img in images], axis=0))
    
    # 生成元数据
    metadata = {
        'source_folder': str(folder),
        'num_frames': len(image_stack),
        'bit_depth': bit_depth,
        'shape': image_stack.shape,
        'dtype': str(image_stack.dtype)
    }
    
    return image_stack, metadata


def get_dm4_folders(root_path: str) -> list:
    """
    扫描根目录，找到所有包含DM4文件的子文件夹
    
    Parameters:
    -----------
    root_path : str
        根目录路径
        
    Returns:
    --------
    folders : list
        包含DM4文件的文件夹列表
    """
    root = Path(root_path)
    dm4_folders = []
    
    for subfolder in root.iterdir():
        if subfolder.is_dir():
            # 检查是否包含dm4文件
            if list(subfolder.glob('*.dm4')):
                dm4_folders.append(str(subfolder))
    
    return dm4_folders

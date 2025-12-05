"""
DM4文件读取模块
包含安全归一化算法，修复黑帧Bug
[Fix OOM]: read_dm4_sequence 现在预分配内存(memmap)而不是使用 list.append
"""
import numpy as np
import dm4
from pathlib import Path
from typing import Optional, Tuple
import concurrent.futures
from tqdm import tqdm
from utils.memory_utils import create_huge_array
import os

def safe_normalize(data: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """
    安全归一化算法 - 修复黑帧Bug
    """
    data = data.astype(np.float32)
    
    q001 = np.percentile(data, 0.1)
    q999 = np.percentile(data, 99.9)
    rng = q999 - q001
    
    if rng < 1e-6:
        glob_min = np.min(data)
        glob_max = np.max(data)
        rng = glob_max - glob_min
        
        if rng > 0:
            data = (data - glob_min) / rng * (2**bit_depth - 1)
        else:
            data = np.zeros_like(data)
    else:
        data = np.clip(data, q001, q999)
        data = (data - q001) / rng * (2**bit_depth - 1)
    
    data = np.nan_to_num(data, nan=0)
    
    if bit_depth == 8:
        return data.astype(np.uint8)
    elif bit_depth == 16:
        return data.astype(np.uint16)
    else:
        return data.astype(np.float32)


def read_single_dm4_into_buffer(filepath: str, 
                                buffer_array: np.ndarray, 
                                index: int,
                                bit_depth: int = 8) -> bool:
    """
    读取单个DM4文件并直接写入 buffer[index]
    """
    try:
        with dm4.DM4File.open(filepath) as dm4data:
            tags = dm4data.read_directory()
            try:
                image_data_tag = tags.named_subdirs['ImageList'].unnamed_subdirs[1].named_subdirs['ImageData']
            except (IndexError, KeyError):
                try:
                    image_data_tag = tags.named_subdirs['ImageList'].unnamed_subdirs[0].named_subdirs['ImageData']
                except Exception:
                    return False
            
            image_tag = image_data_tag.named_tags['Data']
            XDim = dm4data.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[0])
            YDim = dm4data.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[1])
            
            np_array = np.array(dm4data.read_tag_data(image_tag), dtype=np.float32)
            np_array = np.reshape(np_array, (YDim, XDim))
            
            # Normalize and write directly to buffer
            buffer_array[index] = safe_normalize(np_array, bit_depth)
            return True
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False


def get_first_image_shape(filepath: str) -> Optional[Tuple[int, int]]:
    """读取第一个文件以获取尺寸"""
    try:
        with dm4.DM4File.open(filepath) as dm4data:
            tags = dm4data.read_directory()
            try:
                image_data_tag = tags.named_subdirs['ImageList'].unnamed_subdirs[1].named_subdirs['ImageData']
            except:
                image_data_tag = tags.named_subdirs['ImageList'].unnamed_subdirs[0].named_subdirs['ImageData']
            
            XDim = dm4data.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[0])
            YDim = dm4data.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[1])
            return (YDim, XDim)
    except:
        return None

def read_dm4_sequence(folder_path: str, 
                      bit_depth: int = 8,
                      max_workers: int = 8,
                      progress_callback=None) -> Tuple[np.ndarray, dict]:
    """
    读取文件夹中的DM4序列 (OOM Safe)
    """
    folder = Path(folder_path)
    dm4_files = sorted(folder.glob('**/*.dm4'))
    
    if not dm4_files:
        raise ValueError(f"No DM4 files found in {folder_path}")
    
    count = len(dm4_files)
    print(f"Found {count} DM4 files")
    
    # 1. 预读取获取尺寸
    shape = get_first_image_shape(str(dm4_files[0]))
    if not shape:
        raise ValueError("Failed to read dimensions from first file")
    
    H, W = shape
    full_shape = (count, H, W)
    dtype = np.uint8 if bit_depth == 8 else (np.uint16 if bit_depth == 16 else np.float32)
    
    # 2. 预分配大数组 (Memmap if huge)
    image_stack, temp_file = create_huge_array(full_shape, dtype, fill_zeros=False)
    
    # 3. 并行读取并填入
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务：直接传入文件名、目标数组引用、目标索引
        futures = {
            executor.submit(read_single_dm4_into_buffer, str(f), image_stack, i, bit_depth): i 
            for i, f in enumerate(dm4_files)
        }
        
        completed = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=count, desc="Loading DM4"):
            res = future.result()
            completed += 1
            if progress_callback:
                progress_callback(completed, count)
    
    # 4. Flush if memmap
    if hasattr(image_stack, 'flush'):
        image_stack.flush()
    
    metadata = {
        'source_folder': str(folder),
        'num_frames': count,
        'bit_depth': bit_depth,
        'shape': full_shape,
        'dtype': str(dtype),
        'memmap_path': temp_file
    }
    
    return image_stack, metadata

def get_dm4_folders(root_path: str) -> list:
    root = Path(root_path)
    dm4_folders = []
    for subfolder in root.iterdir():
        if subfolder.is_dir():
            if list(subfolder.glob('*.dm4')):
                dm4_folders.append(str(subfolder))
    return dm4_folders
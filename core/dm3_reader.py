"""
DM3 Reader Module using ncempy.
Supports reading DM3 sequences and metadata extraction.
Compatible with MagicImageJ architecture.
"""
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple
import concurrent.futures
from tqdm import tqdm
from utils.memory_utils import create_huge_array, release_memmap_pages
from utils.utils import natural_sort_key
import datetime
import traceback


def _resolve_selected_frame_indices(total_count: int, frame_indices=None) -> list[int]:
    if frame_indices is None:
        return list(range(total_count))

    selected = sorted({int(idx) for idx in frame_indices if 0 <= int(idx) < total_count})
    if not selected:
        raise ValueError("No valid DM3 frames were selected for import.")
    return selected


def _compress_frame_ranges(frame_indices: list[int]) -> list[list[int]]:
    if not frame_indices:
        return []

    ranges = []
    start = frame_indices[0]
    end = frame_indices[0]
    for idx in frame_indices[1:]:
        if idx == end + 1:
            end = idx
            continue
        ranges.append([start, end])
        start = idx
        end = idx
    ranges.append([start, end])
    return ranges

# Try importing ncempy
try:
    import ncempy.io.dm
    NCEMPY_AVAILABLE = True
except ImportError:
    NCEMPY_AVAILABLE = False
    print("Warning: ncempy not found. DM3 reading will fail.")

def safe_normalize(data: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Safe normalization (Copied from dm4_reader for consistency)"""
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

def extract_metadata_centered(dataset_tags: dict, global_tags: dict) -> dict:
    """
    Extract metadata from the specific tags of the matched image.
    Fallback to global tags if not found.
    """
    meta = {
        'pixel_size': 1.0,
        'unit': 'nm',
        'exposure': 0.0,
        'magnification': 0.0,
        'date': datetime.datetime.now().strftime("%Y%m%d"),
        'raw_date': '',
        'total_electrons': 0.0,
        'brightness_scale': 1.0
    }

    # Helper to search deep dict (limited scope)
    def find_key(d, target, partial=False):
        if not isinstance(d, dict): return None
        for k, v in d.items():
            k_str = str(k)
            if partial:
                if target.lower() in k_str.lower(): return v
            else:
                if k_str == target: return v
            
            if isinstance(v, dict):
                res = find_key(v, target, partial)
                if res is not None: return res
        return None

    # 1. Pixel Size & Unit
    # Check Calibrations in the specific dataset tags
    scale = find_key(dataset_tags, "Scale")
    if scale is None: scale = find_key(dataset_tags, "Pixel Size", partial=True)
    if scale is not None and isinstance(scale, (int, float, np.number)):
        meta['pixel_size'] = float(scale)

    units = find_key(dataset_tags, "Units")
    if units is None: units = find_key(dataset_tags, "Unit", partial=True)
    if units and isinstance(units, str):
        meta['unit'] = units
        
    # 2. Exposure
    # Often in ImageTags -> DataBar or Acquisition
    exposure = find_key(dataset_tags, "Exposure Time (s)")
    if exposure is None: exposure = find_key(dataset_tags, "Exposure Time")
    if exposure is None: exposure = find_key(dataset_tags, "Exposure", partial=True)
    if exposure and isinstance(exposure, (int, float, np.number)):
         meta['exposure'] = float(exposure)

    # 3. Magnification
    # Prioritize 'Indicated Magnification'
    mag = find_key(dataset_tags, "Indicated Magnification")
    if mag is None: mag = find_key(dataset_tags, "Indicated Magnification", partial=True)
    if mag is None: mag = find_key(dataset_tags, "Magnification", partial=True)
    if mag and isinstance(mag, (int, float, np.number)):
        meta['magnification'] = float(mag)

    # 4. Date
    acq_date = find_key(dataset_tags, "Acquisition Date")
    if acq_date is None: acq_date = find_key(dataset_tags, "Date", partial=True)
    if acq_date and isinstance(acq_date, str):
        meta['raw_date'] = acq_date
        for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%H:%M:%S"]:
             try:
                 dt = datetime.datetime.strptime(acq_date, fmt)
                 meta['date'] = dt.strftime("%Y%m%d")
                 break
             except: pass
             
    # 5. Brightness Scale
    # Specific to this dataset's calibrations
    br_scale = 1.0
    # Try finding Brightness dict first to be safe
    def find_brightness_scale(d):
        if not isinstance(d, dict): return None
        for k, v in d.items():
            if "brightness" in str(k).lower():
                if isinstance(v, dict) and "Scale" in v:
                    return v['Scale']
            if isinstance(v, dict):
                res = find_brightness_scale(v)
                if res is not None: return res
        return None
        
    found_scale = find_brightness_scale(dataset_tags)
    if found_scale is not None:
        br_scale = float(found_scale)
        
    meta['brightness_scale'] = br_scale
    
    return meta

def read_single_dm3_into_buffer(filepath: str, 
                                buffer_array: np.ndarray, 
                                index: int,
                                bit_depth: int = 8) -> bool:
    try:
        # Use simple getDataset for Image Data
        dm3 = ncempy.io.dm.dmReader(filepath)
        if 'data' not in dm3:
            return False
            
        data = dm3['data']
        
        # Normalize and write
        buffer_array[index] = safe_normalize(data, bit_depth)
        return True
    except Exception as e:
        print(f"Error reading DM3 {filepath}: {e}")
        return False

def get_first_dm3_shape(filepath: str) -> Optional[Tuple[int, int]]:
    try:
        dm3 = ncempy.io.dm.dmReader(filepath)
        if 'data' in dm3:
            return dm3['data'].shape # (H, W) or (Y, X)
        return None
    except:
        return None

def read_dm3_sequence(folder_path: str, 
                      bit_depth: int = 8,
                      max_workers: int = 8,
                      progress_callback=None,
                      frame_indices=None) -> Tuple[np.ndarray, dict]:
    folder = Path(folder_path)
    dm3_files = sorted(folder.glob('**/*.dm3'), key=natural_sort_key)
    
    if not dm3_files:
        raise ValueError(f"No DM3 files found in {folder_path}")
    
    total_count = len(dm3_files)
    selected_indices = _resolve_selected_frame_indices(total_count, frame_indices)
    selected_files = [dm3_files[idx] for idx in selected_indices]
    count = len(selected_files)
    
    # 1. Shape
    shape = get_first_dm3_shape(str(selected_files[0]))
    if not shape:
        raise ValueError("Failed to read dimensions from first DM3 file")
    
    H, W = shape
    full_shape = (count, H, W)
    dtype = np.uint8 if bit_depth == 8 else (np.uint16 if bit_depth == 16 else np.float32)
    
    # 2. Alloc
    image_stack, temp_file = create_huge_array(full_shape, dtype, fill_zeros=False)
    
    # 3. Parallel Load
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(read_single_dm3_into_buffer, str(f), image_stack, i, bit_depth): i 
            for i, f in enumerate(selected_files)
        }
        
        completed = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=count, desc="Loading DM3"):
            future.result()
            completed += 1
            if progress_callback:
                progress_callback(completed, count)
                
    release_memmap_pages(image_stack)
        
    metadata = {
        'source_folder': str(folder),
        'num_frames': count,
        'source_num_frames': total_count,
        'bit_depth': bit_depth,
        'shape': full_shape,
        'dtype': str(dtype),
        'memmap_path': temp_file,
        'frame_selection_applied': count != total_count
    }
    if count != total_count:
        metadata['selected_frame_ranges'] = _compress_frame_ranges(selected_indices)
    
    return image_stack, metadata

def read_dm3_metadata(file_path: str):
    """
    Reads metadata for dose calculation from a single DM3 file.
    Uses fileDM for comprehensive tag access.
    """
    try:
        with ncempy.io.dm.fileDM(file_path) as f:
            all_tags = f.allTags
            dataset = f.getDataset(0)
            data = dataset['data']
            h, w = data.shape
            
            # Locate the correct ImageList index by matching dimensions
            # allTags is flat, e.g. '.ImageList.2.ImageData.Dimensions.1': 2048
            
            best_prefix = ""
            
            # Group dimensions by valid prefixes (e.g. ".ImageList.1")
            candidates = {} # prefix -> [dim1, dim2] (list to collect)
            
            import re
            # Regex to capture: (.ImageList.\d+).ImageData.Dimensions.(\d+)
            dim_pattern = re.compile(r"(\.ImageList\.\d+)\.ImageData\.Dimensions\.(\d+)")
            
            for k, v in all_tags.items():
                m = dim_pattern.search(str(k))
                if m:
                    prefix = m.group(1)
                    if prefix not in candidates:
                        candidates[prefix] = []
                    candidates[prefix].append(v)
            
            # Find match
            for prefix, dims in candidates.items():
                 # dims needs 2 values. Match with h,w
                 if len(dims) >= 2:
                     # Check if fuzzy match or exact match
                     # data.shape is usually (Y, X)
                     if sorted(dims) == sorted([h, w]):
                         # print(f"Match found at {prefix} with dims {dims}")
                         best_prefix = prefix
                         break
            
            if not best_prefix and len(candidates) > 0:
                 # Fallback to the last candidate (often largest?) or just first?
                 # If we have 2048x2048 data, we expect a match.
                 pass

            # Initialize meta
            meta = {
                'pixel_size': 1.0,
                'unit': 'nm',
                'exposure': 0.0,
                'magnification': 0.0,
                'date': datetime.datetime.now().strftime("%Y%m%d"),
                'raw_date': '',
                'total_electrons': 0.0,
                'brightness_scale': 1.0
            }

            # Extract using the prefix
            # Helper to find specific key with prefix
            def get_tag_value(target_suffix, partial=False):
                # target_suffix e.g. ".ImageData.Calibrations.Brightness.Scale"
                # Search exact construction
                full_key = best_prefix + target_suffix
                if full_key in all_tags:
                    return all_tags[full_key]
                
                # Fallback: Search for suffix in keys starting with best_prefix 
                # (useful if exact path varies slightly)
                if partial and best_prefix:
                    for k, v in all_tags.items():
                        if k.startswith(best_prefix) and target_suffix in k:
                            return v
                
                # Fallback Global Search (if prefix failed or not found)
                # But prefer exact match on any key
                if partial:
                     for k, v in all_tags.items():
                         if target_suffix in k:
                            return v
                return None

            # 1. Pixel Size
            # .ImageData.Calibrations.Dimension.1.Scale
            # We usually want Dimension.1 and Dimension.2. Assuming square/isotropic.
            p_scale = get_tag_value(".ImageData.Calibrations.Dimension.1.Scale", partial=True) # Partial to catch .Dimensions.1...
            if p_scale and isinstance(p_scale, (int, float, np.number)):
                meta['pixel_size'] = float(p_scale)
            
            p_unit = get_tag_value(".ImageData.Calibrations.Dimension.1.Units", partial=True)
            if p_unit: meta['unit'] = str(p_unit)

            # 2. Exposure
            exp = get_tag_value("Exposure Time (s)", partial=True)
            if exp and isinstance(exp, (int, float, np.number)):
                meta['exposure'] = float(exp)
                
            # 3. Magnification (Indicated)
            mag = get_tag_value("Indicated Magnification", partial=True)
            if mag is None: mag = get_tag_value("Magnification", partial=True)
            if mag and isinstance(mag, (int, float, np.number)):
                meta['magnification'] = float(mag)
                
            # 4. Date
            date = get_tag_value("Acquisition Date", partial=True)
            if date:
                meta['raw_date'] = str(date)
                for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%H:%M:%S"]:
                     try:
                         dt = datetime.datetime.strptime(str(date), fmt)
                         meta['date'] = dt.strftime("%Y%m%d")
                         break
                     except: pass

            # 5. Brightness Scale
            # .ImageData.Calibrations.Brightness.Scale
            br_scale = get_tag_value(".ImageData.Calibrations.Brightness.Scale")
            if br_scale is None:
                 # Try partial under prefix
                 br_scale = get_tag_value("Brightness.Scale", partial=True)
            
            if br_scale is not None:
                meta['brightness_scale'] = float(br_scale)
            
            # Apply to total sum
            total_electrons = np.sum(data, dtype=np.float64) * meta['brightness_scale']
            meta['total_electrons'] = total_electrons
            
            return data, meta

    except Exception as e:
        print(f"Meta read error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

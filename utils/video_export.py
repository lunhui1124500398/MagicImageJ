"""
视频导出工具 - 终极增强版 (防锁死 + 自动修复分辨率)
修复日志:
- [Critical] 增加 try...finally 结构，确保无论发生什么错误，VideoWriter 都会释放文件锁。
- [Fix] 在写入前尝试删除同名旧文件，防止被占用导致静默失败。
- [Fix] 保持奇数分辨率自动修复逻辑。
"""
import numpy as np
import cv2
from pathlib import Path
import os
from typing import Optional, Callable
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import contextlib

@contextlib.contextmanager
def change_dir(destination):
    """
    上下文管理器：临时切换工作目录
    用于解决 OpenCV VideoWriter 在 Windows 下不支持长路径的问题
    """
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)

def export_to_video(image_stack: np.ndarray,
                   output_path: str,
                   fps: int = 60,
                   codec: str = 'mp4v',
                   quality: int = 100,
                   scale_bar_config: Optional[dict] = None,
                   timestamp_config: Optional[dict] = None,
                   callback: Optional[Callable[[int, int], None]] = None,
                   ) -> bool:
    """
    导出视频 - 健壮性增强版
    """
    out = None
    try:
        # 1. 检查数据维度
        if image_stack.ndim == 3:
            T, H, W = image_stack.shape
            is_color = False
        elif image_stack.ndim == 4:
            T, H, W, C = image_stack.shape
            is_color = True
        else:
            raise ValueError(f"Unsupported image shape: {image_stack.shape}")
        
        full_path = Path(output_path).resolve()
        if not full_path.suffix:
            full_path = full_path.with_suffix('.mp4')
            
        parent_dir = full_path.parent
        file_name = full_path.name

        # 2. [防锁死检测] 尝试清理旧文件
        # 使用 os.remove 结合长路径前缀
        safe_path_str = str(full_path)
        if os.name == 'nt' and not safe_path_str.startswith('\\\\?\\'):
            safe_path_str = '\\\\?\\' + safe_path_str

        if os.path.exists(safe_path_str):
            try:
                os.remove(safe_path_str)
            except PermissionError:
                print(f"Error: File {file_name} is locked by another process.")
                return False
            except Exception as e:
                print(f"Warning: Could not remove existing file: {e}")

        # 3. [分辨率修复] 强制尺寸为偶数
        # 很多编码器(H264等)要求长宽必须是2的倍数
        export_W = W if W % 2 == 0 else W - 1
        export_H = H if H % 2 == 0 else H - 1
        
        needs_resize = (export_W != W) or (export_H != H)
        if needs_resize:
            print(f"Auto-adjusting video output from {W}x{H} to {export_W}x{export_H} (Codec requirement).")

        # 4. 预计算归一化参数 (避免循环内重复计算导致闪烁)
        source_data = image_stack
        glob_min, rng = 0, 1.0
        if source_data.dtype != np.uint8:
            glob_min = source_data.min()
            glob_max = source_data.max()
            rng = glob_max - glob_min
            if rng <= 0: rng = 1.0

        # 5. 初始化 VideoWriter
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            with change_dir(parent_dir):
                out = cv2.VideoWriter(
                    file_name, fourcc, fps, (export_W, export_H), isColor=True
                )

                # 检查是否成功打开
                if not out.isOpened():
                    print(f"Error: VideoWriter failed to open. Codec: {codec}")
                    # 尝试释放并重试 MJPG
                    out.release() 
                    if codec != 'MJPG':
                        print("Retrying with fallback codec 'MJPG'...")
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out = cv2.VideoWriter(file_name, fourcc, fps, (export_W, export_H), isColor=True)
                        if not out.isOpened():
                            return False
                    else:
                        return False

                # 6. 逐帧写入
                total_frames = len(source_data)
                for i, frame in enumerate(tqdm(source_data, desc="Exporting video")):
                    # 归一化
                    if frame.dtype != np.uint8:
                        frame_norm = ((frame - glob_min) / rng * 255).astype(np.uint8)
                    else:
                        frame_norm = frame

                    # 转 RGB
                    if is_color:
                        if frame_norm.shape[-1] == 4: 
                            frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_RGBA2RGB)
                        else: 
                            frame_rgb = frame_norm 
                    else:
                        frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2RGB)

                    # 调整尺寸
                    if needs_resize:
                        frame_rgb = cv2.resize(frame_rgb, (export_W, export_H), interpolation=cv2.INTER_LINEAR)

                    # 绘制 Overlay
                    has_overlay = (scale_bar_config and scale_bar_config.get('enable')) or \
                                  (timestamp_config and timestamp_config.get('enable'))
                    
                    if has_overlay:
                        pil_img = Image.fromarray(frame_rgb)
                        draw = ImageDraw.Draw(pil_img, 'RGBA')

                        if scale_bar_config and scale_bar_config.get('enable', False):
                            _draw_scale_bar_pil(draw, scale_bar_config)

                        if timestamp_config and timestamp_config.get('enable', False):
                            _draw_timestamp_pil(draw, timestamp_config, i)
                        
                        frame_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    out.write(frame_bgr)
                    
                    # 调用回调以更新进度
                    if callback is not None:
                        callback(i + 1, total_frames)
                    
        except Exception as e:
            # 捕获切换目录可能的错误
            print(f"File System Error: {e}")
            return False

        return True

    except Exception as e:
        print(f"Error exporting video: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # === [核心修复] 无论成功失败，必须释放资源 ===
        if out is not None:
            out.release()
            print("VideoWriter released.")

def _get_font(size):
    try: return ImageFont.truetype("arial.ttf", size)
    except: 
        try: return ImageFont.truetype("DejaVuSans.ttf", size)
        except: return ImageFont.load_default()

def _draw_scale_bar_pil(draw, cfg):
    x, y = cfg.get('position', (50, 50))
    length_unit = cfg.get('length', 100)
    ratio = cfg.get('ratio', 1.0)
    unit = cfg.get('unit', 'nm')
    
    if ratio == 0: bar_w = 100
    else: bar_w = int(length_unit / ratio)
    
    bar_h = cfg.get('height', 80)
    thick = cfg.get('thickness', 8)
    font_size = cfg.get('font_size', 36)
    padding = cfg.get('padding', 10)
    
    def to_255(c): return tuple(int(x*255) for x in c[:3])
    
    text_color = to_255(cfg.get('color', (1,1,1,1)))
    bg_color = to_255(cfg.get('bg_color', (0,0,0,1)))
    bg_alpha = int(cfg.get('bg_alpha', 100) / 100 * 255)
    use_bg = cfg.get('use_bg', True)

    if use_bg:
        bg_x = x - padding
        bg_w_total = bar_w + 2 * padding
        draw.rectangle([bg_x, y, bg_x + bg_w_total, y + bar_h], fill=(*bg_color, bg_alpha))

    gap = int(font_size * 0.4)
    total_content_h = thick + gap + font_size
    start_y = y + (bar_h - total_content_h) // 2
    
    draw.rectangle([x, start_y, x + bar_w, start_y + thick], fill=(*text_color, 255))

    font = _get_font(font_size)
    txt = f"{int(length_unit)} {unit}" if length_unit == int(length_unit) else f"{length_unit:.2f} {unit}"
    
    try:
        left, top, right, bottom = font.getbbox(txt)
        text_w = right - left
        text_h = bottom - top
    except:
        text_w, text_h = draw.textsize(txt, font=font)

    text_x = x + (bar_w - text_w) // 2
    text_y = start_y + thick + gap
    draw.text((text_x, text_y), txt, font=font, fill=(*text_color, 255))

def _draw_timestamp_pil(draw, cfg, frame_idx):
    x, y = cfg.get('position', (10, 40))
    font_size = cfg.get('font_size', 32)
    
    def to_255(c): return tuple(int(x*255) for x in c[:3])
    color = to_255(cfg.get('color', (1,1,1,1)))
    
    start = cfg.get('start', 0.0)
    interval = cfg.get('interval', 1.0)
    val = start + frame_idx * interval
    
    fmt = cfg.get('format', 'Custom')
    custom_fmt = cfg.get('custom_fmt', '{:.2f}')
    
    txt = str(val)
    try:
        if fmt == "0": txt = f"{val:.0f}"
        elif fmt == "0.0": txt = f"{val:.1f}"
        elif fmt == "0.00": txt = f"{val:.2f}"
        elif fmt == "00:00": txt = f"{int(val)//60:02d}:{int(val)%60:02d}"
        elif fmt == "Custom": txt = custom_fmt.format(val)
    except: pass
        
    font = _get_font(font_size)
    draw.text((x, y), txt, font=font, fill=(*color, 255))

def get_available_codecs() -> list:
    codecs = ['avc1','H264', 'XVID', 'mp4v', 'MJPG']
    return codecs 
    
def export_to_tiff_stack(image_stack: np.ndarray, output_path: str) -> bool:
    try:
        from tifffile import imwrite
        output_path = Path(output_path)
        if not output_path.suffix: output_path = output_path.with_suffix('.tiff')
        save_path = str(output_path)
        if os.name == 'nt' and not save_path.startswith('\\\\?\\'):
            save_path = '\\\\?\\' + os.path.abspath(save_path)
        imwrite(save_path, image_stack, compression='zlib')
        return True
    except: return False

def export_to_gif(image_stack: np.ndarray, 
                  output_path: str, 
                  fps: int = 10,
                  loop: int = 0,
                  colors: int = 256,
                  scale_bar_config: Optional[dict] = None,
                  timestamp_config: Optional[dict] = None,
                  callback: Optional[Callable[[int, int], None]] = None) -> bool:
    """
    导出 GIF 动图
    
    Args:
        image_stack: numpy 数组 (T, H, W) 或 (T, H, W, C)
        output_path: 输出文件路径
        fps: 帧率，用于计算每帧持续时间 (ms)
        loop: 循环次数，0 表示无限循环
        scale_bar_config: 比例尺配置 (可选)
        timestamp_config: 时间戳配置 (可选)
    
    Returns:
        bool: 成功返回 True，失败返回 False
    """
    try:
        # 1. 检查数据维度
        if image_stack.ndim == 3:
            T, H, W = image_stack.shape
            is_color = False
        elif image_stack.ndim == 4:
            T, H, W, C = image_stack.shape
            is_color = True
        else:
            raise ValueError(f"Unsupported image shape: {image_stack.shape}")
        
        full_path = Path(output_path).resolve()
        if not full_path.suffix:
            full_path = full_path.with_suffix('.gif')
        
        # 确保父目录存在
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 2. 计算每帧持续时间 (毫秒)
        duration_ms = int(1000 / fps) if fps > 0 else 100
        
        # 3. 预计算归一化参数
        glob_min, rng = 0, 1.0
        if image_stack.dtype != np.uint8:
            glob_min = image_stack.min()
            glob_max = image_stack.max()
            rng = glob_max - glob_min
            if rng <= 0: rng = 1.0
        
        # 4. 转换为 PIL Image 列表
        pil_frames = []
        total_gif_frames = len(image_stack)
        for i, frame in enumerate(tqdm(image_stack, desc="Exporting GIF")):
            # 归一化
            if frame.dtype != np.uint8:
                frame_norm = ((frame - glob_min) / rng * 255).astype(np.uint8)
            else:
                frame_norm = frame
            
            # 转 RGB
            if is_color:
                if frame_norm.shape[-1] == 4:
                    frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_RGBA2RGB)
                else:
                    frame_rgb = frame_norm
            else:
                frame_rgb = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2RGB)
            
            # 创建 PIL Image
            pil_img = Image.fromarray(frame_rgb)
            
            # 绘制 Overlay (如果配置了)
            has_overlay = (scale_bar_config and scale_bar_config.get('enable')) or \
                          (timestamp_config and timestamp_config.get('enable'))
            
            if has_overlay:
                draw = ImageDraw.Draw(pil_img, 'RGBA')
                if scale_bar_config and scale_bar_config.get('enable', False):
                    _draw_scale_bar_pil(draw, scale_bar_config)
                if timestamp_config and timestamp_config.get('enable', False):
                    _draw_timestamp_pil(draw, timestamp_config, i)
                # 转回 RGB 模式 (GIF 不支持 RGBA)
                pil_img = pil_img.convert('RGB')
            
            # 转换为调色板模式 (GIF 优化)
            pil_img = pil_img.quantize(colors=colors)
            
            # 调用回调以更新进度
            if callback is not None:
                callback(i + 1, total_gif_frames)
            pil_frames.append(pil_img)
        
        # 5. 保存 GIF
        if pil_frames:
            pil_frames[0].save(
                str(full_path),
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=loop,
                optimize=True
            )
            print(f"GIF saved: {full_path}")
            return True
        return False
        
    except Exception as e:
        print(f"Error exporting GIF: {e}")
        import traceback
        traceback.print_exc()
        return False
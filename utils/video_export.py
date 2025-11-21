"""
视频导出工具 - 增强版
支持 PIL 绘图，支持背景、颜色、字体等高级样式，支持 RGB 输入
修复：移除不存在的 cv2.COLOR_RGB2RGB，直接使用原数据
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

def export_to_video(image_stack: np.ndarray,
                   output_path: str,
                   fps: int = 30,
                   codec: str = 'mp4v',
                   quality: int = 95,
                   scale_bar_config: Optional[dict] = None,
                   timestamp_config: Optional[dict] = None,
                   ) -> bool:
    """
    导出视频 - 支持富文本样式
    """
    try:
        if image_stack.ndim == 3:
            T, H, W = image_stack.shape
            is_color = False
        elif image_stack.ndim == 4:
            T, H, W, C = image_stack.shape
            is_color = True
        else:
            raise ValueError(f"Unsupported image shape: {image_stack.shape}")

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.mp4')

        # 归一化
        if image_stack.dtype != np.uint8:
            stack_min = image_stack.min()
            stack_max = image_stack.max()
            if stack_max > stack_min:
                image_stack = ((image_stack - stack_min) / (stack_max - stack_min) * 255).astype(np.uint8)
            else:
                image_stack = np.zeros_like(image_stack, dtype=np.uint8)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            str(output_path), fourcc, fps, (W, H), isColor=True
        )

        if not out.isOpened():
            raise RuntimeError(f"Failed to create video writer with codec {codec}")

        for i, frame in enumerate(tqdm(image_stack, desc="Exporting video")):
            # 转 RGB (PIL绘图需要)
            if is_color:
                if frame.shape[-1] == 4: 
                    # RGBA -> RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                else: 
                    # RGB -> RGB (直接使用)
                    # 修复：之前误写了 cv2.COLOR_RGB2RGB
                    frame_rgb = frame
            else:
                # Gray -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img, 'RGBA')

            # 1. Draw Scale Bar
            if scale_bar_config and scale_bar_config.get('enable', False):
                _draw_scale_bar_pil(draw, scale_bar_config)

            # 2. Draw Timestamp
            if timestamp_config and timestamp_config.get('enable', False):
                _draw_timestamp_pil(draw, timestamp_config, i)

            # 转回 BGR 用于 OpenCV 保存
            frame_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        return True
    except Exception as e:
        print(f"Error exporting video: {e}")
        return False

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
    
    text_x = x + bar_w / 2
    text_y = start_y + thick + gap
    draw.text((text_x, text_y), txt, font=font, fill=(*text_color, 255), anchor='mt')

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
    draw.text((x, y), txt, font=font, fill=(*color, 255), anchor='lt')

def get_available_codecs() -> list:
    codecs = ['mp4v', 'XVID', 'MJPG', 'H264', 'avc1']
    return codecs 
    
def export_to_tiff_stack(image_stack: np.ndarray, output_path: str) -> bool:
    try:
        from tifffile import imwrite
        output_path = Path(output_path)
        if not output_path.suffix: output_path = output_path.with_suffix('.tiff')
        imwrite(str(output_path), image_stack, compression='zlib')
        return True
    except: return False
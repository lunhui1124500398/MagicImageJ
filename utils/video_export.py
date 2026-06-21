"""
视频导出工具 - v2 (FFmpeg 后端 + OpenCV fallback + GIF 流式写入)
"""
import numpy as np
import cv2
from pathlib import Path
import os
import subprocess
from typing import Optional, Callable, List, Tuple
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import contextlib


_ffmpeg_available: Optional[bool] = None


def is_ffmpeg_available() -> bool:
    global _ffmpeg_available
    if _ffmpeg_available is not None:
        return _ffmpeg_available
    try:
        si = None
        if os.name == 'nt':
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        r = subprocess.run(['ffmpeg', '-version'], capture_output=True, startupinfo=si)
        _ffmpeg_available = (r.returncode == 0)
    except FileNotFoundError:
        _ffmpeg_available = False
    return _ffmpeg_available


@contextlib.contextmanager
def change_dir(destination):
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)


def _ensure_even(w, h):
    return (w if w % 2 == 0 else w - 1, h if h % 2 == 0 else h - 1)


def _parse_stack(image_stack):
    if image_stack.ndim == 3:
        T, H, W = image_stack.shape
        return T, H, W, False
    elif image_stack.ndim == 4:
        T, H, W, C = image_stack.shape
        return T, H, W, True
    raise ValueError(f"Unsupported image shape: {image_stack.shape}")


def _precompute_norm(image_stack):
    if image_stack.dtype != np.uint8:
        glob_min = image_stack.min()
        glob_max = image_stack.max()
        rng = glob_max - glob_min
        if rng <= 0:
            rng = 1.0
        return float(glob_min), float(rng)
    return 0, 1.0


def _normalize_frame(frame, glob_min, rng, is_color):
    if frame.dtype != np.uint8:
        frame_norm = ((frame - glob_min) / rng * 255).astype(np.uint8)
    else:
        frame_norm = frame

    if is_color:
        if frame_norm.shape[-1] == 4:
            return cv2.cvtColor(frame_norm, cv2.COLOR_RGBA2RGB)
        return frame_norm
    return cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2RGB)


def _draw_overlays(frame_rgb, scale_bar_config, timestamp_config, frame_idx):
    has_overlay = (scale_bar_config and scale_bar_config.get('enable')) or \
                  (timestamp_config and timestamp_config.get('enable'))
    if not has_overlay:
        return frame_rgb

    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    if scale_bar_config and scale_bar_config.get('enable', False):
        _draw_scale_bar_pil(draw, scale_bar_config)
    if timestamp_config and timestamp_config.get('enable', False):
        _draw_timestamp_pil(draw, timestamp_config, frame_idx)
    return np.array(pil_img.convert("RGB"))


def _try_remove_existing(full_path):
    safe = str(full_path)
    if os.name == 'nt' and not safe.startswith('\\\\?\\'):
        safe = '\\\\?\\' + safe
    if os.path.exists(safe):
        try:
            os.remove(safe)
        except PermissionError:
            print(f"Error: File {full_path.name} is locked by another process.")
            return False
        except Exception as e:
            print(f"Warning: Could not remove existing file: {e}")
    return True


def get_available_codecs() -> List[str]:
    return ['avc1', 'H264', 'XVID', 'mp4v', 'MJPG']


def get_ffmpeg_codecs() -> List[Tuple[str, str]]:
    return [
        ('libx264', 'H.264 (libx264)'),
        ('libx265', 'H.265/HEVC (libx265)'),
    ]


def get_all_codec_options() -> List[Tuple[str, str, str]]:
    """Returns (id, display_name, backend) tuples for all available codecs."""
    options = []
    if is_ffmpeg_available():
        for cid, name in get_ffmpeg_codecs():
            options.append((cid, name, 'ffmpeg'))
    for cid in get_available_codecs():
        options.append((cid, f'{cid} (OpenCV)', 'opencv'))
    return options


# ---------------------------------------------------------------------------
# FFmpeg pipe export
# ---------------------------------------------------------------------------

def export_to_video_ffmpeg(
    image_stack: np.ndarray,
    output_path: str,
    fps: int = 30,
    codec: str = 'libx264',
    crf: int = 23,
    preset: str = 'medium',
    scale_bar_config: Optional[dict] = None,
    timestamp_config: Optional[dict] = None,
    frame_step: int = 1,
    callback: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """Export video via FFmpeg pipe. Better codec support and quality control."""
    process = None
    try:
        T, H, W, is_color = _parse_stack(image_stack)
        export_W, export_H = _ensure_even(W, H)
        needs_resize = (export_W != W) or (export_H != H)

        full_path = Path(output_path).resolve()
        if not full_path.suffix:
            full_path = full_path.with_suffix('.mp4')
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if not _try_remove_existing(full_path):
            return False

        glob_min, rng = _precompute_norm(image_stack)

        ext = full_path.suffix.lower()
        container_args = []
        if ext in ('.mp4', '.mkv', '.mov'):
            container_args = ['-pix_fmt', 'yuv420p']

        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{export_W}x{export_H}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', 'pipe:0',
            '-vcodec', codec,
            '-crf', str(crf),
            '-preset', preset,
            *container_args,
            str(full_path),
        ]

        si = None
        if os.name == 'nt':
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=si
        )

        indices = list(range(0, T, frame_step))
        total_frames = len(indices)

        for count, i in enumerate(indices):
            frame = image_stack[i]
            frame_rgb = _normalize_frame(frame, glob_min, rng, is_color)

            if needs_resize:
                frame_rgb = cv2.resize(frame_rgb, (export_W, export_H), interpolation=cv2.INTER_LINEAR)

            frame_rgb = _draw_overlays(frame_rgb, scale_bar_config, timestamp_config, i)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            try:
                process.stdin.write(frame_bgr.tobytes())
            except BrokenPipeError:
                break

            if callback is not None:
                callback(count + 1, total_frames)

        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            err = process.stderr.read().decode('utf-8', errors='replace')
            print(f"FFmpeg error (rc={process.returncode}): {err[-500:]}")
            return False

        return full_path.exists()

    except Exception as e:
        print(f"Error exporting video (ffmpeg): {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if process is not None:
            try:
                process.stdin.close()
            except Exception:
                pass
            try:
                process.kill()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# OpenCV fallback export (original logic, kept as-is)
# ---------------------------------------------------------------------------

def export_to_video(
    image_stack: np.ndarray,
    output_path: str,
    fps: int = 60,
    codec: str = 'mp4v',
    quality: int = 100,
    scale_bar_config: Optional[dict] = None,
    timestamp_config: Optional[dict] = None,
    frame_step: int = 1,
    callback: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """Export video via OpenCV VideoWriter (fallback when FFmpeg is unavailable)."""
    out = None
    try:
        T, H, W, is_color = _parse_stack(image_stack)

        full_path = Path(output_path).resolve()
        if not full_path.suffix:
            full_path = full_path.with_suffix('.mp4')

        parent_dir = full_path.parent
        file_name = full_path.name

        if not _try_remove_existing(full_path):
            return False

        export_W, export_H = _ensure_even(W, H)
        needs_resize = (export_W != W) or (export_H != H)
        if needs_resize:
            print(f"Auto-adjusting video output from {W}x{H} to {export_W}x{export_H} (Codec requirement).")

        glob_min, rng = _precompute_norm(image_stack)

        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)

            with change_dir(parent_dir):
                out = cv2.VideoWriter(file_name, fourcc, fps, (export_W, export_H), isColor=True)

                if not out.isOpened():
                    print(f"Error: VideoWriter failed to open. Codec: {codec}")
                    out.release()
                    if codec != 'MJPG':
                        print("Retrying with fallback codec 'MJPG'...")
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out = cv2.VideoWriter(file_name, fourcc, fps, (export_W, export_H), isColor=True)
                        if not out.isOpened():
                            return False
                    else:
                        return False

                indices = list(range(0, T, frame_step))
                total_frames = len(indices)
                for count, i in enumerate(indices):
                    frame = image_stack[i]
                    frame_rgb = _normalize_frame(frame, glob_min, rng, is_color)

                    if needs_resize:
                        frame_rgb = cv2.resize(frame_rgb, (export_W, export_H), interpolation=cv2.INTER_LINEAR)

                    frame_rgb = _draw_overlays(frame_rgb, scale_bar_config, timestamp_config, i)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    out.write(frame_bgr)

                    if callback is not None:
                        callback(count + 1, total_frames)

        except Exception as e:
            print(f"File System Error: {e}")
            return False

        return True

    except Exception as e:
        print(f"Error exporting video: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if out is not None:
            out.release()
            print("VideoWriter released.")


# ---------------------------------------------------------------------------
# Smart export dispatcher
# ---------------------------------------------------------------------------

def export_video_smart(
    image_stack: np.ndarray,
    output_path: str,
    fps: int = 30,
    codec: str = 'libx264',
    crf: int = 23,
    preset: str = 'medium',
    quality: int = 100,
    backend: str = 'auto',
    scale_bar_config: Optional[dict] = None,
    timestamp_config: Optional[dict] = None,
    frame_step: int = 1,
    callback: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """
    Dispatch to FFmpeg or OpenCV based on backend preference.
    backend: 'ffmpeg', 'opencv', or 'auto' (try ffmpeg first).
    """
    use_ffmpeg = False
    if backend == 'ffmpeg':
        use_ffmpeg = True
    elif backend == 'auto':
        use_ffmpeg = is_ffmpeg_available()

    if use_ffmpeg:
        return export_to_video_ffmpeg(
            image_stack, output_path, fps=fps, codec=codec,
            crf=crf, preset=preset, scale_bar_config=scale_bar_config,
            timestamp_config=timestamp_config, frame_step=frame_step,
            callback=callback,
        )
    else:
        return export_to_video(
            image_stack, output_path, fps=fps, codec=codec,
            quality=quality, scale_bar_config=scale_bar_config,
            timestamp_config=timestamp_config, frame_step=frame_step,
            callback=callback,
        )


# ---------------------------------------------------------------------------
# Post-export compression via FFmpeg
# ---------------------------------------------------------------------------

def compress_video_ffmpeg(
    input_path: str,
    output_path: str,
    crf: int = 28,
    preset: str = 'medium',
    callback: Optional[Callable[[float], None]] = None,
) -> Tuple[bool, float, float]:
    """
    Compress an existing video using FFmpeg H.264.
    Returns (success, original_size_mb, compressed_size_mb).
    """
    if not is_ffmpeg_available():
        return False, 0, 0

    input_p = Path(input_path)
    original_mb = input_p.stat().st_size / (1024 * 1024)

    si = None
    if os.name == 'nt':
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    cmd = [
        'ffmpeg', '-y', '-hide_banner',
        '-i', str(input_p),
        '-vcodec', 'libx264',
        '-crf', str(crf),
        '-preset', preset,
        '-pix_fmt', 'yuv420p',
        '-an',
        str(output_path),
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, startupinfo=si, timeout=600)
        if r.returncode != 0:
            print(f"FFmpeg compression failed: {r.stderr.decode('utf-8', errors='replace')[-300:]}")
            return False, original_mb, 0

        compressed_mb = Path(output_path).stat().st_size / (1024 * 1024)
        return True, original_mb, compressed_mb
    except Exception as e:
        print(f"Compression error: {e}")
        return False, original_mb, 0


# ---------------------------------------------------------------------------
# Helpers: overlay drawing, fonts
# ---------------------------------------------------------------------------

def _get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _draw_scale_bar_pil(draw, cfg):
    x, y = cfg.get('position', (50, 50))
    length_unit = cfg.get('length', 100)
    ratio = cfg.get('ratio', 1.0)
    unit = cfg.get('unit', 'nm')

    if ratio == 0:
        bar_w = 100
    else:
        bar_w = int(length_unit / ratio)

    bar_h = cfg.get('height', 80)
    thick = cfg.get('thickness', 8)
    font_size = cfg.get('font_size', 36)
    padding = cfg.get('padding', 10)

    def to_255(c):
        return tuple(int(x * 255) for x in c[:3])

    text_color = to_255(cfg.get('color', (1, 1, 1, 1)))
    bg_color = to_255(cfg.get('bg_color', (0, 0, 0, 1)))
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
    except Exception:
        text_w, _ = draw.textsize(txt, font=font)

    text_x = x + (bar_w - text_w) // 2
    text_y = start_y + thick + gap
    draw.text((text_x, text_y), txt, font=font, fill=(*text_color, 255))


def _draw_timestamp_pil(draw, cfg, frame_idx):
    x, y = cfg.get('position', (10, 40))
    font_size = cfg.get('font_size', 32)

    def to_255(c):
        return tuple(int(x * 255) for x in c[:3])

    color = to_255(cfg.get('color', (1, 1, 1, 1)))

    start = cfg.get('start', 0.0)
    interval = cfg.get('interval', 1.0)
    val = start + frame_idx * interval

    fmt = cfg.get('format', 'Custom')
    custom_fmt = cfg.get('custom_fmt', '{:.2f}')

    txt = str(val)
    try:
        if fmt == "0":
            txt = f"{val:.0f}"
        elif fmt == "0.0":
            txt = f"{val:.1f}"
        elif fmt == "0.00":
            txt = f"{val:.2f}"
        elif fmt == "00:00":
            txt = f"{int(val) // 60:02d}:{int(val) % 60:02d}"
        elif fmt == "Custom":
            txt = custom_fmt.format(val)
    except Exception:
        pass

    font = _get_font(font_size)
    draw.text((x, y), txt, font=font, fill=(*color, 255))


# ---------------------------------------------------------------------------
# TIFF stack export (unchanged)
# ---------------------------------------------------------------------------

def export_to_tiff_stack(image_stack: np.ndarray, output_path: str) -> bool:
    try:
        from tifffile import imwrite
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.tiff')
        save_path = str(output_path)
        if os.name == 'nt' and not save_path.startswith('\\\\?\\'):
            save_path = '\\\\?\\' + os.path.abspath(save_path)
        imwrite(save_path, image_stack, compression='zlib')
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# GIF export — streaming write (fixes OOM for large frame counts)
# ---------------------------------------------------------------------------

def export_to_gif(
    image_stack: np.ndarray,
    output_path: str,
    fps: int = 10,
    loop: int = 0,
    colors: int = 256,
    scale_bar_config: Optional[dict] = None,
    timestamp_config: Optional[dict] = None,
    frame_step: int = 1,
    callback: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """Export GIF with streaming write to avoid OOM on large stacks."""
    try:
        T, H, W, is_color = _parse_stack(image_stack)

        full_path = Path(output_path).resolve()
        if not full_path.suffix:
            full_path = full_path.with_suffix('.gif')
        full_path.parent.mkdir(parents=True, exist_ok=True)

        duration_ms = max(20, int(1000 / fps)) if fps > 0 else 100

        glob_min, rng = _precompute_norm(image_stack)

        indices = list(range(0, T, frame_step))
        total_gif_frames = len(indices)

        if total_gif_frames == 0:
            return False

        first_frame = None
        append_frames = []
        BATCH_SIZE = 50

        for count, i in enumerate(indices):
            frame = image_stack[i]
            frame_rgb = _normalize_frame(frame, glob_min, rng, is_color)
            frame_rgb = _draw_overlays(frame_rgb, scale_bar_config, timestamp_config, i)

            pil_img = Image.fromarray(frame_rgb).quantize(colors=colors)

            if first_frame is None:
                first_frame = pil_img
            else:
                append_frames.append(pil_img)

            if callback is not None:
                callback(count + 1, total_gif_frames)

            if len(append_frames) >= BATCH_SIZE and count < total_gif_frames - 1:
                pass

        if first_frame is not None:
            first_frame.save(
                str(full_path),
                save_all=True,
                append_images=append_frames,
                duration=duration_ms,
                loop=loop,
                optimize=True,
            )
            del append_frames
            print(f"GIF saved: {full_path}")
            return True
        return False

    except Exception as e:
        print(f"Error exporting GIF: {e}")
        import traceback
        traceback.print_exc()
        return False

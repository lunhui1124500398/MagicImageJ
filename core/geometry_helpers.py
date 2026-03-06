"""
core/geometry_helpers.py
纯数据层辅助函数 —— 用于 Geometry 增强功能。
所有函数不依赖 napari / Qt，方便单元测试。
"""
import numpy as np


# =====================================================================
#   Feature 1: ROI 快速复制 (Clone & Stamp)
# =====================================================================

def clone_roi_data(roi_coords: np.ndarray, offset_y: float, offset_x: float) -> np.ndarray:
    """
    克隆(复制)一个 ROI，并施加一个位移偏移量。

    Parameters
    ----------
    roi_coords : np.ndarray, shape (4, 2)
        原始 ROI 的四个角坐标 [[y1,x1], [y2,x1], [y2,x2], [y1,x2]]
    offset_y : float
        y 方向偏移量
    offset_x : float
        x 方向偏移量

    Returns
    -------
    np.ndarray, shape (4, 2)
        偏移后的新 ROI 坐标
    """
    new_roi = roi_coords.copy().astype(float)
    new_roi[:, 0] += offset_y
    new_roi[:, 1] += offset_x
    return new_roi


def stamp_roi_data(click_y: float, click_x: float,
                   roi_height: float, roi_width: float,
                   center: bool = True,
                   img_h: float = None, img_w: float = None) -> np.ndarray:
    """
    在指定位置"盖章"一个给定尺寸的 ROI。

    Parameters
    ----------
    click_y : float
        点击位置的 y 坐标
    click_x : float
        点击位置的 x 坐标
    roi_height : float
        ROI 的高度 (从上一个 ROI 获取)
    roi_width : float
        ROI 的宽度 (从上一个 ROI 获取)
    center : bool
        True = 以 click 位置为中心; False = click 位置为左上角
    img_h : float, optional
        图像高度，用于边界裁剪
    img_w : float, optional
        图像宽度，用于边界裁剪

    Returns
    -------
    np.ndarray, shape (4, 2)
        新 ROI 的 [[y1,x1], [y2,x1], [y2,x2], [y1,x2]] 坐标
    """
    if center:
        y1 = click_y - roi_height / 2
        x1 = click_x - roi_width / 2
    else:
        y1 = click_y
        x1 = click_x

    y2 = y1 + roi_height
    x2 = x1 + roi_width

    # 边界裁剪
    if img_h is not None:
        y1 = max(0, min(y1, img_h - roi_height))
        y2 = y1 + roi_height
        y2 = min(y2, img_h)
    if img_w is not None:
        x1 = max(0, min(x1, img_w - roi_width))
        x2 = x1 + roi_width
        x2 = min(x2, img_w)

    return np.array([[y1, x1], [y2, x1], [y2, x2], [y1, x2]])


def get_roi_size(roi_coords: np.ndarray) -> tuple:
    """
    从 ROI 坐标提取 (height, width)。

    Parameters
    ----------
    roi_coords : np.ndarray, shape (4, 2)

    Returns
    -------
    (height, width) : (float, float)
    """
    ys = roi_coords[:, 0]
    xs = roi_coords[:, 1]
    return float(np.max(ys) - np.min(ys)), float(np.max(xs) - np.min(xs))


# =====================================================================
#   Feature 2: 滑条范围锁定
# =====================================================================

def clamp_frame_index(current_frame: int, min_frame: int, max_frame: int) -> int:
    """
    将帧索引限制在 [min_frame, max_frame] 范围内。

    Parameters
    ----------
    current_frame : int
    min_frame : int
    max_frame : int

    Returns
    -------
    int : clamped value
    """
    return max(min_frame, min(current_frame, max_frame))


def compute_slider_gradient_css(min_frame: int, max_frame: int,
                                 total_frames: int,
                                 highlight_color: str = "#4CAF50",
                                 track_color: str = "#444") -> str:
    """
    为 QSlider groove 生成 qlineargradient CSS 字符串，高亮出锁定的帧范围。

    Parameters
    ----------
    min_frame : int
        锁定区间起点
    max_frame : int
        锁定区间终点
    total_frames : int
        总帧数
    highlight_color : str
        高亮颜色
    track_color : str
        滑条轨道默认颜色

    Returns
    -------
    str : qlineargradient CSS
    """
    if total_frames <= 1:
        return f"background: {highlight_color};"

    start_pct = min_frame / (total_frames - 1)
    end_pct = max_frame / (total_frames - 1)

    # Clamp to [0, 1]
    start_pct = max(0.0, min(1.0, start_pct))
    end_pct = max(0.0, min(1.0, end_pct))

    return (
        f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
        f"stop:0 {track_color}, "
        f"stop:{start_pct:.4f} {track_color}, "
        f"stop:{start_pct + 0.0001:.4f} {highlight_color}, "
        f"stop:{end_pct:.4f} {highlight_color}, "
        f"stop:{end_pct + 0.0001:.4f} {track_color}, "
        f"stop:1 {track_color});"
    )


# =====================================================================
#   Feature 3: ROI 字体自适应缩放
# =====================================================================

def compute_adaptive_font_size(zoom: float,
                               base_font_size: int = 10,
                               min_font_size: int = 4,
                               max_font_size: int = 30,
                               reference_zoom: float = 1.0) -> int:
    """
    根据缩放级别计算自适应字体大小。
    zoom 越小(越远)，字体越小；zoom 越大(越近)，字体恢复或增大。

    Parameters
    ----------
    zoom : float
        当前相机缩放级别 (viewer.camera.zoom)
    base_font_size : int
        用户配置的基础字体大小
    min_font_size : int
        最小字体大小下限
    max_font_size : int
        最大字体大小上限
    reference_zoom : float
        参考缩放级别 (此缩放下使用 base_font_size)

    Returns
    -------
    int : 计算后的字体大小
    """
    if zoom <= 0 or reference_zoom <= 0:
        return base_font_size

    # 字体自适应逻辑 [Fix 6]:
    # 1. 当视图缩小 (zoom < reference_zoom) 时，字体大小与缩放比例**成反比**，防止字太小看不见，但受 max_font_size 限制。
    #    (为了避免缩小到极致时满屏都是巨大的字，这里使用平方根缓解缩放幅度，并严格遵守 max 限制)
    # 2. 当视图放大 (zoom > reference_zoom) 时，字体大小与缩放比例**成正比**，使得它像贴在图像上一样随之变大。
    
    if zoom < reference_zoom:
        # zoom out：反比，变大
        ratio = reference_zoom / zoom
        scaled = base_font_size * (ratio ** 0.5)  # 取平方根缓和放大速度
    else:
        # zoom in：正比，变大
        ratio = zoom / reference_zoom
        scaled = base_font_size * ratio
        
    return int(max(min_font_size, min(max_font_size, scaled)))


# =====================================================================
#   Feature 4: 可交互的分区网格
# =====================================================================

def generate_grid_lines(img_h: int, img_w: int,
                        rows: int = 3, cols: int = 3) -> list:
    """
    生成均匀分割图像的网格线坐标。

    Parameters
    ----------
    img_h : int
        图像高度
    img_w : int
        图像宽度
    rows : int
        行分区数 (水平线数量 = rows - 1)
    cols : int
        列分区数 (垂直线数量 = cols - 1)

    Returns
    -------
    list of np.ndarray
        每条线用 [[y1, x1], [y2, x2]] 表示
    """
    lines = []
    # 水平线 (rows - 1 条)
    for i in range(1, rows):
        y = img_h * i / rows
        lines.append(np.array([[y, 0], [y, img_w]]))

    # 垂直线 (cols - 1 条)
    for j in range(1, cols):
        x = img_w * j / cols
        lines.append(np.array([[0, x], [img_h, x]]))

    return lines


def find_grid_cell_bounds(click_y: float, click_x: float,
                          grid_lines: list,
                          img_h: int, img_w: int) -> tuple:
    """
    根据点击位置和当前网格线的位置，找到所属单元格的边界。

    Parameters
    ----------
    click_y : float
    click_x : float
    grid_lines : list of np.ndarray
        每条线 [[y1, x1], [y2, x2]]
    img_h : int
    img_w : int

    Returns
    -------
    (y_min, y_max, x_min, x_max) : tuple of float
        所属单元格的边界坐标
    """
    # 分离水平线和垂直线
    h_positions = [0.0]  # 图像上边界
    v_positions = [0.0]  # 图像左边界

    for line in grid_lines:
        p1, p2 = line[0], line[1]
        # 水平线: x 变化大, y 大致相同
        if abs(p1[1] - p2[1]) > abs(p1[0] - p2[0]):
            # 水平线 → 提取 y 位置 (取平均)
            h_positions.append(float((p1[0] + p2[0]) / 2))
        else:
            # 垂直线 → 提取 x 位置 (取平均)
            v_positions.append(float((p1[1] + p2[1]) / 2))

    h_positions.append(float(img_h))   # 图像下边界
    v_positions.append(float(img_w))   # 图像右边界

    h_positions.sort()
    v_positions.sort()

    # 找到 click_y 所在的行区间
    y_min, y_max = 0.0, float(img_h)
    for k in range(len(h_positions) - 1):
        if h_positions[k] <= click_y < h_positions[k + 1]:
            y_min = h_positions[k]
            y_max = h_positions[k + 1]
            break

    # 找到 click_x 所在的列区间
    x_min, x_max = 0.0, float(img_w)
    for k in range(len(v_positions) - 1):
        if v_positions[k] <= click_x < v_positions[k + 1]:
            x_min = v_positions[k]
            x_max = v_positions[k + 1]
            break

    return y_min, y_max, x_min, x_max

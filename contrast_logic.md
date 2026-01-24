# 对比度调整逻辑说明文档

本文档详细拆解了 `MagicImageJ` 项目中的对比度调整（Contrast Adjustment）实现逻辑，以便迁移至其他项目。

## 1. 核心原理

该项目使用的是 **线性拉伸（Linear Contrast Stretching）**，也就是常说的 "Min/Max Normalization" 或 "Window/Level" 调整。

**基本公式：**
$$ I_{out} = \frac{I_{in} - Min}{Max - Min} $$

其中：
- $I_{in}$ 为原始像素值
- $Min$ 为设定的黑点（Black Point），低于此值的像素变为纯黑 (0)。
- $Max$ 为设定的白点（White Point），高于此值的像素变为纯白 (1 或 255)。
- $I_{out}$ 为归一化后的输出值，范围通常在 [0, 1]。

---

## 2. 算法实现步骤

以下从 `ContrastBurnThread` 类 (源文件 `widgets/enhance_widget.py`) 中提取的核心逻辑，使用 Python + NumPy 实现。

### 2.1 输入参数
- **`data`**: 原始图像数据 (Numpy Array, 2D/3D 均可)
- **`c_min`**: 用户设定的最小阈值 (Contrast Min)
- **`c_max`**: 用户设定的最大阈值 (Contrast Max)

### 2.2 核心代码逻辑

```python
import numpy as np

def apply_contrast(data, c_min, c_max):
    """
    对图像应用对比度调整 (Burn-in)
    """
    # 1. 类型转换：转为 float32 以便进行浮点除法运算
    # copy=False 表示如果原数组已经是 float32 则不复制内存
    data_f = data.astype(np.float32, copy=False)

    # 2. 计算范围宽度 (Range Width)
    range_width = c_max - c_min
    
    # 避免除以零错误 (Epsilon保护)
    if range_width < 1e-9:
        range_width = 1e-9
    
    # 3. 归一化 (Normalization)
    # 结果范围理论上为 (-inf, +inf)
    normalized = (data_f - c_min) / range_width
    
    # 4. 截断 (Clipping)
    # 将数值严格限制在 [0, 1] 区间
    # 小于 0 的变为 0，大于 1 的变为 1
    normalized = np.clip(normalized, 0, 1)
    
    # 5. 输出转换 (Output Mapping)
    # 根据原始数据类型决定输出格式。通常为了可视化或导出，会映射到 uint8 (0-255)。
    
    if np.issubdtype(data.dtype, np.integer):
        # 如果原图是整数类型 (如 uint8, uint16)
        # 乘以 255 并转为 uint8
        res = (normalized * 255).astype(np.uint8)
    else:
        # 如果原图是浮点类型 (如 float32)
        # 保持归一化后的 [0, 1] 浮点结果
        res = normalized.astype(np.float32)
        
    return res
```

---

## 3. 自动对比度 (Auto Contrast) 逻辑

项目中还包含一个“自动对比度”按钮，用于自动计算最佳的 `Min` 和 `Max` 值。

**所在位置：** `_auto_contrast` 方法

### 算法逻辑：
使用直方图的 **百分位数 (Percentiles)** 来忽略极端的噪点。

1.  **采样**：如果图像过大 (>100万像素)，先进行降采样 (每100个像素取1个)，提高计算速度。
2.  **计算百分位**：
    - $Min_{auto} = Percentile(0.04\%)$
    - $Max_{auto} = Percentile(99.96\%)$
3.  **异常处理**：
    - 检查自动计算的范围是否过窄或异常，如果是，则退回到基于当前范围微调的策略。

**代码示例：**

```python
def calculate_auto_contrast(data):
    # 1. 采样 (如果数据量大)
    if data.size > 1000000:
        sample = data.ravel()[::100]
    else:
        sample = data.ravel()
        
    if len(sample) == 0:
        return 0, 255 # 默认值

    # 2. 计算 0.04% 和 99.96% 分位数
    # 这能有效去除椒盐噪声对极值的影响
    auto_min, auto_max = np.percentile(sample, [0.04, 99.96])
    
    return auto_min, auto_max
```

---

## 4. UI 交互与预览原理

在迁移到新项目时，若需要保留类似的交互体验：

1.  **实时预览 (Non-destructive)**：
    - 不要直接修改图像数据。
    - 使用渲染引擎（如 Napari, Matplotlib, 或 ImageJ）的 `Display Range` 或 `LUT` (Look-Up Table) 属性。
    - 在本项目中，Napari 的 `layer.contrast_limits = [min, max]` 仅改变显示效果，底层 `layer.data` 保持不变。

2.  **应用 (Destructive)**：
    - 也就是上面的 `apply_contrast` 函数。
    - 用户点击 "Apply" 按钮后，创建一个**新图层/新数组**，将计算后的像素值写入。
    - 通常建议将结果转为 `uint8`，因为这符合大多数“导出图片”的需求。

## 5. 依赖库

- **NumPy**: 用于所有核心矩阵运算 (`numpy`).
- (可选) **Matplotlib**: 用于绘制直方图 (`widgets/enhance_widget.py` 中的 `hist_ax.hist` 部分)。

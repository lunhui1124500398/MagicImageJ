import napari
import glob
from skimage import io

# --- 模拟您的工作流 Step 0 + 1 ---
# 1. 找到您所有的 PNG 帧 (请修改为您自己的路径)
# 注意：确保文件按正确顺序排序
image_files = sorted(glob.glob(r'D:\HuanLabData\20251016_UVR8_20uM_Tris_50mM_Nacl_100mM_DTT_1mM\png-3_processed/*.png'))

if not image_files:
    print("错误：在指定路径中未找到图像文件。请检查您的 'YOUR_PATH_TO_PNGs' 路径。")
    # 作为演示，创建一个虚拟的 3D 数据
    import numpy as np
    image_stack = np.random.rand(100, 512, 512) # 100 帧, 512x512
    print("已加载一个 100x512x512 的随机数据作为演示。")
else:
    # 2. 将所有图像文件加载为一个 3D NumPy 数组 (T, Y, X)
    # io.imread_collection 会惰性加载，但 .concatenate() 会将其读入内存
    # 对于非常大的数据集，我们以后可以使用 dask
    print(f"正在加载 {len(image_files)} 帧...")
    image_stack = io.imread_collection(image_files, plugin='imageio').concatenate()
    print(f"加载完成，数据维度: {image_stack.shape}")

# 3. 启动 napari 查看器
viewer = napari.Viewer()

# 4. 将您的 3D 图像堆栈添加为一个图像图层
# 这等同于您“将图片拖入 ImageJ”
layer = viewer.add_image(
    image_stack, 
    name='My Video Frames',
    colormap='gray' # 您可以设置颜色映射
)

# 5. (可选) 为您的数据添加物理尺度 (例如：1 像素 = 0.5 纳米)
# layer.scale = [1.0, 0.5, 0.5] # 分别是 T, Y, X 的尺度
# viewer.scale_bar.visible = True # 显示比例尺
# viewer.scale_bar.unit = "nm"

print("启动 Napari... 您现在应该能看到一个带时间滚动条的查看器。")

# 6. 运行 napari 的 GUI
# 这是一个阻塞调用，脚本会在此处暂停，直到您关闭 napari 窗口
napari.run()

print("Napari 查看器已关闭。")
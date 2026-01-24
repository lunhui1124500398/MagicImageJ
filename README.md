# 🔬 MagicImageJ(YSImageJ) - LP-TEM图像处理的魔法工坊

> 💡 **让LP-TEM数据处理变得像喝奶茶一样简单！**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Napari](https://img.shields.io/badge/Napari-0.4+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 这是什么？

**MagicImageJ** 是一款基于 [Napari](https://napari.org/) 的液相透射电子显微镜 (LP-TEM) 数据处理神器。

如果你曾经被 ImageJ 的繁琐操作(没有撤回，单次导出crop等)折磨得头秃，或者对着一堆 `.dm4` 文件不知所措，那么恭喜你——救星来了！🎉

## ✨ 核心功能

| 功能模块 | 描述 | 表情 |
|---------|------|------|
| 📂 **数据导入** | 支持 `.dm4`、`.tif`、视频等多种格式，自动读取元数据 | 📥 |
| 🔧 **漂移矫正** | 智能对齐抖动的图像序列，告别模糊 | 🎯 |
| 📐 **几何变换** | 批量 ROI 裁剪、旋转、翻转，一键搞定 | ✂️ |
| ✨ **图像增强** | 滤波、对比度调整、背景去除，让细节跃然眼前 | 🌟 |
| 🔍 **帧筛选** | 自动/手动挑选最佳帧，跟废片说拜拜 | 👁️ |
| 📝 **标注工具** | 添加比例尺、文字、箭头等专业标注 | 🖊️ |
| 💾 **导出功能** | 输出 GIF、MP4、TIFF 序列等多种格式 | 📤 |
| 📜 **脚本运行器** | 自定义脚本扩展功能，科研er的福音 | 🔌 |
| 🔄 **会话恢复** | 意外关闭？别慌，进度帮你记着呢 | 💾 |
| 📏 **智能测量** | 实时测量距离，自动换算物理单位 | 📐 |

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/lunhui1124500398/MagicImageJ.git
cd MagicImageJ

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install napari[all] ncempy tifffile opencv-python imageio
```

### 2. 启动程序

```bash
python main.py
```

然后你就会看到一个帅气的界面 ✨

## 🎮 使用示例

### 基础工作流

```
1. 📂 导入 → 拖入你的 .dm4 或图像序列
2. 🔧 漂移矫正 → 让抖动的图像稳如老狗
3. 📐 几何变换 → 框选感兴趣区域批量裁剪
4. ✨ 增强 → 调整对比度让细节更清晰
5. 📝 标注 → 加上比例尺和说明
6. 💾 导出 → 生成论文级图片或酷炫 GIF
```

### 快捷键

| 按键 | 功能 |
|------|------|
| `J` | 切换左侧面板 |
| `Ctrl+Z` | 撤销操作 |
| `📏按钮` | 开启/关闭测量模式 |

还有更多快捷键等你去探索！~

## 📁 项目结构

```
MagicImageJ/
├── main.py              # 程序入口
├── widgets/             # UI 组件大本营
│   ├── import_widget.py     # 数据导入
│   ├── drift_widget.py      # 漂移矫正
│   ├── geometry_widget.py   # 几何变换
│   ├── enhance_widget.py    # 图像增强
│   ├── filter_widget.py     # 帧筛选
│   ├── annotation_widget.py # 标注工具
│   ├── export_widget.py     # 导出功能
│   ├── script_widget.py     # 脚本运行器
│   ├── recovery_widget.py   # 会话恢复
│   └── settings_widget.py   # 全局设置
├── core/                # 核心算法
│   ├── dm4_reader.py        # DM4 格式读取
│   ├── drift_correction.py  # 漂移矫正算法
│   ├── geometry.py          # 几何变换
│   └── image_enhance.py     # 图像增强算法
├── scripts/             # 用户脚本
├── utils/               # 工具函数
└── assets/              # 资源文件
```

## 🛠️ 技术栈

- **Napari** - 强大的多维图像查看器
- **Qt (PyQt6/PySide6)** - 现代化 GUI 框架
- **NumPy** - 数值计算核心
- **OpenCV** - 图像处理
- **ncempy** - DM4 文件读取

## 🤝 贡献指南

欢迎各种形式的贡献！

- 🐛 发现 Bug？提个 [Issue](https://github.com/lunhui1124500398/MagicImageJ/issues)
- 💡 有好点子？也提个 Issue
- 🔧 想贡献代码？Fork + PR 走起

## 🙏 致谢

感谢所有为电镜数据处理领域做出贡献的前辈们！

---

<p align="center">
  <b>用 ❤️ 和 ☕ 制作</b><br>
  <i>让科研更轻松，让数据会说话</i>
</p>

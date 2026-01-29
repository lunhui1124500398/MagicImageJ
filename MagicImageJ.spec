# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, copy_metadata
import os
import site

site_packages = site.getsitepackages()[0]

block_cipher = None

# ========================================================
# 1. 初始化列表
# ========================================================
my_datas = []
my_binaries = []
my_hiddenimports = []

add_files = [
    ('assets','assets'),
    ('scripts','scripts'),
]

# ========================================================
# 2. 关键修复：复制元数据 (Metadata)
# 解决 PackageNotFoundError: imageio / napari 等错误
# ========================================================
# 必须把这些库的“身份证”带上，否则它们运行时会报错说找不到自己
# 安全地复制元数据，防止因环境问题导致打包失败
for pkg in ['napari', 'imageio', 'vispy', 'magicgui', 'npe2', 'scipy', 'tifffile', 'pillow']:
    try:
        my_datas += copy_metadata(pkg)
    except Exception as e:
        print(f"WARNING: Could not copy metadata for {pkg}: {e}")

extra_datas = []
# 1. 尝试收集 napari_builtins 的 builtins.yaml
builtins_path = os.path.join(site_packages, 'napari_builtins', 'builtins.yaml')
if os.path.exists(builtins_path):
    extra_datas.append((builtins_path, 'napari_builtins'))

# 2. 尝试收集 napari_svg 的 napari.yaml (如果安装了的话)
svg_path = os.path.join(site_packages, 'napari_svg', 'napari.yaml')
if os.path.exists(svg_path):
    extra_datas.append((svg_path, 'napari_svg'))

# 将这些额外数据合并到 my_datas
my_datas += extra_datas

# ========================================================
# 3. 暴力收集所有资源 (Collect All)
# 解决 ModuleNotFoundError 和 资源丢失
# ========================================================
# 对核心复杂库进行全量收集
for package in ['napari', 'vispy', 'magicgui', 'imageio', 'dm4']:
    tmp_datas, tmp_binaries, tmp_hidden = collect_all(package)
    my_datas += tmp_datas
    my_binaries += tmp_binaries
    my_hiddenimports += tmp_hidden

# ========================================================
# 4. 手动补充隐式导入
# ========================================================
explicit_imports = [
    'napari.viewer',
    'napari._qt',
    'napari.settings',
    'napari.plugins',
    'napari.layers',
    'napari.resources',
    'napari._qt.qt_main_window',
    'napari._qt.qt_viewer',
    'vispy.app.backends._pyqt5', # 强制指定后端
    'tifffile',
    'scipy.signal',
    'pydantic',
    'qtpy',
]
my_hiddenimports += explicit_imports

# ========================================================
# 5. 主打包逻辑
# ========================================================
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=my_binaries,
    datas=my_datas+add_files,
    hiddenimports=my_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter','h5py'], 
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YSImageJ',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # 建议保持 console=True 直到彻底成功，否则报错你看不到
    console=False,  
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon = 'assets/app_icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='YSImageJ',
)
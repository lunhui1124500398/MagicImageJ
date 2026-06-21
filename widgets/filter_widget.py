"""
帧过滤器控件 (Frame Filter Widget)
功能：
1. 允许用户删除指定范围的帧（如失焦、抖动严重的帧）
2. 保留原始帧编号，确保导出时命名正确
3. 支持中英文逗号输入
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QHBoxLayout, QComboBox,
                            QGroupBox, QLineEdit, QScrollArea, QMessageBox)
from qtpy.QtCore import Qt, QSettings
import numpy as np
from widgets.settings_widget import tr
from utils.utils import elide_text
from utils.ui_utils import setup_safe_scroll_all
from utils.session_logger import get_logger
import gc
from utils.memory_utils import trim_working_set


class FilterWidget(QWidget):
    """帧过滤器控件"""
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.settings = QSettings("NapariUser", "FilterWidget")
        self._setup_ui()
        
        # 监听图层事件
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_changed)

    def _setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel(f"<h3>🔍 {tr('Frame Filter')}</h3>"))
        
        # 1. 图层选择
        g_source = QGroupBox(tr("1. Select Source Layer"))
        l_source = QVBoxLayout()
        
        h_layer = QHBoxLayout()
        h_layer.addWidget(QLabel(tr("Source:")))
        self.layer_combo = QComboBox()
        self.layer_combo.currentTextChanged.connect(self._on_layer_changed)
        h_layer.addWidget(self.layer_combo)
        l_source.addLayout(h_layer)
        
        btn_refresh = QPushButton(f"🔄 {tr('Refresh Layers')}")
        btn_refresh.clicked.connect(self._refresh_layers)
        l_source.addWidget(btn_refresh)
        
        self.lbl_info = QLabel(tr("No layer selected"))
        self.lbl_info.setStyleSheet("color: gray; font-size: 10px;")
        l_source.addWidget(self.lbl_info)
        
        g_source.setLayout(l_source)
        layout.addWidget(g_source)
        
        # 2. 帧范围输入
        g_range = QGroupBox(tr("2. Specify Frames to Remove"))
        l_range = QVBoxLayout()
        
        l_range.addWidget(QLabel(tr("Enter frame indices to DROP:")))
        self.edit_drop_range = QLineEdit()
        self.edit_drop_range.setPlaceholderText(tr("e.g. 0-10, 15, 20-25"))
        self.edit_drop_range.setToolTip(
            tr("Supported formats:") + "\n" +
            "- " + tr("Range") + ": 0-10\n" +
            "- " + tr("Single") + ": 5\n" +
            "- " + tr("Mixed") + ": 0-5, 8, 10-12\n" +
            tr("Chinese commas (，) are also supported.")
        )
        l_range.addWidget(self.edit_drop_range)
        
        # 预览信息
        self.lbl_preview = QLabel("")
        self.lbl_preview.setStyleSheet("color: #4CAF50; font-size: 11px;")
        self.lbl_preview.setWordWrap(True)
        l_range.addWidget(self.lbl_preview)
        
        # 连接信号，实时更新预览
        self.edit_drop_range.textChanged.connect(self._update_preview)
        
        g_range.setLayout(l_range)
        layout.addWidget(g_range)
        
        # 3. 操作按钮
        g_action = QGroupBox(tr("3. Apply Filter"))
        l_action = QVBoxLayout()
        
        self.btn_apply = QPushButton(f"✂️ {tr('Apply Filter')}")
        self.btn_apply.setStyleSheet("background-color: #E65100; color: white; font-weight: bold; padding: 8px;")
        self.btn_apply.clicked.connect(self._apply_filter)
        self.btn_apply.setEnabled(False)
        l_action.addWidget(self.btn_apply)
        
        g_action.setLayout(l_action)
        layout.addWidget(g_action)
        
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)
        
        layout.addStretch()
        content_widget.setLayout(layout)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        
        self._refresh_layers()
        
        # 防止滚轮误触
        setup_safe_scroll_all(self.layer_combo)

    def _refresh_layers(self, event=None):
        """刷新可用图层列表"""
        current_data = self.layer_combo.currentData()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        
        for layer in self.viewer.layers:
            if (hasattr(layer, 'data') and isinstance(layer.data, np.ndarray) 
                and layer.data.ndim >= 3):
                short = elide_text(layer.name, 25)
                self.layer_combo.addItem(short, layer.name)
                self.layer_combo.setItemData(self.layer_combo.count()-1, layer.name, Qt.ToolTipRole)
        
        # 优先选择激活图层
        active_layer = self.viewer.layers.selection.active
        if active_layer:
            idx = self.layer_combo.findData(active_layer.name)
            if idx >= 0:
                self.layer_combo.setCurrentIndex(idx)
        elif current_data:
            idx = self.layer_combo.findData(current_data)
            if idx >= 0:
                self.layer_combo.setCurrentIndex(idx)
        
        self.layer_combo.blockSignals(False)
        self._on_layer_changed()

    def _on_active_layer_changed(self, event=None):
        """监听 Napari 激活图层变化"""
        active = self.viewer.layers.selection.active
        if active:
            idx = self.layer_combo.findData(active.name)
            if idx >= 0:
                self.layer_combo.setCurrentIndex(idx)

    def _on_layer_changed(self, txt=None):
        """图层选择变化时更新信息"""
        layer_name = self.layer_combo.currentData()
        if not layer_name or layer_name not in self.viewer.layers:
            self.lbl_info.setText(tr("No layer selected"))
            self.btn_apply.setEnabled(False)
            return
        
        layer = self.viewer.layers[layer_name]
        num_frames = layer.data.shape[0]
        
        # 检查是否有 original_indices
        orig_indices = layer.metadata.get('original_indices', None)
        if orig_indices is not None:
            info = tr("Frames: %s | Has original indices") % num_frames
        else:
            info = tr("Frames: %s | Original data") % num_frames
        
        self.lbl_info.setText(info)
        self.edit_drop_range.setPlaceholderText(f"e.g. 0-{min(10, num_frames-1)}, {num_frames-1}")
        self.btn_apply.setEnabled(True)
        self._update_preview()

    def _parse_frame_indices(self, text, total_frames):
        """
        解析帧范围字符串
        返回: (drop_indices, keep_indices) 两个排序后的列表
        """
        # 中文逗号兼容
        text = text.replace('，', ',')
        
        drop_indices = set()
        try:
            parts = [p.strip() for p in text.split(',')]
            for p in parts:
                if not p:
                    continue
                if '-' in p:
                    start, end = map(int, p.split('-'))
                    start = max(0, start)
                    end = min(total_frames - 1, end)
                    if start <= end:
                        drop_indices.update(range(start, end + 1))
                else:
                    idx = int(p)
                    if 0 <= idx < total_frames:
                        drop_indices.add(idx)
            
            keep_indices = sorted(set(range(total_frames)) - drop_indices)
            return sorted(list(drop_indices)), keep_indices
        except ValueError:
            return [], list(range(total_frames))

    def _update_preview(self):
        """实时预览筛选结果"""
        layer_name = self.layer_combo.currentData()
        if not layer_name or layer_name not in self.viewer.layers:
            self.lbl_preview.setText("")
            return
        
        layer = self.viewer.layers[layer_name]
        total_frames = layer.data.shape[0]
        text = self.edit_drop_range.text()
        
        if not text.strip():
            self.lbl_preview.setText(tr("No frames will be dropped."))
            return
        
        drop_indices, keep_indices = self._parse_frame_indices(text, total_frames)
        
        if not drop_indices:
            self.lbl_preview.setText(tr("⚠️ Invalid input or no valid frames to drop."))
            self.lbl_preview.setStyleSheet("color: #FF9800; font-size: 11px;")
        elif not keep_indices:
            self.lbl_preview.setText(tr("❌ Cannot drop ALL frames!"))
            self.lbl_preview.setStyleSheet("color: #F44336; font-size: 11px;")
        else:
            self.lbl_preview.setText(
                tr("Drop %s frames, Keep %s frames.") % (len(drop_indices), len(keep_indices))
            )
            self.lbl_preview.setStyleSheet("color: #4CAF50; font-size: 11px;")

    def _apply_filter(self):
        """应用帧筛选"""
        layer_name = self.layer_combo.currentData()
        if not layer_name or layer_name not in self.viewer.layers:
            self.lbl_status.setText(f"❌ {tr('No layer selected')}")
            return
        
        layer = self.viewer.layers[layer_name]
        total_frames = layer.data.shape[0]
        text = self.edit_drop_range.text()
        
        drop_indices, keep_indices = self._parse_frame_indices(text, total_frames)
        
        if not keep_indices:
            QMessageBox.warning(self, tr("Error"), tr("Cannot drop ALL frames!"))
            return
        
        if len(keep_indices) == total_frames:
            QMessageBox.information(self, tr("Info"), tr("No frames dropped. Nothing to do."))
            return
        
        # 1. 切片数据
        new_data = layer.data[keep_indices]
        
        # 2. 处理 original_indices
        existing_originals = layer.metadata.get('original_indices', None)
        if existing_originals is not None:
            # 已有索引：映射
            existing_originals = np.array(existing_originals)
            new_originals = existing_originals[keep_indices].tolist()
        else:
            # 第一次筛选：使用 keep_indices 作为原始索引
            new_originals = keep_indices
        
        # 3. 复制其他元数据
        new_metadata = dict(layer.metadata)  # 浅拷贝
        new_metadata['original_indices'] = new_originals
        new_metadata['source_layer'] = layer_name
        new_metadata['filter_dropped'] = drop_indices
        
        # 4. 创建新图层
        new_layer_name = f"Filtered_{layer_name}"
        # 避免重名
        count = 1
        base_name = new_layer_name
        while new_layer_name in self.viewer.layers:
            new_layer_name = f"{base_name}_v{count}"
            count += 1
        
        new_layer = self.viewer.add_image(
            new_data,
            name=new_layer_name,
            colormap='gray',
            metadata=new_metadata
        )
        gc.collect()
        trim_working_set()

        # 5. 隐藏源图层，激活新图层
        layer.visible = False
        self.viewer.layers.selection.active = new_layer
        
        self.lbl_status.setText(
            f"✅ {tr('Created: %s (%s frames)') % (new_layer_name, len(keep_indices))}"
        )
        
        # 6. 记录日志
        try:
            get_logger().log_action("filter", "apply_filter", {
                "source_layer": layer_name,
                "dropped_count": len(drop_indices),
                "kept_count": len(keep_indices),
                "original_indices_sample": new_originals[:5] if len(new_originals) > 5 else new_originals
            })
        except Exception as e:
            print(f"Logger error: {e}")

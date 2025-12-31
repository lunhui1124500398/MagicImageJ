"""
Recovery Dialog - 会话恢复对话框

功能:
- 检测未完成的会话
- 显示可恢复操作列表
- 用户选择性恢复 (跳过被 undo 的操作)
"""
from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QListWidget, QListWidgetItem,
                            QGroupBox, QMessageBox, QCheckBox, QProgressDialog)
from qtpy.QtCore import Qt
from pathlib import Path
import json
from widgets.settings_widget import GlobalConfig, tr


class RecoveryDialog(QDialog):
    """会话恢复对话框"""
    
    def __init__(self, session_data: dict, parent=None):
        super().__init__(parent)
        self.session_data = session_data
        self.selected_actions = []
        self.setWindowTitle(tr("Session Recovery"))
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # 标题
        title = QLabel(f"<h3>🔄 {tr('Incomplete Session Detected')}</h3>")
        layout.addWidget(title)
        
        # 会话信息
        meta = self.session_data.get("metadata", {})
        info_text = f"""
        <b>{tr('Session ID')}:</b> {self.session_data.get('session_id', 'N/A')}<br>
        <b>{tr('Created')}:</b> {self.session_data.get('created_at', 'N/A')}<br>
        <b>{tr('Substance')}:</b> {meta.get('substance', 'N/A')}<br>
        <b>{tr('Dataset')}:</b> {meta.get('dataset_id', 'N/A')}
        """
        info_label = QLabel(info_text)
        info_label.setStyleSheet("background: #333; padding: 10px; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # 校验和状态
        if not self.session_data.get("_checksum_valid", True):
            warn = QLabel(f"⚠️ {tr('Warning: Log file may have been modified')}")
            warn.setStyleSheet("color: #FFA500; font-weight: bold;")
            layout.addWidget(warn)
        
        # === [增强] 数据源要求 - 智能检测 ===
        self.data_sources = self._get_data_sources()
        if self.data_sources:
            g_data = QGroupBox(f"📁 {tr('Required Data Sources')}")
            g_data.setStyleSheet("QGroupBox { color: #FFA500; font-weight: bold; }")
            d_layout = QVBoxLayout()
            
            warn_text = QLabel(f"<b style='color: #FFA500;'>⚠️ {tr('Important: You must load these files FIRST before recovery can work!')}</b>")
            warn_text.setWordWrap(True)
            d_layout.addWidget(warn_text)
            
            # 检查每个数据源路径是否存在
            for i, src in enumerate(self.data_sources):
                src_type = src.get("type", "")
                src_path = src.get("path", "")
                path_exists = Path(src_path).exists() if src_path != "N/A" else False
                src["exists"] = path_exists
                
                h_src = QHBoxLayout()
                
                # 状态图标
                if path_exists:
                    status = f"<span style='color: #4CAF50;'>✅ {tr('Data source found')}</span>"
                else:
                    status = f"<span style='color: #F44336;'>❌ {tr('Data source NOT found (may have been moved)')}</span>"
                
                src_label = QLabel(f"<b>{src_type}:</b> {src_path}<br>{status}")
                src_label.setWordWrap(True)
                src_label.setStyleSheet("margin-left: 10px;")
                h_src.addWidget(src_label, stretch=1)
                
                # 操作按钮
                btn_auto = QPushButton(tr("Auto Import"))
                btn_auto.setEnabled(path_exists)
                btn_auto.setToolTip(src_path)
                btn_auto.clicked.connect(lambda checked, idx=i: self._auto_import_source(idx))
                h_src.addWidget(btn_auto)
                
                btn_manual = QPushButton(tr("Manual Select"))
                btn_manual.clicked.connect(lambda checked, idx=i: self._manual_select_source(idx))
                h_src.addWidget(btn_manual)
                
                d_layout.addLayout(h_src)
            
            g_data.setLayout(d_layout)
            layout.addWidget(g_data)
        
        # 操作列表
        g_actions = QGroupBox(tr("Recoverable Actions"))
        g_layout = QVBoxLayout()
        
        self.action_list = QListWidget()
        self.action_list.setSelectionMode(QListWidget.MultiSelection)
        
        # 获取有效操作 (排除被 undo 的)
        effective_actions = self._get_effective_actions()
        
        for action in effective_actions:
            widget_name = action.get("widget", "unknown")
            action_name = action.get("action", "unknown")
            timestamp = action.get("timestamp", "")[:19]  # 截取到秒
            
            # 创建显示文本
            display_text = f"[{widget_name}] {action_name} - {timestamp}"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, action)  # 存储完整数据
            item.setSelected(True)  # 默认全选
            self.action_list.addItem(item)
        
        g_layout.addWidget(self.action_list)
        
        # 全选/取消全选
        h_select = QHBoxLayout()
        btn_select_all = QPushButton(tr("Select All"))
        btn_select_all.clicked.connect(lambda: self._select_all(True))
        btn_deselect_all = QPushButton(tr("Deselect All"))
        btn_deselect_all.clicked.connect(lambda: self._select_all(False))
        h_select.addWidget(btn_select_all)
        h_select.addWidget(btn_deselect_all)
        h_select.addStretch()
        g_layout.addLayout(h_select)
        
        g_actions.setLayout(g_layout)
        layout.addWidget(g_actions)
        
        # 提示信息
        hint = QLabel(f"<i style='color: gray;'>{tr('Only selected actions will be displayed. Undone actions are automatically excluded.')}</i>")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        
        # 按钮
        h_btn = QHBoxLayout()
        
        btn_skip = QPushButton(tr("Skip Recovery"))
        btn_skip.clicked.connect(self.reject)
        
        btn_recover = QPushButton(f"✅ {tr('Recover Selected')}")
        btn_recover.setStyleSheet("background-color: #2E7D32; font-weight: bold;")
        btn_recover.clicked.connect(self._on_recover_clicked)
        
        h_btn.addWidget(btn_skip)
        h_btn.addStretch()
        h_btn.addWidget(btn_recover)
        
        layout.addLayout(h_btn)
        self.setLayout(layout)
    
    def _get_effective_actions(self):
        """获取有效操作 (排除被 undo 的和 system 操作)"""
        actions = self.session_data.get("actions", [])
        
        # 找出所有被 undo 的 action id
        undone_ids = set()
        for action in actions:
            if action.get("action") == "undo":
                target = action.get("params", {}).get("target")
                if target:
                    undone_ids.add(target)
        
        # 过滤: 排除 undo 操作本身 和 被撤回的操作
        effective = []
        for action in actions:
            action_id = action.get("id", "")
            action_type = action.get("action", "")
            widget = action.get("widget", "")
            
            # 跳过 undo 操作和 system 操作
            if action_type == "undo" or widget == "system":
                continue
            
            # 跳过被撤回的操作
            if action_id in undone_ids:
                continue
            
            effective.append(action)
        
        return effective
    
    def _get_data_sources(self):
        """从日志中提取数据源信息"""
        sources = []
        actions = self.session_data.get("actions", [])
        
        for action in actions:
            widget = action.get("widget", "")
            action_type = action.get("action", "")
            params = action.get("params", {})
            
            # 识别各种导入操作
            if widget == "import":
                if action_type == "create_archive":
                    sources.append({
                        "type": "DM4 Archive",
                        "path": params.get("archive_path", "N/A")
                    })
                elif action_type == "load_png_sequence":
                    sources.append({
                        "type": "PNG Sequence",
                        "path": params.get("source_path", "N/A")
                    })
                elif action_type == "load_tiff_stack":
                    sources.append({
                        "type": "TIFF Stack",
                        "path": params.get("source_path", "N/A")
                    })
        
        return sources
    
    def _auto_import_source(self, idx: int):
        """自动导入指定数据源"""
        if idx >= len(self.data_sources):
            return
        
        src = self.data_sources[idx]
        src_type = src.get("type", "")
        src_path = src.get("path", "")
        
        if not Path(src_path).exists():
            QMessageBox.warning(self, tr("Error"), f"{tr('Data source NOT found (may have been moved)')}\n{src_path}")
            return
        
        # 根据类型调用对应的导入函数
        try:
            if src_type == "PNG Sequence":
                self._load_png_sequence(src_path)
            elif src_type == "TIFF Stack":
                self._load_tiff_stack(src_path)
            elif src_type == "DM4 Archive":
                # DM4 归档只需设置路径，用户需要在 Import tab 完成加载
                QMessageBox.information(self, tr("Data Source Import"), 
                    f"DM4 Archive path set. Please use Import tab to load images.\n{src_path}")
            
            QMessageBox.information(self, tr("Data Source Import"), f"✅ {src_type} loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"Import failed: {e}")
    
    def _manual_select_source(self, idx: int):
        """手动选择数据源"""
        from qtpy.QtWidgets import QFileDialog
        
        if idx >= len(self.data_sources):
            return
        
        src = self.data_sources[idx]
        src_type = src.get("type", "")
        
        if src_type == "PNG Sequence":
            folder = QFileDialog.getExistingDirectory(self, tr("Select PNG Sequence Folder"))
            if folder:
                self._load_png_sequence(folder)
                QMessageBox.information(self, tr("Data Source Import"), f"✅ PNG Sequence loaded!")
        elif src_type == "TIFF Stack":
            file_path, _ = QFileDialog.getOpenFileName(self, tr("Select TIFF File"), "", "TIFF (*.tiff *.tif)")
            if file_path:
                self._load_tiff_stack(file_path)
                QMessageBox.information(self, tr("Data Source Import"), f"✅ TIFF Stack loaded!")
        elif src_type == "DM4 Archive":
            QMessageBox.information(self, tr("Data Source Import"), 
                "DM4 loading requires the full Import workflow.\nPlease use Import tab after closing this dialog.")
    
    def _load_png_sequence(self, folder_path: str):
        """加载 PNG 序列"""
        import cv2
        import numpy as np
        
        folder = Path(folder_path)
        png_files = sorted(folder.glob("*.png"))
        if not png_files:
            raise ValueError(f"No PNG files found in {folder_path}")
        
        frames = []
        for f in png_files:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is not None:
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frames.append(img)
        
        if not frames:
            raise ValueError("Could not read any valid images")
        
        stack = np.array(frames)
        name = f"Recovered_PNG_{folder.name}"
        
        # 获取 viewer (从 parent)
        if hasattr(self.parent(), 'viewer'):
            self.parent().viewer.add_image(stack, name=name, colormap='gray')
    
    def _load_tiff_stack(self, file_path: str):
        """加载 TIFF Stack"""
        import tifffile
        import numpy as np
        
        stack = tifffile.imread(file_path)
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        elif stack.ndim == 4:
            stack = stack[..., 0]
        
        name = f"Recovered_TIFF_{Path(file_path).stem}"
        
        if hasattr(self.parent(), 'viewer'):
            self.parent().viewer.add_image(stack, name=name, colormap='gray')
    
    def _select_all(self, select: bool):
        """全选/取消全选"""
        for i in range(self.action_list.count()):
            self.action_list.item(i).setSelected(select)
    
    def _on_recover_clicked(self):
        """用户点击恢复"""
        self.selected_actions = []
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            if item.isSelected():
                self.selected_actions.append(item.data(Qt.UserRole))
        
        self.accept()
    
    def get_selected_actions(self):
        """获取用户选择的操作"""
        return self.selected_actions


def check_and_show_recovery(parent=None) -> list:
    """
    检查是否有未完成会话并显示恢复对话框
    
    Args:
        parent: 父窗口
    
    Returns:
        用户选择恢复的操作列表，如果跳过则返回空列表
    """
    from utils.session_logger import SessionLogger
    
    # 检查是否需要询问
    ask_on_recovery = bool(GlobalConfig.get("session_ask_on_recovery"))
    
    # 查找未完成会话
    incomplete = SessionLogger.find_incomplete_sessions()
    
    if not incomplete:
        return []
    
    # 取最近的一个
    latest_log = max(incomplete, key=lambda p: p.stat().st_mtime)
    
    # 加载日志
    session_data = SessionLogger.load_from_file(latest_log)
    if not session_data:
        return []
    
    # 检查是否有有效操作
    actions = session_data.get("actions", [])
    if not actions:
        return []
    
    if ask_on_recovery:
        # 显示对话框
        dialog = RecoveryDialog(session_data, parent)
        if dialog.exec_() == QDialog.Accepted:
            selected = dialog.get_selected_actions()
            # 标记旧会话为已恢复
            _mark_session_recovered(latest_log)
            return selected
        else:
            # 用户选择跳过，标记为放弃
            _mark_session_abandoned(latest_log)
            return []
    else:
        # 自动恢复所有有效操作
        effective = _get_effective_actions_from_data(session_data)
        _mark_session_recovered(latest_log)
        return effective


def _get_effective_actions_from_data(session_data: dict) -> list:
    """从日志数据获取有效操作"""
    actions = session_data.get("actions", [])
    undone_ids = set()
    for action in actions:
        if action.get("action") == "undo":
            target = action.get("params", {}).get("target")
            if target:
                undone_ids.add(target)
    
    return [a for a in actions 
            if a.get("action") != "undo" 
            and a.get("widget") != "system"
            and a.get("id") not in undone_ids]


def _mark_session_recovered(log_path: Path):
    """标记会话为已恢复"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data["status"] = "recovered"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except:
        pass


def _mark_session_abandoned(log_path: Path):
    """标记会话为已放弃"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data["status"] = "abandoned"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except:
        pass

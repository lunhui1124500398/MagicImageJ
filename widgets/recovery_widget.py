"""
Recovery Widget - 独立的会话恢复组件

功能:
- 显示所有历史会话（不仅仅是未完成的）
- 支持手动选择日志文件
- 数据源智能检测与导入
- 选择性恢复操作
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QListWidget, QListWidgetItem,
                            QGroupBox, QMessageBox, QSplitter, QFrame,
                            QFileDialog, QScrollArea)
from qtpy.QtCore import Qt
from pathlib import Path
import json
from widgets.settings_widget import GlobalConfig, tr


class RecoveryWidget(QWidget):
    """独立的会话恢复组件 - 作为主界面Tab"""
    
    # 状态图标映射
    STATUS_ICONS = {
        "completed": "✅",
        "in_progress": "⚠️",
        "crashed": "💥",
        "recovered": "🔄",
        "abandoned": "🗑️"
    }
    
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.sessions = []  # 会话列表
        self.current_session = None  # 当前选中的会话数据
        self.data_sources = []  # 当前会话的数据源
        self.manual_log_path = None  # 手动选择的日志路径
        self._setup_ui()
        self._refresh_sessions()
        self._setup_shortcuts()
    
    def _setup_shortcuts(self):
        """设置键盘快捷键 (使用配置的快捷键)"""
        from qtpy.QtWidgets import QShortcut
        from qtpy.QtGui import QKeySequence
        
        # 使用配置的快捷键
        star_key = str(GlobalConfig.get("shortcut_session_star") or "S")
        label_key = str(GlobalConfig.get("shortcut_session_label") or "L")
        
        # 收藏快捷键
        shortcut_star = QShortcut(QKeySequence(star_key), self)
        shortcut_star.activated.connect(self._toggle_session_star)
        
        # 编辑标签快捷键
        shortcut_label = QShortcut(QKeySequence(label_key), self)
        shortcut_label.activated.connect(self._edit_session_label)
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # === 顶部标题栏 ===
        header = QHBoxLayout()
        title = QLabel(f"<h3>🔄 {tr('Session Recovery')}</h3>")
        header.addWidget(title)
        header.addStretch()
        
        btn_refresh = QPushButton(f"🔃 {tr('Refresh Sessions')}")
        btn_refresh.clicked.connect(self._refresh_sessions)
        btn_refresh.setStyleSheet("padding: 4px 12px;")
        header.addWidget(btn_refresh)
        layout.addLayout(header)
        
        # === 手动模式区域 ===
        g_manual = QGroupBox(f"📂 {tr('Manual Mode')}")
        m_layout = QHBoxLayout()
        
        btn_select_log = QPushButton(f"📄 {tr('Select Log File')}")
        btn_select_log.clicked.connect(self._select_log_file)
        m_layout.addWidget(btn_select_log)
        
        self.lbl_manual_path = QLabel(f"{tr('Current')}: <i>{tr('Not selected')}</i>")
        self.lbl_manual_path.setStyleSheet("color: #888;")
        m_layout.addWidget(self.lbl_manual_path, stretch=1)
        
        g_manual.setLayout(m_layout)
        layout.addWidget(g_manual)
        
        # === 主内容区 - 使用 Splitter 分割 ===
        splitter = QSplitter(Qt.Horizontal)
        
        # --- 左侧: 会话列表 ---
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_sessions = QLabel(f"<b>📋 {tr('Recent Sessions')}</b>")
        left_layout.addWidget(lbl_sessions)
        
        self.session_list = QListWidget()
        self.session_list.currentItemChanged.connect(self._on_session_selected)
        self.session_list.itemDoubleClicked.connect(self._on_session_double_clicked)  # 双击编辑标签
        self.session_list.setMinimumWidth(220)
        left_layout.addWidget(self.session_list)
        
        left_panel.setLayout(left_layout)
        splitter.addWidget(left_panel)
        
        # --- 右侧: 会话详情 ---
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        
        self.details_panel = QWidget()
        self.details_layout = QVBoxLayout()
        self.details_layout.setAlignment(Qt.AlignTop)
        self.details_panel.setLayout(self.details_layout)
        right_scroll.setWidget(self.details_panel)
        splitter.addWidget(right_scroll)
        
        # 设置 splitter 比例
        splitter.setSizes([280, 420])
        layout.addWidget(splitter, stretch=1)
        
        self.setLayout(layout)
        
        # 初始显示空状态
        self._show_empty_state()
    
    def _show_empty_state(self):
        """显示无会话时的空状态"""
        self._clear_details()
        
        empty_label = QLabel(f"""
            <div style='text-align: center; padding: 40px; color: #888;'>
                <p style='font-size: 48px; margin-bottom: 10px;'>📭</p>
                <p style='font-size: 14px;'>{tr('No sessions found')}</p>
                <p style='font-size: 12px; color: #666;'>{tr('Click Refresh to check again')}</p>
                <p style='font-size: 12px; color: #666;'>{tr('Or use Manual Mode to select a log file')}</p>
            </div>
        """)
        empty_label.setAlignment(Qt.AlignCenter)
        self.details_layout.addWidget(empty_label)
    
    def _clear_details(self):
        """清空详情面板"""
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                elif item.layout():
                    clear_layout(item.layout())
        
        clear_layout(self.details_layout)
    
    def _select_log_file(self):
        """手动选择日志文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            tr("Select Log File"), 
            str(Path.home() / ".napari_tem" / "sessions"),
            "Session Log (*.json)"
        )
        
        if file_path:
            self.manual_log_path = Path(file_path)
            self.lbl_manual_path.setText(f"{tr('Current')}: <b>{self.manual_log_path.name}</b>")
            self.lbl_manual_path.setStyleSheet("color: #4CAF50;")
            
            # 加载并显示该会话
            self._load_manual_session()
    
    def _load_manual_session(self):
        """加载手动选择的会话"""
        from utils.session_logger import SessionLogger
        
        if not self.manual_log_path or not self.manual_log_path.exists():
            return
        
        session_data = SessionLogger.load_from_file(self.manual_log_path)
        if session_data:
            session_data["_log_path"] = str(self.manual_log_path)
            self.current_session = session_data
            
            # 取消列表选中
            self.session_list.clearSelection()
            
            # 显示详情
            self._show_session_details()
    
    def _refresh_sessions(self):
        """刷新会话列表 - 显示所有会话"""
        from utils.session_logger import SessionLogger
        
        self.session_list.clear()
        self.sessions = []
        self.manual_log_path = None
        self.lbl_manual_path.setText(f"{tr('Current')}: <i>{tr('Not selected')}</i>")
        self.lbl_manual_path.setStyleSheet("color: #888;")
        
        # 获取所有会话
        all_sessions = SessionLogger.find_all_sessions(limit=50)
        
        if not all_sessions:
            self._show_empty_state()
            return
        
        for log_path in all_sessions:
            session_data = SessionLogger.load_from_file(log_path)
            if session_data:
                session_data["_log_path"] = str(log_path)
                self.sessions.append(session_data)
                
                # 获取会话信息
                status = session_data.get("status", "unknown")
                icon = self.STATUS_ICONS.get(status, "❓")
                created = session_data.get("created_at", "")[:16].replace("T", " ")
                meta = session_data.get("metadata", {})
                substance = meta.get("substance", "N/A")
                dataset = meta.get("dataset_id", "")
                
                # 收藏和标签状态 (新增)
                starred = session_data.get("starred", False)
                label = session_data.get("label", "")
                star_icon = "⭐ " if starred else ""
                label_text = f"[{label}] " if label else ""
                
                # 创建列表项
                display = f"{star_icon}{icon} {label_text}{created} | {substance}"
                if dataset:
                    display += f"/{dataset}"
                
                item = QListWidgetItem(display)
                item.setToolTip(f"Status: {status}\nLabel: {label or '(none)'}\nPath: {log_path}")
                item.setData(Qt.UserRole, len(self.sessions) - 1)
                self.session_list.addItem(item)
        
        # 显示空状态如果没有会话
        if self.session_list.count() == 0:
            self._show_empty_state()
    
    def _on_session_selected(self, current, previous):
        """会话选中时显示详情"""
        if current is None:
            return
        
        # 清除手动选择状态
        self.manual_log_path = None
        self.lbl_manual_path.setText(f"{tr('Current')}: <i>{tr('Not selected')}</i>")
        self.lbl_manual_path.setStyleSheet("color: #888;")
        
        idx = current.data(Qt.UserRole)
        if idx is None or idx >= len(self.sessions):
            return
        
        self.current_session = self.sessions[idx]
        self._show_session_details()
    
    def _show_session_details(self):
        """显示选中会话的详情"""
        self._clear_details()
        
        if not self.current_session:
            self._show_empty_state()
            return
        
        session = self.current_session
        meta = session.get("metadata", {})
        status = session.get("status", "unknown")
        status_display = tr(status) if status in ["completed", "in_progress", "crashed", "recovered", "abandoned"] else status
        icon = self.STATUS_ICONS.get(status, "❓")
        
        # === 1. 会话信息 ===
        g_info = QGroupBox(f"📄 {tr('Session Details')}")
        info_layout = QVBoxLayout()
        
        info_text = f"""
        <table style='margin: 5px;'>
            <tr><td><b>{tr('Session ID')}:</b></td><td>{session.get('session_id', 'N/A')}</td></tr>
            <tr><td><b>{tr('Created')}:</b></td><td>{session.get('created_at', 'N/A')[:19]}</td></tr>
            <tr><td><b>{tr('Substance')}:</b></td><td>{meta.get('substance', 'N/A')}</td></tr>
            <tr><td><b>{tr('Dataset')}:</b></td><td>{meta.get('dataset_id', 'N/A')}</td></tr>
            <tr><td><b>Status:</b></td><td>{icon} {status_display}</td></tr>
        </table>
        """
        lbl_info = QLabel(info_text)
        lbl_info.setStyleSheet("background: #333; padding: 10px; border-radius: 5px;")
        info_layout.addWidget(lbl_info)
        
        # 校验和警告
        if not session.get("_checksum_valid", True):
            warn = QLabel(f"⚠️ {tr('Warning: Log file may have been modified')}")
            warn.setStyleSheet("color: #FFA500; font-weight: bold; padding: 5px;")
            info_layout.addWidget(warn)
        
        g_info.setLayout(info_layout)
        self.details_layout.addWidget(g_info)
        
        # === 2. 数据源 ===
        self.data_sources = self._get_data_sources(session)
        if self.data_sources:
            g_data = QGroupBox(f"📁 {tr('Required Data Sources')}")
            g_data.setStyleSheet("QGroupBox { color: #FFA500; font-weight: bold; }")
            d_layout = QVBoxLayout()
            
            warn_text = QLabel(f"<b style='color: #FFA500;'>⚠️ {tr('Please load data sources first')}</b>")
            warn_text.setWordWrap(True)
            d_layout.addWidget(warn_text)
            
            for i, src in enumerate(self.data_sources):
                src_type = src.get("type", "")
                src_path = src.get("path", "")
                path_exists = Path(src_path).exists() if src_path != "N/A" else False
                src["exists"] = path_exists
                
                h_src = QHBoxLayout()
                
                if path_exists:
                    status_str = f"<span style='color: #4CAF50;'>✅ {tr('Data source found')}</span>"
                else:
                    status_str = f"<span style='color: #F44336;'>❌ {tr('Data source NOT found (may have been moved)')}</span>"
                
                src_label = QLabel(f"<b>{tr(src_type)}:</b><br><small>{src_path}</small><br>{status_str}")
                src_label.setWordWrap(True)
                h_src.addWidget(src_label, stretch=1)
                
                btn_auto = QPushButton(tr("Auto Import"))
                btn_auto.setEnabled(path_exists)
                btn_auto.clicked.connect(lambda checked, idx=i: self._auto_import_source(idx))
                h_src.addWidget(btn_auto)
                
                btn_manual = QPushButton(tr("Manual Select"))
                btn_manual.clicked.connect(lambda checked, idx=i: self._manual_select_source(idx))
                h_src.addWidget(btn_manual)
                
                d_layout.addLayout(h_src)
            
            g_data.setLayout(d_layout)
            self.details_layout.addWidget(g_data)
        
        # === 3. 可恢复操作列表 ===
        g_actions = QGroupBox(tr("Recoverable Actions"))
        a_layout = QVBoxLayout()
        
        self.action_list = QListWidget()
        self.action_list.setSelectionMode(QListWidget.MultiSelection)
        self.action_list.setMaximumHeight(200)
        
        effective_actions = self._get_effective_actions(session)
        
        if effective_actions:
            for action in effective_actions:
                widget_name = action.get("widget", "unknown")
                action_name = action.get("action", "unknown")
                timestamp = action.get("timestamp", "")[:19]
                
                display_text = f"[{widget_name}] {action_name} - {timestamp}"
                
                item = QListWidgetItem(display_text)
                item.setData(Qt.UserRole, action)
                item.setSelected(True)
                self.action_list.addItem(item)
        else:
            self.action_list.addItem(QListWidgetItem("(No recoverable actions)"))
        
        a_layout.addWidget(self.action_list)
        
        # 全选/取消全选
        h_select = QHBoxLayout()
        btn_select_all = QPushButton(tr("Select All"))
        btn_select_all.clicked.connect(lambda: self._select_all_actions(True))
        btn_deselect_all = QPushButton(tr("Deselect All"))
        btn_deselect_all.clicked.connect(lambda: self._select_all_actions(False))
        h_select.addWidget(btn_select_all)
        h_select.addWidget(btn_deselect_all)
        h_select.addStretch()
        a_layout.addLayout(h_select)
        
        g_actions.setLayout(a_layout)
        self.details_layout.addWidget(g_actions)
        
        # === 4. 会话管理按钮 (新增: 收藏/标签) ===
        h_mgmt = QHBoxLayout()
        
        # 收藏按钮
        is_starred = session.get("starred", False)
        star_btn_text = f"⭐ {tr('Unstar Session')}" if is_starred else f"☆ {tr('Star Session')}"
        btn_star = QPushButton(star_btn_text)
        btn_star.clicked.connect(self._toggle_session_star)
        if is_starred:
            btn_star.setStyleSheet("background-color: #FFA500;")
        h_mgmt.addWidget(btn_star)
        
        # 标签编辑按钮
        btn_label = QPushButton(f"🏷️ {tr('Edit Label')}")
        btn_label.clicked.connect(self._edit_session_label)
        h_mgmt.addWidget(btn_label)
        
        h_mgmt.addStretch()
        self.details_layout.addLayout(h_mgmt)
        
        # === 5. 恢复/放弃按钮 ===
        h_btns = QHBoxLayout()
        
        btn_abandon = QPushButton(f"🗑️ {tr('Abandon Session')}")
        btn_abandon.setStyleSheet("background-color: #8B0000;")
        btn_abandon.clicked.connect(self._abandon_session)
        
        btn_recover = QPushButton(f"✅ {tr('Start Recovery')}")
        btn_recover.setStyleSheet("background-color: #2E7D32; font-weight: bold; padding: 8px 16px;")
        btn_recover.clicked.connect(self._recover_selected)
        
        h_btns.addWidget(btn_abandon)
        h_btns.addStretch()
        h_btns.addWidget(btn_recover)
        
        self.details_layout.addLayout(h_btns)
        
        # 提示
        hint = QLabel(f"<i style='color: gray;'>{tr('Only selected actions will be displayed. Undone actions are automatically excluded.')}</i>")
        hint.setWordWrap(True)
        self.details_layout.addWidget(hint)
        
        self.details_layout.addStretch()
    
    def _toggle_session_star(self):
        """切换当前会话的收藏状态"""
        from utils.session_logger import SessionLogger
        
        if not self.current_session:
            return
        
        log_path = self.current_session.get("_log_path")
        if not log_path:
            return
        
        current_starred = self.current_session.get("starred", False)
        new_starred = not current_starred
        
        if SessionLogger.update_session_file(Path(log_path), starred=new_starred):
            # 更新本地状态
            self.current_session["starred"] = new_starred
            # 刷新列表和详情
            self._refresh_sessions()
            # 重新选中当前会话并显示详情
            self._show_session_details()
    
    def _edit_session_label(self):
        """编辑当前会话的标签"""
        from utils.session_logger import SessionLogger
        from qtpy.QtWidgets import QInputDialog
        
        if not self.current_session:
            return
        
        log_path = self.current_session.get("_log_path")
        if not log_path:
            return
        
        current_label = self.current_session.get("label", "")
        
        new_label, ok = QInputDialog.getText(
            self, tr("Session Label"), 
            tr("Enter label for this session:"),
            text=current_label
        )
        
        if ok:
            if SessionLogger.update_session_file(Path(log_path), label=new_label):
                # 更新本地状态
                self.current_session["label"] = new_label
                # 刷新列表和详情
                self._refresh_sessions()
                self._show_session_details()
    
    def _on_session_double_clicked(self, item):
        """双击会话列表项时编辑标签"""
        if item is None:
            return
        
        idx = item.data(Qt.UserRole)
        if idx is not None and idx < len(self.sessions):
            self.current_session = self.sessions[idx]
            self._edit_session_label()
    
    def _get_data_sources(self, session_data):
        """从会话数据中提取数据源信息"""
        sources = []
        actions = session_data.get("actions", [])
        
        for action in actions:
            widget = action.get("widget", "")
            action_type = action.get("action", "")
            params = action.get("params", {})
            
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
    
    def _get_effective_actions(self, session_data):
        """获取有效操作 (排除 undo 和 system)"""
        actions = session_data.get("actions", [])
        
        undone_ids = set()
        for action in actions:
            if action.get("action") == "undo":
                target = action.get("params", {}).get("target")
                if target:
                    undone_ids.add(target)
        
        effective = []
        for action in actions:
            action_id = action.get("id", "")
            action_type = action.get("action", "")
            widget = action.get("widget", "")
            
            if action_type == "undo" or widget == "system":
                continue
            
            if action_id in undone_ids:
                continue
            
            effective.append(action)
        
        return effective
    
    def _select_all_actions(self, select: bool):
        """全选/取消全选操作"""
        for i in range(self.action_list.count()):
            self.action_list.item(i).setSelected(select)
    
    def _auto_import_source(self, idx: int):
        """自动导入数据源"""
        if idx >= len(self.data_sources):
            return
        
        src = self.data_sources[idx]
        src_type = src.get("type", "")
        src_path = src.get("path", "")
        
        if not Path(src_path).exists():
            QMessageBox.warning(self, tr("Error"), 
                f"{tr('Data source NOT found (may have been moved)')}\n{src_path}")
            return
        
        try:
            if src_type == "PNG Sequence":
                self._load_png_sequence(src_path)
            elif src_type == "TIFF Stack":
                self._load_tiff_stack(src_path)
            elif src_type == "DM4 Archive":
                QMessageBox.information(self, tr("Data Source Import"), 
                    tr("DM4 Archive path detected.\nPlease use Import tab to load the images.\n\n%s") % src_path)
                return
            
            QMessageBox.information(self, tr("Data Source Import"), 
                tr("✅ %s loaded successfully!") % tr(src_type))
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), tr("Import failed: %s") % e)
    
    def _manual_select_source(self, idx: int):
        """手动选择数据源"""
        if idx >= len(self.data_sources):
            return
        
        src = self.data_sources[idx]
        src_type = src.get("type", "")
        
        if src_type == "PNG Sequence":
            folder = QFileDialog.getExistingDirectory(self, tr("Select PNG Sequence Folder"))
            if folder:
                try:
                    self._load_png_sequence(folder)
                    QMessageBox.information(self, tr("Data Source Import"), tr("✅ PNG Sequence loaded!"))
                except Exception as e:
                    QMessageBox.critical(self, tr("Error"), tr("Import failed: %s") % e)
        elif src_type == "TIFF Stack":
            file_path, _ = QFileDialog.getOpenFileName(self, tr("Select TIFF File"), "", "TIFF (*.tiff *.tif)")
            if file_path:
                try:
                    self._load_tiff_stack(file_path)
                    QMessageBox.information(self, tr("Data Source Import"), tr("✅ TIFF Stack loaded!"))
                except Exception as e:
                    QMessageBox.critical(self, tr("Error"), tr("Import failed: %s") % e)
        elif src_type == "DM4 Archive":
            QMessageBox.information(self, tr("Data Source Import"), 
                tr("DM4 loading requires the full Import workflow.\nPlease use Import tab."))
    
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
        self.viewer.add_image(stack, name=name, colormap='gray')
    
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
        self.viewer.add_image(stack, name=name, colormap='gray')
    
    def _auto_detect_and_import_sources(self) -> str:
        """
        自动检测并导入数据源
        
        Returns:
            "success" - 成功导入
            "cancelled" - 用户取消
            "failed" - 检测失败
        """
        if not self.current_session:
            return "failed"
        
        # 从 import 操作中获取数据源路径
        actions = self.current_session.get("actions", [])
        import_actions = [a for a in actions if a.get("widget") == "import"]
        
        if not import_actions:
            return "failed"
        
        # 检测数据源
        detected_sources = []
        for action in import_actions:
            action_type = action.get("action", "")
            params = action.get("params", {})
            
            if action_type == "create_archive":
                archive_path = params.get("archive_path", "")
                if archive_path and Path(archive_path).exists():
                    detected_sources.append({
                        "type": "DM4 Archive",
                        "path": archive_path,
                        "exists": True
                    })
            elif action_type == "load_dm4_sequence":
                # 检测 DM4 序列导入
                source_path = params.get("source_path", "")
                if source_path:
                    detected_sources.append({
                        "type": "DM4 Sequence",
                        "path": source_path,
                        "exists": Path(source_path).exists()
                    })
            elif action_type == "load_png_sequence":
                source_path = params.get("source_path", "")
                if source_path:
                    detected_sources.append({
                        "type": "PNG Sequence",
                        "path": source_path,
                        "exists": Path(source_path).exists()
                    })
            elif action_type == "load_tiff_stack":
                source_path = params.get("source_path", "")
                if source_path:
                    detected_sources.append({
                        "type": "TIFF Stack",
                        "path": source_path,
                        "exists": Path(source_path).exists()
                    })
        
        if not detected_sources:
            return "failed"
        
        # 构建检测报告对话框
        msg_text = f"<b>{tr('Data Source Detection')}</b><br><br>"
        msg_text += f"{tr('Auto-detected data source')}:<br><br>"
        
        for src in detected_sources:
            status = "✅" if src["exists"] else "❌"
            msg_text += f"{status} <b>{src['type']}</b><br>"
            msg_text += f"&nbsp;&nbsp;{src['path']}<br><br>"
        
        msg_text += f"{tr('Do you want to import this?')}"
        
        # 弹窗询问
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(tr("Data Source Detection"))
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(msg_text)
        
        btn_import = msg_box.addButton(f"✅ {tr('Import Detected')}", QMessageBox.ActionRole)
        btn_manual = msg_box.addButton(f"🔍 {tr('Manual Select')}", QMessageBox.ActionRole)
        btn_skip = msg_box.addButton(tr("Skip Import"), QMessageBox.RejectRole)
        
        msg_box.exec_()
        choice = msg_box.clickedButton()
        
        if choice == btn_skip or choice is None:
            return "cancelled"
        
        if choice == btn_manual:
            # 手动选择
            return self._show_manual_import_dialog()
        
        if choice == btn_import:
            # 自动导入检测到的数据源
            for src in detected_sources:
                if not src["exists"]:
                    continue  # 跳过不存在的源
                try:
                    if src["type"] == "PNG Sequence":
                        self._load_png_sequence(src["path"])
                        return "success"
                    elif src["type"] == "TIFF Stack":
                        self._load_tiff_stack(src["path"])
                        return "success"
                    elif src["type"] in ("DM4 Archive", "DM4 Sequence"):
                        # 直接加载 DM4 序列
                        self._load_dm4_sequence(src["path"])
                        return "success"
                except Exception as e:
                    QMessageBox.critical(self, tr("Error"), f"Import failed: {e}")
                    # 导入失败，给用户手动选择的机会
                    return self._show_manual_import_dialog()
            
            # 如果没有有效的源可导入
            return "failed"
        
        return "cancelled"
    
    def _manual_select_data_source(self) -> str:
        """手动选择数据源"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(tr("Manual Select"))
        msg_box.setText(tr("Select data source type:"))
        
        btn_png = msg_box.addButton(tr("PNG Sequence"), QMessageBox.ActionRole)
        btn_tiff = msg_box.addButton(tr("TIFF Stack"), QMessageBox.ActionRole)
        btn_cancel = msg_box.addButton(QMessageBox.Cancel)
        
        msg_box.exec_()
        choice = msg_box.clickedButton()
        
        if choice == btn_cancel:
            return "cancelled"
        
        try:
            if choice == btn_png:
                folder = QFileDialog.getExistingDirectory(self, tr("Select PNG Sequence Folder"))
                if folder:
                    self._load_png_sequence(folder)
                    return "success"
            elif choice == btn_tiff:
                file_path, _ = QFileDialog.getOpenFileName(self, tr("Select TIFF File"), "", "TIFF (*.tiff *.tif)")
                if file_path:
                    self._load_tiff_stack(file_path)
                    return "success"
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"Import failed: {e}")
            return "failed"
        
        return "cancelled"
    
    def _show_manual_import_dialog(self) -> str:
        """
        显示手动导入对话框
        
        Returns:
            "success" - 导入成功
            "cancelled" - 用户取消
            "continue" - 用户选择继续（不导入）
        """
        # 获取日志中的源路径信息（如果有）
        source_hints = []
        if self.current_session:
            actions = self.current_session.get("actions", [])
            for action in actions:
                if action.get("widget") == "import":
                    params = action.get("params", {})
                    source_path = params.get("source_path", params.get("archive_path", ""))
                    if source_path:
                        source_hints.append(source_path)
        
        # 构建消息
        msg_text = f"<b>{tr('Manual Data Import')}</b><br><br>"
        msg_text += f"⚠️ {tr('No image layers detected')}<br><br>"
        
        if source_hints:
            msg_text += f"<b>{tr('Detected paths from session log')}:</b><br>"
            for hint in source_hints[:3]:
                exists_icon = "✅" if Path(hint).exists() else "❌"
                msg_text += f"{exists_icon} {hint}<br>"
            msg_text += "<br>"
        else:
            msg_text += f"{tr('No source path found in session log.')}<br><br>"
        
        msg_text += f"{tr('Please select how to import data')}:"
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(tr("Manual Import"))
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(msg_text)
        
        btn_dm4 = msg_box.addButton(f"📂 {tr('DM4 Folder')}", QMessageBox.ActionRole)
        btn_png = msg_box.addButton(f"📂 {tr('PNG Sequence')}", QMessageBox.ActionRole)
        btn_tiff = msg_box.addButton(f"📂 {tr('TIFF Stack')}", QMessageBox.ActionRole)
        btn_continue = msg_box.addButton(f"⏭️ {tr('Continue Without Import')}", QMessageBox.ActionRole)
        btn_cancel = msg_box.addButton(tr("Cancel"), QMessageBox.RejectRole)
        
        msg_box.exec_()
        choice = msg_box.clickedButton()
        
        if choice == btn_cancel:
            return "cancelled"
        elif choice == btn_continue:
            return "continue"
        
        try:
            # 使用最后一个检测到的路径作为起始目录
            start_path = str(Path(source_hints[0]).parent) if source_hints and Path(source_hints[0]).parent.exists() else ""
            # 如果有直接检测到的路径，优先使用
            if source_hints and Path(source_hints[0]).exists():
                start_path = source_hints[0]
            
            if choice == btn_dm4:
                folder = QFileDialog.getExistingDirectory(self, tr("Select DM4 Folder"), start_path)
                if folder:
                    self._load_dm4_sequence(folder)
                    return "success"
            elif choice == btn_png:
                folder = QFileDialog.getExistingDirectory(self, tr("Select PNG Sequence Folder"), start_path)
                if folder:
                    self._load_png_sequence(folder)
                    return "success"
            elif choice == btn_tiff:
                file_path, _ = QFileDialog.getOpenFileName(self, tr("Select TIFF File"), start_path, "TIFF (*.tiff *.tif)")
                if file_path:
                    self._load_tiff_stack(file_path)
                    return "success"
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"Import failed: {e}")
        
        return "cancelled"
    
    def _confirm_export_actions(self, export_actions: list) -> str:
        """
        确认导出操作
        
        Returns:
            "proceed" - 继续导出
            "skip" - 跳过导出
        """
        # 统计导出操作
        export_count = len(export_actions)
        
        msg_text = f"<b>{tr('Export Confirmation')}</b><br><br>"
        msg_text += f"{tr('The session contains export operations. Do you want to re-export?')}<br><br>"
        msg_text += f"共 {export_count} 个导出操作<br><br>"
        
        # 显示导出详情
        for i, action in enumerate(export_actions[:3]):
            params = action.get("params", {})
            export_type = params.get("format", "unknown")
            output_path = params.get("output_path", "N/A")
            msg_text += f"• {export_type}: {Path(output_path).name if output_path != 'N/A' else 'N/A'}<br>"
        
        if export_count > 3:
            msg_text += f"• ... 还有 {export_count - 3} 个<br>"
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(tr("Export Confirmation"))
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(msg_text)
        
        btn_proceed = msg_box.addButton(f"✅ {tr('Start Recovery')}", QMessageBox.AcceptRole)
        btn_skip = msg_box.addButton(f"⏭️ {tr('Skip Export')}", QMessageBox.ActionRole)
        btn_cancel = msg_box.addButton(tr("Cancel"), QMessageBox.RejectRole)
        
        msg_box.exec_()
        choice = msg_box.clickedButton()
        
        if choice == btn_cancel or choice is None:
            return "cancelled"
        elif choice == btn_skip:
            return "skip"
        
        return "proceed"
    
    
    def _recover_selected(self):
        """恢复选中的操作"""
        if not self.current_session:
            return
        
        selected_actions = []
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            if item.isSelected() and item.data(Qt.UserRole):
                selected_actions.append(item.data(Qt.UserRole))
        
        if not selected_actions:
            QMessageBox.warning(self, tr("Session Recovery"), 
                tr("No actions selected for recovery."))
            return
        
        # === 智能数据源检测 ===
        image_layers = [l for l in self.viewer.layers if hasattr(l, 'data') and l.data is not None]
        
        if not image_layers:
            # 没有图层，尝试自动检测数据源
            auto_detect = GlobalConfig.get("session_auto_detect_source")
            
            if auto_detect:
                # 尝试从日志中检测数据源
                import_result = self._auto_detect_and_import_sources()
                
                if import_result == "cancelled":
                    return
                elif import_result == "failed":
                    # 自动检测失败，提供手动导入选项
                    manual_result = self._show_manual_import_dialog()
                    if manual_result == "cancelled":
                        return
            else:
                # 不自动检测，直接提供手动导入选项
                manual_result = self._show_manual_import_dialog()
                if manual_result == "cancelled":
                    return
        
        # === 检查是否有 export 操作，提前询问 ===
        export_actions = [a for a in selected_actions if a.get("widget") == "export"]
        if export_actions and GlobalConfig.get("session_confirm_export"):
            export_choice = self._confirm_export_actions(export_actions)
            if export_choice == "cancelled":
                return  # 用户取消恢复
            elif export_choice == "skip":
                # 从选中列表中移除 export 操作
                selected_actions = [a for a in selected_actions if a.get("widget") != "export"]
        
        # === 智能处理：多次 update_batch_rois 只保留最后一次 ===
        # 找到最后一个 update_batch_rois 的索引
        last_roi_update_idx = -1
        for i, action in enumerate(selected_actions):
            if action.get("widget") == "geometry" and action.get("action") == "update_batch_rois":
                last_roi_update_idx = i
        
        # 过滤掉除最后一次外的所有 update_batch_rois
        if last_roi_update_idx >= 0:
            filtered_actions = []
            for i, action in enumerate(selected_actions):
                if action.get("widget") == "geometry" and action.get("action") == "update_batch_rois":
                    if i == last_roi_update_idx:
                        filtered_actions.append(action)
                    # 跳过其他的 update_batch_rois
                else:
                    filtered_actions.append(action)
            selected_actions = filtered_actions
        
        # 执行恢复操作 - 链式恢复
        success_count = 0
        failed_actions = []
        skipped_actions = []
        aborted = False
        last_result_layer = None  # 跟踪上一个操作产生的图层名
        
        for action in selected_actions:
            widget = action.get("widget", "")
            action_type = action.get("action", "")
            params = action.get("params", {})
            
            try:
                # 传入上一个结果图层名，用于链式操作
                result, result_layer = self._replay_action(widget, action_type, params, last_result_layer)
                if result == "success":
                    success_count += 1
                    if result_layer:
                        last_result_layer = result_layer  # 更新链式图层
                elif result == "skipped":
                    skipped_actions.append(f"[{widget}] {action_type}")
                elif result == "abort":
                    # 用户请求取消所有后续步骤
                    aborted = True
                    break
                else:
                    failed_actions.append(f"[{widget}] {action_type}")
            except Exception as e:
                failed_actions.append(f"[{widget}] {action_type}: {str(e)[:50]}")
        
        # 生成恢复报告
        report_lines = [f"✅ 成功恢复: {success_count} 个操作"]
        
        if skipped_actions:
            report_lines.append(f"\n⏭️ 跳过 (需手动操作): {len(skipped_actions)} 个")
            for s in skipped_actions[:5]:
                report_lines.append(f"  • {s}")
            if len(skipped_actions) > 5:
                report_lines.append(f"  ... 还有 {len(skipped_actions) - 5} 个")
        
        if failed_actions:
            report_lines.append(f"\n❌ 失败: {len(failed_actions)} 个")
            for f in failed_actions[:3]:
                report_lines.append(f"  • {f}")
        
        # 添加参数参考
        if skipped_actions:
            report_lines.append("\n📋 操作参数参考 (请按以下参数手动操作):")
            for action in selected_actions:
                widget = action.get("widget", "")
                action_type = action.get("action", "")
                if f"[{widget}] {action_type}" in skipped_actions:
                    params = action.get("params", {})
                    param_str = self._format_params(params)
                    report_lines.append(f"\n[{widget}] {action_type}:")
                    report_lines.append(param_str)
        
        # 标记会话为已恢复
        log_path = Path(self.current_session.get("_log_path", ""))
        if log_path.exists():
            self._mark_session_status(log_path, "recovered")
        
        # 显示恢复结果
        QMessageBox.information(self, tr("Session Recovery"), 
            "\n".join(report_lines))
        
        # 更新状态栏
        self.viewer.status = f"✅ Session recovery: {success_count} success, {len(skipped_actions)} skipped"
        
        # 刷新列表
        self._refresh_sessions()
    
    def _ask_recovery_confirm(self, title: str, message: str) -> str:
        """
        显示恢复确认对话框，支持取消所有后续步骤
        
        Returns:
            "yes" - 继续此操作
            "skip" - 跳过此操作
            "abort" - 取消所有后续操作
        """
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        
        btn_yes = msg_box.addButton(f"✅ {tr('Continue')}", QMessageBox.AcceptRole)
        btn_skip = msg_box.addButton(f"⏭️ {tr('Skip This')}", QMessageBox.ActionRole)
        btn_abort = msg_box.addButton(f"🛑 {tr('Cancel All')}", QMessageBox.RejectRole)
        
        msg_box.exec_()
        choice = msg_box.clickedButton()
        
        if choice == btn_yes:
            return "yes"
        elif choice == btn_skip:
            return "skip"
        else:
            return "abort"
    
    def _replay_action(self, widget: str, action_type: str, params: dict, last_result_layer: str = None) -> tuple:
        """
        重放单个操作
        
        Args:
            last_result_layer: 上一个操作产生的图层名，用于链式操作
        
        Returns:
            (status, result_layer_name)
            status: "success" / "skipped" / "failed"
            result_layer_name: 操作产生的新图层名（如果有）
        """
        import numpy as np
        
        # 获取恢复模式
        recovery_mode = GlobalConfig.get("session_recovery_mode") or "review"
        
        # === 导入操作 - 通常已经手动完成，跳过 ===
        if widget == "import":
            return ("skipped", None)  # 导入操作需要用户手动完成
        
        # === 漂移矫正 - 自动重放 ===
        if widget == "drift" and action_type == "apply_correction":
            return self._replay_drift(params, recovery_mode, last_result_layer)
        
        # === 图像增强 - 可以尝试自动重放 ===
        if widget == "enhance":
            return self._replay_enhance(action_type, params, recovery_mode, last_result_layer)
        
        # === 几何变换 - 自动重放 ===
        if widget == "geometry":
            return self._replay_geometry(action_type, params, recovery_mode, last_result_layer)
        
        # === 导出 ===
        if widget == "export":
            return ("skipped", None)  # 导出需要手动确认路径
        
        return ("skipped", None)
    
    def _replay_geometry(self, action_type: str, params: dict, recovery_mode: str, last_result_layer: str = None) -> tuple:
        """重放几何变换操作"""
        import numpy as np
        
        try:
            source_layer_name = params.get("source_layer", "")
            
            # 链式恢复：优先使用上一个操作产生的图层
            if last_result_layer and last_result_layer in self.viewer.layers:
                source_layer = self.viewer.layers[last_result_layer]
            elif source_layer_name and source_layer_name in self.viewer.layers:
                source_layer = self.viewer.layers[source_layer_name]
            else:
                # 使用第一个图像图层
                image_layers = [l for l in self.viewer.layers 
                               if hasattr(l, 'data') and l.data is not None and l.data.ndim >= 2]
                if not image_layers:
                    return ("failed", None)
                source_layer = image_layers[0]
            
            image_stack = np.array(source_layer.data)
            
            # === 旋转 ===
            if action_type == "rotate":
                angle = params.get("angle", 0)
                expand = params.get("expand", True)
                
                if recovery_mode == "review":
                    confirm = self._ask_recovery_confirm(
                        tr("Session Recovery"),
                        f"重放旋转操作:\n\n"
                        f"Source: {source_layer.name}\n"
                        f"Angle: {angle}°\n"
                        f"Expand: {expand}\n\n"
                        f"{tr('Continue with this result?')}"
                    )
                    if confirm == "abort":
                        return ("abort", None)
                    elif confirm == "skip":
                        return ("skipped", None)
                
                # 执行旋转 - 添加进度对话框
                from core.geometry import rotate_image_stack
                from qtpy.QtWidgets import QProgressDialog, QApplication
                from qtpy.QtCore import Qt
                
                # 使用忙碌进度条（无限循环）
                progress = QProgressDialog(tr("Replaying rotation..."), None, 0, 0, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(0)  # 触发显示
                progress.show()
                QApplication.processEvents()  # 确保进度条显示
                
                try:
                    rotated = rotate_image_stack(image_stack, angle, center=None, expand=expand)
                finally:
                    progress.close()
                
                new_name = f"{source_layer.name}_recovered_rotated"
                self.viewer.add_image(rotated, name=new_name, colormap='gray')
                
                # 隐藏源图层
                source_layer.visible = False
                
                if recovery_mode == "review":
                    QMessageBox.information(self, tr("Session Recovery"), 
                        f"✅ 旋转重放成功\nAngle: {angle}°\nResult: {new_name}")
                
                return ("success", new_name)
            
            # === 翻转 ===
            elif action_type == "flip":
                direction = params.get("direction", "horizontal")
                
                if recovery_mode == "review":
                    confirm = self._ask_recovery_confirm(
                        tr("Session Recovery"),
                        f"重放翻转操作:\n\n"
                        f"Source: {source_layer.name}\n"
                        f"Direction: {direction}\n\n"
                        f"{tr('Continue with this result?')}"
                    )
                    if confirm == "abort":
                        return ("abort", None)
                    elif confirm == "skip":
                        return ("skipped", None)
                
                # 执行翻转
                from core.geometry import flip_image_stack
                flipped = flip_image_stack(image_stack, direction)
                
                suffix = "FlipH" if direction == 'horizontal' else "FlipV"
                new_name = f"{source_layer.name}_recovered_{suffix}"
                self.viewer.add_image(flipped, name=new_name, colormap='gray')
                
                # 隐藏源图层
                source_layer.visible = False
                
                if recovery_mode == "review":
                    QMessageBox.information(self, tr("Session Recovery"), 
                        f"✅ 翻转重放成功\nDirection: {direction}\nResult: {new_name}")
                
                return ("success", new_name)
            
            # === 裁剪 ===
            elif action_type == "crop":
                bbox = params.get("bbox")
                if not bbox or len(bbox) != 4:
                    return ("failed", None)
                
                if recovery_mode == "review":
                    confirm = self._ask_recovery_confirm(
                        tr("Session Recovery"),
                        f"重放裁剪操作:\n\n"
                        f"Source: {source_layer.name}\n"
                        f"Bbox: {bbox}\n\n"
                        f"{tr('Continue with this result?')}"
                    )
                    if confirm == "abort":
                        return ("abort", None)
                    elif confirm == "skip":
                        return ("skipped", None)
                
                # 执行裁剪
                from core.geometry import crop_image_stack
                cropped = crop_image_stack(image_stack, tuple(bbox))
                
                new_name = f"{source_layer.name}_recovered_cropped"
                self.viewer.add_image(cropped, name=new_name, colormap='gray')
                
                # 隐藏源图层
                source_layer.visible = False
                
                if recovery_mode == "review":
                    QMessageBox.information(self, tr("Session Recovery"), 
                        f"✅ 裁剪重放成功\nBbox: {bbox}\nResult: {new_name}")
                
                return ("success", new_name)
            
            # === 批量 ROI 恢复 ===
            elif action_type == "update_batch_rois":
                rois = params.get("rois", [])
                if not rois:
                    return ("skipped", None)
                
                if recovery_mode == "review":
                    confirm = self._ask_recovery_confirm(
                        tr("Session Recovery"),
                        f"恢复批量 ROI:\n\n"
                        f"ROI 数量: {len(rois)}\n\n"
                        f"{tr('Continue with this result?')}"
                    )
                    if confirm == "abort":
                        return ("abort", None)
                    elif confirm == "skip":
                        return ("skipped", None)
                
                # 创建或获取 Batch_ROI 图层
                import napari
                if "Batch_ROI" in self.viewer.layers:
                    self.viewer.layers.remove("Batch_ROI")
                
                # 构建 shapes 数据
                shapes_data = []
                labels = []
                frame_ranges = []
                
                for roi in rois:
                    coords = roi.get("coordinates", [])
                    if coords:
                        shapes_data.append(np.array(coords))
                        labels.append(roi.get("label", ""))
                        frame_ranges.append(roi.get("frame_range", ""))
                
                if shapes_data:
                    # 从全局配置读取 ROI 样式
                    box_col = GlobalConfig.get("style_batch_box_color") or 'yellow'
                    width = int(GlobalConfig.get("style_batch_width") or 2)
                    txt_col = GlobalConfig.get("style_batch_text_color") or 'white'
                    font_size = int(GlobalConfig.get("style_batch_font_size") or 12)
                    
                    # 预先生成 frame_infos
                    frame_infos = []
                    for fr in frame_ranges:
                        if fr and str(fr).strip():
                            frame_infos.append(f"[{fr}]")
                        else:
                            frame_infos.append("")
                    
                    # 创建 shapes 时直接传入正确的 features
                    roi_layer = self.viewer.add_shapes(
                        shapes_data,
                        name="Batch_ROI",
                        shape_type='rectangle',
                        edge_color=box_col,
                        face_color=[0, 1, 0, 0.05],
                        edge_width=width,
                        text={
                            'string': '{label}\n{frame_info}', 
                            'size': font_size, 
                            'color': txt_col, 
                            'anchor': 'upper_left', 
                            'translation': [-5, -5]
                        },
                        features={
                            'label': labels,
                            'frame_range': frame_ranges,
                            'frame_info': frame_infos
                        }
                    )
                    
                    if recovery_mode == "review":
                        QMessageBox.information(self, tr("Session Recovery"), 
                            f"✅ 恢复了 {len(shapes_data)} 个 ROI")
                    
                    return ("success", "Batch_ROI")
                
                return ("failed", None)
            
            # 其他 geometry 操作暂时跳过
            return ("skipped", None)
            
        except Exception as e:
            print(f"Geometry replay failed: {e}")
            import traceback
            traceback.print_exc()
            return ("failed", None)
    
    def _replay_drift(self, params: dict, recovery_mode: str, last_result_layer: str = None) -> tuple:
        """重放漂移矫正操作"""
        import numpy as np
        from qtpy.QtWidgets import QProgressDialog
        from qtpy.QtCore import Qt
        
        try:
            # 获取参数
            source_layer_name = params.get("source_layer", "")
            roi_bbox = params.get("roi_bbox")
            template_frame = params.get("template_frame", 0)
            kernel_size = params.get("kernel_size", 11)
            
            if not roi_bbox or len(roi_bbox) != 4:
                return ("failed", None)
            
            # 链式恢复：优先使用上一个操作产生的图层
            if last_result_layer and last_result_layer in self.viewer.layers:
                source_layer = self.viewer.layers[last_result_layer]
            elif source_layer_name and source_layer_name in self.viewer.layers:
                source_layer = self.viewer.layers[source_layer_name]
            else:
                # 使用第一个图像图层
                image_layers = [l for l in self.viewer.layers 
                               if hasattr(l, 'data') and l.data is not None and l.data.ndim == 3]
                if not image_layers:
                    return ("failed", None)
                source_layer = image_layers[0]
            
            image_stack = np.array(source_layer.data)
            
            if image_stack.ndim != 3:
                return ("failed", None)
            
            # Review 模式：先询问用户
            if recovery_mode == "review":
                confirm = self._ask_recovery_confirm(
                    tr("Session Recovery"),
                    f"{tr('Replaying drift correction...')}\n\n"
                    f"Source: {source_layer.name}\n"
                    f"ROI: {roi_bbox}\n"
                    f"Template Frame: {template_frame}\n"
                    f"Kernel: {kernel_size}\n\n"
                    f"{tr('Continue with this result?')}"
                )
                if confirm == "abort":
                    return ("abort", None)
                elif confirm == "skip":
                    return ("skipped", None)
            
            # 显示进度对话框
            from qtpy.QtWidgets import QProgressDialog, QApplication
            progress = QProgressDialog(tr("Replaying drift correction..."), None, 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()
            
            # 导入核心算法
            from core.drift_correction import calculate_drift_curve, apply_drift_correction
            
            try:
                # Step 1: 计算漂移曲线
                progress.setLabelText("Calculating drift curve...")
                
                def progress_cb(c, t):
                    progress.setValue(int(c / t * 50))  # 0-50%
                    QApplication.processEvents()
                
                drifts = calculate_drift_curve(
                    image_stack,
                    tuple(roi_bbox),
                    template_frame,
                    max_workers=8,
                    kernel_size=kernel_size,
                    progress_callback=progress_cb
                )
                
                # Step 2: 应用漂移矫正
                progress.setLabelText("Applying drift correction...")
                
                def progress_cb2(c, t):
                    progress.setValue(50 + int(c / t * 50))  # 50-100%
                    QApplication.processEvents()
                
                corrected = apply_drift_correction(
                    image_stack,
                    drifts,
                    max_workers=8,
                    progress_callback=progress_cb2
                )
                
                # Step 3: 添加结果图层 (这个步骤可能很慢，更新状态)
                progress.setLabelText("Rendering result...")
                progress.setValue(99)
                QApplication.processEvents()
                
                # 添加结果图层
                new_name = f"{source_layer.name}_recovered_corrected"
                new_layer = self.viewer.add_image(
                    corrected, 
                    name=new_name, 
                    colormap='gray',
                    metadata=source_layer.metadata.copy() if hasattr(source_layer, 'metadata') else {}
                )
                
                # Review 模式：显示结果确认
                if recovery_mode == "review":
                    max_drift = np.max(np.abs(drifts), axis=0)
                    QMessageBox.information(
                        self,
                        tr("Session Recovery"),
                        f"✅ {tr('Drift correction replayed successfully')}\n\n"
                        f"Max X drift: {max_drift[0]:.1f} px\n"
                        f"Max Y drift: {max_drift[1]:.1f} px\n"
                        f"Result layer: {new_name}"
                    )
                
                # 隐藏源图层
                source_layer.visible = False
                
                return ("success", new_name)
                
            finally:
                progress.close()
            
        except Exception as e:
            print(f"Drift replay failed: {e}")
            import traceback
            traceback.print_exc()
            return ("failed", None)
    
    def _replay_enhance(self, action_type: str, params: dict, recovery_mode: str = "auto", last_result_layer: str = None) -> tuple:
        """重放图像增强操作"""
        import numpy as np
        
        try:
            # 解析日志结构 - enhance 日志格式为 {"source": ..., "params": {...}}
            if "params" in params:
                # 新格式：嵌套 params
                source_name = params.get("source", "")
                inner_params = params.get("params", {})
            else:
                # 旧格式：直接参数
                source_name = params.get("source_layer", "")
                inner_params = params
            
            # 链式恢复：优先使用上一个操作产生的图层
            if last_result_layer and last_result_layer in self.viewer.layers:
                source_layer = self.viewer.layers[last_result_layer]
            elif source_name and source_name in self.viewer.layers:
                source_layer = self.viewer.layers[source_name]
            else:
                # 使用第一个图像图层
                image_layers = [l for l in self.viewer.layers 
                               if hasattr(l, 'data') and l.data is not None and l.data.ndim >= 2]
                if not image_layers:
                    return ("failed", None)
                source_layer = image_layers[0]
            
            data = np.array(source_layer.data)
            
            # Review 模式：先询问用户
            if recovery_mode == "review":
                title_str = "Filter Enhancement" if action_type == "filter_enhancement" else "Enhancement"
                confirm = self._ask_recovery_confirm(
                    tr("Session Recovery"),
                    f"Replaying {title_str}:\n\n"
                    f"Source: {source_layer.name}\n"
                    f"{tr('Continue with this result?')}"
                )
                if confirm == "abort":
                    return ("abort", None)
                elif confirm == "skip":
                    return ("skipped", None)
            
            if action_type == "filter_enhancement":
                # 解析增强参数 - 注意字段名和 enhance_widget 保持一致
                use_gaussian = inner_params.get("use_gaussian", False)
                sigma = inner_params.get("sigma", 0.8)
                ksize = inner_params.get("ksize", 3)
                use_avg = inner_params.get("use_average", False)
                window = inner_params.get("average_window", 3)
                
                # 使用核心增强函数 - 添加进度对话框
                from core.image_enhance import enhance_image_stack
                from qtpy.QtWidgets import QProgressDialog, QApplication
                from qtpy.QtCore import Qt
                
                progress = QProgressDialog(tr("Replaying enhancement..."), None, 0, 100, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(0)
                progress.show()
                QApplication.processEvents()
                
                def progress_cb(pct):
                    progress.setValue(int(pct))
                    QApplication.processEvents()
                
                try:
                    result = enhance_image_stack(
                        data,
                        use_gaussian=use_gaussian,
                        ksize=ksize,
                        sigma=sigma,
                        use_average=use_avg,
                        average_window=window,
                        progress_callback=progress_cb
                    )
                    
                    # 渲染结果时保持进度条
                    progress.setLabelText("Rendering result...")
                    progress.setValue(99)
                    QApplication.processEvents()
                    
                    # 添加新图层
                    new_name = f"Enh_{source_layer.name}"
                    self.viewer.add_image(result, name=new_name, colormap='gray')
                    
                    # 隐藏源图层
                    source_layer.visible = False
                    
                    return ("success", new_name)
                    
                finally:
                    progress.close()
            
            elif action_type == "contrast_adjustment":
                # 对比度调整 - 日志格式为 {"source": ..., "min": c_min, "max": c_max}
                # 注意：对比度日志不是嵌套格式
                source_from_log = params.get("source", "")
                
                # 如果有指定源图层且存在，使用它
                if source_from_log and source_from_log in self.viewer.layers:
                    source_layer = self.viewer.layers[source_from_log]
                    data = np.array(source_layer.data)
                
                c_min = params.get("min", 0)
                c_max = params.get("max", 255)
                
                # 执行对比度调整 - 和 ContrastBurnThread 相同的逻辑
                data_f = data.astype(np.float32)
                range_width = c_max - c_min
                if range_width < 1e-9:
                    range_width = 1e-9
                
                # 归一化到 [0, 1]
                normalized = (data_f - c_min) / range_width
                normalized = np.clip(normalized, 0, 1)
                
                # 映射到 uint8
                result = (normalized * 255).astype(np.uint8)
                
                new_name = f"Contrast_{source_layer.name}"
                self.viewer.add_image(result, name=new_name, colormap='gray')
                
                # 隐藏源图层
                source_layer.visible = False
                
                return ("success", new_name)
            
        except Exception as e:
            print(f"Enhance replay failed: {e}")
            return ("failed", None)
        
        return ("skipped", None)
    
    def _format_params(self, params: dict) -> str:
        """格式化参数为可读字符串"""
        lines = []
        for key, value in params.items():
            if isinstance(value, (list, tuple)) and len(value) > 10:
                value = f"[{len(value)} items]"
            elif isinstance(value, float):
                value = f"{value:.3f}"
            lines.append(f"  {key}: {value}")
        return "\n".join(lines) if lines else "  (no params)"
    
    def _abandon_session(self):
        """放弃当前会话"""
        if not self.current_session:
            return
        
        reply = QMessageBox.question(self, tr("Abandon Session"), 
            tr("Are you sure you want to abandon this session? This cannot be undone."),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply != QMessageBox.Yes:
            return
        
        log_path = Path(self.current_session.get("_log_path", ""))
        if log_path.exists():
            self._mark_session_status(log_path, "abandoned")
        
        QMessageBox.information(self, tr("Session Recovery"), 
            f"🗑️ {tr('Session abandoned')}")
        
        self._refresh_sessions()
    
    def _mark_session_status(self, log_path: Path, status: str):
        """更新会话状态"""
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["status"] = status
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error updating session status: {e}")
    
    # === 数据加载方法 ===
    
    def _load_dm4_sequence(self, folder_path: str):
        """加载 DM4 序列文件夹"""
        from core.dm4_reader import read_dm4_sequence
        from qtpy.QtWidgets import QProgressDialog
        from qtpy.QtCore import Qt
        
        progress = QProgressDialog(tr("Loading DM4 sequence..."), None, 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        
        def callback(current, total):
            progress.setValue(int(current / total * 100))
        
        try:
            stack, metadata = read_dm4_sequence(folder_path, bit_depth=8, max_workers=8, progress_callback=callback)
            progress.close()
            
            name = f"Original_{Path(folder_path).name}"
            if len(name) > 30:
                name = name[:15] + "..." + name[-10:]
            
            self.viewer.add_image(stack, name=name, metadata=metadata, colormap='gray')
        except Exception as e:
            progress.close()
            raise e
    
    def _load_png_sequence(self, folder_path: str):
        """加载 PNG 序列文件夹"""
        import cv2
        from qtpy.QtWidgets import QProgressDialog
        from qtpy.QtCore import Qt
        
        folder = Path(folder_path)
        png_files = sorted(list(folder.glob("*.png")))
        
        if not png_files:
            raise ValueError(f"No PNG files found in {folder_path}")
        
        progress = QProgressDialog(f"Loading {len(png_files)} PNG files...", None, 0, len(png_files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        
        try:
            frames = []
            for i, f in enumerate(png_files):
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    frames.append(img)
                progress.setValue(i + 1)
            
            progress.close()
            
            if not frames:
                raise ValueError("Could not read any valid images.")
            
            stack = np.array(frames)
            name = f"PNG_{folder.name}"
            if len(name) > 30:
                name = name[:15] + "..." + name[-10:]
            
            self.viewer.add_image(stack, name=name, colormap='gray')
        except Exception as e:
            progress.close()
            raise e
    
    def _load_tiff_stack(self, file_path: str):
        """加载 TIFF Stack 文件"""
        import tifffile
        
        stack = tifffile.imread(file_path)
        
        # 确保是3D数组 (T, H, W)
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        elif stack.ndim == 4:
            stack = stack[..., 0]
        
        name = f"TIFF_{Path(file_path).stem}"
        if len(name) > 30:
            name = name[:15] + "..." + name[-10:]
        
        self.viewer.add_image(stack, name=name, colormap='gray')

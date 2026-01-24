"""
脚本运行器组件 (Script Runner Widget)
允许用户运行内置脚本或自定义脚本（如视频压缩、批处理等）
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QHBoxLayout, QListWidget, QListWidgetItem,
                            QTextEdit, QGroupBox, QScrollArea, QFileDialog,
                            QProgressBar, QMessageBox, QSplitter)
from qtpy.QtCore import Signal, QThread, QProcess, Qt, QSettings
from qtpy.QtGui import QFont
from pathlib import Path
import os
import sys

from widgets.settings_widget import tr
from utils.utils import resource_path


class ScriptRunnerThread(QThread):
    """后台执行脚本的线程"""
    output_received = Signal(str)
    error_received = Signal(str)
    finished_signal = Signal(int)  # exit code
    
    def __init__(self, script_path: str, args: list = None):
        super().__init__()
        self.script_path = script_path
        self.args = args or []
        self.process = None
        self._should_stop = False
    
    def run(self):
        """运行脚本"""
        try:
            import subprocess
            
            # 构建命令
            if self.script_path.endswith('.py'):
                cmd = [sys.executable, self.script_path] + self.args
            elif self.script_path.endswith('.bat') or self.script_path.endswith('.cmd'):
                cmd = ['cmd', '/c', self.script_path] + self.args
            elif self.script_path.endswith('.ps1'):
                cmd = ['powershell', '-ExecutionPolicy', 'Bypass', '-File', self.script_path] + self.args
            else:
                cmd = [self.script_path] + self.args
            
            # 启动进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            # 实时读取输出
            while True:
                if self._should_stop:
                    self.process.terminate()
                    break
                    
                line = self.process.stdout.readline()
                if line:
                    self.output_received.emit(line.rstrip())
                    
                err_line = self.process.stderr.readline()
                if err_line:
                    self.error_received.emit(err_line.rstrip())
                
                if not line and not err_line and self.process.poll() is not None:
                    break
            
            # 读取剩余输出
            remaining_out, remaining_err = self.process.communicate()
            if remaining_out:
                self.output_received.emit(remaining_out.rstrip())
            if remaining_err:
                self.error_received.emit(remaining_err.rstrip())
            
            self.finished_signal.emit(self.process.returncode)
            
        except Exception as e:
            self.error_received.emit(f"Error: {str(e)}")
            self.finished_signal.emit(-1)
    
    def stop(self):
        self._should_stop = True


class ScriptWidget(QWidget):
    """脚本运行器组件"""
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.runner_thread = None
        self.scripts_dir = self._get_scripts_dir()
        self.settings = QSettings("NapariUser", "ScriptRunner")
        
        self._setup_ui()
        self._scan_scripts()
    
    def _get_scripts_dir(self) -> Path:
        """获取脚本目录路径"""
        # 优先使用软件目录下的 scripts 文件夹
        if getattr(sys, 'frozen', False):
            # 打包后的路径
            base_dir = Path(sys._MEIPASS)
        else:
            # 开发环境路径
            base_dir = Path(__file__).parent.parent
        
        scripts_dir = base_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        return scripts_dir
    
    def _setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel(f"<h3>📜 {tr('Script Runner')}</h3>"))
        
        # 使用 Splitter 分割脚本列表和控制台
        splitter = QSplitter(Qt.Vertical)
        
        # === 1. 脚本列表区域 ===
        top_widget = QWidget()
        top_layout = QVBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # 脚本列表
        g_scripts = QGroupBox(tr("Available Scripts"))
        l_scripts = QVBoxLayout()
        
        self.script_list = QListWidget()
        self.script_list.setMinimumHeight(100)
        self.script_list.currentItemChanged.connect(self._on_script_selected)
        l_scripts.addWidget(self.script_list)
        
        # 刷新按钮
        h_refresh = QHBoxLayout()
        btn_refresh = QPushButton(f"🔄 {tr('Refresh')}")
        btn_refresh.clicked.connect(self._scan_scripts)
        h_refresh.addWidget(btn_refresh)
        
        btn_open_folder = QPushButton(f"📂 {tr('Open Scripts Folder')}")
        btn_open_folder.clicked.connect(self._open_scripts_folder)
        h_refresh.addWidget(btn_open_folder)
        
        l_scripts.addLayout(h_refresh)
        g_scripts.setLayout(l_scripts)
        top_layout.addWidget(g_scripts)
        
        # 脚本描述
        g_desc = QGroupBox(tr("Description"))
        l_desc = QVBoxLayout()
        self.lbl_description = QLabel(tr("Select a script to view its description."))
        self.lbl_description.setWordWrap(True)
        self.lbl_description.setStyleSheet("color: #AAA; font-size: 10px;")
        l_desc.addWidget(self.lbl_description)
        g_desc.setLayout(l_desc)
        top_layout.addWidget(g_desc)
        
        # 运行控制
        h_run = QHBoxLayout()
        self.btn_run = QPushButton(f"▶️ {tr('Run Script')}")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        self.btn_run.clicked.connect(self._run_script)
        self.btn_run.setEnabled(False)
        h_run.addWidget(self.btn_run)
        
        self.btn_stop = QPushButton(f"⏹️ {tr('Stop')}")
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; padding: 6px;")
        self.btn_stop.clicked.connect(self._stop_script)
        self.btn_stop.setEnabled(False)
        h_run.addWidget(self.btn_stop)
        
        top_layout.addLayout(h_run)
        top_widget.setLayout(top_layout)
        splitter.addWidget(top_widget)
        
        # === 2. 控制台输出区域 ===
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        g_console = QGroupBox(tr("Console Output"))
        l_console = QVBoxLayout()
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(80)
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #333;
            }
        """)
        l_console.addWidget(self.console)
        
        # 清空按钮
        btn_clear = QPushButton(f"🗑️ {tr('Clear Console')}")
        btn_clear.clicked.connect(self.console.clear)
        l_console.addWidget(btn_clear)
        
        g_console.setLayout(l_console)
        bottom_layout.addWidget(g_console)
        bottom_widget.setLayout(bottom_layout)
        splitter.addWidget(bottom_widget)
        
        # 设置初始比例
        splitter.setSizes([200, 150])
        
        layout.addWidget(splitter)
        
        content_widget.setLayout(layout)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
    
    def _scan_scripts(self):
        """扫描 scripts 目录下的脚本"""
        self.script_list.clear()
        
        if not self.scripts_dir.exists():
            self.scripts_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # 支持的脚本类型
        extensions = ['.py', '.bat', '.cmd', '.ps1']
        
        for f in sorted(self.scripts_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in extensions:
                # 尝试读取脚本描述 (从 docstring 或注释)
                desc = self._get_script_description(f)
                
                item = QListWidgetItem(f"📄 {f.stem}")
                item.setData(Qt.UserRole, str(f))  # 存储完整路径
                item.setData(Qt.UserRole + 1, desc)  # 存储描述
                item.setToolTip(str(f))
                self.script_list.addItem(item)
        
        if self.script_list.count() == 0:
            self.lbl_description.setText(tr("No scripts found. Place .py/.bat/.ps1 files in the 'scripts' folder."))
    
    def _get_script_description(self, script_path: Path) -> str:
        """从脚本文件中提取描述"""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read(2000)  # 只读取开头部分
            
            # Python: 提取 docstring
            if script_path.suffix == '.py':
                # 尝试匹配 """ 或 '''
                import re
                match = re.search(r'^[\s\S]*?["\'{3}]([\s\S]*?)["\'{3}]', content)
                if match:
                    return match.group(1).strip()[:200]
                # 尝试匹配开头的 # 注释
                lines = content.split('\n')
                comments = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('#'):
                        comments.append(line[1:].strip())
                    elif line and not line.startswith(('#', '"', "'")):
                        break
                if comments:
                    return ' '.join(comments[:3])
            
            # Batch/PowerShell: 提取 REM 或 # 注释
            elif script_path.suffix in ['.bat', '.cmd']:
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.upper().startswith('REM '):
                        return line[4:].strip()[:200]
            
            elif script_path.suffix == '.ps1':
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('#') and not line.startswith('#!'):
                        return line[1:].strip()[:200]
            
            return tr("No description available.")
        except:
            return tr("Could not read script.")
    
    def _on_script_selected(self, current, previous):
        """脚本选中时更新描述"""
        if current:
            desc = current.data(Qt.UserRole + 1)
            self.lbl_description.setText(desc if desc else tr("No description available."))
            self.btn_run.setEnabled(True)
        else:
            self.lbl_description.setText(tr("Select a script to view its description."))
            self.btn_run.setEnabled(False)
    
    def _open_scripts_folder(self):
        """打开脚本文件夹"""
        import subprocess
        if os.name == 'nt':
            os.startfile(str(self.scripts_dir))
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(self.scripts_dir)])
        else:
            subprocess.run(['xdg-open', str(self.scripts_dir)])
    
    def _run_script(self):
        """运行选中的脚本"""
        item = self.script_list.currentItem()
        if not item:
            return
        
        script_path = item.data(Qt.UserRole)
        if not script_path or not Path(script_path).exists():
            QMessageBox.warning(self, tr("Error"), tr("Script file not found."))
            return
        
        # 获取上次导出的文件路径作为参数 (可选)
        last_export = self.settings.value("last_export_path", "")
        args = [last_export] if last_export else []
        
        # 清空控制台
        self.console.clear()
        self.console.append(f"[Run] {tr('Running')}: {Path(script_path).name}\n")
        self.console.append(f"[Path] {script_path}\n")
        self.console.append("-" * 50 + "\n")
        
        # 读取脚本内容以进行检测
        try:
             with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read(5000)
        except:
            content = ""

        # 1. 检测是否需要 Napari Viewer (in-process execution)
        if "def main(viewer):" in content or "def main(viewer," in content:
            self._run_in_process_script(script_path)
        # 2. 检测是否为 GUI 脚本 (独立进程)
        elif self._is_gui_script_content(content):
             self._run_gui_script(script_path, args)
        # 3. 普通脚本 (子进程后台)
        else:
            self._run_console_script(script_path, args)

    def _is_gui_script(self, script_path: str) -> bool:
        """(Legacy wrapper)"""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read(5000)
            return self._is_gui_script_content(content)
        except:
            return False

    def _is_gui_script_content(self, content: str) -> bool:
         # 检测常见 GUI 关键字
        gui_keywords = ['tkinter', 'Tk()', 'mainloop', 'PyQt', 'QApplication', 'QWidget']
        return any(kw in content for kw in gui_keywords)

    def _run_in_process_script(self, script_path: str):
        """在当前进程中运行脚本 (阻塞式，用于操作 viewer)"""
        self.console.append("[Info] Executing in-process (Napari Viewer Access)...\n")
        try:
            import importlib.util
            
            # 动态加载模块
            spec = importlib.util.spec_from_file_location("dynamic_script_module", script_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["dynamic_script_module"] = module
                spec.loader.exec_module(module)
                
                # 检查并调用 main(viewer)
                if hasattr(module, 'main'):
                    self.console.append("[Info] Calling main(viewer)...\n")
                    module.main(self.viewer)
                    self.console.append(f"✅ {tr('Script executed successfully.')}\n")
                else:
                    self.console.append("[Error] Function 'main(viewer)' not found in script.\n")
            else:
                 self.console.append("[Error] Failed to load script module.\n")
                 
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            self.console.append(f"[Error] Exception in script:\n{err_msg}\n")
            self.console.append(f"❌ {tr('Script failed.')}\n")
        finally:
             if "dynamic_script_module" in sys.modules:
                 del sys.modules["dynamic_script_module"]
    
    def _run_gui_script(self, script_path: str, args: list):
        """运行 GUI 脚本 (直接启动，不捕获输出)"""
        import subprocess
        
        self.console.append("[Info] GUI script detected, launching in separate window...\n")
        self.console.append("[Info] 检测到 GUI 脚本，正在独立窗口中启动...\n")
        
        try:
            # 构建命令
            if script_path.endswith('.py'):
                cmd = [sys.executable, script_path] + args
            else:
                cmd = [script_path] + args
            
            # 直接启动，不捕获输出，不隐藏窗口
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            self.console.append("[OK] Script launched successfully.\n")
            self.console.append("[OK] 脚本已成功启动。\n")
            self.console.append("-" * 50)
            
        except Exception as e:
            self.console.append(f"[Error] {str(e)}\n")
    
    def _run_console_script(self, script_path: str, args: list):
        """运行控制台脚本 (后台线程，捕获输出)"""
        # 启动线程
        self.runner_thread = ScriptRunnerThread(script_path, args)
        self.runner_thread.output_received.connect(self._on_output)
        self.runner_thread.error_received.connect(self._on_error)
        self.runner_thread.finished_signal.connect(self._on_finished)
        self.runner_thread.start()
        
        # 更新 UI 状态
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.script_list.setEnabled(False)
    
    def _stop_script(self):
        """停止运行中的脚本"""
        if self.runner_thread and self.runner_thread.isRunning():
            self.runner_thread.stop()
            self.console.append(f"\n⏹️ {tr('Script stopped by user.')}")
    
    def _on_output(self, text):
        """处理标准输出"""
        self.console.append(text)
        # 自动滚动到底部
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _on_error(self, text):
        """处理错误输出"""
        self.console.append(f"<span style='color: #FF6B6B;'>{text}</span>")
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _on_finished(self, exit_code):
        """脚本执行完成"""
        self.console.append("-" * 50)
        if exit_code == 0:
            self.console.append(f"✅ {tr('Script finished successfully.')}")
        else:
            self.console.append(f"❌ {tr('Script finished with exit code')}: {exit_code}")
        
        # 恢复 UI 状态
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.script_list.setEnabled(True)
    
    def set_last_export_path(self, path: str):
        """设置上次导出的路径 (供 ExportWidget 调用)"""
        self.settings.setValue("last_export_path", path)

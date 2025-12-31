"""
Session Logger - 集中式日志管理与会话恢复系统

功能:
- 统一记录所有 Widget 操作
- 异步写入防止UI阻塞
- 崩溃保护 (atexit + signal)
- SHA256 完整性校验
- 撤回操作追踪 (追加 undo record)
"""
import json
import hashlib
import datetime
import uuid
import atexit
import signal
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from qtpy.QtCore import QSettings

# --- Numpy JSON Encoder ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, int)): return int(obj)
        elif isinstance(obj, (np.floating, float)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


class SessionLogger:
    """
    会话日志单例类
    
    使用方式:
        from utils.session_logger import SessionLogger
        logger = SessionLogger.get_instance()
        logger.log_action("drift", "apply_correction", {"roi": [...], "kernel": 11})
        logger.log_undo("action_id_xxx")
    """
    _instance: Optional['SessionLogger'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        if SessionLogger._instance is not None:
            raise RuntimeError("Use SessionLogger.get_instance()")
        
        self.session_id = str(uuid.uuid4())[:8]
        self.created_at = datetime.datetime.now().isoformat()
        self.status = "in_progress"
        self.metadata: Dict[str, Any] = {
            "substance": "",
            "dataset_id": "",
            "archive_path": ""
        }
        self.actions: List[Dict[str, Any]] = []
        self._action_counter = 0
        self._save_lock = threading.Lock()
        self._log_path: Optional[Path] = None
        
        # 注册崩溃保护
        atexit.register(self._on_exit)
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except:
            pass  # Windows 下某些信号不可用
    
    @classmethod
    def get_instance(cls) -> 'SessionLogger':
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = SessionLogger()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """重置实例 (用于测试或新会话)"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.status = "abandoned"
                cls._instance._save_sync()
            cls._instance = None
    
    # =========================================================================
    # 路径管理
    # =========================================================================
    def get_log_directory(self) -> Path:
        """
        获取日志存储目录 (按优先级)
        1. 归档路径 (archive_path)
        2. 用户配置的默认路径 (session_log_dir)
        3. 固定目录 (~/.napari_tem/sessions/)
        """
        # 延迟导入避免循环依赖
        from widgets.settings_widget import GlobalConfig
        
        # 1. 归档路径
        archive = QSettings("NapariUser", "Global").value("archive_path", "")
        if archive and Path(archive).exists():
            return Path(archive)
        
        # 2. 用户配置的默认路径
        user_default = GlobalConfig.get("session_log_dir")
        if user_default and Path(user_default).exists():
            return Path(user_default)
        
        # 3. 固定目录
        fallback = Path.home() / ".napari_tem" / "sessions"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    
    def _generate_filename(self) -> str:
        """生成日志文件名: {Date}_{Substance}_{DatasetID}_{SessionID}_session.json"""
        from widgets.settings_widget import GlobalConfig
        
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        substance = self.metadata.get("substance") or GlobalConfig.get("session_substance_default") or "Unknown"
        dataset_id = self.metadata.get("dataset_id") or GlobalConfig.get("session_dataset_default") or "ds0"
        
        # 清理非法字符
        substance = "".join(c for c in substance if c.isalnum() or c in "_-")
        dataset_id = "".join(c for c in dataset_id if c.isalnum() or c in "_-")
        
        # 加入 session_id 防止同一天同样品的会话互相覆盖
        return f"{date_str}_{substance}_{dataset_id}_{self.session_id}_session.json"
    
    def get_log_path(self) -> Path:
        """获取当前日志文件路径"""
        if self._log_path is None:
            self._log_path = self.get_log_directory() / self._generate_filename()
        return self._log_path
    
    def update_log_path(self):
        """更新日志路径 (当元数据变化时调用)"""
        self._log_path = None  # 强制重新生成
    
    # =========================================================================
    # 元数据管理
    # =========================================================================
    def set_metadata(self, substance: str = None, dataset_id: str = None, archive_path: str = None):
        """更新会话元数据"""
        if substance is not None:
            self.metadata["substance"] = substance
        if dataset_id is not None:
            self.metadata["dataset_id"] = dataset_id
        if archive_path is not None:
            self.metadata["archive_path"] = archive_path
        
        self.update_log_path()  # 元数据变化可能影响文件名
    
    # =========================================================================
    # 操作记录
    # =========================================================================
    def log_action(self, widget: str, action: str, params: Dict[str, Any] = None, 
                   result: str = "success") -> str:
        """
        记录一个操作
        
        Args:
            widget: 模块名 (import, drift, geometry, enhance, export)
            action: 操作名 (load_dm4, apply_correction, export_crops, etc.)
            params: 操作参数
            result: 结果状态 (success, failed, cancelled)
        
        Returns:
            action_id: 操作唯一标识 (用于 undo)
        """
        self._action_counter += 1
        action_id = f"{widget}_{action}_{self._action_counter}"
        
        entry = {
            "id": action_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "widget": widget,
            "action": action,
            "params": params or {},
            "result": result
        }
        
        self.actions.append(entry)
        self._save_async()
        
        return action_id
    
    def log_undo(self, target_action_id: str):
        """
        记录撤回操作
        
        Args:
            target_action_id: 被撤回的操作 ID
        """
        self._action_counter += 1
        
        entry = {
            "id": f"undo_{self._action_counter}",
            "timestamp": datetime.datetime.now().isoformat(),
            "widget": "system",
            "action": "undo",
            "params": {"target": target_action_id},
            "result": "success"
        }
        
        self.actions.append(entry)
        self._save_async()
    
    def get_effective_actions(self) -> List[Dict[str, Any]]:
        """
        获取有效操作列表 (排除被撤回的操作)
        
        用于恢复时只重放有效操作
        """
        undone_ids = set()
        for entry in self.actions:
            if entry.get("action") == "undo":
                target = entry.get("params", {}).get("target")
                if target:
                    undone_ids.add(target)
        
        return [a for a in self.actions 
                if a.get("action") != "undo" and a.get("id") not in undone_ids]
    
    # =========================================================================
    # 保存与校验
    # =========================================================================
    def _compute_checksum(self) -> str:
        """计算 actions 的 SHA256 校验和"""
        actions_str = json.dumps(self.actions, sort_keys=True, cls=NumpyEncoder)
        return hashlib.sha256(actions_str.encode('utf-8')).hexdigest()
    
    def _to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "status": self.status,
            "metadata": self.metadata,
            "actions": self.actions,
            "checksum": self._compute_checksum()
        }
    
    def _save_sync(self):
        """同步保存 (阻塞)"""
        with self._save_lock:
            try:
                path = self.get_log_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self._to_dict(), f, indent=2, cls=NumpyEncoder)
                
                # 保存后尝试清理旧日志
                self._cleanup_old_sessions_async()
            except Exception as e:
                print(f"[SessionLogger] Save failed: {e}")
    
    def _save_async(self):
        """异步保存 (非阻塞)"""
        thread = threading.Thread(target=self._save_sync, daemon=True)
        thread.start()
    
    def _cleanup_old_sessions_async(self):
        """异步清理旧会话日志"""
        thread = threading.Thread(target=self._cleanup_old_sessions, daemon=True)
        thread.start()
    
    def _cleanup_old_sessions(self):
        """清理旧会话日志，保留最近 N 个"""
        from widgets.settings_widget import GlobalConfig
        
        max_keep = int(GlobalConfig.get("session_max_keep") or 20)
        
        try:
            log_dir = self.get_log_directory()
            session_files = list(log_dir.glob("*_session.json"))
            
            if len(session_files) <= max_keep:
                return
            
            # 按修改时间排序，最新的在前
            session_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # 删除多余的旧文件
            for old_file in session_files[max_keep:]:
                try:
                    old_file.unlink()
                    print(f"[SessionLogger] Cleaned up old log: {old_file.name}")
                except:
                    pass
        except Exception as e:
            print(f"[SessionLogger] Cleanup failed: {e}")
    
    def _on_exit(self):
        """程序退出时保存"""
        if self.status == "in_progress":
            self.status = "completed"
        self._save_sync()
    
    def _signal_handler(self, signum, frame):
        """信号处理器 (崩溃保护)"""
        self.status = "crashed"
        self._save_sync()
    
    # =========================================================================
    # 恢复相关
    # =========================================================================
    def mark_completed(self):
        """标记会话完成"""
        self.status = "completed"
        self._save_sync()
    
    @staticmethod
    def find_incomplete_sessions(search_dir: Path = None) -> List[Path]:
        """
        查找未完成的会话日志
        
        Args:
            search_dir: 搜索目录，默认使用标准路径
        
        Returns:
            未完成会话的日志文件路径列表
        """
        incomplete = []
        
        # 搜索目录列表
        search_dirs = SessionLogger._get_search_dirs(search_dir)
        
        for dir_path in search_dirs:
            for json_file in dir_path.glob("*_session.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if data.get("status") == "in_progress":
                        incomplete.append(json_file)
                except:
                    pass
        
        return incomplete
    
    @staticmethod
    def find_all_sessions(search_dir: Path = None, limit: int = 50) -> List[Path]:
        """
        查找所有会话日志（不限状态）
        
        Args:
            search_dir: 搜索目录，默认使用标准路径
            limit: 最大返回数量
        
        Returns:
            会话日志文件路径列表，按修改时间倒序
        """
        all_sessions = []
        
        search_dirs = SessionLogger._get_search_dirs(search_dir)
        
        for dir_path in search_dirs:
            for json_file in dir_path.glob("*_session.json"):
                all_sessions.append(json_file)
        
        # 按修改时间排序，最新的在前
        all_sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return all_sessions[:limit]
    
    @staticmethod
    def _get_search_dirs(search_dir: Path = None) -> List[Path]:
        """获取搜索目录列表"""
        search_dirs = []
        if search_dir:
            search_dirs.append(search_dir)
        else:
            # 默认搜索固定目录
            fallback = Path.home() / ".napari_tem" / "sessions"
            if fallback.exists():
                search_dirs.append(fallback)
            
            # 也搜索归档路径
            archive = QSettings("NapariUser", "Global").value("archive_path", "")
            if archive and Path(archive).exists():
                search_dirs.append(Path(archive))
        
        return search_dirs
    
    @staticmethod
    def load_from_file(path: Path) -> Optional[Dict[str, Any]]:
        """加载日志文件并验证"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证校验和
            stored_checksum = data.get("checksum", "")
            actions_str = json.dumps(data.get("actions", []), sort_keys=True)
            computed_checksum = hashlib.sha256(actions_str.encode('utf-8')).hexdigest()
            
            if stored_checksum and stored_checksum != computed_checksum:
                data["_checksum_valid"] = False
                print(f"[SessionLogger] Warning: Checksum mismatch for {path}")
            else:
                data["_checksum_valid"] = True
            
            return data
        except Exception as e:
            print(f"[SessionLogger] Failed to load {path}: {e}")
            return None


# =========================================================================
# 便捷函数
# =========================================================================
def get_logger() -> SessionLogger:
    """获取全局 SessionLogger 实例"""
    return SessionLogger.get_instance()

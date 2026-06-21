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
try:
    from qtpy.QtCore import QSettings
except ImportError:
    from PyQt6.QtCore import QSettings

# --- Numpy JSON Encoder ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, int)): return int(obj)
        elif isinstance(obj, (np.floating, float)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# Phase 2 (2026-05-29): Recovery Layer A — Chapter detection
# Import actions that trigger a new chapter boundary
IMPORT_ACTION_TYPES = {"load_dm4_sequence", "load_png_sequence", "load_tiff_stack"}


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
        self.starred = False  # 收藏状态
        self.label = ""  # 用户自定义标签
        self.metadata: Dict[str, Any] = {
            "substance": "",
            "dataset_id": "",
            "archive_path": ""
        }
        self.layer_aliases: Dict[str, str] = {}
        self.actions: List[Dict[str, Any]] = []
        # Phase 2: chapters segment actions into logical "load-bounded" groups
        self.chapters: List[Dict[str, Any]] = []
        # Phase 6 (2026-05-29): edit_records express user edits (disable/enable/delete/
        # update_params) without mutating original actions array.
        self.edit_records: List[Dict[str, Any]] = []
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

    @classmethod
    def resume_from_file(cls, path: Path) -> 'SessionLogger':
        """
        从已有 session 文件恢复并继续记录。
        新操作会追加到原有 actions 列表中，保留 session_id 和历史。
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.status = "abandoned"
                cls._instance._save_sync()
                cls._instance = None

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            inst = object.__new__(cls)
            inst.session_id = data.get("session_id", str(uuid.uuid4())[:8])
            inst.created_at = data.get("created_at", datetime.datetime.now().isoformat())
            inst.status = "in_progress"
            inst.starred = data.get("starred", False)
            inst.label = data.get("label", "")
            inst.metadata = data.get("metadata", {"substance": "", "dataset_id": "", "archive_path": ""})
            inst.layer_aliases = data.get("layer_aliases", {})
            inst.actions = data.get("actions", [])
            # Phase 2: backward compat — wrap whole legacy actions array into one "current" chapter
            inst.chapters = data.get("chapters", [])
            if not inst.chapters and inst.actions:
                inst.chapters = [{
                    "id": f"ch_{uuid.uuid4().hex[:8]}",
                    "start_action_index": 0,
                    "label": "(legacy session, pre-chapter)",
                    "status": "current",
                    "created_at": inst.created_at,
                }]
            # Phase 6: edit_records (may not exist on older sessions)
            inst.edit_records = data.get("edit_records", [])
            inst._save_lock = threading.Lock()
            inst._save_pending = False
            inst._log_path = path

            max_counter = 0
            for a in inst.actions:
                parts = a.get("id", "").rsplit("_", 1)
                if len(parts) == 2:
                    try:
                        max_counter = max(max_counter, int(parts[-1]))
                    except ValueError:
                        pass
            inst._action_counter = max_counter

            atexit.register(inst._on_exit)
            try:
                signal.signal(signal.SIGTERM, inst._signal_handler)
                signal.signal(signal.SIGINT, inst._signal_handler)
            except:
                pass

            inst.log_action("system", "session_resumed", {
                "resumed_at": datetime.datetime.now().isoformat(),
                "previous_action_count": len(inst.actions) - 1
            })

            cls._instance = inst
            return inst
    
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
        self._save_async()
    
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
            "result": result,
            # Phase 3 (2026-05-29): per-action state for replay filtering.
            # current | historical | disabled (Phase 5 will add trial/committed)
            "state": "current",
        }

        # Phase 2 (2026-05-29): Chapter boundary detection
        # Import actions start a new chapter; previous chapters become historical.
        is_import = (widget == "import" and action in IMPORT_ACTION_TYPES)
        if is_import:
            # Demote any current chapter to historical
            # Phase 3: also flip its actions' state -> historical
            # Phase 5 (2026-05-29): include trials too — they are exploratory AND
            #   belong to the closed-out chapter, so should not replay either.
            for ch in self.chapters:
                if ch.get("status") == "current":
                    ch["status"] = "historical"
                    start = ch.get("start_action_index", 0)
                    for a in self.actions[start:]:
                        if a.get("state") in ("current", "trial"):
                            a["state"] = "historical"
            self.chapters.append({
                "id": f"ch_{uuid.uuid4().hex[:8]}",
                "start_action_index": len(self.actions),
                "label": self._auto_chapter_label(action, params),
                "status": "current",
                "created_at": entry["timestamp"],
            })
        elif not self.chapters:
            # Bootstrap: first non-import action without any chapter yet.
            self.chapters.append({
                "id": f"ch_{uuid.uuid4().hex[:8]}",
                "start_action_index": 0,
                "label": "session_start",
                "status": "current",
                "created_at": entry["timestamp"],
            })

        self.actions.append(entry)
        self._save_async()

        return action_id

    # ----------------------------------------------------------------------
    # Phase 3 (2026-05-29): per-action state management
    # ----------------------------------------------------------------------
    def set_action_state(self, action_id: str, state: str) -> bool:
        """Update the state of a single action (current|historical|disabled|trial|committed).
        Returns True if updated, False if action_id not found."""
        if state not in ("current", "historical", "disabled", "trial", "committed"):
            raise ValueError(f"Invalid state: {state}")
        for a in self.actions:
            if a.get("id") == action_id:
                a["state"] = state
                self._save_async()
                return True
        return False

    # ----------------------------------------------------------------------
    # Phase 5 (2026-05-29): trial / committed state machine
    # ----------------------------------------------------------------------
    def log_action_trial(self, widget: str, action: str, params: Dict[str, Any] = None,
                         result: str = "success") -> str:
        """Same as log_action but writes state='trial'.
        Use for exploratory operations (drift parameter tweaks, enhance previews, etc.)
        that the user is iterating on before committing."""
        aid = self.log_action(widget, action, params, result)
        # Override the state set by log_action
        for a in self.actions:
            if a.get("id") == aid:
                a["state"] = "trial"
                break
        self._save_async()
        return aid

    def log_action_committed(self, widget: str, action: str, params: Dict[str, Any] = None,
                              result: str = "success") -> str:
        """Same as log_action but writes state='committed'.
        Use for deterministic / final operations (import, geometry flip/crop, batch export,
        annotation burn) that are not exploratory."""
        aid = self.log_action(widget, action, params, result)
        for a in self.actions:
            if a.get("id") == aid:
                a["state"] = "committed"
                break
        self._save_async()
        return aid

    def commit_last_action_of(self, widget: str, action_type: str,
                                chapter_id: Optional[str] = None) -> Optional[str]:
        """Mark the most recent (widget+action_type) action in the given chapter
        (default: current chapter) as 'committed'. Useful as a manual or tab-switch hook.
        Returns the action_id that was committed, or None if nothing matched."""
        # Resolve chapter
        target_ch = None
        if chapter_id is None:
            target_ch = self.get_current_chapter()
        else:
            target_ch = next((c for c in self.chapters if c.get("id") == chapter_id), None)
        if target_ch is None:
            return None

        sorted_chs = sorted(self.chapters, key=lambda c: c.get("start_action_index", 0))
        idx_in_sorted = next(
            (i for i, c in enumerate(sorted_chs) if c["id"] == target_ch["id"]), -1
        )
        if idx_in_sorted < 0:
            return None
        start = target_ch.get("start_action_index", 0)
        end = (sorted_chs[idx_in_sorted + 1].get("start_action_index", len(self.actions))
               if idx_in_sorted + 1 < len(sorted_chs) else len(self.actions))

        last_idx = -1
        for i in range(start, end):
            a = self.actions[i]
            if a.get("widget") == widget and a.get("action") == action_type:
                if a.get("state") in ("trial", "current"):
                    last_idx = i
        if last_idx < 0:
            return None
        self.actions[last_idx]["state"] = "committed"
        self._save_async()
        return self.actions[last_idx].get("id")

    # ----------------------------------------------------------------------
    # Phase 6 (2026-05-29): edit_records
    # ----------------------------------------------------------------------
    def add_edit_record(self, edit_type: str, target_action_id: str,
                          new_params: Optional[Dict[str, Any]] = None) -> str:
        """Append an edit_record. Original actions array stays immutable.

        edit_type: 'disable' | 'enable' | 'delete' | 'update_params' | 'reset'
        target_action_id: id of the action this edit applies to
        new_params: dict (required for 'update_params', ignored otherwise)

        'reset' (Issue #6): 丢弃此前对该 action 的所有编辑，回到原始(章节自动判定)
        状态——这是唯一能让一条操作回到 historical(羊皮纸) 的途径，因为 historical
        是分章降级时自动赋的、不是用户可直接选择的状态。
        """
        if edit_type not in ("disable", "enable", "delete", "update_params", "reset"):
            raise ValueError(f"Invalid edit_type: {edit_type}")
        # Verify target exists (warning only — soft fail to keep things resilient)
        if not any(a.get("id") == target_action_id for a in self.actions):
            print(f"[SessionLogger] add_edit_record: target action {target_action_id} not in current session — accepting anyway")
        rec_id = f"edit_{uuid.uuid4().hex[:8]}"
        record = {
            "id": rec_id,
            "edit_type": edit_type,
            "target_action_id": target_action_id,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if edit_type == "update_params":
            record["new_params"] = new_params or {}
        self.edit_records.append(record)
        self._save_async()
        return rec_id

    def compute_effective_actions(self) -> List[Dict[str, Any]]:
        """Apply all edit_records on top of actions and return the resolved replay list.

        Algorithm:
          1. Build effective actions = list of dict copies of self.actions
          2. For each edit_record in chronological order:
             - disable: set effective[idx]['state'] = 'disabled'
             - enable: set effective[idx]['state'] = 'current' (if was disabled/deleted)
             - delete: set effective[idx]['state'] = '_deleted' (internal marker)
             - update_params: shallow-merge new_params into effective[idx]['params']
          3. Run the Phase 5 effective_actions_for_replay logic on the resolved list,
             filtering historical/disabled/_deleted as well.

        NOTE: This is a pure / non-mutating computation. self.actions stays immutable.
        """
        # Step 1: deep-copy actions
        eff = [dict(a) for a in self.actions]
        for a in eff:
            if "params" in a and isinstance(a["params"], dict):
                a["params"] = dict(a["params"])

        # Step 2: apply edit_records in order
        for er in self.edit_records:
            tgt = er.get("target_action_id")
            idx = next((i for i, a in enumerate(eff) if a.get("id") == tgt), -1)
            if idx == -1:
                continue
            etype = er.get("edit_type")
            if etype == "disable":
                eff[idx]["state"] = "disabled"
            elif etype == "enable":
                eff[idx]["state"] = "current"
            elif etype == "delete":
                eff[idx]["state"] = "_deleted"
            elif etype == "reset":
                # Issue #6: 回到原始(章节自动判定)状态并还原参数，丢弃此前对该 action 的
                # 所有编辑——这是让操作能重新回到 historical(羊皮纸) 的唯一途径。
                orig = next((a for a in self.actions if a.get("id") == tgt), None)
                if orig is not None:
                    eff[idx]["state"] = orig.get("state", "current")
                    if isinstance(orig.get("params"), dict):
                        eff[idx]["params"] = dict(orig["params"])
            elif etype == "update_params":
                eff[idx]["params"] = {**eff[idx].get("params", {}), **er.get("new_params", {})}

        # Step 3: run replay-style filter
        undone = set()
        for a in eff:
            if a.get("action") == "undo":
                t = a.get("params", {}).get("target")
                if t:
                    undone.add(t)

        sorted_chs = sorted(self.chapters, key=lambda c: c.get("start_action_index", 0))
        def find_ch(i):
            cur = None
            for ch in sorted_chs:
                if ch.get("start_action_index", 0) <= i:
                    cur = ch.get("id")
                else:
                    break
            return cur

        committed_keys = set()
        last_trial_idx_by_key = {}
        for i, a in enumerate(eff):
            if a.get("widget") == "system" or a.get("action") == "undo":
                continue
            if a.get("id") in undone:
                continue
            st = a.get("state", "current")
            if st in ("historical", "disabled", "_deleted"):
                continue
            key = (find_ch(i), a.get("widget"), a.get("action"))
            if st in ("committed", "current"):
                committed_keys.add(key)
            elif st == "trial":
                last_trial_idx_by_key[key] = i

        out = []
        for i, a in enumerate(eff):
            if a.get("widget") == "system" or a.get("action") == "undo":
                continue
            if a.get("id") in undone:
                continue
            st = a.get("state", "current")
            if st in ("historical", "disabled", "_deleted"):
                continue
            if st in ("committed", "current"):
                out.append(a)
                continue
            key = (find_ch(i), a.get("widget"), a.get("action"))
            if key in committed_keys:
                continue
            if last_trial_idx_by_key.get(key) == i:
                out.append(a)
        return out

    def effective_actions_for_replay(self) -> List[Dict[str, Any]]:
        """Phase 5 (2026-05-29): smart replay filter.

        Returns the action subset a replay engine should execute. Rules:
          - skip widget=system, action=undo, undone targets
          - skip state in {historical, disabled}
          - include all 'committed' and 'current' (legacy) actions
          - for 'trial' actions: per (chapter, widget, action_type) group,
            include only the LAST trial IF no committed/current of that key exists.
            (Earlier trials were exploratory and superseded by the last attempt.)
        """
        undone = set()
        for a in self.actions:
            if a.get("action") == "undo":
                tgt = a.get("params", {}).get("target")
                if tgt:
                    undone.add(tgt)

        # Index actions by (chapter_id, widget, action_type) for trial dedup
        sorted_chs = sorted(self.chapters, key=lambda c: c.get("start_action_index", 0))
        def find_ch(i):
            cur = None
            for ch in sorted_chs:
                if ch.get("start_action_index", 0) <= i:
                    cur = ch.get("id")
                else:
                    break
            return cur

        # Pass 1: collect committed/current keys and last trial index per key
        committed_keys = set()
        last_trial_idx_by_key = {}
        for i, a in enumerate(self.actions):
            if a.get("widget") == "system" or a.get("action") == "undo":
                continue
            if a.get("id") in undone:
                continue
            st = a.get("state", "current")
            if st in ("historical", "disabled"):
                continue
            key = (find_ch(i), a.get("widget"), a.get("action"))
            if st in ("committed", "current"):
                committed_keys.add(key)
            elif st == "trial":
                last_trial_idx_by_key[key] = i

        # Pass 2: build result
        out = []
        for i, a in enumerate(self.actions):
            if a.get("widget") == "system" or a.get("action") == "undo":
                continue
            if a.get("id") in undone:
                continue
            st = a.get("state", "current")
            if st in ("historical", "disabled"):
                continue
            if st in ("committed", "current"):
                out.append(a)
                continue
            # trial: include only if this is the last trial AND no committed/current supersedes
            key = (find_ch(i), a.get("widget"), a.get("action"))
            if key in committed_keys:
                continue
            if last_trial_idx_by_key.get(key) == i:
                out.append(a)
        return out

    # ----------------------------------------------------------------------
    # Phase 2: Chapter management
    # ----------------------------------------------------------------------
    def _auto_chapter_label(self, action: str, params: Optional[Dict[str, Any]]) -> str:
        """Auto-generate a chapter label from an import action."""
        p = params or {}
        src = (p.get("source_path") or p.get("archive_path") or "").rstrip("/\\")
        # Extract last path component for compactness
        if src:
            try:
                src_name = Path(src).name or src
            except Exception:
                src_name = src
        else:
            src_name = ""
        n_frames = p.get("total_frames")
        if n_frames is None:
            n_frames = p.get("frame_count")
        if n_frames is None:
            # Try to infer from frame_indices list length
            fi = p.get("frame_indices")
            if isinstance(fi, list):
                n_frames = len(fi)
        n_str = f" ({n_frames}f)" if n_frames else ""

        if action == "load_dm4_sequence":
            return f"DM4: {src_name}{n_str}" if src_name else f"DM4{n_str}"
        if action == "load_png_sequence":
            return f"PNG: {src_name}{n_str}" if src_name else f"PNG{n_str}"
        if action == "load_tiff_stack":
            return f"TIFF: {src_name}{n_str}" if src_name else f"TIFF{n_str}"
        return action

    def get_chapters(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of the chapters list."""
        return list(self.chapters)

    def get_current_chapter(self) -> Optional[Dict[str, Any]]:
        """Return the chapter currently being appended to, or None if no chapters yet."""
        for ch in reversed(self.chapters):
            if ch.get("status") == "current":
                return ch
        return self.chapters[-1] if self.chapters else None

    def get_chapter_for_action_index(self, idx: int) -> Optional[Dict[str, Any]]:
        """Find the chapter that contains the action at the given index."""
        if not self.chapters:
            return None
        sorted_chs = sorted(self.chapters, key=lambda c: c.get("start_action_index", 0))
        cur = None
        for ch in sorted_chs:
            if ch.get("start_action_index", 0) <= idx:
                cur = ch
            else:
                break
        return cur

    def get_actions_for_chapter(self, chapter_id: str) -> List[Dict[str, Any]]:
        """Return the slice of actions belonging to the chapter with the given id."""
        if not self.chapters:
            return []
        sorted_chs = sorted(self.chapters, key=lambda c: c.get("start_action_index", 0))
        target = None
        target_next_start = len(self.actions)
        for i, ch in enumerate(sorted_chs):
            if ch.get("id") == chapter_id:
                target = ch
                if i + 1 < len(sorted_chs):
                    target_next_start = sorted_chs[i + 1].get("start_action_index", len(self.actions))
                break
        if target is None:
            return []
        return list(self.actions[target.get("start_action_index", 0):target_next_start])
    
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
        """计算 actions 的 SHA256 校验和.

        Phase 3 (2026-05-29): Strip mutable replay-control fields (`state`) from
        the hash so that user toggling current↔historical via Recovery UI does
        NOT invalidate integrity. Old sessions had no `state` field so the
        filter is a no-op for them — same hash as before.
        """
        sanitized = [
            {k: v for k, v in a.items() if k != "state"}
            for a in self.actions
        ]
        actions_str = json.dumps(sanitized, sort_keys=True, cls=NumpyEncoder)
        return hashlib.sha256(actions_str.encode('utf-8')).hexdigest()
    
    def _to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        d = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "status": self.status,
            "starred": self.starred,
            "label": self.label,
            "metadata": self.metadata,
            "actions": self.actions,
            "checksum": self._compute_checksum()
        }
        if self.layer_aliases:
            d["layer_aliases"] = self.layer_aliases
        # Phase 2: persist chapters (omit if empty so old readers don't see the key)
        if self.chapters:
            d["chapters"] = self.chapters
        # Phase 6: persist edit_records (omit if empty)
        if self.edit_records:
            d["edit_records"] = self.edit_records
        return d

    def add_layer_alias(self, original_name: str, new_name: str):
        """Register a layer alias mapping (original → new name after re-import)."""
        self.layer_aliases[original_name] = new_name
        self._save_async()

    def resolve_layer_name(self, target_name: str) -> str:
        """Resolve a layer name through aliases. Returns the mapped name or original."""
        return self.layer_aliases.get(target_name, target_name)
    
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
        """异步保存 (非阻塞, debounced to avoid thread pile-up)"""
        with self._save_lock:
            if getattr(self, '_save_pending', False):
                return
            self._save_pending = True
        thread = threading.Thread(target=self._save_async_worker, daemon=True)
        thread.start()

    def _save_async_worker(self):
        try:
            self._save_sync()
        finally:
            with self._save_lock:
                self._save_pending = False

    def _cleanup_old_sessions_async(self):
        """异步清理旧会话日志"""
        thread = threading.Thread(target=self._cleanup_old_sessions, daemon=True)
        thread.start()
    
    def _cleanup_old_sessions(self):
        """清理旧会话日志，保留最近 N 个 (跳过收藏的和归档路径下的)"""
        from widgets.settings_widget import GlobalConfig
        
        max_keep = int(GlobalConfig.get("session_max_keep") or 20)
        
        try:
            log_dir = self.get_log_directory()
            session_files = list(log_dir.glob("*_session.json"))
            
            # 获取归档路径 (归档路径下的session不清理)
            archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
            archive_dir = Path(archive_path) if archive_path else None
            
            # 分类: 受保护的 vs 普通的
            protected_files = []
            normal_files = []
            
            for f in session_files:
                # 检查是否在归档路径下
                if archive_dir and archive_dir.exists():
                    try:
                        f.resolve().relative_to(archive_dir.resolve())
                        protected_files.append(f)
                        continue
                    except ValueError:
                        pass  # 不在归档路径下
                
                # 检查是否被收藏
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                    if data.get("starred", False):
                        protected_files.append(f)
                        continue
                except:
                    pass
                
                normal_files.append(f)
            
            # 只清理普通文件，且只有当普通文件超出限制时才清理
            if len(normal_files) <= max_keep:
                return
            
            # 按修改时间排序，最新的在前 (带错误处理)
            def safe_mtime(p):
                try:
                    return p.stat().st_mtime
                except:
                    return 0  # 文件可能已被删除，排到最前面不被清理
            
            normal_files.sort(key=safe_mtime, reverse=True)
            
            # 删除多余的旧文件
            for old_file in normal_files[max_keep:]:
                try:
                    if old_file.exists():  # 确保文件存在再删除
                        old_file.unlink()
                        print(f"[SessionLogger] Cleaned up old log: {old_file.name}")
                except Exception as del_e:
                    print(f"[SessionLogger] Failed to delete {old_file.name}: {del_e}")
        except Exception as e:
            print(f"[SessionLogger] Cleanup error: {e}")
    
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
            
            # 也搜索当前归档路径
            archive = QSettings("NapariUser", "Global").value("archive_path", "")
            if archive and Path(archive).exists():
                search_dirs.append(Path(archive))
            
            # 搜索保存的搜索路径列表
            saved_paths = SessionLogger.get_saved_search_paths()
            for p in saved_paths:
                if p not in search_dirs:
                    search_dirs.append(p)
        
        return search_dirs
    
    @staticmethod
    def get_saved_search_paths() -> List[Path]:
        """获取保存的搜索路径列表 (自动过滤无效路径)"""
        settings = QSettings("NapariUser", "Recovery")
        saved = settings.value("saved_search_paths", [])
        if not saved:
            return []
        
        # 返回有效路径
        valid_paths = []
        for p in saved:
            path = Path(p) if isinstance(p, str) else p
            if path.exists():
                valid_paths.append(path)
        
        return valid_paths
    
    @staticmethod
    def get_all_saved_search_paths() -> List[dict]:
        """获取所有保存的搜索路径（包括无效的），返回路径和状态"""
        settings = QSettings("NapariUser", "Recovery")
        saved = settings.value("saved_search_paths", [])
        if not saved:
            return []
        
        result = []
        for p in saved:
            path = Path(p) if isinstance(p, str) else p
            result.append({
                "path": str(path),
                "exists": path.exists()
            })
        
        return result
    
    @staticmethod
    def add_search_path(path: str) -> bool:
        """添加搜索路径到保存列表"""
        settings = QSettings("NapariUser", "Recovery")
        saved = settings.value("saved_search_paths", []) or []
        
        if path not in saved:
            saved.append(path)
            settings.setValue("saved_search_paths", saved)
            return True
        return False
    
    @staticmethod
    def remove_search_path(path: str) -> bool:
        """从保存列表移除搜索路径"""
        settings = QSettings("NapariUser", "Recovery")
        saved = settings.value("saved_search_paths", []) or []
        
        if path in saved:
            saved.remove(path)
            settings.setValue("saved_search_paths", saved)
            return True
        return False
    
    @staticmethod
    def update_search_path(old_path: str, new_path: str) -> bool:
        """更新搜索路径"""
        settings = QSettings("NapariUser", "Recovery")
        saved = settings.value("saved_search_paths", []) or []
        
        if old_path in saved:
            idx = saved.index(old_path)
            saved[idx] = new_path
            settings.setValue("saved_search_paths", saved)
            return True
        return False
    
    @staticmethod
    def cleanup_invalid_paths() -> int:
        """清理所有无效路径，返回清理数量"""
        settings = QSettings("NapariUser", "Recovery")
        saved = settings.value("saved_search_paths", []) or []
        
        valid = [p for p in saved if Path(p).exists()]
        removed_count = len(saved) - len(valid)
        
        if removed_count > 0:
            settings.setValue("saved_search_paths", valid)
        
        return removed_count
    
    @staticmethod
    def try_relocate_path(old_path: str) -> Optional[str]:
        """
        尝试使用 Everything 搜索引擎查找移动后的路径
        
        Args:
            old_path: 失效的旧路径
            
        Returns:
            新路径字符串，未找到返回 None
        """
        try:
            from everytools import EveryTools
            
            # 按文件夹名搜索
            folder_name = Path(old_path).name
            et = EveryTools()
            results = et.search(folder_name)
            
            if not results:
                return None
            
            for result in results[:10]:  # 只检查前10个结果
                result_path = Path(result)
                # 检查是否包含 session 日志文件
                if result_path.is_dir():
                    session_files = list(result_path.glob("*_session.json"))
                    if session_files:
                        return str(result_path)
            
            return None
            
        except ImportError:
            # everytools 未安装
            return None
        except Exception as e:
            print(f"[SessionLogger] Everything search failed: {e}")
            return None
    
    @staticmethod
    def check_everything_available() -> bool:
        """检测 Everything 搜索引擎是否可用"""
        try:
            from everytools import EveryTools
            # 执行一次简单搜索测试
            et = EveryTools()
            results = et.search("test")
            # 如果返回 None 或空列表，说明 Everything 未运行
            return results is not None
        except ImportError:
            print("[SessionLogger] everytools not installed. Run: pip install everytools")
            return False
        except Exception as e:
            print(f"[SessionLogger] Everything check failed: {e}")
            print("[SessionLogger] Make sure Everything is running (portable version needs manual start)")
            return False
    
    @staticmethod
    def get_everything_status() -> tuple:
        """
        获取 Everything 状态详情
        
        Returns:
            (available: bool, message: str)
        """
        try:
            from everytools import EveryTools
            try:
                et = EveryTools()
                results = et.search("napari")  # 使用更可能有结果的关键词
                # everytools 返回 None 表示 Everything 未运行
                # 返回空列表 [] 表示 Everything 运行但无结果（这是正常的）
                if results is None:
                    return (False, "Everything 未运行 (请启动 Everything.exe)")
                else:
                    return (True, "Everything 搜索可用")
            except Exception as e:
                error_str = str(e).lower()
                if "ipc" in error_str or "not running" in error_str:
                    return (False, "Everything 未运行 (请启动 Everything.exe)")
                return (False, f"Everything 错误: {e}")
        except ImportError:
            return (False, "everytools 未安装 (pip install everytools)")
    
    @staticmethod
    def load_from_file(path: Path) -> Optional[Dict[str, Any]]:
        """加载日志文件并验证"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证校验和 (Phase 3: state field stripped to keep checksum stable
            # across user-driven state toggles in Recovery UI)
            stored_checksum = data.get("checksum", "")
            raw_actions = data.get("actions", [])
            sanitized_actions = [
                {k: v for k, v in a.items() if k != "state"}
                for a in raw_actions
            ]
            actions_str = json.dumps(sanitized_actions, sort_keys=True, cls=NumpyEncoder)
            computed_checksum = hashlib.sha256(actions_str.encode('utf-8')).hexdigest()
            
            if stored_checksum and stored_checksum != computed_checksum:
                data["_checksum_valid"] = False
                print(f"[SessionLogger] Warning: Checksum mismatch for {path}")
            else:
                data["_checksum_valid"] = True
            
            # 确保兼容旧版本 (没有 starred/label 字段)
            data.setdefault("starred", False)
            data.setdefault("label", "")
            
            return data
        except Exception as e:
            print(f"[SessionLogger] Failed to load {path}: {e}")
            return None
    
    @staticmethod
    def update_session_file(path: Path, starred: bool = None, label: str = None) -> bool:
        """
        更新已存在session文件的收藏/标签状态
        
        Args:
            path: session文件路径
            starred: 新的收藏状态 (None表示不修改)
            label: 新的标签 (None表示不修改)
        
        Returns:
            是否更新成功
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            modified = False
            if starred is not None and data.get("starred") != starred:
                data["starred"] = starred
                modified = True
            if label is not None and data.get("label") != label:
                data["label"] = label
                modified = True
            
            if modified:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                print(f"[SessionLogger] Updated session: {path.name}")
            
            return True
        except Exception as e:
            print(f"[SessionLogger] Failed to update {path}: {e}")
            return False
    
    @staticmethod
    def get_session_summary(path: Path) -> Optional[Dict[str, Any]]:
        """
        获取session摘要信息 (用于列表展示)
        
        Returns:
            包含 session_id, created_at, status, starred, label, 
            actions_count, metadata, is_archive_session, auto_label 的字典
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            meta = data.get("metadata", {})
            
            # 判断是否为归档路径下的会话
            # 检查当前归档路径
            archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
            is_archive = bool(archive_path and str(path).startswith(str(archive_path)))
            
            # 同时检查保存的搜索路径列表
            if not is_archive:
                saved_paths = QSettings("NapariUser", "Recovery").value("saved_search_paths", []) or []
                path_str = str(path)
                for saved in saved_paths:
                    if path_str.startswith(str(saved)):
                        is_archive = True
                        break
            
            # 自动生成标签 (substance-ds[x])
            auto_label = ""
            substance = meta.get("substance", "")
            ds = meta.get("dataset_id", "")
            if substance or ds:
                auto_label = f"{substance}-{ds}" if substance and ds else (substance or ds)
            
            return {
                "path": path,
                "session_id": data.get("session_id", "unknown"),
                "created_at": data.get("created_at", ""),
                "status": data.get("status", "unknown"),
                "starred": data.get("starred", False),
                "label": data.get("label", ""),
                "actions_count": len(data.get("actions", [])),
                "metadata": meta,
                "file_size": path.stat().st_size,
                "modified_time": path.stat().st_mtime,
                "is_archive_session": is_archive,
                "auto_label": auto_label
            }
        except Exception as e:
            print(f"[SessionLogger] Failed to get summary for {path}: {e}")
            return None


# =========================================================================
# 便捷函数
# =========================================================================
def get_logger() -> SessionLogger:
    """获取全局 SessionLogger 实例"""
    return SessionLogger.get_instance()

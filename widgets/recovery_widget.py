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
                            QFileDialog, QScrollArea, QCheckBox, QDialog,
                            QMenu, QAction, QComboBox)
from qtpy.QtCore import Qt, QTimer
from pathlib import Path
import json
from widgets.settings_widget import GlobalConfig, tr
import numpy as np
import gc
import hashlib
import math
from utils.memory_utils import trim_working_set, create_huge_array, release_memmap_pages
from utils.session_logger import (
    NumpyEncoder, effective_replay_actions, compute_actions_checksum,
    build_logged_output_map, IMAGE_PRODUCER_ACTIONS,
)


# ---------------------------------------------------------------------------
# Param-edit validation (2026-05-29)
# See discussions/2026-05-29_param_edit_validation_design.md
#
# Plan A (全锁): in ActionDetailDialog, lock value cells that are meaningless or
# dangerous to hand-edit, so the user can't change them and be misled (the
# original roi_count 2->3 confusion). Only scalar replay INPUTS stay editable
# (template_frame / angle / kernel_size / bbox / roi_bbox / clip_limit / ...).
# Plan C (范围+sanity): editable values are range/sanity-checked on save and a
# violation hard-blocks the write. Note (finding 1): once roi_count is read-only
# the doc's "roi_count == len(rois)" check is moot — replay reads `rois`, never
# `roi_count`, and neither can be desynced now — so C is range/sanity on inputs.
# ---------------------------------------------------------------------------
# Derived metadata: computed from other fields; editing has no replay effect.
_RO_DERIVED_KEYS = {"roi_count", "proposal_count", "frame_count", "n_frames",
                    "total_frames", "total_source_frames", "source_total_frames"}
# Measured OUTPUTS recorded for info (finding 2); replay re-measures, so a no-op.
_RO_MEASURED_KEYS = {"max_shift", "max_shift_x", "max_shift_y"}
# Layer NAME references; renaming here can make replay fail to find the layer.
_RO_LAYER_KEYS = {"source_layer", "source_layer_name", "view_layer", "data_layer",
                  "target_layer", "layer_name"}
# Large structured objects; editing as raw JSON is error-prone.
_RO_STRUCT_KEYS = {"rois", "frame_info", "frame_indices"}
# Small structured values that ARE meaningful to hand-edit (kept editable).
_EDITABLE_STRUCT_WHITELIST = {"bbox", "roi_bbox"}


def _param_readonly_reason(key, value):
    """Return a human reason string if this param key should be read-only in the
    editor, else None. Centralized + Qt-free so it is unit-testable."""
    if key in _EDITABLE_STRUCT_WHITELIST:
        return None
    if key in _RO_DERIVED_KEYS or key.endswith("_count"):
        return tr("Derived from other fields (e.g. roi_count = len(rois)); "
                  "editing it has no effect on recovery.")
    if key in _RO_MEASURED_KEYS:
        return tr("A measured output; recovery re-measures this value, "
                  "so editing it has no effect.")
    if key in _RO_LAYER_KEYS:
        return tr("Layer reference; renaming it here can make recovery "
                  "fail to find the source layer.")
    if key in _RO_STRUCT_KEYS or isinstance(value, (list, dict)):
        return tr("Structured data — edit it in its own panel, "
                  "not as raw JSON here.")
    return None


def _validate_param_changes(changed, total_frames=None):
    """Range / sanity check edited params (Plan C). Returns a list of error
    strings; an empty list means OK. Pure (Qt-free) for unit testing.
    total_frames=None skips the frame upper-bound check (still rejects < 0)."""

    def _as_int(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float) and math.isfinite(v) and float(v).is_integer():
            return int(v)
        return None

    def _is_num(v):
        return (isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(float(v)))

    errs = []
    for k, v in changed.items():
        if k == "template_frame" or k.endswith("_frame"):
            iv = _as_int(v)
            if iv is None or iv < 0:
                errs.append(tr("%s: must be a non-negative integer frame index (got %r)") % (k, v))
            elif total_frames is not None and iv >= total_frames:
                errs.append(tr("%s: frame %s is out of range [0, %s]") % (k, iv, total_frames - 1))
        elif k in ("bbox", "roi_bbox"):
            if (not isinstance(v, (list, tuple)) or len(v) != 4
                    or not all(_is_num(x) for x in v)):
                errs.append(tr("%s: must be 4 numbers [x1, y1, x2, y2] (got %r)") % (k, v))
            else:
                x1, y1, x2, y2 = v
                if not (x2 > x1 and y2 > y1):
                    errs.append(tr("%s: requires x2 > x1 and y2 > y1 (got %s)") % (k, list(v)))
        elif k == "clip_limit":
            if not _is_num(v) or float(v) <= 0:
                errs.append(tr("%s: must be a positive finite number (got %r)") % (k, v))
        elif k == "kernel_size":
            iv = _as_int(v)
            if iv is None or iv <= 0:
                errs.append(tr("%s: must be a positive integer (got %r)") % (k, v))
        elif k == "angle":
            if not _is_num(v):
                errs.append(tr("%s: must be a finite number (got %r)") % (k, v))
    return errs


def _action_display_class(action, undone_ids):
    """Decide how the Recovery action list should treat an action w.r.t. undo.

    Background (2026-05-30 bug): the drift "Apply → Ctrl+Z → retry another
    kernel" loop logs an `undo` record per reverted preview. Replay already
    excludes undone actions (session_logger.effective_actions_for_replay), but
    `_populate_action_list` was *hiding* every undone action — so repeated drift
    trials collapsed to just the last non-undone one and looked "overwritten".

    A reverted **trial** is still exploration history the user wants to see (and
    may deliberately re-select), so we keep it visible but dimmed + unchecked.
    Non-trial undone actions (e.g. a reverted crop) stay hidden as before.

    Returns: 'undone_trial' | 'hidden' | 'normal'.
    """
    if action.get("id") in undone_ids:
        return "undone_trial" if action.get("state") == "trial" else "hidden"
    return "normal"


# ---------------------------------------------------------------------------
# Replay memory bounding (2026-07-01)
#
# Chained replay used to add_image() every intermediate full stack and never
# remove it (only .visible=False), so drift + 49GB rotate-expand + crop + enhance
# all stayed resident and each add_image read the whole stack for auto-contrast.
# These two pure helpers back the fix: evict each consumed intermediate, and give
# add_image explicit sampled contrast_limits so it skips the whole-array read.
# ---------------------------------------------------------------------------
def plan_replay_evictions(produced_per_step):
    """Given the ordered list of layer names each replay step *produced* (None for
    steps that don't advance the image chain, e.g. ROI/annotation updates), return
    the ordered list of intermediate layer names to evict.

    Rule: when a step produces P, the previously-produced layer it consumed is no
    longer needed and is evicted. The original source (never in this list) and the
    final produced layer are never evicted. Pure + unit-testable."""
    evict = []
    produced = set()
    last = None
    for p in produced_per_step:
        if not p:
            continue
        if last is not None and last in produced and last != p:
            evict.append(last)
        produced.add(p)
        last = p
    return evict


def plan_replay_retention(produced_per_step, mode="lean", keep_set=None):
    """给定各步产出的图层名 (None=不推进链条的步), 返回 (evict, spill):
    随链条推进被消费掉的中间层里, 哪些【移除】(省内存) vs 哪些【保留但写盘】。
    原始源层与最终产出层永不出现在两者中 (最终层保持 live)。

      mode 'lean'   -> evict == plan_replay_evictions(...),  spill == []      (现行为)
      mode 'full'   -> evict == [],  spill == 每个被消费中间层                 (全保留·写盘)
      mode 'custom' -> keep_set 内的写盘, 其余移除 (未选=lean)
    纯函数, 可单测。"""
    consumed = plan_replay_evictions(produced_per_step)  # 所有被消费的中间层
    if mode == "full":
        return [], list(consumed)
    if mode == "custom":
        keep = set(keep_set or ())
        evict = [c for c in consumed if c not in keep]
        spill = [c for c in consumed if c in keep]
        return evict, spill
    # lean (默认)
    return list(consumed), []


def _sample_stack_pixels(data, max_frames=8, pixel_step=97):
    """Sample pixels without paging in a whole memmap: a few evenly-spaced frames,
    strided. Mirrors enhance_widget._sample_array."""
    a = np.asarray(data) if not isinstance(data, np.ndarray) else data
    if a.ndim >= 3 and a.shape[0] > max_frames:
        idx = np.linspace(0, a.shape[0] - 1, max_frames, dtype=int)
        return np.concatenate([np.asarray(a[int(i)]).ravel()[::pixel_step] for i in idx])
    if a.size > 1_000_000:
        return a.ravel()[::pixel_step]
    return a.ravel()


def compute_replay_contrast_limits(stack):
    """Sampled (lo, hi) contrast limits for a possibly-huge/memmap stack, or None
    if there is no meaningful range. Passing these to napari add_image avoids the
    full-array auto-contrast scan that spikes RAM on big rotate-expand results."""
    try:
        sample = _sample_stack_pixels(stack)
        if sample is None or sample.size == 0:
            return None
        lo, hi = np.percentile(sample, [0.5, 99.5])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = float(np.min(sample)), float(np.max(sample))
        if hi <= lo:
            return None
        return (float(lo), float(hi))
    except Exception:
        return None


class ActionDetailDialog(QDialog):
    """Issue #5: Session 操作详情对话框。

    上半部只读：widget / action / 当前状态 / 受哪些 edit_records 影响 / 时间 / 结果。
    下半部可编辑 params（键只读、值可改，按 JSON 解析）。点击 OK 仅返回相对原始
    params 真正变化的键，由调用方写成非破坏性的 update_params edit_record——
    不触碰原始 actions[]，因此不破坏会话校验和。
    """

    def __init__(self, action, session=None, parent=None):
        super().__init__(parent)
        from qtpy.QtWidgets import (QVBoxLayout, QFormLayout, QLabel, QTableWidget,
                                    QTableWidgetItem, QDialogButtonBox, QHeaderView,
                                    QGroupBox)
        self._action = action if isinstance(action, dict) else {}
        self._session = session if isinstance(session, dict) else {}
        params = self._action.get("params", {})
        self._orig_params = params if isinstance(params, dict) else {}

        self.setWindowTitle(tr("Action Details"))
        self.resize(560, 580)
        root = QVBoxLayout(self)

        # --- 只读概要 ---
        info_box = QGroupBox(tr("Overview (read-only)"))
        form = QFormLayout()

        def _ro(v):
            lab = QLabel("" if v is None else str(v))
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lab.setWordWrap(True)
            return lab

        form.addRow(tr("Widget:"), _ro(self._action.get("widget", "")))
        form.addRow(tr("Action:"), _ro(self._action.get("action", "")))
        form.addRow(tr("State:"), _ro(self._action.get("state", "current")))
        form.addRow(tr("Action ID:"), _ro(self._action.get("id", "")))
        form.addRow(tr("Timestamp:"), _ro(self._action.get("timestamp", "")))
        my_id = self._action.get("id")
        affecting = [er.get("edit_type", "?") for er in self._session.get("edit_records", [])
                     if er.get("target_action_id") == my_id]
        if affecting:
            form.addRow(tr("Edits applied:"), _ro(", ".join(affecting)))
        result = self._action.get("result")
        if result not in (None, ""):
            form.addRow(tr("Result:"), _ro(result))
        info_box.setLayout(form)
        root.addWidget(info_box)

        # --- 可编辑参数 ---
        edit_box = QGroupBox(tr("Parameters (editable — saved as a non-destructive edit)"))
        from qtpy.QtWidgets import QVBoxLayout as _VBox
        ebl = _VBox()
        hint = QLabel(tr("Gray rows are read-only (derived / measured / structured / "
                         "layer refs). Edit an editable value then click OK; values "
                         "are parsed as JSON (e.g. 12, 1.5, true, [1,2], \"text\")."))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        ebl.addWidget(hint)

        from qtpy.QtGui import QColor, QBrush
        self._row_keys = list(self._orig_params.keys())
        self._editable_keys = []     # keys whose value cell stays editable (Plan A)
        self._table = QTableWidget(len(self._row_keys), 2)
        self._table.setHorizontalHeaderLabels([tr("Key"), tr("Value")])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for r, k in enumerate(self._row_keys):
            val = self._orig_params[k]
            key_item = QTableWidgetItem(str(k))
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)  # 键不可改
            self._table.setItem(r, 0, key_item)
            val_item = QTableWidgetItem(self._value_to_text(val))
            reason = _param_readonly_reason(k, val)
            if reason is not None:
                # Plan A: 派生/测量/结构化/层名引用 → 只读 + 灰显 + tooltip 说明原因
                val_item.setFlags(val_item.flags() & ~Qt.ItemIsEditable)
                gray = QBrush(QColor("#888888"))
                val_item.setForeground(gray)
                key_item.setForeground(gray)
                val_item.setToolTip(reason)
                key_item.setToolTip(reason)
            else:
                self._editable_keys.append(k)
            self._table.setItem(r, 1, val_item)
        ebl.addWidget(self._table)
        if not self._row_keys:
            none_lab = QLabel(tr("(This action has no editable parameters.)"))
            none_lab.setStyleSheet("color: #888;")
            ebl.addWidget(none_lab)
        edit_box.setLayout(ebl)
        root.addWidget(edit_box)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    @staticmethod
    def _value_to_text(v):
        if isinstance(v, str):
            return v
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    @staticmethod
    def _parse_text(text, original):
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            # 解析失败：原值本是字符串就按文本保存，否则也退回原始文本
            return text

    def accept(self):
        # [Issue #5] 用户点 OK 或按 Enter 时, 正在编辑的单元格可能还没提交,
        # item.text() 仍是旧值 → get_changed_params() 误判为"无改动"。先把焦点移出
        # 表格触发 delegate 提交, 再接受对话框。
        try:
            self._table.setFocus(Qt.OtherFocusReason)
        except Exception:
            pass
        # Plan C: 提交前做范围/sanity 校验; 不通过则弹窗+保持对话框打开(保留用户已输入),
        # 不写 edit_record。校验在对话框内完成, 故 exec_() 返回 Accepted 即保证合法。
        changed = self.get_changed_params()
        errs = self._validate_changes(changed) if changed else []
        if errs:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, tr("Invalid Parameter Edit"),
                tr("These edits were NOT saved. Fix the values and click OK again:")
                + "\n\n• " + "\n• ".join(errs)
            )
            return  # keep dialog open, edits intact
        super().accept()

    def get_changed_params(self):
        """返回相对原始 params 真正发生变化的键值（仅这些键写入 update_params）。
        只读(派生/测量/结构化/层名)键不在 _editable_keys 中, 不参与改动检测。"""
        changed = {}
        for r, k in enumerate(self._row_keys):
            if k not in self._editable_keys:
                continue
            item = self._table.item(r, 1)
            if item is None:
                continue
            new_val = self._parse_text(item.text(), self._orig_params.get(k))
            if new_val != self._orig_params.get(k):
                changed[k] = new_val
        return changed

    def _resolve_total_frames(self):
        """Best-effort frame count for range-checking frame indices: the nearest
        PRECEDING import action's total_frames, else the max found in the session.
        Returns None if no import action recorded a frame count."""
        actions = self._session.get("actions", []) if isinstance(self._session, dict) else []

        def frames_of(a):
            p = a.get("params", {}) or {}
            for kk in ("total_frames", "total_source_frames", "source_total_frames",
                       "frame_count", "n_frames"):
                vv = p.get(kk)
                if isinstance(vv, (int, float)) and not isinstance(vv, bool) and vv > 0:
                    return int(vv)
            return None

        my_id = self._action.get("id")
        my_idx = next((i for i, a in enumerate(actions) if a.get("id") == my_id), None)
        if my_idx is not None:
            for j in range(my_idx, -1, -1):
                a = actions[j]
                if a.get("widget") == "import" or str(a.get("action", "")).startswith("load_"):
                    f = frames_of(a)
                    if f:
                        return f
        best = None
        for a in actions:
            f = frames_of(a)
            if f is not None and (best is None or f > best):
                best = f
        return best

    def _validate_changes(self, changed):
        """Plan C: range/sanity check the edited (editable) params before saving."""
        return _validate_param_changes(changed, self._resolve_total_frames())


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
        self._manual_mode_active = False  # 手动模式激活标志
        self._last_sessions_mtime = 0  # 上次检测到的 session 目录/文件 mtime (含文件级以捕获 in-place 编辑)
        self._last_sessions_count = 0  # 上次检测到的文件数
        # Phase 2 (2026-05-29): chapter selection state
        self._session_chapters_sorted = []     # sorted chapter list for current session
        self._included_chapter_ids = None      # set of chapter ids replay will include; None = no chapters
        self._chapter_checkboxes = []          # list of (chapter_id, QCheckBox)
        self._setup_ui()
        self._refresh_sessions()
        self._setup_shortcuts()

        # 定时器仅做 dirty-check，不全量刷新
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._check_sessions_dirty)
        self._refresh_timer.start(5000)

        # 首次使用时检测 Everything
        QTimer.singleShot(1000, self._check_everything_hint)
    
    def _setup_shortcuts(self):
        """设置键盘快捷键 (使用配置的快捷键)"""
        from qtpy.QtWidgets import QShortcut
        from qtpy.QtGui import QKeySequence
        
        # 使用配置的快捷键
        star_key = str(GlobalConfig.get("shortcut_session_star") or "S")
        label_key = str(GlobalConfig.get("shortcut_session_label") or "L")
        delete_key = str(GlobalConfig.get("shortcut_delete_session") or "Delete")
        
        # 收藏快捷键
        shortcut_star = QShortcut(QKeySequence(star_key), self)
        shortcut_star.activated.connect(self._toggle_session_star)
        
        # 编辑标签快捷键
        shortcut_label = QShortcut(QKeySequence(label_key), self)
        shortcut_label.activated.connect(self._edit_session_label)
        
        # 删除会话快捷键
        shortcut_delete = QShortcut(QKeySequence(delete_key), self)
        shortcut_delete.activated.connect(self._delete_current_session)
    
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
        
        btn_manage_paths = QPushButton(f"🗂️ {tr('Manage Paths')}")
        btn_manage_paths.clicked.connect(self._show_path_manager)
        m_layout.addWidget(btn_manage_paths)
        
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
        
        # === 筛选复选框 ===
        filter_layout = QHBoxLayout()
        self.chk_archive_only = QCheckBox(f"📦 {tr('Archive Only')}")
        self.chk_archive_only.stateChanged.connect(self._refresh_sessions)
        filter_layout.addWidget(self.chk_archive_only)
        
        self.chk_starred_only = QCheckBox(f"⭐ {tr('Starred Only')}")
        self.chk_starred_only.stateChanged.connect(self._refresh_sessions)
        filter_layout.addWidget(self.chk_starred_only)
        filter_layout.addStretch()
        left_layout.addLayout(filter_layout)
        
        self.session_list = QListWidget()
        self.session_list.currentItemChanged.connect(self._on_session_selected)
        self.session_list.itemDoubleClicked.connect(self._on_session_double_clicked)  # 双击编辑标签
        self.session_list.setContextMenuPolicy(Qt.CustomContextMenu)  # 启用右键菜单
        self.session_list.customContextMenuRequested.connect(self._show_session_context_menu)
        self.session_list.setMinimumWidth(220)
        left_layout.addWidget(self.session_list)
        
        # 删除会话按钮
        btn_delete_session = QPushButton(f"🗑️ {tr('Delete Session')}")
        btn_delete_session.clicked.connect(self._delete_current_session)
        btn_delete_session.setToolTip(tr("Delete the selected session from disk"))
        left_layout.addWidget(btn_delete_session)
        
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
            session_data["_is_manual"] = True  # 标记为手动选择
            self.current_session = session_data
            self._manual_mode_active = True  # 激活手动模式
            
            # 阻止列表选择事件覆盖
            self.session_list.blockSignals(True)
            self.session_list.clearSelection()
            self.session_list.blockSignals(False)
            
            # 显示详情
            self._show_session_details()
    
    def _check_sessions_dirty(self):
        """定时检查 session 目录是否有变化, 仅在变化时才刷新列表。

        Phase 2 fix: 用 max(file.stat().st_mtime for file in *.json) 而不是目录 mtime —
        在 Windows 上目录 mtime 不会因目录内文件 in-place 编辑 (save_async 写回) 而变化,
        导致同一文件 starred/label 修改后, 列表无法自动刷新。
        """
        from utils.session_logger import SessionLogger
        search_dirs = SessionLogger._get_search_dirs()
        combined_mtime = 0.0
        combined_count = 0
        for d in search_dirs:
            try:
                # Include directory mtime (catches add/remove)
                combined_mtime = max(combined_mtime, d.stat().st_mtime)
                for f in d.glob("*_session.json"):
                    combined_count += 1
                    try:
                        combined_mtime = max(combined_mtime, f.stat().st_mtime)
                    except OSError:
                        pass
            except OSError:
                continue
        if combined_mtime != self._last_sessions_mtime or combined_count != self._last_sessions_count:
            self._last_sessions_mtime = combined_mtime
            self._last_sessions_count = combined_count
            self._refresh_sessions(preserve_details=True)

    def _refresh_sessions(self, preserve_details=False):
        """刷新会话列表 - 支持筛选、归档优先排序、实时刷新"""
        from utils.session_logger import SessionLogger

        # 保存当前选中的 session_id 以便恢复
        current_session_id = None
        if self.current_session:
            current_session_id = self.current_session.get("session_id")
        current_idx = None
        if self.session_list.currentItem():
            current_idx = self.session_list.currentItem().data(Qt.UserRole)
        
        self.session_list.clear()
        self.sessions = []
        
        # 获取所有会话
        all_sessions = SessionLogger.find_all_sessions(limit=50)
        
        if not all_sessions:
            self._show_empty_state()
            return
        
        # 获取筛选条件
        archive_only = self.chk_archive_only.isChecked()
        starred_only = self.chk_starred_only.isChecked()
        
        # 加载并处理会话数据
        session_summaries = []
        for log_path in all_sessions:
            summary = SessionLogger.get_session_summary(log_path)
            if not summary:
                continue
            
            # 应用筛选条件
            is_archive = summary.get("is_archive_session", False)
            is_starred = summary.get("starred", False)
            
            if archive_only and not is_archive:
                continue
            if starred_only and not is_starred:
                continue
            
            # 加载完整数据
            session_data = SessionLogger.load_from_file(log_path)
            if session_data:
                session_data["_log_path"] = str(log_path)
                session_data["_is_archive"] = is_archive
                session_data["_auto_label"] = summary.get("auto_label", "")
                session_summaries.append((summary, session_data))
        
        # 排序：归档 > 收藏 > 其余(按修改时间倒序), 让常驻会话置顶
        session_summaries.sort(key=lambda x: (
            not x[0].get("is_archive_session", False),
            not x[0].get("starred", False),
            -x[0].get("modified_time", 0),
        ))
        
        for summary, session_data in session_summaries:
            self.sessions.append(session_data)
            
            # 获取显示信息
            status = session_data.get("status", "unknown")
            status_icon = self.STATUS_ICONS.get(status, "❓")
            created = session_data.get("created_at", "")[:16].replace("T", " ")
            
            # 图标：收藏 ⭐ 和归档 📦
            starred = session_data.get("starred", False)
            is_archive = session_data.get("_is_archive", False)
            star_icon = "⭐ " if starred else ""
            archive_icon = "📦 " if is_archive else ""
            
            # 标签：优先用户标签，否则用自动标签
            user_label = session_data.get("label", "")
            auto_label = session_data.get("_auto_label", "")
            display_label = user_label or auto_label
            label_text = f"[{display_label}] " if display_label else ""
            
            # 创建列表项: ⭐ 📦 ✅ [NaCl-ds1] 2024-01-01 12:00
            display = f"{star_icon}{archive_icon}{status_icon} {label_text}{created}"
            
            item = QListWidgetItem(display)
            item.setToolTip(f"Status: {status}\nLabel: {display_label or '(none)'}\nPath: {session_data.get('_log_path', '')}")
            item.setData(Qt.UserRole, len(self.sessions) - 1)
            self.session_list.addItem(item)
        
        # 恢复选中项 (手动模式时跳过，避免覆盖手动选择的会话)
        if not self._manual_mode_active and current_session_id:
            # 按 session_id 恢复，而非 row index
            self.session_list.blockSignals(True)
            restored = False
            for i in range(self.session_list.count()):
                idx = self.session_list.item(i).data(Qt.UserRole)
                if idx is not None and idx < len(self.sessions):
                    if self.sessions[idx].get("session_id") == current_session_id:
                        self.session_list.setCurrentRow(i)
                        restored = True
                        break
            self.session_list.blockSignals(False)
            if not restored and not preserve_details:
                self._show_session_details()
        elif not self._manual_mode_active and not preserve_details:
            pass  # 无之前选中，不做动作

        # 显示空状态如果没有会话
        if self.session_list.count() == 0:
            self._show_empty_state()
    
    def _on_session_selected(self, current, previous):
        """会话选中时显示详情"""
        if current is None:
            return
        
        # 用户主动选择列表项时，解除手动模式
        if self._manual_mode_active:
            self._manual_mode_active = False
        
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

        # === 2.5. Chapter 选择 (Phase 2 - 2026-05-29) ===
        # Surface chapter segmentation so users can choose to include historical loads.
        # Default: only "current" chapters are checked. Replay obeys the active set.
        actions_raw = session.get("actions", [])
        chapters_raw = session.get("chapters", [])
        if not chapters_raw and actions_raw:
            # Backward compat: legacy session — wrap as single current chapter
            chapters_raw = [{
                "id": "_legacy",
                "start_action_index": 0,
                "label": "(legacy session)",
                "status": "current",
            }]
        self._session_chapters_sorted = sorted(
            chapters_raw, key=lambda c: c.get("start_action_index", 0)
        )
        self._included_chapter_ids = {
            ch["id"] for ch in self._session_chapters_sorted
            if ch.get("status") == "current"
        }
        self._chapter_checkboxes = []

        if len(self._session_chapters_sorted) > 1:
            g_chapters = QGroupBox(tr("Chapters (toggle to include historical loads)"))
            ch_layout = QVBoxLayout()
            hint = QLabel(
                f"<i style='color:#888;'>{tr('Each load_dm4 / load_png / load_tiff starts a new chapter. By default only the current chapter is replayed.')}</i>"
            )
            hint.setWordWrap(True)
            ch_layout.addWidget(hint)
            for ch in self._session_chapters_sorted:
                cb = QCheckBox(self._format_chapter_label(ch, actions_raw))
                is_current = ch.get("status") == "current"
                cb.setChecked(is_current)
                if not is_current:
                    cb.setStyleSheet("color: #888;")
                cb.toggled.connect(
                    lambda v, cid=ch["id"]: self._on_chapter_toggled(cid, v)
                )
                ch_layout.addWidget(cb)
                self._chapter_checkboxes.append((ch["id"], cb))
            g_chapters.setLayout(ch_layout)
            self.details_layout.addWidget(g_chapters)

        # === 3. 可恢复操作列表 ===
        g_actions = QGroupBox(tr("Recoverable Actions"))
        a_layout = QVBoxLayout()

        self.action_list = QListWidget()
        self.action_list.setSelectionMode(QListWidget.MultiSelection)
        self.action_list.setMaximumHeight(200)
        # Phase 6 (2026-05-29): right-click menu for edit_records (disable/enable/delete)
        self.action_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.action_list.customContextMenuRequested.connect(self._on_action_context_menu)
        # Issue #5: 双击操作 → 打开详情/参数编辑对话框
        self.action_list.itemDoubleClicked.connect(self._on_action_double_clicked)

        # Phase 6: Apply existing edit_records to the in-memory session before render
        # so visuals reflect prior user decisions immediately.
        self._apply_edit_records_inplace(session)

        self._populate_action_list(session)

        a_layout.addWidget(self.action_list)
        
        # 全选/取消全选 + Phase 5: Commit selected trials
        h_select = QHBoxLayout()
        btn_select_all = QPushButton(tr("Select All"))
        btn_select_all.clicked.connect(lambda: self._select_all_actions(True))
        btn_deselect_all = QPushButton(tr("Deselect All"))
        btn_deselect_all.clicked.connect(lambda: self._select_all_actions(False))
        h_select.addWidget(btn_select_all)
        h_select.addWidget(btn_deselect_all)
        h_select.addStretch()
        # Phase 5 (2026-05-29): Commit selected trials to committed
        btn_commit = QPushButton(f"✅ {tr('Commit Selected Trials')}")
        btn_commit.setToolTip(tr(
            "Bookkeeping only: marks the selected trial as the CHOSEN version for "
            "future replay (writes state in the session JSON). It does NOT re-apply "
            "anything to the image — the result was already produced live when you "
            "clicked Apply. Nothing on the canvas changes."
        ))
        btn_commit.clicked.connect(self._commit_selected_trials)
        h_select.addWidget(btn_commit)
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
        
        # === 内存保留模式 (F2, 与 auto/review 正交): lean / full / custom ===
        h_ret = QHBoxLayout()
        h_ret.addWidget(QLabel(tr("Memory Retention:")))
        self.retention_combo = QComboBox()
        self.retention_combo.addItem(tr("Lean (evict intermediates)"), "lean")
        self.retention_combo.addItem(tr("Full (keep all, spill to disk)"), "full")
        self.retention_combo.addItem(tr("Custom (choose steps to keep)"), "custom")
        _cur_ret = GlobalConfig.get("session_recovery_retention") or "lean"
        _ret_idx = self.retention_combo.findData(_cur_ret)
        self.retention_combo.setCurrentIndex(_ret_idx if _ret_idx >= 0 else 0)
        self.retention_combo.currentIndexChanged.connect(
            lambda _i: GlobalConfig.set("session_recovery_retention", self.retention_combo.currentData()))
        h_ret.addWidget(self.retention_combo)
        h_ret.addStretch()
        self.details_layout.addLayout(h_ret)

        # === 5. 恢复/放弃按钮 ===
        h_btns = QHBoxLayout()

        btn_abandon = QPushButton(f"🗑️ {tr('Abandon Session')}")
        btn_abandon.setStyleSheet("background-color: #8B0000;")
        btn_abandon.clicked.connect(self._abandon_session)

        btn_continue = QPushButton(f"▶️ {tr('Continue Session')}")
        btn_continue.setStyleSheet("background-color: #1565C0; font-weight: bold; padding: 8px 16px;")
        btn_continue.setToolTip(tr("Resume recording to this session (append new actions)"))
        btn_continue.clicked.connect(self._continue_session)

        btn_recover = QPushButton(f"✅ {tr('Start Recovery')}")
        btn_recover.setStyleSheet("background-color: #2E7D32; font-weight: bold; padding: 8px 16px;")
        btn_recover.clicked.connect(self._recover_selected)

        h_btns.addWidget(btn_abandon)
        h_btns.addStretch()
        h_btns.addWidget(btn_continue)
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
    
    def _edit_session_label(self, session=None):
        """编辑当前会话的标签"""
        from utils.session_logger import SessionLogger
        from qtpy.QtWidgets import QInputDialog
        
        # 兼容信号(bool)和直接调用
        target_session = session if isinstance(session, dict) else self.current_session
        
        if not target_session:
            return
        
        log_path = target_session.get("_log_path")
        if not log_path:
            return
        
        current_label = target_session.get("label", "")
        
        new_label, ok = QInputDialog.getText(
            self, tr("Session Label"), 
            tr("Enter label for this session:"),
            text=current_label
        )
        
        if ok:
            if SessionLogger.update_session_file(Path(log_path), label=new_label):
                # 更新本地状态
                target_session["label"] = new_label
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
            self._edit_session_label(self.current_session)
    
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
        """获取有效操作 (排除 undo 和 system, 按 chapter 过滤 - Phase 2)。

        chapter 过滤规则:
        - 如果 session 有 chapters 字段: 只返回 self._included_chapter_ids 集合内的 action
        - 如果 session 没有 chapters 字段 (旧 session): 视为全部 current, 全部返回
        - 如果 self._included_chapter_ids 未初始化 (调用 _get_data_sources 等场景): 不做 chapter 过滤
        """
        actions = session_data.get("actions", [])
        chapters = session_data.get("chapters", [])

        # Backward compat: legacy session with actions but no chapters
        if not chapters and actions:
            chapters = [{
                "id": "_legacy",
                "start_action_index": 0,
                "status": "current",
            }]

        # Use the user-toggleable inclusion set if available; else default to current
        included = None
        if hasattr(self, '_included_chapter_ids') and self._included_chapter_ids is not None:
            included = self._included_chapter_ids
        elif chapters:
            included = {ch["id"] for ch in chapters if ch.get("status") == "current"}

        sorted_chs = sorted(chapters, key=lambda c: c.get("start_action_index", 0))

        def find_chapter_id(action_idx):
            cur = None
            for ch in sorted_chs:
                if ch.get("start_action_index", 0) <= action_idx:
                    cur = ch["id"]
                else:
                    break
            return cur

        undone_ids = set()
        for action in actions:
            if action.get("action") == "undo":
                target = action.get("params", {}).get("target")
                if target:
                    undone_ids.add(target)

        effective = []
        for i, action in enumerate(actions):
            action_id = action.get("id", "")
            action_type = action.get("action", "")
            widget = action.get("widget", "")

            if action_type == "undo" or widget == "system":
                continue
            if action_id in undone_ids:
                continue
            if included is not None and find_chapter_id(i) not in included:
                continue

            effective.append(action)

        return effective

    # ------------------------------------------------------------------
    # Phase 2 (2026-05-29): Chapter UI helpers
    # ------------------------------------------------------------------
    def _format_chapter_label(self, ch, actions):
        """Build the display label for a chapter checkbox."""
        label = ch.get("label", "(no label)")
        status = ch.get("status", "current")
        start = ch.get("start_action_index", 0)
        # Find end of chapter (start of next chapter or len(actions))
        idx_in_sorted = next(
            (i for i, c in enumerate(self._session_chapters_sorted) if c["id"] == ch["id"]),
            -1
        )
        if 0 <= idx_in_sorted < len(self._session_chapters_sorted) - 1:
            end = self._session_chapters_sorted[idx_in_sorted + 1].get(
                "start_action_index", len(actions)
            )
        else:
            end = len(actions)
        n_actions = max(0, end - start)
        status_text = tr("current") if status == "current" else tr("historical")
        return f"[{status_text}] {label} — {n_actions} {tr('actions')}"

    def _on_chapter_toggled(self, chapter_id, included):
        if not hasattr(self, '_included_chapter_ids') or self._included_chapter_ids is None:
            self._included_chapter_ids = set()
        if included:
            self._included_chapter_ids.add(chapter_id)
        else:
            self._included_chapter_ids.discard(chapter_id)
        # Refresh the action list to reflect new chapter inclusion
        if self.current_session is not None:
            self._populate_action_list(self.current_session)

    def _populate_action_list(self, session):
        """Fill (or refill) self.action_list based on chapter inclusion + action state.

        Phase 3 (2026-05-29): render per-action state:
          - current   → normal text, selected by default (replay)
          - historical→ gray text, NOT selected (user can manually select)
          - disabled  → red strikethrough, NOT selected (Phase 6 will use; we render now)
        Selection state is transient; we do NOT mutate the session JSON here.
        """
        from qtpy.QtGui import QFont, QColor, QBrush

        if not hasattr(self, 'action_list') or self.action_list is None:
            return
        self.action_list.clear()
        # Phase 3: bypass chapter filter for state-only rendering. We want to show
        # historical chapter actions too so the user can see and selectively
        # include them. The _get_effective_actions chapter filter is honored only
        # at "Start Recovery" time (which reads action_list selection).
        all_actions = session.get("actions", [])

        # [Issue #5] 哪些 action 被 update_params 编辑过 → 列表项加 ✏️ 标记, 让"改了参数"看得见
        edited_ids = {er.get("target_action_id") for er in session.get("edit_records", [])
                      if er.get("edit_type") == "update_params"}

        # 默认勾选必须与真正会重放的"有效集"一致 (否则出现: 两条 enhance 都勾上但只
        # 跑最后一条 / 点全选反而取消被取代的行 / 甚至勾了却"恢复 0 个")。用同一个
        # effective_replay_actions 作为唯一真相; 失败则回退到旧的按 state 勾选。
        try:
            eff_ids = {a.get("id") for a in effective_replay_actions(
                all_actions, session.get("chapters", []), session.get("edit_records", []))}
        except Exception:
            eff_ids = None

        # undone filter
        undone_ids = set()
        for a in all_actions:
            if a.get("action") == "undo":
                tgt = a.get("params", {}).get("target")
                if tgt:
                    undone_ids.add(tgt)

        # Chapter id lookup helper (for highlighting chapters in title)
        chapters = session.get("chapters", [])
        sorted_chs = sorted(chapters, key=lambda c: c.get("start_action_index", 0))
        def find_chapter_id(action_idx):
            cur = None
            for ch in sorted_chs:
                if ch.get("start_action_index", 0) <= action_idx:
                    cur = ch["id"]
                else:
                    break
            return cur

        included = self._included_chapter_ids if self._included_chapter_ids is not None else None

        shown = 0
        for i, action in enumerate(all_actions):
            widget_name = action.get("widget", "unknown")
            action_name = action.get("action", "unknown")
            # 隐藏纯记账动作 (system/undo/会话 resume/restore/recovery 链接) —— 它们不是
            # 管线操作, 重放 session_restored 还会递归去加载源会话 (见 _is_meta_action)。
            if (widget_name in ("system", "recovery")
                    or action_name in ("undo", "session_resumed",
                                        "session_restored", "session_start")):
                continue
            disp_class = _action_display_class(action, undone_ids)
            if disp_class == "hidden":
                continue
            is_undone_trial = (disp_class == "undone_trial")
            # Skip if outside selected chapters (chapter checkbox controls this)
            if included is not None and find_chapter_id(i) not in included:
                continue

            timestamp = action.get("timestamp", "")[:19]
            state = action.get("state", "current")

            # Phase 5/6 (2026-05-29): also render trial / committed / _deleted visual classes
            prefix = ""
            if state == "historical":
                prefix = "📜 "
            elif state == "disabled":
                prefix = "🚫 "
            elif state == "_deleted":
                prefix = "🗑️ "
            elif state == "trial":
                prefix = "🧪 "
            elif state == "committed":
                prefix = "✅ "
            edit_marker = " ✏️" if action.get("id") in edited_ids else ""
            undone_marker = " ↩️" if is_undone_trial else ""
            display_text = f"{prefix}[{widget_name}] {action_name} - {timestamp}{edit_marker}{undone_marker}"

            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, action)
            if edit_marker:
                item.setToolTip(tr("Parameters edited (double-click to view/edit)."))

            if state == "historical":
                item.setForeground(QBrush(QColor("#888888")))
                item.setSelected(False)
            elif state == "disabled":
                font = QFont()
                font.setStrikeOut(True)
                item.setFont(font)
                item.setForeground(QBrush(QColor("#D32F2F")))
                item.setSelected(False)
            elif state == "_deleted":
                font = QFont()
                font.setStrikeOut(True)
                item.setFont(font)
                item.setForeground(QBrush(QColor("#B71C1C")))
                item.setSelected(False)
            elif state == "trial":
                if is_undone_trial:
                    # Reverted (Ctrl+Z) trial kept as exploration history: dim + italic,
                    # never auto-selected (replay excludes undone). Still selectable so
                    # the user can deliberately re-include it.
                    f = QFont()
                    f.setItalic(True)
                    item.setFont(f)
                    item.setForeground(QBrush(QColor("#9E9E9E")))
                    item.setSelected(False)
                    item.setToolTip(tr("Reverted with Ctrl+Z (kept as history); "
                                       "not replayed unless you select it."))
                else:
                    # Light gray + selected only if it's in the effective replay set
                    # (falls back to the last-non-undone-trial heuristic if eff unavailable)
                    item.setForeground(QBrush(QColor("#AAAAAA")))
                    if eff_ids is not None:
                        item.setSelected(action.get("id") in eff_ids)
                    else:
                        item.setSelected(self._is_last_trial_of_key(all_actions, i, undone_ids))
            elif state == "committed":
                # Slight emphasis but readable
                item.setForeground(QBrush(QColor("#4CAF50")))
                item.setSelected(eff_ids is None or action.get("id") in eff_ids)
            else:  # current (legacy/default)
                item.setSelected(eff_ids is None or action.get("id") in eff_ids)

            self.action_list.addItem(item)
            shown += 1

        if shown == 0:
            self.action_list.addItem(
                QListWidgetItem(tr("(No actions in selected chapters)"))
            )

    # Phase 6 (2026-05-29): right-click action menu for edit_records.
    def _on_action_context_menu(self, pos):
        from qtpy.QtWidgets import QMenu
        item = self.action_list.itemAt(pos)
        if item is None:
            return
        payload = item.data(Qt.UserRole)
        if not isinstance(payload, dict):
            return
        menu = QMenu(self.action_list)
        act_detail = menu.addAction(f"🔍 {tr('View / Edit Details')}")
        menu.addSeparator()
        act_enable = menu.addAction(f"✅ {tr('Enable (clear disable/delete)')}")
        act_disable = menu.addAction(f"🚫 {tr('Disable')}")
        # Issue #6: 回到原始(章节自动判定)状态——唯一能让操作重新回到 historical(羊皮纸) 的入口
        act_reset = menu.addAction(f"🔄 {tr('Reset to original state')}")
        menu.addSeparator()
        act_delete = menu.addAction(f"🗑️ {tr('Delete')}")
        chosen = menu.exec_(self.action_list.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == act_detail:
            self._open_action_detail(payload)
        elif chosen == act_enable:
            self._add_edit_record_to_session("enable", payload.get("id"))
        elif chosen == act_disable:
            self._add_edit_record_to_session("disable", payload.get("id"))
        elif chosen == act_reset:
            self._add_edit_record_to_session("reset", payload.get("id"))
        elif chosen == act_delete:
            self._add_edit_record_to_session("delete", payload.get("id"))

    def _on_action_double_clicked(self, item):
        """Issue #5: 双击一条操作 → 打开详情/参数编辑对话框。"""
        if item is None:
            return
        payload = item.data(Qt.UserRole)
        if not isinstance(payload, dict):
            return
        self._open_action_detail(payload)

    def _open_action_detail(self, action):
        """Issue #5: 详情对话框——上半只读(widget/action/state/时间/结果)，下半可编辑参数。
        参数改动写成非破坏性的 update_params edit_record，不触碰原始 actions[] 与校验和。"""
        if not isinstance(action, dict):
            return
        dlg = ActionDetailDialog(action, session=self.current_session, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            changed = dlg.get_changed_params()
            if changed:
                self._add_edit_record_to_session(
                    "update_params", action.get("id"), new_params=changed
                )
                # [Issue #5] 明确反馈, 否则用户"看不出改了什么"
                summary = ", ".join(f"{k}={v}" for k, v in changed.items())
                QMessageBox.information(
                    self, tr("Parameter Edit"),
                    tr("Saved %d change(s): %s\n\nThe action now shows ✏️ in the list; "
                       "recovery will replay with the new parameters.") % (len(changed), summary)
                )
            else:
                QMessageBox.information(
                    self, tr("Parameter Edit"), tr("No parameter changes detected.")
                )

    def _add_edit_record_to_session(self, edit_type, target_action_id, new_params=None):
        """Append an edit_record to the session JSON and refresh UI.

        Edit_records do NOT mutate actions[], so checksum stays valid.
        new_params: 仅 edit_type=='update_params' 时使用（被编辑的参数键值）。
        """
        import uuid as _uuid

        if not self.current_session or not target_action_id:
            return

        # [Issue #5] 若编辑的是当前活动(正在记录)的会话, 直接写文件会被实时 logger 的下次保存覆盖
        # (这正是 edit_records 落盘后又变空的原因)。这种情况改走 logger.add_edit_record 持久化。
        # 用 _instance 而非 get_logger(), 避免在 recovery 里误创建一个新 logger。
        try:
            from utils.session_logger import SessionLogger
            live = SessionLogger._instance
        except Exception:
            live = None
        if (live is not None and getattr(live, 'session_id', None)
                and live.session_id == self.current_session.get('session_id')):
            try:
                if edit_type == "update_params":
                    live.add_edit_record(edit_type, target_action_id, new_params=new_params or {})
                else:
                    live.add_edit_record(edit_type, target_action_id)
            except Exception as e:
                QMessageBox.critical(self, tr("Edit Record"), str(e))
                return
            self.current_session["edit_records"] = list(getattr(live, 'edit_records', []))
            self._apply_edit_records_inplace(self.current_session)
            self._populate_action_list(self.current_session)
            try:
                self.action_list.viewport().update()
            except Exception:
                pass
            return

        log_path_str = self.current_session.get("_log_path", "")
        if not log_path_str:
            return
        log_path = Path(log_path_str)
        if not log_path.exists():
            QMessageBox.warning(self, tr("Edit Record"), tr("Session file not found."))
            return
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, tr("Edit Record"), str(e))
            return
        recs = data.get("edit_records", [])
        rec = {
            "id": f"edit_{_uuid.uuid4().hex[:8]}",
            "edit_type": edit_type,
            "target_action_id": target_action_id,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
        if edit_type == "update_params":
            rec["new_params"] = new_params or {}
        recs.append(rec)
        data["edit_records"] = recs
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, tr("Edit Record"), str(e))
            return
        # Apply same change to in-memory session
        self.current_session["edit_records"] = recs
        # Visual update: re-derive effective state by applying edit_records locally
        self._apply_edit_records_inplace(self.current_session)
        self._populate_action_list(self.current_session)
        self.action_list.viewport().update()

    def _apply_edit_records_inplace(self, session_data):
        """For UI rendering: mutate a COPY of actions' state field according to edit_records.

        This does NOT touch session JSON's actions array on disk (we only write edit_records).
        It only updates the in-memory dicts used by _populate_action_list so the
        right styling appears immediately after the user right-clicks.
        """
        actions = session_data.get("actions", [])
        # Issue #6: 捕获一次原始状态/参数（首次应用前 actions[] 即磁盘上的章节原始态），
        # 供 reset 回退使用；这些 _orig_* 键只存在于内存，不会写回磁盘。
        for a in actions:
            if "_orig_state" not in a:
                a["_orig_state"] = a.get("state", "current")
            if "_orig_params" not in a and isinstance(a.get("params"), dict):
                a["_orig_params"] = dict(a["params"])
        # 每次都先复位到原始再按序重放全部 edit_records，避免多次调用时状态/参数累积串味
        for a in actions:
            a["state"] = a.get("_orig_state", a.get("state", "current"))
            if "_orig_params" in a:
                a["params"] = dict(a["_orig_params"])

        recs = session_data.get("edit_records", [])
        if not recs:
            return
        # Index actions for fast lookup
        actions_by_id = {a.get("id"): a for a in actions}
        for er in recs:
            tgt = er.get("target_action_id")
            a = actions_by_id.get(tgt)
            if a is None:
                continue
            etype = er.get("edit_type")
            if etype == "disable":
                a["state"] = "disabled"
            elif etype == "enable":
                a["state"] = "current"
            elif etype == "delete":
                a["state"] = "_deleted"
            elif etype == "reset":
                # Issue #6: 回到原始状态(含 historical)并还原参数
                a["state"] = a.get("_orig_state", "current")
                if "_orig_params" in a:
                    a["params"] = dict(a["_orig_params"])
            elif etype == "update_params":
                # Issue #5: 把编辑后的参数并入显示，使详情对话框再次打开时反映改动
                if isinstance(a.get("params"), dict):
                    a["params"] = {**a["params"], **er.get("new_params", {})}
                else:
                    a["params"] = dict(er.get("new_params", {}))

    # Phase 5 (2026-05-29): Commit selected trial actions to "committed" in JSON.
    def _commit_selected_trials(self):
        """Walk action_list. For each selected item whose state is 'trial', flip to
        'committed' and write back to the session JSON file in place."""
        if not self.current_session:
            return
        log_path_str = self.current_session.get("_log_path", "")
        if not log_path_str:
            QMessageBox.warning(self, tr("Commit Trials"), tr("Session log path missing."))
            return
        log_path = Path(log_path_str)
        if not log_path.exists():
            QMessageBox.warning(self, tr("Commit Trials"), tr("Session file not found."))
            return

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, tr("Commit Trials"), str(e))
            return

        targets = set()
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            if item is None or not item.isSelected():
                continue
            payload = item.data(Qt.UserRole)
            if not isinstance(payload, dict):
                continue
            if payload.get("state") == "trial":
                aid = payload.get("id")
                if aid:
                    targets.add(aid)

        if not targets:
            QMessageBox.information(
                self, tr("Commit Trials"),
                tr("No trial actions selected. Select trial rows in the list first.")
            )
            return

        changed = 0
        for a in data.get("actions", []):
            if a.get("id") in targets and a.get("state") == "trial":
                a["state"] = "committed"
                changed += 1

        if changed == 0:
            QMessageBox.information(self, tr("Commit Trials"), tr("Nothing to commit."))
            return

        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, tr("Commit Trials"), str(e))
            return

        # Refresh detail view (re-load session_data via _refresh_sessions)
        QMessageBox.information(
            self, tr("Commit Trials"),
            (tr("Committed %d trial action(s).") % changed) + "\n\n" +
            tr("This only records which version replay should use; it does not "
               "re-apply anything to the image (the canvas is unchanged).")
        )
        # Update in-memory current_session
        for a in self.current_session.get("actions", []):
            if a.get("id") in targets and a.get("state") == "trial":
                a["state"] = "committed"
        self._populate_action_list(self.current_session)

    # Phase 5 (2026-05-29): trial-dedup helper for UI selection
    def _is_last_trial_of_key(self, all_actions, action_idx, undone_ids=None):
        """Return True if the action at action_idx is the last trial of its
        (chapter, widget, action_type) group AND no committed/current supersedes it.

        Mirrors session_logger.effective_actions_for_replay logic so the UI
        selection matches what replay will actually execute. undone_ids (2026-05-30):
        trials reverted with Ctrl+Z are excluded from "last trial" selection, exactly
        as replay excludes them — so the auto-checked trial is the last NON-undone one.
        """
        undone_ids = undone_ids or set()
        action = all_actions[action_idx]
        if action.get("state") != "trial":
            return False
        if action.get("id") in undone_ids:
            return False
        widget = action.get("widget")
        action_type = action.get("action")

        # Find chapter membership for action_idx via session_data chapters
        if not hasattr(self, '_session_chapters_sorted') or not self._session_chapters_sorted:
            return True
        def find_ch(i):
            cur = None
            for ch in self._session_chapters_sorted:
                if ch.get("start_action_index", 0) <= i:
                    cur = ch.get("id")
                else:
                    break
            return cur
        my_ch = find_ch(action_idx)

        # Look across all actions of the same key in the same chapter
        last_trial_idx = -1
        has_committed_or_current = False
        for i, a in enumerate(all_actions):
            if a.get("widget") == widget and a.get("action") == action_type:
                if find_ch(i) == my_ch:
                    st = a.get("state", "current")
                    if st in ("committed", "current"):
                        has_committed_or_current = True
                    elif st == "trial" and a.get("id") not in undone_ids:
                        last_trial_idx = i
        if has_committed_or_current:
            return False
        return last_trial_idx == action_idx
    
    def _select_all_actions(self, select: bool):
        """全选/取消全选操作。

        全选时只勾选"有效重放集"内的行 —— 排除已撤销 / 被取代的 trial 与对同源图层的
        重复 re-tune(如两次 drift、两次 enhance)——让勾选状态如实反映真正会重放的操作,
        避免"全选"把探索性中间步骤也跑一遍(内存翻倍 + 结果错误)。取消全选照常清空。
        """
        if not select:
            for i in range(self.action_list.count()):
                self.action_list.item(i).setSelected(False)
            return

        eff_ids = None
        try:
            _sess = self.current_session or {}
            _eff = effective_replay_actions(
                _sess.get("actions", []),
                _sess.get("chapters", []),
                _sess.get("edit_records", []),
            )
            eff_ids = {a.get("id") for a in _eff}
        except Exception as _eff_err:
            print(f"[recovery] select-all effective filter skipped: {_eff_err}")
            eff_ids = None

        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            if eff_ids is None:
                item.setSelected(True)
            else:
                a = item.data(Qt.UserRole)
                item.setSelected(bool(a) and a.get("id") in eff_ids)
    
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
                # DM4 需要完整导入流程，但可以提前设置好路径
                from qtpy.QtCore import QSettings
                QSettings("NapariUser", "Importer").setValue("last_folder", src_path)
                
                reply = QMessageBox.question(self, tr("DM4 Auto-Import"),
                    f"{tr('DM4 Archive detected. Auto-set path and switch to Import tab?')}\n\n{src_path}",
                    QMessageBox.Yes | QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    # 使用辅助方法切换到 Import 标签页
                    self._switch_to_import_tab()
                    
                    QMessageBox.information(self, tr("Path Set"), 
                        f"{tr('DM4 folder path set. Click Load Images to proceed.')}\n{src_path}")
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
            folder = QFileDialog.getExistingDirectory(self, tr("Select DM4 Folder"))
            if folder:
                from qtpy.QtCore import QSettings
                QSettings("NapariUser", "Importer").setValue("last_folder", folder)
                
                # 使用辅助方法切换到 Import 标签页
                self._switch_to_import_tab()
                
                QMessageBox.information(self, tr("Path Set"), 
                    f"{tr('DM4 folder path set. Click Load Images to proceed.')}\n{folder}")
    
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

        # Phase 7 (2026-05-29): preallocated stack (memmap-safe)
        from utils.memory_utils import stack_frames_preallocated
        stack = stack_frames_preallocated(frames)
        name = f"Recovered_PNG_{folder.name}"
        self.viewer.add_image(stack, name=name, colormap='gray')
        gc.collect()
        trim_working_set()

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
        gc.collect()
        trim_working_set()

    def _register_recovery_source_alias(self, loaded_name):
        """恢复导入源后, 把每个记录里的 import 层名 -> 刚加载的源层名 写进
        current_session['layer_aliases']。这样即便用户用导出的液池 PNG(层名 PNG_...)
        来恢复, replay 的第一步(drift, source_layer=Original_...)也能经 alias 命中,
        不再依赖模糊前缀匹配。后续 rotate/crop/enhance 走链式 last_result_layer。"""
        if not loaded_name or not self.current_session:
            return
        aliases = self.current_session.setdefault("layer_aliases", {})
        for a in self.current_session.get("actions", []):
            if a.get("widget") != "import":
                continue
            rec = (a.get("params", {}) or {}).get("layer_name")
            if rec and rec != loaded_name:
                aliases[rec] = loaded_name

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
                        "exists": Path(source_path).exists(),
                        # [Issue] 记下导入时的选帧, 恢复时按原样复现 (否则会加载全部帧)
                        "frame_selection": params.get("frame_selection", ""),
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
                        loaded = self._load_png_sequence(src["path"])
                        self._register_recovery_source_alias(loaded)
                        return "success"
                    elif src["type"] == "TIFF Stack":
                        loaded = self._load_tiff_stack(src["path"])
                        self._register_recovery_source_alias(loaded)
                        return "success"
                    elif src["type"] in ("DM4 Archive", "DM4 Sequence"):
                        # 直接加载 DM4 序列 (带上导入时的选帧, 复现 0-20 这类范围)
                        loaded = self._load_dm4_sequence(src["path"], src.get("frame_selection", ""))
                        self._register_recovery_source_alias(loaded)
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
                    self._register_recovery_source_alias(self._load_png_sequence(folder))
                    return "success"
            elif choice == btn_tiff:
                file_path, _ = QFileDialog.getOpenFileName(self, tr("Select TIFF File"), "", "TIFF (*.tiff *.tif)")
                if file_path:
                    self._register_recovery_source_alias(self._load_tiff_stack(file_path))
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
                    # 即便手动选目录, 也复现日志里的选帧 (如 0-20)
                    self._load_dm4_sequence(folder, self._logged_dm4_frame_selection())
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
        # 用类型守卫排除 Shapes/Points（list 型 .data），避免误判为图像源
        image_layers = [l for l in self.viewer.layers if self._is_image_layer_data(l)]
        
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

        # === 权威过滤：只重放"有效重放集" ===
        # 无论用户如何勾选(尤其"全选"), 都用 effective_replay_actions 收敛到真正会
        # 影响最终结果的操作: 排除已撤销(undo)的动作、被后续试探取代的 trial、以及对
        # 同一源图层的重复 re-tune(本会话的两次 drift、两次 enhance)。这是让"全选恢复
        # 只跑第二次 drift"、且不会 enhance-of-enhance 的关键。
        self._logged_output_name = {}
        try:
            _sess = self.current_session or {}
            _eff = effective_replay_actions(
                _sess.get("actions", []),
                _sess.get("chapters", []),
                _sess.get("edit_records", []),
            )
            _eff_ids = {a.get("id") for a in _eff}
            selected_actions = [a for a in selected_actions if a.get("id") in _eff_ids]

            # 持久重放映射 (跨多次分步恢复): logged 输出层名 -> 本次重放实际产层。
            # 每个会话独立; 换会话则清空。map 内容在下方 replay 循环里逐步写入,
            # 供 _find_best_matching_layer 的 step 1.5 消费, 让下一次(单独一次点击的)
            # 分步恢复能把 "Corrected_v2_X" 正确解析到上次产生的漂移矫正层。
            sid = _sess.get("session_id")
            if getattr(self, "_replay_map_session_id", None) != sid:
                self._replay_layer_map = {}
                self._replay_map_session_id = sid
            self._logged_output_name = build_logged_output_map(_eff, IMAGE_PRODUCER_ACTIONS)
        except Exception as _eff_err:
            print(f"[recovery] effective-replay filter skipped: {_eff_err}")

        # 有效过滤后若一个都不剩 (勾选的全被撤销/取代)，明确告知而非静默"恢复 0 个"。
        if not selected_actions:
            QMessageBox.information(
                self, tr("Session Recovery"),
                tr("All selected actions were superseded or undone — nothing to replay. "
                   "Select the effective (non-grayed) rows."))
            return

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
        
        # === 智能处理：annotation 的 update_params/pre_burn_params ===
        # 只保留最后一次 update_params (或 burn_in 之前的那一次 pre_burn_params)
        # 因为用户调整过程中的中间状态通常不需要恢复
        annotation_updates = []
        for i, action in enumerate(selected_actions):
            if action.get("widget") == "annotation" and action.get("action") in ("update_params", "pre_burn_params"):
                annotation_updates.append(i)
        
        if len(annotation_updates) > 1:
            # 保留最后一次，或者如果有 burn_in，保留 burn_in 之前的那一个
            has_burn_in = any(a.get("widget") == "annotation" and a.get("action") == "burn_in" for a in selected_actions)
            
            if has_burn_in:
                # 只需要保留 burn_in 本身 (它包含 params)，删除所有 update_params/pre_burn_params
                filtered_actions = [a for a in selected_actions 
                                  if not (a.get("widget") == "annotation" and a.get("action") in ("update_params", "pre_burn_params"))]
            else:
                # 没有 burn_in，只保留最后一个 update_params
                last_update_idx = annotation_updates[-1]
                filtered_actions = []
                for i, action in enumerate(selected_actions):
                    if action.get("widget") == "annotation" and action.get("action") in ("update_params", "pre_burn_params"):
                        if i == last_update_idx:
                            filtered_actions.append(action)
                        # 跳过其他的
                    else:
                        filtered_actions.append(action)
            selected_actions = filtered_actions
        
        # 执行恢复操作 - 链式恢复
        success_count = 0
        failed_actions = []
        skipped_actions = []
        aborted = False
        last_result_layer = None  # 跟踪上一个操作产生的图层名

        # 内存封顶 (2026-07-01): 记录 replay 自己产生的图像中间层, 每当链条推进就把上
        # 一步刚被消费的中间层从 viewer 移除 —— 只保留原始源层 + 最新层, 避免整段
        # drift/rotate/crop/enhance 的整帧栈同时驻留 (旧行为只 .visible=False, OOM 根因)。
        # 只对"产生新图像层"的操作维护此链; ROI/annotation 等不推进也不驱逐。
        # 用共享的 IMAGE_PRODUCER_ACTIONS (含 contrast_adjustment; 旧本地集写成
        # "contrast" 与实际动作名不符, 导致对比度层不进驱逐链而泄漏)。
        _IMAGE_PRODUCERS = IMAGE_PRODUCER_ACTIONS
        replay_produced = set()
        chain_last = None

        # 中间层保留模式 (F2, 与 session_recovery_mode 的 auto/review 正交):
        #   lean   = 驱逐已消费中间层 (省内存, 默认)
        #   full   = 全部保留但写盘 (可逐步核对, RAM 有界)
        #   custom = 用户选保留哪些步骤 (选中写盘, 其余驱逐)
        retention = GlobalConfig.get("session_recovery_retention") or "lean"
        keep_ids = set()
        if retention == "custom":
            keep_ids = self._prompt_custom_retention(selected_actions, _IMAGE_PRODUCERS)
            if keep_ids is None:  # 用户取消
                return
        keep_by_layer = {}   # 产出层名 -> 是否保留(写盘)
        kept_count = 0
        spill_count = 0
        evict_count = 0

        # UI 提示：开始重放
        self.viewer.status = "Starting session replay..."

        for action in selected_actions:
            widget = action.get("widget", "")
            action_type = action.get("action", "")
            params = action.get("params", {})
            is_image_producer = (widget, action_type) in _IMAGE_PRODUCERS
            consumed = chain_last if is_image_producer else None

            try:
                # 传入上一个结果图层名，用于链式操作
                result, result_layer = self._replay_action(widget, action_type, params, last_result_layer)
                if result == "success":
                    success_count += 1
                    if result_layer:
                        last_result_layer = result_layer  # 更新链式图层
                        if is_image_producer:
                            # 记录本步产出层的保留意图 (full=全留; custom=按 action id)
                            keep_this = (retention == "full"
                                         or (retention == "custom" and action.get("id") in keep_ids))
                            keep_by_layer[result_layer] = keep_this
                            # 处理刚被这一步消费掉的中间层 (仅限 replay 自产层, 从不动原始源)
                            if consumed and consumed in replay_produced and consumed != result_layer:
                                if keep_by_layer.get(consumed, False):
                                    self._spill_layer_to_disk(consumed)
                                    spill_count += 1
                                    kept_count += 1
                                else:
                                    self._evict_replay_layer(consumed)
                                    evict_count += 1
                                replay_produced.discard(consumed)
                            replay_produced.add(result_layer)
                            chain_last = result_layer
                            # 记录 logged 输出名 -> 实际产层, 供后续(可能是另一次分步
                            # 恢复的)步骤按 logged 源名解析到本层。依赖 result_layer 为
                            # napari 实际层名 (见各 _replay_* 返回 new_layer.name)。
                            lo = self._logged_output_name.get(action.get("id"))
                            if lo:
                                self._replay_layer_map[lo] = result_layer
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

        # 中间层保留结果 (无论哪种模式都显示, 让"图层消失"不再是谜)
        report_lines.append(
            tr("kept %d, spilled %d to disk, removed %d intermediates")
            % (kept_count, spill_count, evict_count))

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

        # === Auto-continuation: 恢复后自动续写到原 session ===
        auto_continued = False
        if GlobalConfig.get("session_auto_continue_after_recovery") and log_path.exists() and success_count > 0:
            try:
                from utils.session_logger import SessionLogger
                logger = SessionLogger.resume_from_file(log_path)
                auto_continued = True
                logger.log_action("recovery", "session_restored", {
                    "source_session_id": self.current_session.get("session_id", "unknown"),
                    "source_log_path": str(log_path),
                    "success_count": success_count,
                    "skipped_count": len(skipped_actions),
                    "failed_count": len(failed_actions),
                    "recovered_widgets": list(set(a.get("widget", "") for a in selected_actions))
                })
            except Exception as e:
                print(f"[RecoveryWidget] Auto-continue failed: {e}")

        if not auto_continued:
            # 回退：记录到当前活跃的 session（如果有）
            try:
                from utils.session_logger import get_logger
                recovered_session_id = self.current_session.get("session_id", "unknown")
                get_logger().log_action("recovery", "session_restored", {
                    "source_session_id": recovered_session_id,
                    "source_log_path": str(log_path),
                    "success_count": success_count,
                    "skipped_count": len(skipped_actions),
                    "failed_count": len(failed_actions),
                    "recovered_widgets": list(set(a.get("widget", "") for a in selected_actions))
                })
            except Exception as e:
                print(f"[RecoveryWidget] Failed to log recovery event: {e}")
        
        # 设置激活图层并同步给 Geometry 面板
        if last_result_layer and last_result_layer in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[last_result_layer]

            # 恢复后只显示【最终结果层】, 隐藏所有其它图像层 —— 包括原始层、未被本
            # 次链条消费的孤儿中间层、以及【上一次分步恢复】遗留的可见层。旧行为里
            # 每步只 .visible=False 其直接源, 断链/跨调用的层会一直亮着, 导致"裁剪层
            # 和别的层一起显示"。full/custom 保留模式下这些层仍在列表里(可手动重显)。
            try:
                for _l in self.viewer.layers:
                    if self._is_image_layer_data(_l):
                        _l.visible = (_l.name == last_result_layer)
            except Exception as _vis_err:
                print(f"[recovery] post-replay visibility cleanup skipped: {_vis_err}")

            # 手动同步 Geometry widget 的下拉框，确保重放后的数据被正确选中
            try:
                view_target = last_result_layer
                data_target = last_result_layer
                
                # 获取最新生成的图像图层列表 (修复: 安全检查 ndim 防止 Shapes/Points 等图层数据类型为 list 而导致崩溃)
                image_layers = [l.name for l in self.viewer.layers 
                                if hasattr(l, 'data') and hasattr(l.data, 'ndim') and l.data.ndim >= 2]
                
                if image_layers:
                    view_target = image_layers[-1]
                    data_target = image_layers[-1]
                    
                    # 验证：如果最新图层包含 Enh，说明这是视图层。数据层需要往前找一个不包含 Enh 的图层。
                    if "Enh" in view_target:
                        for layer_name in reversed(image_layers):
                            if "Enh" not in layer_name:
                                data_target = layer_name
                                break

                for dock in self.viewer.window._dock_widgets.values():
                    widget = dock.widget()
                    if hasattr(widget, 'batch_data_combo') and hasattr(widget, 'batch_view_combo'):
                        idx_data = widget.batch_data_combo.findData(data_target)
                        if idx_data >= 0: widget.batch_data_combo.setCurrentIndex(idx_data)
                        
                        idx_view = widget.batch_view_combo.findData(view_target)
                        if idx_view >= 0: widget.batch_view_combo.setCurrentIndex(idx_view)
            except Exception as e:
                print(f"[RecoveryWidget] Failed to sync geometry widget: {e}")

        if auto_continued:
            report_lines.append(f"\n▶️ 已自动续写到原 session（后续操作将追加记录）")

        # 显示恢复结果 (原为 QMessageBox, 现在改用 QDialog 解决信息太长截断的问题)
        from qtpy.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QPushButton
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("Session Recovery Report"))
        dialog.resize(600, 400)
        dialog_layout = QVBoxLayout(dialog)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText("\n".join(report_lines))
        dialog_layout.addWidget(text_edit)
        
        btn_ok = QPushButton(tr("OK"))
        btn_ok.clicked.connect(dialog.accept)
        btn_ok.setStyleSheet("background-color: #2196F3; color: white; padding: 6px; font-weight: bold;")
        dialog_layout.addWidget(btn_ok)
        
        dialog.exec_()
        
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
        
        # === 标注恢复 - 恢复参数或执行烧录 ===
        if widget == "annotation":
            return self._replay_annotation(action_type, params, recovery_mode, last_result_layer)
        
        # === 导出 ===
        if widget == "export":
            return ("skipped", None)  # 导出需要手动确认路径
        
        # === 会话恢复链接 - 可以链式恢复到源会话 ===
        if widget == "recovery" and action_type == "session_restored":
            return self._replay_session_link(params, recovery_mode)
        
        return ("skipped", None)
        
    @staticmethod
    def _is_image_layer_data(layer) -> bool:
        """仅当图层持有 >=2D 的数组型图像时返回 True。

        排除 Shapes/Points 等图层（其 .data 是 Python list，没有 .ndim），
        这些图层此前会让匹配器在 `l.data.ndim` 处抛出
        'list' object has no attribute 'ndim'。
        """
        data = getattr(layer, 'data', None)
        return data is not None and hasattr(data, 'ndim') and data.ndim >= 2

    def _find_best_matching_layer(self, target_name: str, last_result_layer: str) -> 'napari.layers.Layer':
        """
        智能图层匹配：在重放时找到最合适的源图层。
        优先顺序：
        1. 链式首选：上一步产生的 last_result_layer
        2. 全名精确匹配（目标名在画布中存在）
        2.5. Layer alias 匹配（session 中记录的 Original→PNG 映射）
        3. 前缀特征模糊匹配（去除 _recovered 干扰）
        4. 后备：第一个有效的图像图层

        注意：每一条按名字命中的捷径都必须再次校验类型——画布里可能存在
        一个与目标同名/同别名的 Shapes 图层（如 Batch_ROI），若直接返回会在
        后续 .data.ndim 处崩溃。命中但类型不符时继续向下一策略回退。
        """
        image_layers = [l for l in self.viewer.layers if self._is_image_layer_data(l)]
        if not image_layers:
            return None

        # 1. 如果有链式的上一步结果，优先使用
        if last_result_layer and last_result_layer in self.viewer.layers:
            cand = self.viewer.layers[last_result_layer]
            if self._is_image_layer_data(cand):
                return cand

        # 1.5 持久重放映射: logged 源名 -> 实际重放层。跨多次【分步恢复】保留 ——
        #     修复根因: 实时命名 (Corrected_v2_X) 与重放命名 (X_recovered_corrected)
        #     不一致, 单靠名字匹配会桥接失败。见 build_logged_output_map。
        replay_map = getattr(self, "_replay_layer_map", {}) or {}
        if target_name and target_name in replay_map:
            mapped = replay_map[target_name]
            if mapped in self.viewer.layers and self._is_image_layer_data(self.viewer.layers[mapped]):
                return self.viewer.layers[mapped]

        # 2. 精确匹配
        if target_name and target_name in self.viewer.layers:
            cand = self.viewer.layers[target_name]
            if self._is_image_layer_data(cand):
                return cand

        # 2.5. Alias 匹配
        if target_name and self.current_session:
            aliases = self.current_session.get("layer_aliases", {})
            aliased_name = aliases.get(target_name, "")
            if aliased_name and aliased_name in self.viewer.layers:
                cand = self.viewer.layers[aliased_name]
                if self._is_image_layer_data(cand):
                    return cand
            for orig, mapped in aliases.items():
                if mapped and mapped in self.viewer.layers and orig == target_name:
                    cand = self.viewer.layers[mapped]
                    if self._is_image_layer_data(cand):
                        return cand

        # 3. 模糊特征匹配
        if target_name:
            import re
            import os
            # 从目标名提取有效的特征前缀或块。
            # 常见场景：源名字是 Original_cropped, 重放后画布里的名字变成了 Original_recovered_cropped
            # 策略：如果画布里有图层包含了所有的单词块，那就是它。
            # 或者暴力一点：去掉所有 "_recovered" 标志，再进行匹配
            
            clean_target = target_name.replace("_recovered", "").replace("__", "_")
            
            best_layer = None
            best_score = -1
            
            for layer in image_layers:
                clean_layer_name = layer.name.replace("_recovered", "").replace("__", "_")
                
                # 如果去掉 recovered 后名字完全一致
                if clean_layer_name == clean_target:
                    return layer
                    
                # 评估重叠程度 (比如 base name 相同且后缀匹配)
                # 简单的前缀匹配
                if clean_layer_name.startswith(clean_target) or clean_target.startswith(clean_layer_name):
                    score = len(os.path.commonprefix([clean_layer_name, clean_target]))
                    if score > best_score:
                        best_score = score
                        best_layer = layer
            
            if best_layer:
                return best_layer

        # 4. 兜底：仅当画布只有一个图像层（无歧义）时才用它。旧代码在多层时也
        #    return image_layers[0]，会静默选中原始层（漂移矫正被丢弃且报 success）。
        #    多层歧义现在返回 None → 该步"明确失败"进 failed 列表，而非给出错误结果。
        if len(image_layers) == 1:
            return image_layers[0]
        return None

    def _add_replay_image(self, array, name, colormap='gray', metadata=None):
        """add_image for a replay result with explicit sampled contrast_limits so
        napari skips its whole-array auto-contrast read (the RAM spike on big
        rotate-expand stacks). Post-add gc + Working Set trim as before."""
        cl = compute_replay_contrast_limits(array)
        kwargs = {"name": name, "colormap": colormap}
        if metadata is not None:
            kwargs["metadata"] = metadata
        if cl is not None:
            kwargs["contrast_limits"] = list(cl)
        layer = self.viewer.add_image(array, **kwargs)
        gc.collect()
        trim_working_set()
        return layer

    def _evict_replay_layer(self, name):
        """Remove a consumed replay-intermediate layer to bound memory, then trim.
        Only called on layers replay itself produced (never the original source)."""
        if not name:
            return
        try:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
                gc.collect()
                trim_working_set()
        except Exception as e:
            print(f"[recovery] evict intermediate '{name}' failed: {e}")

    def _spill_layer_to_disk(self, name):
        """保留图层但把其数组移到磁盘 memmap, 界住 RAM (恢复"全内存/自定义"模式用)。
        已是 memmap 的大层 (replay 生产者对大数据本就用 create_huge_array) → 只释放缓存页;
        小 RAM 数组 → 复制进强制磁盘 memmap 再换入 (先存 contrast_limits, 避免 napari 整读)。"""
        if not name:
            return
        try:
            if name not in self.viewer.layers:
                return
            layer = self.viewer.layers[name]
            data = layer.data
            if isinstance(data, np.memmap):
                release_memmap_pages(data)  # 已在磁盘, 只需吐掉 OS 缓存页
                return
            cl = getattr(layer, "contrast_limits", None)
            arr = np.asarray(data)
            mm, _ = create_huge_array(arr.shape, arr.dtype, force_disk=True)
            if arr.ndim >= 3:
                for i in range(arr.shape[0]):
                    mm[i] = arr[i]
            else:
                mm[...] = arr
            release_memmap_pages(mm)
            layer.data = mm  # 换成磁盘数组; napari 会重算对比度, 故下面复原
            if cl is not None:
                try:
                    layer.contrast_limits = cl
                except Exception:
                    pass
            del arr
            gc.collect()
            trim_working_set()
        except Exception as e:
            print(f"[recovery] spill intermediate '{name}' to disk failed: {e}")

    def _prompt_custom_retention(self, selected_actions, image_producers):
        """自定义模式: 弹框让用户勾选要【保留(写盘)】哪些中间步骤。
        返回要保留的 action id 集合; 用户取消返回 None; 无中间步骤返回空集。
        (最终结果层永远保留, 不在此列; 未勾选的中间层按省内存驱逐。)"""
        _LABELS = {
            ("drift", "apply_correction"): "Drift",
            ("enhance", "filter_enhancement"): "Enhance (filter)",
            ("enhance", "contrast_adjustment"): "Contrast",
            ("geometry", "rotate"): "Rotate",
            ("geometry", "flip"): "Flip",
            ("geometry", "crop"): "Crop",
        }
        steps = []
        for a in selected_actions:
            key = (a.get("widget", ""), a.get("action", ""))
            if key in image_producers:
                steps.append((a.get("id"), _LABELS.get(key, f"{key[0]}:{key[1]}")))
        if not steps:
            return set()

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("Select intermediate steps to keep"))
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(tr("Select intermediate steps to keep")))
        checks = []
        for aid, label in steps:
            cb = QCheckBox(label)
            v.addWidget(cb)
            checks.append((aid, cb))
        h = QHBoxLayout()
        btn_cancel = QPushButton(tr("Cancel"))
        btn_ok = QPushButton(tr("OK"))
        btn_ok.setDefault(True)
        h.addStretch()
        h.addWidget(btn_cancel)
        h.addWidget(btn_ok)
        v.addLayout(h)
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        if not dlg.exec():
            return None
        return {aid for aid, cb in checks if cb.isChecked()}

    def _replay_geometry(self, action_type: str, params: dict, recovery_mode: str, last_result_layer: str = None) -> tuple:
        """重放几何变换操作"""
        import numpy as np
        
        try:
            source_layer_name = params.get("source_layer", "")
            
            # 使用智能图层匹配
            source_layer = self._find_best_matching_layer(source_layer_name, last_result_layer)
            if not source_layer:
                return ("failed", None)
            
            image_stack = source_layer.data

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
                # 用 napari 实际层名 (重名时会被自动改成 "X [1]") 覆盖请求名, 否则
                # 二次恢复会把链/映射/驱逐都指到旧层 (见 build_logged_output_map)。
                new_name = self._add_replay_image(rotated, new_name).name

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
                new_name = self._add_replay_image(flipped, new_name).name

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
                new_name = self._add_replay_image(cropped, new_name).name

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
            
            # 使用智能匹配，并且增加 ndim 检查
            source_layer = self._find_best_matching_layer(source_layer_name, last_result_layer)
            if not source_layer or not hasattr(source_layer, 'data') or source_layer.data.ndim < 3:
                return ("failed", None)
            
            image_stack = source_layer.data

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
                    progress_callback=progress_cb
                )

                gc.collect()
                trim_working_set()

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
                new_layer = self._add_replay_image(
                    corrected, new_name,
                    metadata=source_layer.metadata.copy() if hasattr(source_layer, 'metadata') else {}
                )
                new_name = new_layer.name  # napari 实际层名 (重名自动改名时 != 请求名)

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
                # 旧格式/对比度：直接参数。对比度日志记的是 "source" (不是 source_layer),
                # 之前只读 source_layer → 空目标 → 落到兜底选错层。回退读 source。
                source_name = params.get("source_layer", "") or params.get("source", "")
                inner_params = params
            
            source_layer = self._find_best_matching_layer(source_name, last_result_layer)
            if not source_layer:
                return ("failed", None)
            
            data = source_layer.data

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
                    new_name = self._add_replay_image(result, new_name).name

                    source_layer.visible = False

                    return ("success", new_name)
                    
                finally:
                    progress.close()
            
            elif action_type == "contrast_adjustment":
                # 对比度调整 - 日志格式为 {"source": ..., "min": c_min, "max": c_max}
                # 源图层已由上面的 prelude 经 _find_best_matching_layer 解析 (含持久
                # 映射/别名/模糊)。此处不再用"精确名 in layers"覆盖它 —— 那会退回到
                # 只认精确名, 在实时/重放命名不一致时反而选错层。
                c_min = params.get("min", 0)
                c_max = params.get("max", 255)
                
                # 添加进度对话框，防止用户认为程序卡死
                from qtpy.QtWidgets import QProgressDialog, QApplication
                from qtpy.QtCore import Qt
                
                progress = QProgressDialog(tr("Applying contrast adjustment..."), None, 0, 100, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(0)
                progress.show()
                QApplication.processEvents()
                
                try:
                    from utils.memory_utils import create_huge_array, release_memmap_pages

                    range_width = c_max - c_min
                    if range_width < 1e-9:
                        range_width = 1e-9

                    if data.ndim == 3:
                        n = data.shape[0]
                        result, _ = create_huge_array(data.shape, np.uint8)
                        for i in range(n):
                            frame_f = data[i].astype(np.float32)
                            frame_f = np.clip((frame_f - c_min) / range_width, 0, 1)
                            result[i] = (frame_f * 255).astype(np.uint8)
                            if i % 50 == 0:
                                progress.setValue(10 + int(i / n * 85))
                                QApplication.processEvents()
                        release_memmap_pages(result)
                    else:
                        data_f = data.astype(np.float32)
                        normalized = np.clip((data_f - c_min) / range_width, 0, 1)
                        result = (normalized * 255).astype(np.uint8)

                    progress.setLabelText(tr("Rendering result..."))
                    progress.setValue(95)
                    QApplication.processEvents()

                    new_name = f"Contrast_{source_layer.name}"
                    new_name = self._add_replay_image(result, new_name).name

                    source_layer.visible = False

                    progress.setValue(100)

                    return ("success", new_name)
                    
                finally:
                    progress.close()
            
        except Exception as e:
            print(f"Enhance replay failed: {e}")
            return ("failed", None)
        
        return ("skipped", None)
    
    def _replay_annotation(self, action_type: str, params: dict, recovery_mode: str, last_result_layer: str = None) -> tuple:
        """
        重放标注操作
        
        action_type:
            - "update_params" / "pre_burn_params": 恢复 UI 参数 (可编辑模式)
            - "burn_in": 执行烧录 (生成新图层)
        """
        try:
            # 找到 AnnotationWidget 实例
            annotation_widget = self._find_annotation_widget()
            if annotation_widget is None:
                print("[RecoveryWidget] AnnotationWidget not found")
                return ("skipped", None)
            
            # === Case 1: 参数恢复 (可编辑模式) ===
            if action_type in ("update_params", "pre_burn_params"):
                if recovery_mode == "review":
                    confirm = self._ask_recovery_confirm(
                        tr("Session Recovery"),
                        f"{tr('Restore annotation parameters')}?\n\n"
                        f"{tr('This will update scale bar and timestamp settings.')}\n\n"
                        f"{tr('Continue with this result?')}"
                    )
                    if confirm == "abort":
                        return ("abort", None)
                    elif confirm == "skip":
                        return ("skipped", None)
                
                # 应用参数
                annotation_widget.apply_params(params)
                
                if recovery_mode == "review":
                    from qtpy.QtWidgets import QMessageBox
                    QMessageBox.information(
                        self,
                        tr("Session Recovery"),
                        f"✅ {tr('Annotation parameters restored')}\n\n"
                        f"{tr('You can now preview and adjust the settings.')}"
                    )
                
                return ("success", None)
            
            # === Case 2: 烧录操作 ===
            elif action_type == "burn_in":
                # 从 params 中提取嵌套的参数
                inner_params = params.get("params", params)
                source_layer_name = params.get("source_layer", "")
                
                if recovery_mode == "review":
                    from qtpy.QtWidgets import QMessageBox
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle(tr("Session Recovery"))
                    msg_box.setText(
                        f"{tr('This is a burn-in operation')}.\n\n"
                        f"{tr('How would you like to recover?')}"
                    )
                    
                    btn_params_only = msg_box.addButton(
                        f"📝 {tr('Restore Parameters Only')} ({tr('Editable')})", 
                        QMessageBox.ActionRole
                    )
                    btn_execute = msg_box.addButton(
                        f"🔥 {tr('Execute Burn-in')} ({tr('New Layer')})", 
                        QMessageBox.AcceptRole
                    )
                    btn_skip = msg_box.addButton(tr("Skip"), QMessageBox.RejectRole)
                    
                    msg_box.exec_()
                    choice = msg_box.clickedButton()
                    
                    if choice == btn_skip or choice is None:
                        return ("skipped", None)
                    
                    if choice == btn_params_only:
                        # 仅恢复参数
                        annotation_widget.apply_params(inner_params)
                        QMessageBox.information(
                            self,
                            tr("Session Recovery"),
                            f"✅ {tr('Annotation parameters restored (editable mode)')}"
                        )
                        return ("success", None)
                    
                    # 继续执行烧录
                    pass
                
                # 执行烧录：先应用参数，再触发烧录
                annotation_widget.apply_params(inner_params)
                
                # 确保源图层存在
                if source_layer_name and source_layer_name in self.viewer.layers:
                    # 设置源图层
                    idx = annotation_widget.layer_combo.findData(source_layer_name)
                    if idx >= 0:
                        annotation_widget.layer_combo.setCurrentIndex(idx)
                elif last_result_layer and last_result_layer in self.viewer.layers:
                    idx = annotation_widget.layer_combo.findData(last_result_layer)
                    if idx >= 0:
                        annotation_widget.layer_combo.setCurrentIndex(idx)
                
                # 触发烧录
                annotation_widget._apply_to_new_layer()
                
                # 获取结果图层名
                result_layer_name = params.get("result_layer", "")
                
                return ("success", result_layer_name if result_layer_name else None)
            
            return ("skipped", None)
            
        except Exception as e:
            print(f"[RecoveryWidget] Annotation replay failed: {e}")
            import traceback
            traceback.print_exc()
            return ("failed", None)
    
    def _replay_session_link(self, params: dict, recovery_mode: str) -> tuple:
        """
        处理 session_restored 操作 - 链式恢复到源会话
        """
        from qtpy.QtWidgets import QMessageBox
        import json
        
        source_session_id = params.get("source_session_id", "unknown")
        source_log_path = params.get("source_log_path", "")
        
        if recovery_mode == "review":
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle(tr("Session Recovery"))
            msg_box.setText(
                f"{tr('This action was recovered from another session')}.\n\n"
                f"Source Session: {source_session_id}\n"
                f"Path: {source_log_path}\n\n"
                f"{tr('What would you like to do?')}"
            )
            
            btn_load = msg_box.addButton(f"🔗 {tr('Load Source Session')}", QMessageBox.AcceptRole)
            btn_skip = msg_box.addButton(tr("Skip"), QMessageBox.RejectRole)
            
            msg_box.exec_()
            choice = msg_box.clickedButton()
            
            if choice != btn_load:
                return ("skipped", None)
        
        # 检查源会话文件是否存在
        source_path = Path(source_log_path)
        if not source_path.exists():
            QMessageBox.warning(
                self,
                tr("Session Recovery"),
                f"⚠️ {tr('Source session file not found')}:\n\n{source_log_path}\n\n"
                f"{tr('The original session may have been moved or deleted.')}"
            )
            return ("skipped", None)
        
        # 加载源会话
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                source_session = json.load(f)
            source_session["_log_path"] = str(source_path)
            
            # 显示源会话供用户选择恢复
            self.current_session = source_session
            self._show_session_details()
            
            QMessageBox.information(
                self,
                tr("Session Recovery"),
                f"✅ {tr('Source session loaded')}\n\n"
                f"Session: {source_session_id}\n"
                f"{tr('Please select the actions you want to recover from this session.')}"
            )
            
            return ("success", None)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                tr("Error"),
                f"Failed to load source session: {e}"
            )
            return ("failed", None)
    
    def _find_annotation_widget(self):
        """
        在 Napari 窗口中查找 AnnotationWidget 实例
        """
        try:
            from widgets.annotation_widget import AnnotationWidget
            
            # 方法 1: 通过 viewer.window 的 dock widgets 查找 (使用公共 API)
            # Napari 的 dock_widgets 返回的 values 直接就是 widget 本身，而不是 QDockWidget
            if hasattr(self.viewer, 'window') and hasattr(self.viewer.window, 'dock_widgets'):
                for widget in self.viewer.window.dock_widgets.values():
                    if isinstance(widget, AnnotationWidget):
                        return widget
            
            # 方法 2: 通过 Qt 遍历所有子控件查找
            from qtpy.QtWidgets import QApplication
            for widget in QApplication.allWidgets():
                if isinstance(widget, AnnotationWidget):
                    return widget
            
            return None
        except ImportError:
            print("[RecoveryWidget] Could not import AnnotationWidget")
            return None
    
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

    def _continue_session(self):
        """断点续写：从已有 session 恢复并继续记录新操作"""
        from utils.session_logger import SessionLogger

        if not self.current_session:
            return

        log_path = Path(self.current_session.get("_log_path", ""))
        if not log_path.exists():
            QMessageBox.warning(self, tr("Continue Session"),
                tr("Session file not found on disk."))
            return

        session_id = self.current_session.get("session_id", "?")
        action_count = len(self.current_session.get("actions", []))
        reply = QMessageBox.question(
            self, tr("Continue Session"),
            f"{tr('Resume recording to session')} [{session_id}]?\n\n"
            f"{tr('Existing actions')}: {action_count}\n"
            f"{tr('New actions will be appended to this session.')}\n\n"
            f"{tr('The current active session (if any) will be closed.')}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        try:
            logger = SessionLogger.resume_from_file(log_path)
            QMessageBox.information(self, tr("Continue Session"),
                f"▶️ {tr('Session resumed')}: [{logger.session_id}]\n"
                f"{tr('Total actions')}: {len(logger.actions)}\n\n"
                f"{tr('All subsequent operations will be recorded to this session.')}")
            self._refresh_sessions()
        except Exception as e:
            QMessageBox.critical(self, tr("Continue Session"),
                f"{tr('Failed to resume session')}:\n{e}")

    def _mark_session_status(self, log_path: Path, status: str):
        """更新会话状态并重算 checksum"""
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["status"] = status
            # 必须与 load_from_file/_compute_checksum 一致地【剥掉 state】再哈希, 否则
            # 恢复时写回的 checksum 含 state, 下次加载(剥 state)重算必不一致 → 工具自
            # 己写的文件被误报"可能被篡改"。
            data["checksum"] = compute_actions_checksum(data.get("actions", []))
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            print(f"Error updating session status: {e}")
    
    # === 数据加载方法 ===
    
    def _logged_dm4_frame_selection(self) -> str:
        """从当前 session 的 load_dm4_sequence 动作取导入选帧字符串(没有则返回空)。"""
        try:
            for a in (self.current_session or {}).get("actions", []):
                if a.get("action") == "load_dm4_sequence":
                    return a.get("params", {}).get("frame_selection", "") or ""
        except Exception:
            pass
        return ""

    @staticmethod
    def _parse_frame_selection(text):
        """把 '0-20' / '0-30,50' 解析成有序索引列表; 空/all/global → None(=全部帧)。
        用于恢复 DM4 导入时复现用户当初选的帧范围。"""
        if not text or str(text).strip().lower() in ("", "all", "global"):
            return None
        out = set()
        for part in str(text).split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part.lstrip('-'):  # 避免负号干扰; 帧号非负
                try:
                    a, b = part.split('-', 1)
                    a, b = int(a.strip()), int(b.strip())
                    if a > b:
                        a, b = b, a
                    out.update(range(a, b + 1))
                except Exception:
                    continue
            else:
                try:
                    out.add(int(part))
                except Exception:
                    continue
        return sorted(out) if out else None

    def _load_dm4_sequence(self, folder_path: str, frame_selection: str = ""):
        """加载 DM4 序列文件夹。frame_selection 非空时只加载这些帧(复现导入时的选帧)。"""
        from core.dm4_reader import read_dm4_sequence
        from qtpy.QtWidgets import QProgressDialog
        from qtpy.QtCore import Qt

        # [Issue] 恢复时复现导入选帧: 解析日志里的 frame_selection (如 '0-20') → frame_indices
        frame_indices = self._parse_frame_selection(frame_selection)

        progress = QProgressDialog(tr("Loading DM4 sequence..."), None, 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        def callback(current, total):
            progress.setValue(int(current / total * 100))

        try:
            stack, metadata = read_dm4_sequence(folder_path, bit_depth=8, max_workers=8,
                                                progress_callback=callback, frame_indices=frame_indices)
            progress.close()
            
            name = f"Original_{Path(folder_path).name}"
            if len(name) > 30:
                name = name[:15] + "..." + name[-10:]

            self._add_replay_image(stack, name, metadata=metadata)
            return name
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
            # 2026-07-01: 逐帧读入"预分配(大数据自动 memmap)"的栈, 不再先攒一个
            # 全量 frames 列表再 stack —— 5000+ 帧恢复不再一次性 ~24GB×2 驻留 RAM。
            from utils.memory_utils import load_png_stack_memmap
            stack = load_png_stack_memmap(
                png_files, progress_cb=lambda k: progress.setValue(k))
            progress.close()

            if stack is None or getattr(stack, "size", 0) == 0:
                raise ValueError("Could not read any valid images.")

            name = f"PNG_{folder.name}"
            if len(name) > 30:
                name = name[:15] + "..." + name[-10:]

            self._add_replay_image(stack, name)
            return name
        except Exception as e:
            progress.close()
            raise e

    def _load_tiff_stack(self, file_path: str):
        """加载 TIFF Stack 文件"""
        import tifffile

        # 2026-07-01: 优先 memmap 打开 TIFF (大栈不整读进 RAM); 失败再 imread。
        try:
            stack = tifffile.memmap(file_path)
        except Exception:
            stack = tifffile.imread(file_path)

        # 确保是3D数组 (T, H, W)
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        elif stack.ndim == 4:
            stack = stack[..., 0]

        name = f"TIFF_{Path(file_path).stem}"
        if len(name) > 30:
            name = name[:15] + "..." + name[-10:]

        self._add_replay_image(stack, name)
        return name

    def _show_path_manager(self):
        """显示搜索路径管理器对话框"""
        from utils.session_logger import SessionLogger
        
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("Manage Search Paths"))
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout()
        
        # 说明
        layout.addWidget(QLabel(f"<b>{tr('Saved Search Paths')}</b><br><i>{tr('Sessions from these folders will be shown in the list.')}</i>"))
        
        # 路径列表
        path_list = QListWidget()
        paths_data = SessionLogger.get_all_saved_search_paths()
        
        for item_data in paths_data:
            path = item_data["path"]
            exists = item_data["exists"]
            icon = "✅" if exists else "❌"
            item = QListWidgetItem(f"{icon} {path}")
            item.setData(Qt.UserRole, path)
            if not exists:
                item.setForeground(Qt.gray)
            path_list.addItem(item)
        
        layout.addWidget(path_list)
        
        # 按钮区域 - 第一行：路径管理
        btn_layout = QHBoxLayout()
        
        btn_add = QPushButton(f"➕ {tr('Add Path')}")
        def add_path():
            folder = QFileDialog.getExistingDirectory(dialog, tr("Select Archive Folder"))
            if folder:
                if SessionLogger.add_search_path(folder):
                    new_item = QListWidgetItem(f"✅ {folder}")
                    new_item.setData(Qt.UserRole, folder)  # 修复：设置 UserRole 数据
                    path_list.addItem(new_item)
                    self._refresh_sessions()
        btn_add.clicked.connect(add_path)
        btn_layout.addWidget(btn_add)
        
        # 导入 Session 文件按钮
        btn_import_session = QPushButton(f"📄 {tr('Import Session File')}")
        def import_session_file():
            file_path, _ = QFileDialog.getOpenFileName(
                dialog, tr("Select Session Log File"), "", 
                "Session JSON (*.json)"
            )
            if file_path:
                # 将 session 文件复制到默认目录并添加父文件夹到搜索路径
                import shutil
                src = Path(file_path)
                parent_folder = src.parent
                
                # 将父文件夹添加到搜索路径
                if SessionLogger.add_search_path(str(parent_folder)):
                    new_item = QListWidgetItem(f"✅ {parent_folder}")
                    new_item.setData(Qt.UserRole, str(parent_folder))
                    path_list.addItem(new_item)
                
                self._refresh_sessions()
                QMessageBox.information(dialog, tr("Import Success"), 
                    f"{tr('Session imported successfully!')}\n\n{tr('Path added')}: {parent_folder}")
        btn_import_session.clicked.connect(import_session_file)
        btn_layout.addWidget(btn_import_session)
        btn_layout.addWidget(btn_add)
        
        btn_locate = QPushButton(f"🔍 {tr('Locate')}")
        def locate_path():
            item = path_list.currentItem()
            if not item:
                QMessageBox.warning(dialog, tr("Error"), tr("Please select a path first."))
                return
            old_path = item.data(Qt.UserRole)
            if not old_path:
                QMessageBox.warning(dialog, tr("Error"), tr("Invalid path data."))
                return
            if Path(old_path).exists():
                QMessageBox.information(dialog, tr("Path Valid"), tr("This path is valid."))
                return
            
            # 尝试 Everything 自动查找
            new_path = SessionLogger.try_relocate_path(old_path)
            if new_path:
                SessionLogger.update_search_path(old_path, new_path)
                item.setText(f"✅ {new_path}")
                item.setData(Qt.UserRole, new_path)
                item.setForeground(Qt.white)
                QMessageBox.information(dialog, tr("Path Located"), f"{tr('Found at')}:\n{new_path}")
                self._refresh_sessions()
            else:
                # Everything 未找到，手动选择
                reply = QMessageBox.question(dialog, tr("Not Found"), 
                    tr("Could not auto-locate. Browse manually?"),
                    QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    folder = QFileDialog.getExistingDirectory(dialog, tr("Manual Select Folder"))
                    if folder:
                        SessionLogger.update_search_path(old_path, folder)
                        item.setText(f"✅ {folder}")
                        item.setData(Qt.UserRole, folder)
                        item.setForeground(Qt.white)
                        self._refresh_sessions()
        btn_locate.clicked.connect(locate_path)
        btn_layout.addWidget(btn_locate)
        
        btn_remove = QPushButton(f"🗑️ {tr('Remove')}")
        def remove_path():
            item = path_list.currentItem()
            if item:
                path = item.data(Qt.UserRole)
                if path:
                    SessionLogger.remove_search_path(path)
                path_list.takeItem(path_list.row(item))
                self._refresh_sessions()
        btn_remove.clicked.connect(remove_path)
        btn_layout.addWidget(btn_remove)
        
        btn_cleanup = QPushButton(f"🧹 {tr('Cleanup Invalid')}")
        def cleanup_invalid():
            count = SessionLogger.cleanup_invalid_paths()
            if count > 0:
                QMessageBox.information(dialog, tr("Cleanup Complete"), f"{tr('Removed')} {count} {tr('invalid paths')}")
                dialog.accept()
                self._show_path_manager()  # 重新打开刷新
            else:
                QMessageBox.information(dialog, tr("Cleanup Complete"), tr("No invalid paths found."))
        btn_cleanup.clicked.connect(cleanup_invalid)
        btn_layout.addWidget(btn_cleanup)
        
        layout.addLayout(btn_layout)
        
        # Everything 状态 - 显示详细诊断信息
        available, status_msg = SessionLogger.get_everything_status()
        if available:
            layout.addWidget(QLabel(f"<span style='color:#4CAF50'>✅ {status_msg}</span>"))
        else:
            lbl_hint = QLabel(f"<span style='color:#FF9800'>⚠️ {status_msg}</span><br>"
                              f"<a href='https://voidtools.com/'>{tr('Download')} Everything</a>")
            lbl_hint.setOpenExternalLinks(True)
            layout.addWidget(lbl_hint)
        
        # 关闭按钮
        btn_close = QPushButton(tr("Close"))
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def _check_everything_hint(self):
        """首次使用时检测 Everything 并显示提示"""
        from utils.session_logger import SessionLogger
        from qtpy.QtCore import QSettings
        
        settings = QSettings("NapariUser", "Recovery")
        if settings.value("everything_hint_shown", False):
            return
        
        if not SessionLogger.check_everything_available():
            QMessageBox.information(self, tr("Tip"),
                f"{tr('Everything search engine not detected.')}\n\n"
                f"{tr('Installing it enables')}:\n"
                f"• {tr('Auto-locate moved archive folders')}\n"
                f"• {tr('Millisecond full-disk search')}\n\n"
                f"{tr('Download')}: https://voidtools.com/")
        
        settings.setValue("everything_hint_shown", True)
    
    def _show_session_context_menu(self, pos):
        """显示会话右键菜单"""
        item = self.session_list.itemAt(pos)
        if not item:
            return
        
        idx = item.data(Qt.UserRole)
        if idx is None or idx >= len(self.sessions):
            return
        
        session = self.sessions[idx]
        log_path = session.get("_log_path", "")
        
        menu = QMenu(self)
        
        # 在文件浏览器中打开
        action_open_folder = QAction(f"📂 {tr('Open in Explorer')}", self)
        action_open_folder.triggered.connect(lambda: self._open_in_explorer(log_path))
        menu.addAction(action_open_folder)
        
        menu.addSeparator()
        
        # 收藏/取消收藏
        starred = session.get("starred", False)
        star_text = tr("Unstar Session") if starred else tr("Star Session")
        action_star = QAction(f"{'⭐' if not starred else '☆'} {star_text}", self)
        action_star.triggered.connect(lambda: self._toggle_star_session(session))
        menu.addAction(action_star)
        
        # 编辑标签
        action_label = QAction(f"🏷️ {tr('Edit Label')}", self)
        action_label.triggered.connect(lambda: self._edit_session_label(session))
        menu.addAction(action_label)
        
        # 合并到其他 session
        action_merge = QAction(f"🔗 {tr('Merge into...')}", self)
        action_merge.triggered.connect(lambda: self._merge_session(session))
        menu.addAction(action_merge)

        menu.addSeparator()

        # 删除会话
        action_delete = QAction(f"🗑️ {tr('Delete Session')}", self)
        action_delete.triggered.connect(lambda: self._delete_session(session))
        menu.addAction(action_delete)

        menu.exec_(self.session_list.mapToGlobal(pos))
    
    def _open_in_explorer(self, log_path):
        """在文件浏览器中打开会话所在文件夹"""
        import subprocess
        import os
        
        if not log_path:
            return
        
        path = Path(log_path)
        folder = path.parent if path.is_file() else path
        
        # 检查是否需要确认
        from qtpy.QtCore import QSettings
        settings = QSettings("NapariUser", "Recovery")
        ask_confirm = settings.value("ask_open_explorer", True, type=bool)
        
        if ask_confirm:
            msg = QMessageBox(self)
            msg.setWindowTitle(tr("Open in Explorer"))
            msg.setText(f"{tr('Open folder in file explorer?')}\n\n{folder}")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            
            chk_dont_ask = QCheckBox(tr("Don't ask again"))
            msg.setCheckBox(chk_dont_ask)
            
            if msg.exec_() != QMessageBox.Yes:
                return
            
            if chk_dont_ask.isChecked():
                settings.setValue("ask_open_explorer", False)
        
        # 打开文件浏览器
        if os.name == 'nt':  # Windows
            subprocess.run(['explorer', str(folder)])
        else:  # macOS / Linux
            subprocess.run(['open' if os.uname().sysname == 'Darwin' else 'xdg-open', str(folder)])
    
    def _delete_current_session(self):
        """删除当前选中的会话"""
        if not self.current_session:
            QMessageBox.warning(self, tr("Error"), tr("Please select a session first."))
            return
        self._delete_session(self.current_session)
    
    def _delete_session(self, session):
        """删除指定会话"""
        log_path = session.get("_log_path", "")
        if not log_path:
            return
        
        reply = QMessageBox.question(self, tr("Delete Session"),
            f"{tr('Are you sure you want to delete this session?')}\n\n{Path(log_path).name}",
            QMessageBox.Yes | QMessageBox.No)
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            path = Path(log_path)
            if path.exists():
                path.unlink()
            
            QMessageBox.information(self, tr("Session deleted"), tr("Session has been deleted."))
            self._refresh_sessions()
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"{tr('Failed to delete session')}: {e}")
    
    def _toggle_star_session(self, session):
        """切换会话收藏状态"""
        from utils.session_logger import SessionLogger
        log_path = session.get("_log_path", "")
        if not log_path:
            return
        
        current_starred = session.get("starred", False)
        SessionLogger.update_session_file(Path(log_path), starred=not current_starred)
        self._refresh_sessions()
    
    def _edit_session_label(self, session):
        """编辑会话标签"""
        from qtpy.QtWidgets import QInputDialog
        from utils.session_logger import SessionLogger
        
        log_path = session.get("_log_path", "")
        if not log_path:
            return
        
        current_label = session.get("label", "")
        new_label, ok = QInputDialog.getText(self, tr("Edit Label"), 
            tr("Enter label for this session:"), text=current_label)
        
        if ok:
            SessionLogger.update_session_file(Path(log_path), label=new_label)
            self._refresh_sessions()
    
    def _switch_to_import_tab(self):
        """切换到 Import 标签页"""
        from qtpy.QtWidgets import QTabWidget

        try:
            # 方法1: 向上查找 QTabWidget 父组件
            parent = self.parent()
            tab_widget = None

            while parent is not None:
                if isinstance(parent, QTabWidget):
                    tab_widget = parent
                    break
                # 检查父组件的子组件中是否有 QTabWidget
                for child in parent.children():
                    if isinstance(child, QTabWidget):
                        tab_widget = child
                        break
                if tab_widget:
                    break
                parent = parent.parent()

            if tab_widget:
                # 查找 Import 标签页
                for i in range(tab_widget.count()):
                    tab_text = tab_widget.tabText(i)
                    if 'Import' in tab_text or '导入' in tab_text:
                        tab_widget.setCurrentIndex(i)
                        print(f"[Recovery] Switched to Import tab (index {i})")
                        return True

            print("[Recovery] Could not find tab widget")
            return False

        except Exception as e:
            print(f"[Recovery] Tab switch error: {e}")
            return False

    # =========================================================================
    # Session Merge
    # =========================================================================

    def _merge_session(self, source_session):
        """将 source_session (B) 合并到用户选择的 target_session (A) 中。"""
        from qtpy.QtWidgets import QInputDialog
        from utils.session_logger import SessionLogger

        source_id = source_session.get("session_id", "?")
        source_path = Path(source_session.get("_log_path", ""))
        if not source_path.exists():
            QMessageBox.warning(self, tr("Merge"), tr("Source session file not found."))
            return

        # 选择目标 session
        candidates = []
        candidate_map = {}
        for s in self.sessions:
            sid = s.get("session_id", "")
            if sid == source_id:
                continue
            meta = s.get("metadata", {})
            label = s.get("label", "") or s.get("_auto_label", "")
            display = f"[{sid}] {label} ({meta.get('substance', '')} / {meta.get('dataset_id', '')})"
            candidates.append(display)
            candidate_map[display] = s

        if not candidates:
            QMessageBox.information(self, tr("Merge"), tr("No other sessions available for merge."))
            return

        chosen, ok = QInputDialog.getItem(self, tr("Merge into..."),
            f"{tr('Select target session to merge into')}:\n"
            f"(Source: [{source_id}])",
            candidates, 0, False)
        if not ok or chosen not in candidate_map:
            return

        target_session = candidate_map[chosen]
        target_path = Path(target_session.get("_log_path", ""))
        if not target_path.exists():
            QMessageBox.warning(self, tr("Merge"), tr("Target session file not found."))
            return

        # 确认
        source_actions = source_session.get("actions", [])
        target_actions = target_session.get("actions", [])
        reply = QMessageBox.question(self, tr("Confirm Merge"),
            f"{tr('Merge session')} [{source_id}] ({len(source_actions)} actions)\n"
            f"{tr('into session')} [{target_session.get('session_id', '?')}] ({len(target_actions)} actions)?\n\n"
            f"{tr('The source session will be marked as merged.')}",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        # 执行合并
        try:
            self._execute_merge(source_path, target_path)
            QMessageBox.information(self, tr("Merge"), tr("Sessions merged successfully."))
            self._refresh_sessions()
        except Exception as e:
            QMessageBox.critical(self, tr("Merge"), f"{tr('Merge failed')}: {e}")

    def _execute_merge(self, source_path: Path, target_path: Path):
        """执行合并: B 的 actions rebase 并按时间戳排序插入 A."""
        with open(source_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        with open(target_path, 'r', encoding='utf-8') as f:
            target_data = json.load(f)

        # 计算 target 的 max action counter
        max_counter = 0
        for a in target_data.get("actions", []):
            parts = a.get("id", "").rsplit("_", 1)
            if len(parts) == 2:
                try:
                    max_counter = max(max_counter, int(parts[-1]))
                except ValueError:
                    pass

        # Rebase source actions: 更新 action_id，重映射 undo target 引用
        id_map = {}
        rebased_actions = []
        for a in source_data.get("actions", []):
            old_id = a.get("id", "")
            max_counter += 1
            new_id = f"{target_data.get('session_id', 'merged')}_{max_counter:04d}"
            id_map[old_id] = new_id
            a["id"] = new_id
            # Rebase undo target references
            if a.get("action") == "undo":
                old_target = a.get("params", {}).get("target", "")
                if old_target in id_map:
                    a["params"]["target"] = id_map[old_target]
            rebased_actions.append(a)

        # 合并: 按时间戳排序
        all_actions = target_data.get("actions", []) + rebased_actions
        all_actions.sort(key=lambda a: a.get("timestamp", ""))
        target_data["actions"] = all_actions

        # 合并 layer_aliases
        source_aliases = source_data.get("layer_aliases", {})
        if source_aliases:
            target_aliases = target_data.get("layer_aliases", {})
            target_aliases.update(source_aliases)
            target_data["layer_aliases"] = target_aliases

        # 重算 checksum
        target_data["checksum"] = compute_actions_checksum(target_data["actions"])

        # 保存 target
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(target_data, f, indent=2, cls=NumpyEncoder)

        # 标记 source 为 merged
        source_data["status"] = "merged"
        source_data["merged_into"] = target_data.get("session_id", "")
        with open(source_path, 'w', encoding='utf-8') as f:
            json.dump(source_data, f, indent=2, cls=NumpyEncoder)

"""
Helpers for inferring the current MagicImageJ session root for auto-crop tools.

The goal is to avoid asking users to manually pick a session folder when
MagicImageJ already knows enough context through:

1. RecoveryWidget.current_session
2. Session logger metadata / import actions
3. Global QSettings archive_path
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QDockWidget
except ImportError:
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QDockWidget

from utils.session_logger import SessionLogger


def normalize_existing_dir(path_like: Any) -> Optional[Path]:
    if not path_like:
        return None
    try:
        path = Path(str(path_like)).resolve()
    except Exception:
        return None
    if path.exists() and path.is_dir():
        return path
    return None


def find_session_root_from_path(path_like: Any) -> Optional[Path]:
    path = normalize_existing_dir(path_like)
    if path is None:
        return None

    candidates = [path] + list(path.parents)
    for candidate in candidates:
        if (candidate / "processing_log.json").exists():
            return candidate
    return None


def extract_session_root_from_session_data(session_data: dict[str, Any] | None) -> Optional[Path]:
    if not session_data:
        return None

    metadata = session_data.get("metadata", {})
    archive_path = metadata.get("archive_path")
    session_root = find_session_root_from_path(archive_path)
    if session_root is not None:
        return session_root

    actions = session_data.get("actions", [])
    for action in reversed(actions):
        params = action.get("params", {})
        for key in ("archive_path", "source_path"):
            session_root = find_session_root_from_path(params.get(key))
            if session_root is not None:
                return session_root

    log_path = session_data.get("_log_path")
    if log_path:
        log_parent = find_session_root_from_path(Path(log_path).parent)
        if log_parent is not None:
            return log_parent

    return None


def find_recovery_widget(viewer: Any) -> Any | None:
    if viewer is None:
        return None
    try:
        qt_window = viewer.window._qt_window
    except Exception:
        return None

    try:
        from widgets.recovery_widget import RecoveryWidget
    except Exception:
        return None

    for dock in qt_window.findChildren(QDockWidget):
        widget = dock.widget()
        if widget is None:
            continue
        if isinstance(widget, RecoveryWidget):
            return widget
        for child in widget.findChildren(RecoveryWidget):
            return child
    return None


def find_current_session_root(
    viewer: Any | None = None,
    recovery_widget: Any | None = None,
) -> Optional[Path]:
    if recovery_widget is None and viewer is not None:
        recovery_widget = find_recovery_widget(viewer)

    if recovery_widget is not None:
        current_session = getattr(recovery_widget, "current_session", None)
        session_root = extract_session_root_from_session_data(current_session)
        if session_root is not None:
            return session_root

    archive_path = QSettings("NapariUser", "Global").value("archive_path", "")
    session_root = find_session_root_from_path(archive_path)
    if session_root is not None:
        return session_root

    for log_path in SessionLogger.find_all_sessions(limit=10):
        session_data = SessionLogger.load_from_file(log_path)
        session_root = extract_session_root_from_session_data(session_data)
        if session_root is not None:
            return session_root

    return None

"""
Phase 2 data-layer test — Chapter management in SessionLogger.

Tests:
1. New chapter auto-created on first import action
2. Subsequent imports demote prior chapter to historical, start new "current"
3. Non-import actions append to current chapter without creating new ones
4. resume_from_file wraps legacy session (no chapters) as 1 current chapter
5. get_actions_for_chapter returns correct slice
6. get_chapter_for_action_index works
7. _to_dict serializes chapters; reading back preserves them
8. _compute_checksum is stable across chapter additions (chapters NOT in checksum)
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _fresh_logger():
    """Create a logger, bypassing singleton lock for isolated test."""
    from utils.session_logger import SessionLogger
    SessionLogger._instance = None
    return SessionLogger.get_instance()


def test_first_import_creates_chapter():
    logger = _fresh_logger()
    assert logger.chapters == []
    logger.log_action("import", "load_png_sequence", {
        "source_path": "C:/path/to/data",
        "total_frames": 261,
    })
    assert len(logger.chapters) == 1
    ch = logger.chapters[0]
    assert ch["status"] == "current"
    assert ch["start_action_index"] == 0
    assert "data" in ch["label"].lower() or "PNG" in ch["label"]
    assert "261f" in ch["label"]
    print(f"[OK] first import created chapter: '{ch['label']}'")


def test_non_import_appends_no_chapter():
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 100})
    n_chs_before = len(logger.chapters)
    logger.log_action("geometry", "flip_horizontal", {})
    logger.log_action("drift", "apply_correction", {})
    assert len(logger.chapters) == n_chs_before
    print("[OK] non-import actions append without creating chapters")


def test_second_import_demotes_prior():
    logger = _fresh_logger()
    logger.log_action("import", "load_dm4_sequence", {"archive_path": "X1", "total_frames": 1800})
    logger.log_action("geometry", "flip", {})
    logger.log_action("import", "load_dm4_sequence", {"archive_path": "X2", "total_frames": 260})
    assert len(logger.chapters) == 2
    assert logger.chapters[0]["status"] == "historical"
    assert logger.chapters[1]["status"] == "current"
    assert logger.chapters[1]["start_action_index"] == 2
    print("[OK] second import demoted prior chapter and started new")


def test_bootstrap_chapter_on_first_non_import():
    """If first action is NOT an import, a session_start chapter is bootstrapped."""
    logger = _fresh_logger()
    logger.log_action("drift", "apply_correction", {})
    assert len(logger.chapters) == 1
    assert logger.chapters[0]["label"] == "session_start"
    print("[OK] bootstrap session_start chapter on first non-import action")


def test_get_chapter_for_action_index():
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 100})  # idx 0 -> ch0
    logger.log_action("geometry", "flip", {})  # idx 1 -> ch0
    logger.log_action("import", "load_png_sequence", {"total_frames": 200})  # idx 2 -> ch1
    logger.log_action("drift", "apply", {})  # idx 3 -> ch1

    ch_at_0 = logger.get_chapter_for_action_index(0)
    ch_at_1 = logger.get_chapter_for_action_index(1)
    ch_at_2 = logger.get_chapter_for_action_index(2)
    ch_at_3 = logger.get_chapter_for_action_index(3)
    assert ch_at_0["id"] == ch_at_1["id"]
    assert ch_at_2["id"] == ch_at_3["id"]
    assert ch_at_0["id"] != ch_at_2["id"]
    print(f"[OK] chapter membership: actions 0/1 -> {ch_at_0['id']}, 2/3 -> {ch_at_2['id']}")


def test_get_actions_for_chapter():
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 100})  # 0
    logger.log_action("geometry", "flip", {})  # 1
    logger.log_action("geometry", "crop", {})  # 2
    logger.log_action("import", "load_png_sequence", {"total_frames": 200})  # 3
    logger.log_action("drift", "apply", {})  # 4

    ch1 = logger.chapters[0]
    ch2 = logger.chapters[1]
    acts1 = logger.get_actions_for_chapter(ch1["id"])
    acts2 = logger.get_actions_for_chapter(ch2["id"])
    assert len(acts1) == 3
    assert acts1[0]["action"] == "load_png_sequence"
    assert acts1[1]["action"] == "flip"
    assert acts1[2]["action"] == "crop"
    assert len(acts2) == 2
    assert acts2[0]["action"] == "load_png_sequence"
    assert acts2[1]["action"] == "apply"
    print(f"[OK] ch1 has 3 actions, ch2 has 2 actions (correct slicing)")


def test_to_dict_includes_chapters():
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 50})
    d = logger._to_dict()
    assert "chapters" in d
    assert len(d["chapters"]) == 1
    assert "checksum" in d
    print("[OK] _to_dict includes chapters field")


def test_checksum_stable_across_chapter_changes():
    """Adding/changing chapters should NOT change actions checksum."""
    logger = _fresh_logger()
    logger.log_action("geometry", "flip", {})
    cks_before = logger._compute_checksum()
    # Bootstrap chapter was added; checksum should still match actions-only hash
    logger.log_action("import", "load_png_sequence", {"total_frames": 10})
    cks_after = logger._compute_checksum()
    # cks changed because actions changed (1 -> 2 entries). But chapter metadata changes alone shouldn't.
    # Simulate: tweak chapter without touching actions
    logger.chapters[-1]["status"] = "historical"
    cks_after2 = logger._compute_checksum()
    assert cks_after == cks_after2, "chapter status mutation should not affect actions checksum"
    print(f"[OK] checksum stable under chapter metadata mutation")


def test_resume_from_file_legacy():
    """Legacy session (no chapters field) should be wrapped as 1 current chapter."""
    from utils.session_logger import SessionLogger

    tmp = Path(tempfile.gettempdir()) / "phase2_legacy_session.json"
    tmp.write_text(json.dumps({
        "session_id": "abcd1234",
        "created_at": "2026-05-01T00:00:00",
        "status": "in_progress",
        "starred": False,
        "label": "",
        "metadata": {"substance": "X", "dataset_id": "ds1", "archive_path": ""},
        "actions": [
            {"id": "import_load_png_1", "timestamp": "2026-05-01T00:00:01",
             "widget": "import", "action": "load_png_sequence",
             "params": {"total_frames": 100}, "result": "success"},
            {"id": "geometry_flip_2", "timestamp": "2026-05-01T00:00:02",
             "widget": "geometry", "action": "flip", "params": {}, "result": "success"},
        ],
        "checksum": "ignored",
    }), encoding='utf-8')

    SessionLogger._instance = None
    logger = SessionLogger.resume_from_file(tmp)
    assert len(logger.chapters) >= 1
    assert logger.chapters[0]["status"] == "current"
    assert logger.chapters[0]["start_action_index"] == 0
    print(f"[OK] legacy session resumed; got {len(logger.chapters)} chapter(s)")

    # Continue logging — new import should bring up a new current chapter and demote the wrapped one
    logger.log_action("import", "load_png_sequence", {"total_frames": 200})
    # First chapter (legacy wrap) should now be historical
    assert logger.chapters[0]["status"] == "historical"
    assert logger.chapters[-1]["status"] == "current"
    print(f"[OK] post-resume import demoted legacy wrap chapter")

    tmp.unlink()


def test_get_chapters_returns_copy():
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 10})
    chs = logger.get_chapters()
    chs.append({"fake": "data"})
    # Mutation should not affect the logger's internal state
    assert len(logger.chapters) == 1
    print("[OK] get_chapters() returns a shallow copy")


def test_backward_compat_to_dict_empty():
    """Logger with no actions should not produce chapters key."""
    logger = _fresh_logger()
    d = logger._to_dict()
    assert "chapters" not in d
    print("[OK] empty session does not include chapters in _to_dict")


if __name__ == "__main__":
    print("=== Phase 2 chapter tests ===")
    test_first_import_creates_chapter()
    test_non_import_appends_no_chapter()
    test_second_import_demotes_prior()
    test_bootstrap_chapter_on_first_non_import()
    test_get_chapter_for_action_index()
    test_get_actions_for_chapter()
    test_to_dict_includes_chapters()
    test_checksum_stable_across_chapter_changes()
    test_resume_from_file_legacy()
    test_get_chapters_returns_copy()
    test_backward_compat_to_dict_empty()
    print("\nAll Phase 2 chapter tests passed.")

"""
Phase 3 data-layer test — per-action state management.

Tests:
1. New action gets state="current" by default
2. On new chapter creation, prior chapter actions flip to state="historical"
3. set_action_state mutates state correctly
4. effective_actions_for_replay() skips historical/disabled
5. Checksum stable across state mutations (sanitized hash)
6. load_from_file checksum verification works with state field
7. Re-running Phase 2 tests still pass (no regression)
"""
import json
import sys
import tempfile
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _fresh_logger():
    from utils.session_logger import SessionLogger
    SessionLogger._instance = None
    return SessionLogger.get_instance()


def test_new_action_state_default():
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 50})
    assert logger.actions[0].get("state") == "current"
    print("[OK] new action defaults to state='current'")


def test_chapter_boundary_flips_state():
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 100})  # idx 0
    logger.log_action("geometry", "flip", {})  # idx 1
    logger.log_action("geometry", "crop", {})  # idx 2
    # All actions in chapter 1 should be 'current' so far
    assert all(a["state"] == "current" for a in logger.actions)

    # New import — chapter 1 actions flip to historical
    logger.log_action("import", "load_dm4_sequence", {"total_frames": 200})  # idx 3
    states = [a["state"] for a in logger.actions]
    assert states[0] == "historical"
    assert states[1] == "historical"
    assert states[2] == "historical"
    assert states[3] == "current"
    print(f"[OK] new chapter demoted prior actions: states={states}")


def test_set_action_state():
    logger = _fresh_logger()
    logger.log_action("geometry", "flip", {})
    aid = logger.actions[0]["id"]

    ok = logger.set_action_state(aid, "disabled")
    assert ok
    assert logger.actions[0]["state"] == "disabled"

    bad = logger.set_action_state("nonexistent_id", "current")
    assert not bad
    print("[OK] set_action_state works on valid ids, returns False on unknown")


def test_set_action_state_validation():
    logger = _fresh_logger()
    logger.log_action("geometry", "flip", {})
    aid = logger.actions[0]["id"]
    try:
        logger.set_action_state(aid, "bogus_state")
        assert False, "should raise"
    except ValueError:
        pass
    print("[OK] set_action_state rejects invalid state values")


def test_effective_actions_for_replay():
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 10})  # 0 current
    logger.log_action("geometry", "flip", {})  # 1 current
    logger.log_action("import", "load_dm4_sequence", {"total_frames": 5})  # 2 current, 0/1 -> historical
    logger.log_action("drift", "apply", {})  # 3 current

    eff = logger.effective_actions_for_replay()
    # Only current actions (2, 3) should appear; 0 and 1 are historical
    eff_ids = [a["id"] for a in eff]
    assert logger.actions[0]["id"] not in eff_ids
    assert logger.actions[1]["id"] not in eff_ids
    assert logger.actions[2]["id"] in eff_ids
    assert logger.actions[3]["id"] in eff_ids

    # Disable one current action
    logger.set_action_state(logger.actions[3]["id"], "disabled")
    eff = logger.effective_actions_for_replay()
    eff_ids = [a["id"] for a in eff]
    assert logger.actions[3]["id"] not in eff_ids
    print("[OK] effective_actions_for_replay skips historical + disabled")


def test_checksum_stable_across_state_changes():
    logger = _fresh_logger()
    logger.log_action("geometry", "flip", {})
    logger.log_action("geometry", "crop", {})
    cks_before = logger._compute_checksum()

    # Mutate state
    aid = logger.actions[0]["id"]
    logger.set_action_state(aid, "disabled")
    cks_after = logger._compute_checksum()

    assert cks_before == cks_after, (
        f"checksum should be stable across state changes\n"
        f"  before: {cks_before}\n"
        f"  after:  {cks_after}"
    )
    print(f"[OK] checksum stable across state mutations: {cks_before[:16]}...")


def test_checksum_changes_on_non_state_field():
    logger = _fresh_logger()
    logger.log_action("geometry", "flip", {})
    cks_before = logger._compute_checksum()

    # Mutate something that IS part of the hash
    logger.actions[0]["params"] = {"tampered": True}
    cks_after = logger._compute_checksum()

    assert cks_before != cks_after, "tampering with params should change checksum"
    print(f"[OK] checksum DOES change for non-state field mutation")


def test_load_from_file_legacy_no_state():
    """A legacy session JSON (no state field) should load with checksum-valid=True."""
    from utils.session_logger import SessionLogger

    actions = [
        {"id": "geometry_flip_1", "timestamp": "2026-05-01T00:00:01",
         "widget": "geometry", "action": "flip", "params": {}, "result": "success"},
    ]
    sanitized = [{k: v for k, v in a.items() if k != "state"} for a in actions]
    actions_str = json.dumps(sanitized, sort_keys=True)
    cks = hashlib.sha256(actions_str.encode('utf-8')).hexdigest()

    data = {
        "session_id": "phase3test",
        "created_at": "2026-05-01T00:00:00",
        "status": "completed",
        "starred": False,
        "label": "",
        "metadata": {"substance": "", "dataset_id": "", "archive_path": ""},
        "actions": actions,
        "checksum": cks,
    }

    tmp = Path(tempfile.gettempdir()) / "phase3_legacy_session.json"
    tmp.write_text(json.dumps(data), encoding='utf-8')

    loaded = SessionLogger.load_from_file(tmp)
    assert loaded is not None
    assert loaded.get("_checksum_valid") is True
    print("[OK] legacy session (no state field) loads with valid checksum")

    tmp.unlink()


def test_load_from_file_with_state_mutation_keeps_valid():
    """A session JSON saved AFTER state mutation should still verify."""
    from utils.session_logger import SessionLogger

    logger = _fresh_logger()
    logger.log_action("geometry", "flip", {})
    aid = logger.actions[0]["id"]
    logger.set_action_state(aid, "historical")

    tmp = Path(tempfile.gettempdir()) / "phase3_mutated_session.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(logger._to_dict(), f)

    loaded = SessionLogger.load_from_file(tmp)
    assert loaded is not None
    assert loaded.get("_checksum_valid") is True
    # State should be preserved
    assert loaded["actions"][0]["state"] == "historical"
    print("[OK] mutated-state session JSON loads with valid checksum + state preserved")

    tmp.unlink()


if __name__ == "__main__":
    print("=== Phase 3 state tests ===")
    test_new_action_state_default()
    test_chapter_boundary_flips_state()
    test_set_action_state()
    test_set_action_state_validation()
    test_effective_actions_for_replay()
    test_checksum_stable_across_state_changes()
    test_checksum_changes_on_non_state_field()
    test_load_from_file_legacy_no_state()
    test_load_from_file_with_state_mutation_keeps_valid()
    print("\nAll Phase 3 state tests passed.")

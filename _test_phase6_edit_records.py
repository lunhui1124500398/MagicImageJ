"""
Phase 6 data-layer test — edit_records (disable/enable/delete/update_params).

Tests:
1. add_edit_record returns id, stores record
2. invalid edit_type raises ValueError
3. compute_effective_actions: disable
4. compute_effective_actions: delete
5. compute_effective_actions: enable (after disable, restore to current)
6. compute_effective_actions: update_params merges new_params
7. Multiple edit_records on same action: last write wins (chronological apply)
8. Original actions array UNCHANGED after edit_records added (immutability)
9. Checksum stable across edit_record additions (edit_records not in checksum)
10. Phase 5 regression: replay still works without edit_records
"""
import sys
import hashlib
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _fresh_logger():
    from utils.session_logger import SessionLogger
    SessionLogger._instance = None
    return SessionLogger.get_instance()


def test_add_edit_record():
    logger = _fresh_logger()
    aid = logger.log_action("geometry", "flip", {})
    rid = logger.add_edit_record("disable", aid)
    assert rid.startswith("edit_")
    assert len(logger.edit_records) == 1
    assert logger.edit_records[0]["edit_type"] == "disable"
    assert logger.edit_records[0]["target_action_id"] == aid
    print(f"[OK] add_edit_record returns id={rid}")


def test_invalid_edit_type():
    logger = _fresh_logger()
    logger.log_action("geometry", "flip", {})
    try:
        logger.add_edit_record("bogus", "any_id")
        assert False, "should raise"
    except ValueError:
        pass
    print("[OK] invalid edit_type rejected with ValueError")


def test_compute_disable():
    logger = _fresh_logger()
    aid1 = logger.log_action("geometry", "flip", {})
    aid2 = logger.log_action("geometry", "crop", {})
    logger.add_edit_record("disable", aid1)
    eff = logger.compute_effective_actions()
    eff_ids = [a["id"] for a in eff]
    assert aid1 not in eff_ids
    assert aid2 in eff_ids
    # Original actions unchanged
    assert logger.actions[0]["state"] == "current", "actions[] should NOT mutate"
    print("[OK] disable edit_record removes action from effective list")


def test_compute_delete():
    logger = _fresh_logger()
    aid1 = logger.log_action("geometry", "flip", {})
    aid2 = logger.log_action("geometry", "crop", {})
    logger.add_edit_record("delete", aid1)
    eff = logger.compute_effective_actions()
    eff_ids = [a["id"] for a in eff]
    assert aid1 not in eff_ids
    assert aid2 in eff_ids
    print("[OK] delete edit_record removes action from effective list")


def test_compute_enable_after_disable():
    logger = _fresh_logger()
    aid = logger.log_action("geometry", "flip", {})
    logger.add_edit_record("disable", aid)
    logger.add_edit_record("enable", aid)
    eff = logger.compute_effective_actions()
    eff_ids = [a["id"] for a in eff]
    assert aid in eff_ids  # enabled again
    print("[OK] enable edit_record restores disabled action")


def test_compute_update_params():
    logger = _fresh_logger()
    aid = logger.log_action("drift", "apply_correction", {"kernel_size": 11, "max_shift": 5})
    logger.add_edit_record("update_params", aid, new_params={"kernel_size": 21})
    eff = logger.compute_effective_actions()
    assert eff[0]["params"]["kernel_size"] == 21
    assert eff[0]["params"]["max_shift"] == 5  # original unchanged via merge
    # Original action unchanged
    assert logger.actions[0]["params"]["kernel_size"] == 11
    print("[OK] update_params merges new_params, original immutable")


def test_multiple_edits_chronological():
    logger = _fresh_logger()
    aid = logger.log_action("drift", "apply_correction", {"k": 1})
    logger.add_edit_record("update_params", aid, new_params={"k": 5})
    logger.add_edit_record("update_params", aid, new_params={"k": 10})
    eff = logger.compute_effective_actions()
    assert eff[0]["params"]["k"] == 10  # last write wins
    print("[OK] multiple edits applied chronologically, last write wins")


def test_actions_array_immutable():
    """After many edit_records, actions[] should be identical to what we started with."""
    logger = _fresh_logger()
    aid1 = logger.log_action("geometry", "flip", {"a": 1})
    aid2 = logger.log_action("drift", "apply_correction", {"k": 11})
    snapshot1 = json.dumps(logger.actions, sort_keys=True)
    logger.add_edit_record("disable", aid1)
    logger.add_edit_record("update_params", aid2, new_params={"k": 99})
    logger.add_edit_record("delete", aid2)
    snapshot2 = json.dumps(logger.actions, sort_keys=True)
    assert snapshot1 == snapshot2, "actions[] should not mutate via edit_records"
    print("[OK] actions array unchanged after add_edit_record calls")


def test_checksum_stable_across_edit_records():
    logger = _fresh_logger()
    logger.log_action("geometry", "flip", {})
    aid2 = logger.log_action("drift", "apply_correction", {"k": 11})
    cks_before = logger._compute_checksum()
    logger.add_edit_record("disable", aid2)
    logger.add_edit_record("update_params", aid2, new_params={"k": 99})
    cks_after = logger._compute_checksum()
    assert cks_before == cks_after, "edit_records should not affect actions checksum"
    print("[OK] checksum stable across edit_records additions")


def test_phase5_regression_compute():
    """compute_effective_actions should also work without any edit_records (acts like
    effective_actions_for_replay)."""
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 100})
    logger.log_action_trial("drift", "apply_correction", {"k": 11})
    logger.log_action_trial("drift", "apply_correction", {"k": 13})
    logger.log_action_trial("drift", "apply_correction", {"k": 15})
    eff = logger.compute_effective_actions()
    # Same as Phase 5: import (current) + last trial (k=15)
    drift_in_eff = [a for a in eff if a.get("widget") == "drift"]
    assert len(drift_in_eff) == 1
    assert drift_in_eff[0]["params"]["k"] == 15
    print("[OK] compute_effective_actions = effective_actions_for_replay when no edit_records")


if __name__ == "__main__":
    print("=== Phase 6 edit_records tests ===")
    test_add_edit_record()
    test_invalid_edit_type()
    test_compute_disable()
    test_compute_delete()
    test_compute_enable_after_disable()
    test_compute_update_params()
    test_multiple_edits_chronological()
    test_actions_array_immutable()
    test_checksum_stable_across_edit_records()
    test_phase5_regression_compute()
    print("\nAll Phase 6 edit_records tests passed.")

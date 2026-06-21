"""
Phase 5 data-layer test — trial / committed state machine.

Tests:
1. log_action_trial creates action with state="trial"
2. log_action_committed creates action with state="committed"
3. commit_last_action_of promotes last trial of given type to committed
4. effective_actions_for_replay: trial dedup logic
   - if committed exists for (chapter, widget, action_type) → skip all trials of that key
   - else: keep ONLY the last trial of that key
5. Multiple chapters: trial dedup respects chapter boundary
6. Phase 3 + Phase 4 regression: no impact
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _fresh_logger():
    from utils.session_logger import SessionLogger
    SessionLogger._instance = None
    return SessionLogger.get_instance()


def test_log_action_trial():
    logger = _fresh_logger()
    aid = logger.log_action_trial("drift", "apply_correction", {"k": 11})
    assert logger.actions[-1]["state"] == "trial"
    assert logger.actions[-1]["id"] == aid
    print("[OK] log_action_trial sets state='trial'")


def test_log_action_committed():
    logger = _fresh_logger()
    aid = logger.log_action_committed("geometry", "flip", {"axis": "h"})
    assert logger.actions[-1]["state"] == "committed"
    print("[OK] log_action_committed sets state='committed'")


def test_commit_last_action_of():
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 100})
    logger.log_action_trial("drift", "apply_correction", {"k": 11})
    logger.log_action_trial("drift", "apply_correction", {"k": 13})
    logger.log_action_trial("drift", "apply_correction", {"k": 15})  # this is the last trial

    aid = logger.commit_last_action_of("drift", "apply_correction")
    assert aid is not None
    assert logger.actions[-1]["state"] == "committed"
    assert logger.actions[-1]["params"]["k"] == 15
    # The earlier trials are still trial
    assert logger.actions[1]["state"] == "trial"
    assert logger.actions[2]["state"] == "trial"
    print("[OK] commit_last_action_of promoted the latest trial (k=15)")


def test_commit_no_match():
    logger = _fresh_logger()
    logger.log_action_trial("drift", "apply_correction", {"k": 11})
    # Try to commit a non-matching type
    aid = logger.commit_last_action_of("enhance", "apply_enhance")
    assert aid is None
    print("[OK] commit_last_action_of returns None when nothing matches")


def test_replay_trial_dedup_no_committed():
    """When no committed exists for (chapter, widget, action), keep the LAST trial only."""
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 100})  # current
    logger.log_action_trial("drift", "apply_correction", {"k": 11})
    logger.log_action_trial("drift", "apply_correction", {"k": 13})
    logger.log_action_trial("drift", "apply_correction", {"k": 15})

    eff = logger.effective_actions_for_replay()
    # Should have: import (current) + only the last drift trial (k=15)
    drift_in_eff = [a for a in eff if a.get("widget") == "drift"]
    assert len(drift_in_eff) == 1
    assert drift_in_eff[0]["params"]["k"] == 15
    print(f"[OK] trial dedup: kept only last drift trial (k=15); total eff = {len(eff)}")


def test_replay_trial_skipped_when_committed_exists():
    """When a committed exists for the (chapter, widget, action_type) key, skip all trials of that key."""
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 100})  # current
    logger.log_action_trial("drift", "apply_correction", {"k": 11})
    logger.log_action_trial("drift", "apply_correction", {"k": 13})
    logger.log_action_committed("drift", "apply_correction", {"k": 99})  # committed superseding

    eff = logger.effective_actions_for_replay()
    drift_in_eff = [a for a in eff if a.get("widget") == "drift"]
    # All trials skipped; only the committed remains
    assert len(drift_in_eff) == 1
    assert drift_in_eff[0]["params"]["k"] == 99
    assert drift_in_eff[0]["state"] == "committed"
    print("[OK] trials skipped when committed exists for the same key")


def test_trial_dedup_respects_chapter_boundary():
    """Different chapters have independent (widget+action_type) keys."""
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 100})  # ch1 current
    logger.log_action_trial("drift", "apply_correction", {"k": 11})            # ch1 trial
    logger.log_action_trial("drift", "apply_correction", {"k": 13})            # ch1 trial (last)
    logger.log_action("import", "load_dm4_sequence", {"total_frames": 50})    # ch2 current; demotes ch1
    logger.log_action_trial("drift", "apply_correction", {"k": 21})            # ch2 trial
    logger.log_action_trial("drift", "apply_correction", {"k": 23})            # ch2 trial (last)

    eff = logger.effective_actions_for_replay()
    # ch1 actions all become historical, skipped
    # ch2 actions: import (current) + last trial (k=23)
    drift_in_eff = [a for a in eff if a.get("widget") == "drift"]
    assert len(drift_in_eff) == 1
    assert drift_in_eff[0]["params"]["k"] == 23
    # Total eff includes ch2 import + ch2 last drift trial = 2 actions
    assert len(eff) == 2
    print("[OK] trial dedup respects chapter boundary")


def test_phase3_regression():
    """Phase 3 logic (no trials used) should still work after Phase 5 changes."""
    logger = _fresh_logger()
    logger.log_action("import", "load_png_sequence", {"total_frames": 50})
    logger.log_action("geometry", "flip", {})
    logger.log_action("import", "load_dm4_sequence", {"total_frames": 30})  # ch2 starts
    logger.log_action("drift", "apply_correction", {})

    # ch1 actions should be historical, ch2 current
    states = [a["state"] for a in logger.actions]
    assert states == ["historical", "historical", "current", "current"]

    eff = logger.effective_actions_for_replay()
    eff_ids = [a["id"] for a in eff]
    # Only ch2 actions should be included
    assert logger.actions[0]["id"] not in eff_ids  # ch1 historical
    assert logger.actions[1]["id"] not in eff_ids
    assert logger.actions[2]["id"] in eff_ids
    assert logger.actions[3]["id"] in eff_ids
    print("[OK] Phase 3 regression: state transitions still correct")


def test_checksum_with_phase5_states():
    """Trial / committed states should still strip cleanly for checksum stability."""
    logger = _fresh_logger()
    logger.log_action_trial("drift", "apply_correction", {"k": 11})
    cks_a = logger._compute_checksum()
    logger.commit_last_action_of("drift", "apply_correction")
    cks_b = logger._compute_checksum()
    assert cks_a == cks_b, "state change trial->committed should not break checksum"
    print("[OK] checksum stable: trial -> committed")


if __name__ == "__main__":
    print("=== Phase 5 trial/committed tests ===")
    test_log_action_trial()
    test_log_action_committed()
    test_commit_last_action_of()
    test_commit_no_match()
    test_replay_trial_dedup_no_committed()
    test_replay_trial_skipped_when_committed_exists()
    test_trial_dedup_respects_chapter_boundary()
    test_phase3_regression()
    test_checksum_with_phase5_states()
    print("\nAll Phase 5 trial/committed tests passed.")

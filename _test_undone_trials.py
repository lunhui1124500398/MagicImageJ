"""
Regression test for the 2026-05-30 bug: repeated drift trials "overwritten" in
Recovery. Root cause: the drift "Apply -> Ctrl+Z -> retry" loop logs an `undo`
record per reverted preview, and the Recovery list hid ALL undone actions, so
repeated drift trials collapsed to the last non-undone one.

Fix is display-only: undone *trials* stay visible as exploration history (dimmed,
unchecked), while replay (session_logger.effective_actions_for_replay) already
excludes them. This test pins:
  - _action_display_class(): undone trial -> visible-as-history, undone non-trial -> hidden
  - _is_last_trial_of_key(undone_ids): auto-selected trial == last NON-undone trial
  - the UI auto-selection set == effective_actions_for_replay() (replay parity)

Run from the MagicImageJ env:  python _test_undone_trials.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from widgets.recovery_widget import _action_display_class, RecoveryWidget  # noqa: E402
from utils.session_logger import SessionLogger  # noqa: E402

_passed = 0
_failed = 0


def check(label, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label}")


# Mirror the real session 20260530_..._ca87f33b: import + 3 drift trials, the
# first two reverted with Ctrl+Z (undo), the last one kept.
def make_actions():
    return [
        {"id": "import_1", "widget": "import", "action": "load_dm4_sequence", "state": "current"},
        {"id": "d2", "widget": "drift", "action": "apply_correction", "state": "trial"},
        {"id": "undo_3", "widget": "system", "action": "undo", "params": {"target": "d2"}},
        {"id": "d4", "widget": "drift", "action": "apply_correction", "state": "trial"},
        {"id": "undo_5", "widget": "system", "action": "undo", "params": {"target": "d4"}},
        {"id": "d6", "widget": "drift", "action": "apply_correction", "state": "trial"},
    ]


CHAPTERS = [{"id": "ch1", "start_action_index": 0, "status": "current"}]


class _Dummy:
    """Minimal stand-in: _is_last_trial_of_key only reads _session_chapters_sorted,
    so we call it as an unbound method on this — no QWidget / QApplication needed."""
    pass


def lasttrial(inst, actions, idx, undone):
    return RecoveryWidget._is_last_trial_of_key(inst, actions, idx, undone)


# ---------------------------------------------------------------------------
print("\n[display] _action_display_class:")
undone = {"d2", "d4"}
acts = make_actions()
check("undone trial d2 -> 'undone_trial' (kept as history)",
      _action_display_class(acts[1], undone) == "undone_trial")
check("undone trial d4 -> 'undone_trial'",
      _action_display_class(acts[3], undone) == "undone_trial")
check("non-undone trial d6 -> 'normal'",
      _action_display_class(acts[5], undone) == "normal")
check("non-undone current import -> 'normal'",
      _action_display_class(acts[0], undone) == "normal")
check("undone NON-trial (current crop) -> 'hidden'",
      _action_display_class({"id": "c1", "state": "current"}, {"c1"}) == "hidden")
check("undone NON-trial (committed) -> 'hidden'",
      _action_display_class({"id": "x", "state": "committed"}, {"x"}) == "hidden")

# ---------------------------------------------------------------------------
print("\n[selection] _is_last_trial_of_key skips undone trials:")
inst = _Dummy()                                # no Qt init needed
inst._session_chapters_sorted = CHAPTERS

# Scenario 1: first two trials undone, last (d6) kept.
acts = make_actions()
check("S1 d2(idx1) undone -> not last trial", lasttrial(inst, acts, 1, undone) is False)
check("S1 d4(idx3) undone -> not last trial", lasttrial(inst, acts, 3, undone) is False)
check("S1 d6(idx5) kept   -> IS last trial",  lasttrial(inst, acts, 5, undone) is True)

# Scenario 2: the LAST trial (d6) is the one undone -> auto-select falls back to d4.
undone2 = {"d6"}
check("S2 d6(idx5) undone -> not last trial", lasttrial(inst, acts, 5, undone2) is False)
check("S2 d4(idx3) now last non-undone -> IS last trial",
      lasttrial(inst, acts, 3, undone2) is True)
check("S2 d2(idx1) still not last", lasttrial(inst, acts, 1, undone2) is False)

# Back-compat: no undone_ids passed -> last trial overall (idx5).
check("compat: no undone_ids, d6(idx5) is last", lasttrial(inst, acts, 5, None) is True)
check("compat: no undone_ids, d2(idx1) not last", lasttrial(inst, acts, 1, None) is False)

# ---------------------------------------------------------------------------
print("\n[parity] UI auto-selection == effective_actions_for_replay:")


def ui_autoselected_ids(inst, actions, undone_ids):
    """Replicate _populate_action_list's auto-check decision per action."""
    out = []
    for i, a in enumerate(actions):
        if a.get("widget") == "system" or a.get("action") == "undo":
            continue
        cls = _action_display_class(a, undone_ids)
        if cls == "hidden":
            continue
        st = a.get("state", "current")
        if cls == "undone_trial":
            continue  # shown but never auto-checked
        if st in ("current", "committed"):
            out.append(a["id"])
        elif st == "trial":
            if RecoveryWidget._is_last_trial_of_key(inst, actions, i, undone_ids):
                out.append(a["id"])
    return out


def replay_ids(actions, chapters):
    lg = object.__new__(SessionLogger)        # bare instance: method reads only .actions/.chapters
    lg.actions = actions
    lg.chapters = chapters
    return [a["id"] for a in lg.effective_actions_for_replay()]

# Scenario 1
acts = make_actions()
ui1 = ui_autoselected_ids(inst, acts, undone)
rp1 = replay_ids(acts, CHAPTERS)
check(f"S1 replay ids == [import_1, d6]  (got {rp1})", rp1 == ["import_1", "d6"])
check(f"S1 UI auto-select == replay  (ui={ui1})", set(ui1) == set(rp1))

# Scenario 2 (last trial undone): replay/UI should fall back to d4
acts2 = make_actions()
acts2[2]["params"]["target"] = "d2"   # keep d2 undone
acts2[4]["params"]["target"] = "d6"   # now d6 is undone instead of d4
undone_s2 = {"d2", "d6"}
ui2 = ui_autoselected_ids(inst, acts2, undone_s2)
rp2 = replay_ids(acts2, CHAPTERS)
check(f"S2 replay ids == [import_1, d4]  (got {rp2})", rp2 == ["import_1", "d4"])
check(f"S2 UI auto-select == replay  (ui={ui2})", set(ui2) == set(rp2))

# ---------------------------------------------------------------------------
print(f"\n==== _test_undone_trials: {_passed} passed, {_failed} failed ====")
sys.exit(1 if _failed else 0)

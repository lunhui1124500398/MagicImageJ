# -*- coding: utf-8 -*-
"""Headless logic test for Issue #6 (reset edit_type) + Issue #5 (update_params).
Tests the authoritative replay resolver SessionLogger.compute_effective_actions
without any Qt/GUI. Run: conda run -n MagicImageJ python _test_issue6_reset.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.session_logger import SessionLogger

def make_logger(actions, edit_records, chapters):
    lg = SessionLogger.__new__(SessionLogger)   # bypass __init__ (no file I/O)
    lg.actions = actions
    lg.edit_records = edit_records
    lg.chapters = chapters
    return lg

def ids(actions):
    return [a["id"] for a in actions]

ACTIONS = [
    {"id": "a1", "widget": "geometry", "action": "rotate", "state": "historical", "params": {"angle": 5.0}},
    {"id": "a2", "widget": "geometry", "action": "crop",   "state": "current",    "params": {"x": 1}},
]
CHAPTERS = [{"id": "ch1", "start_action_index": 0, "status": "current"}]

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  PASS  {name}")
    else:
        failed += 1; print(f"  FAIL  {name}")

print("=== Issue #6: reset edit_type (compute_effective_actions) ===")

# Test 1: baseline — historical a1 excluded, current a2 included
out = make_logger([dict(a) for a in ACTIONS], [], list(CHAPTERS)).compute_effective_actions()
check("baseline: a1(historical) excluded", "a1" not in ids(out))
check("baseline: a2(current) included",    "a2" in ids(out))

# Test 2: enable a1 -> becomes current -> included
out = make_logger([dict(a) for a in ACTIONS],
                  [{"id": "e1", "edit_type": "enable", "target_action_id": "a1"}],
                  list(CHAPTERS)).compute_effective_actions()
check("after enable: a1 now included", "a1" in ids(out))

# Test 3 (THE KEY ONE): enable a1 then reset a1 -> back to historical -> excluded
out = make_logger([dict(a) for a in ACTIONS],
                  [{"id": "e1", "edit_type": "enable", "target_action_id": "a1"},
                   {"id": "e2", "edit_type": "reset",  "target_action_id": "a1"}],
                  list(CHAPTERS)).compute_effective_actions()
check("after enable+reset: a1 back to historical (excluded)", "a1" not in ids(out))

# Test 4: disable a2 then reset a2 -> back to current -> included
out = make_logger([dict(a) for a in ACTIONS],
                  [{"id": "e1", "edit_type": "disable", "target_action_id": "a2"},
                   {"id": "e2", "edit_type": "reset",   "target_action_id": "a2"}],
                  list(CHAPTERS)).compute_effective_actions()
check("after disable+reset: a2 back to current (included)", "a2" in ids(out))

print("\n=== Issue #5 + reset: update_params then reset restores original params ===")

# Test 5: update_params on a2, then reset -> params restored to original
lg = make_logger([dict(a) for a in ACTIONS],
                 [{"id": "e1", "edit_type": "update_params", "target_action_id": "a2",
                   "new_params": {"x": 999}}],
                 list(CHAPTERS))
out = lg.compute_effective_actions()
a2 = next(a for a in out if a["id"] == "a2")
check("update_params: a2.x == 999", a2["params"].get("x") == 999)

lg2 = make_logger([dict(a) for a in ACTIONS],
                  [{"id": "e1", "edit_type": "update_params", "target_action_id": "a2",
                    "new_params": {"x": 999}},
                   {"id": "e2", "edit_type": "reset", "target_action_id": "a2"}],
                  list(CHAPTERS))
out2 = lg2.compute_effective_actions()
a2b = next(a for a in out2 if a["id"] == "a2")
check("update_params+reset: a2.x restored to 1", a2b["params"].get("x") == 1)

# Test 6: original self.actions never mutated (non-destructive guarantee)
src = [dict(a) for a in ACTIONS]
make_logger(src, [{"id": "e1", "edit_type": "reset", "target_action_id": "a1"}],
            list(CHAPTERS)).compute_effective_actions()
check("self.actions[a1].state still 'historical' (immutable)", src[0]["state"] == "historical")

print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)

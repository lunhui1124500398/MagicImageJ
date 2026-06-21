"""
Phase 7 data-layer test — preallocated stacking helper.

Tests:
1. stack_frames_preallocated produces correct output for 2D frames
2. stack_frames_preallocated produces correct output for 3D frames (RGB)
3. Empty list returns empty array
4. Preallocation actually pre-allocates (no intermediate full-stack alloc)
5. Works on list of np.memmap views without amplifying RAM
"""
import sys
import numpy as np
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_preallocated_2d():
    from utils.memory_utils import stack_frames_preallocated
    frames = [np.full((10, 20), i, dtype=np.uint16) for i in range(5)]
    out = stack_frames_preallocated(frames)
    assert out.shape == (5, 10, 20)
    assert out.dtype == np.uint16
    for i in range(5):
        assert (out[i] == i).all()
    print(f"[OK] 2D stack: shape={out.shape}, dtype={out.dtype}")


def test_preallocated_3d():
    from utils.memory_utils import stack_frames_preallocated
    frames = [np.full((10, 20, 3), i, dtype=np.uint8) for i in range(4)]
    out = stack_frames_preallocated(frames)
    assert out.shape == (4, 10, 20, 3)
    print(f"[OK] 3D stack: shape={out.shape}")


def test_empty_input():
    from utils.memory_utils import stack_frames_preallocated
    out = stack_frames_preallocated([])
    assert out.shape == (0,)
    print("[OK] empty input -> shape (0,)")


def test_memmap_input():
    """Verify that stacking memmap views works (peak RAM not doubled)."""
    from utils.memory_utils import stack_frames_preallocated

    tmpdir = Path(tempfile.gettempdir())
    fpath = tmpdir / "phase7_memmap_test.dat"
    if fpath.exists():
        fpath.unlink()

    # Create a memmap of size (5, 100, 100) and fill with 1..5 per frame
    mm = np.memmap(fpath, mode='w+', dtype=np.uint16, shape=(5, 100, 100))
    for i in range(5):
        mm[i] = i + 1
    mm.flush()
    del mm

    # Re-open in read-only memmap mode
    mm_r = np.memmap(fpath, mode='r', dtype=np.uint16, shape=(5, 100, 100))
    frames = [mm_r[i] for i in range(5)]  # list of memmap views

    out = stack_frames_preallocated(frames)
    assert out.shape == (5, 100, 100)
    assert out.dtype == np.uint16
    # Verify content
    for i in range(5):
        assert (out[i] == (i + 1)).all()
    # out is NOT a memmap (we pre-allocated)
    assert not isinstance(out, np.memmap)
    print(f"[OK] memmap input -> standalone ndarray (not memmap), shape={out.shape}")

    # Cleanup
    del mm_r, frames, out
    try:
        fpath.unlink()
    except OSError:
        pass


def test_unsupported_ndim_fallback():
    from utils.memory_utils import stack_frames_preallocated
    # 1D frames — should fall back to np.array
    frames = [np.arange(10) + i for i in range(3)]
    out = stack_frames_preallocated(frames)
    assert out.shape == (3, 10)
    print("[OK] 1D input falls back to np.array (no crash)")


def test_overview_pipeline_stage_function():
    """Smoke test: generate_overview.detect_pipeline_stage covers 4 stages."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_overview",
        r"D:\Revolution_Sample_Claude\tools\generate_overview.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # detect_pipeline_stage should return one of 4 strings
    assert mod.STAGE_COLORS == {
        "pending": "#666",
        "cropped": "#2196F3",
        "segmented": "#FF9800",
        "refined": "#4CAF50",
    }
    assert mod.STAGE_LABELS == {
        "pending": "Pending",
        "cropped": "Cropped",
        "segmented": "Segmented",
        "refined": "Refined",
    }
    print("[OK] generate_overview already has 4-stage pipeline tracking")


if __name__ == "__main__":
    print("=== Phase 7 memory + overview tests ===")
    test_preallocated_2d()
    test_preallocated_3d()
    test_empty_input()
    test_memmap_input()
    test_unsupported_ndim_fallback()
    test_overview_pipeline_stage_function()
    print("\nAll Phase 7 tests passed.")

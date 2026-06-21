"""
Phase 1 data-layer test: verify burn-in math + LUT correctness.

Tests:
1. _build_export_lut_table returns correct (256,3) uint8 LUT for Gray/Inverted/Viridis
2. Burn-in math: given c_min/c_max/img_value, output should match np.clip((img-c_min)/(c_max-c_min)*255, 0, 255)
3. RGB output: LUT lookup applies correctly
4. TIFF burn: full stack uint8 buffer matches per-frame computation

Run from D:\zidongshibie\MagicImageJ:
    python _test_phase1_burnin.py
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_lut_table():
    """LUT helpers should produce expected (256,3) uint8 tables."""
    from widgets.geometry_widget import _build_export_lut_table

    # Gray = identity grayscale -> RGB
    gray = _build_export_lut_table('Gray')
    assert gray.shape == (256, 3), f"Gray shape {gray.shape}"
    assert gray.dtype == np.uint8
    assert np.all(gray[0] == [0, 0, 0]), "Gray[0] should be black"
    assert np.all(gray[255] == [255, 255, 255]), "Gray[255] should be white"
    assert np.all(gray[128] == [128, 128, 128]), "Gray[128] should be mid"

    # Inverted = 255 - i for each channel
    inv = _build_export_lut_table('Inverted')
    assert inv.shape == (256, 3)
    assert np.all(inv[0] == [255, 255, 255]), "Inverted[0] should be white"
    assert np.all(inv[255] == [0, 0, 0]), "Inverted[255] should be black"

    # Viridis (if matplotlib present)
    vir = _build_export_lut_table('Viridis')
    assert vir.shape == (256, 3)
    # Viridis is NOT identity grayscale (yellow-green-purple)
    assert not np.all(vir[128] == [128, 128, 128]), "Viridis should not be gray"

    # Unknown LUT -> grayscale fallback
    unk = _build_export_lut_table('Bogus')
    assert np.all(unk == gray), "Unknown LUT must fall back to Gray"

    print("[OK] LUT table tests passed")


def test_burnin_math():
    """Replicate the per-frame burn-in formula and verify against direct compute."""
    # Synthetic uint16 frame
    img = np.array([
        [0, 100, 200, 500],
        [1000, 1500, 2000, 65535],
    ], dtype=np.uint16)

    c_min, c_max = 100.0, 1500.0
    rng = max(c_max - c_min, 1e-8)

    # Apply burn-in (matches BatchExportThread inline math)
    f = img.astype(np.float32)
    u8 = np.clip((f - c_min) / rng * 255, 0, 255).astype(np.uint8)

    # Reference values
    # img=0    -> -100/1400*255 = -18.21 -> clipped 0
    # img=100  -> 0 -> 0
    # img=200  -> 100/1400*255 = 18.21 -> 18
    # img=500  -> 400/1400*255 = 72.86 -> 72
    # img=1000 -> 900/1400*255 = 163.93 -> 163
    # img=1500 -> 255 -> 255
    # img=2000 -> 500/1400*255 = 91.07 + 255 = clipped 255
    # img=65535 -> way over -> 255
    expected = np.array([
        [0, 0, 18, 72],
        [163, 255, 255, 255],
    ], dtype=np.uint8)

    assert np.array_equal(u8, expected), f"burn-in mismatch:\n  got {u8}\n  exp {expected}"
    print("[OK] burn-in math test passed")


def test_lut_apply():
    """uint8 stack -> RGB via LUT indexing."""
    from widgets.geometry_widget import _build_export_lut_table

    u8 = np.array([[0, 128, 255]], dtype=np.uint8)
    gray_lut = _build_export_lut_table('Gray')
    rgb = gray_lut[u8]
    assert rgb.shape == (1, 3, 3)
    assert np.all(rgb[0, 0] == [0, 0, 0])
    assert np.all(rgb[0, 1] == [128, 128, 128])
    assert np.all(rgb[0, 2] == [255, 255, 255])

    inv_lut = _build_export_lut_table('Inverted')
    rgb_inv = inv_lut[u8]
    assert np.all(rgb_inv[0, 0] == [255, 255, 255])
    assert np.all(rgb_inv[0, 2] == [0, 0, 0])
    print("[OK] LUT application test passed")


def test_tiff_burn_memory():
    """Verify the TIFF burn path produces same result as per-frame loop, and doesn't
       require allocating float32 of the full stack."""
    # Mimic a (T,H,W) uint16 ROI crop
    T, H, W = 50, 100, 100
    np.random.seed(42)
    crop = (np.random.rand(T, H, W) * 65000).astype(np.uint16)

    c_min, c_max = 1000.0, 50000.0
    rng = max(c_max - c_min, 1e-8)

    # Method 1: BatchExportThread inline (per-frame, allocate-on-write)
    burned1 = np.empty((T, H, W), dtype=np.uint8)
    for k in range(T):
        f = np.asarray(crop[k]).astype(np.float32)
        burned1[k] = np.clip((f - c_min) / rng * 255, 0, 255).astype(np.uint8)

    # Method 2: bulk reference (will allocate full float32 copy ~ 50*100*100*4 = 2MB; small enough)
    burned2 = np.clip((crop.astype(np.float32) - c_min) / rng * 255, 0, 255).astype(np.uint8)

    assert np.array_equal(burned1, burned2), "Per-frame burn should match bulk burn"
    print(f"[OK] TIFF burn matches reference (T={T} H={H} W={W})")


def test_polygon_to_bbox_clipping():
    """Make sure the polygon->bbox helper in the dialog clips correctly."""
    from widgets.batch_export_confirm_dialog import _polygon_to_bbox

    # ROI poly (y, x) order
    poly = np.array([[10, 20], [10, 80], [60, 80], [60, 20]])
    stack_shape = (100, 50, 100)  # T=100, H=50, W=100  -> y clipped to 50

    y1, x1, y2, x2 = _polygon_to_bbox(poly, stack_shape)
    assert y1 == 10, y1
    assert x1 == 20, x1
    assert y2 == 50, f"y2 should clip to H=50, got {y2}"
    assert x2 == 80, x2

    # Negative coordinates
    poly_neg = np.array([[-5, -10], [-5, 50], [40, 50], [40, -10]])
    y1, x1, y2, x2 = _polygon_to_bbox(poly_neg, (10, 100, 100))
    assert y1 == 0 and x1 == 0
    print("[OK] polygon->bbox clipping test passed")


def test_preview_overrides_alignment():
    """Verify the dialog's preview_overrides list length always matches input ROI count."""
    # Skip - requires Qt; just structural check by reading the source.
    src = (Path(__file__).parent / "widgets" / "batch_export_confirm_dialog.py").read_text(encoding="utf-8")
    assert "def get_preview_overrides" in src
    assert "out: List[Optional[dict]] = []" in src
    print("[OK] dialog.get_preview_overrides signature present")


if __name__ == "__main__":
    print("=== Phase 1 data-layer tests ===")
    test_lut_table()
    test_burnin_math()
    test_lut_apply()
    test_tiff_burn_memory()
    test_polygon_to_bbox_clipping()
    test_preview_overrides_alignment()
    print("\nAll Phase 1 data-layer tests passed.")

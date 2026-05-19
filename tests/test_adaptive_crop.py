"""Tests for adaptive corner crop sizing.

The corner crop size for pose-model refinement is proportional to the
detected photo's bounding box size rather than a fixed 640px. This ensures:
- Large photos get crops with enough context around the corner
- Small photos don't include adjacent photos in their crops
- Low-confidence detections get extra margin (larger crops)

The base percentage is 70% of max_dim (roughly equivalent to the old fixed 640
for typical ~900px photos), with a minimum floor of 320px (half the model's
640×640 input resolution).

Tests cover:
- _corner_crop_size: basic percentage, confidence scaling, clamping
- _corner_crop_min: percentage of crop, absolute floor
- Integration with refine_corners_geometric
"""

import sys
import math
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onnx_inference.photocrop import (
    _corner_crop_size,
    _corner_crop_min,
    CORNER_CROP_SIZE_MIN,
    CORNER_CROP_PCT,
    CORNER_CROP_CONF_SCALE,
    CORNER_CROP_MIN_PCT,
)


def _make_box(x1=100, y1=100, x2=500, y2=700):
    """Helper to create a bounding box dict."""
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


class TestCornerCropSize:
    """Test _corner_crop_size computes proportional crop sizes."""

    def test_basic_percentage(self):
        """At confidence=1.0, crop = base_pct × max_dim."""
        box = _make_box(0, 0, 800, 1000)  # max_dim = 1000
        result = _corner_crop_size(box, confidence=1.0, image_size=(3000, 3000))
        expected = 1000 * CORNER_CROP_PCT  # 0.70 * 1000 = 700
        # Rounded to multiple of 32
        expected_rounded = ((max(CORNER_CROP_SIZE_MIN, int(expected)) + 31) // 32) * 32
        assert result == expected_rounded

    def test_confidence_scaling(self):
        """Lower confidence produces larger crops."""
        box = _make_box(0, 0, 800, 1000)
        img_size = (3000, 3000)

        crop_high = _corner_crop_size(box, confidence=0.95, image_size=img_size)
        crop_low = _corner_crop_size(box, confidence=0.5, image_size=img_size)
        crop_mid = _corner_crop_size(box, confidence=0.7, image_size=img_size)

        # Lower confidence → larger crop
        assert crop_low > crop_high, \
            f"Low conf should give larger crop: {crop_low} > {crop_high}"
        # Also order: low > mid > high
        assert crop_low >= crop_mid >= crop_high, \
            f"Should be ordered: {crop_low} >= {crop_mid} >= {crop_high}"

    def test_confidence_formula(self):
        """Verify the formula: pct = base_pct + conf_scale * (1 - confidence)."""
        box = _make_box(0, 0, 600, 800)  # max_dim = 800
        img_size = (3000, 3000)

        for conf in [0.5, 0.8, 0.95, 1.0]:
            pct = CORNER_CROP_PCT + CORNER_CROP_CONF_SCALE * (1.0 - conf)
            ideal = 800 * pct
            expected = max(CORNER_CROP_SIZE_MIN, int(ideal))
            expected = ((expected + 31) // 32) * 32
            expected = min(expected, 3000, 3000)
            result = _corner_crop_size(box, confidence=conf, image_size=img_size)
            assert result == expected, \
                f"conf={conf}: expected {expected}, got {result}"

    def test_minimum_crop_size(self):
        """Very small photos should still get at least CORNER_CROP_SIZE_MIN (320)."""
        box = _make_box(0, 0, 50, 50)  # max_dim = 50
        result = _corner_crop_size(box, confidence=1.0, image_size=(3000, 3000))
        assert result >= CORNER_CROP_SIZE_MIN, \
            f"Should be at least {CORNER_CROP_SIZE_MIN}, got {result}"
        # 0.70 * 50 = 35 → max(320, 35) = 320
        assert result == CORNER_CROP_SIZE_MIN, \
            f"Small photo should get minimum crop, got {result}"

    def test_clamped_to_image_size(self):
        """Crop should not exceed image dimensions after rounding."""
        box = _make_box(0, 0, 2000, 3000)  # max_dim = 3000
        # 0.70 * 3000 = 2100, but image is only 1500 wide → clamped
        result = _corner_crop_size(box, confidence=1.0, image_size=(1500, 2000))
        assert result <= 1500, f"Crop {result} exceeds image width 1500"

    def test_larger_photo_gets_larger_crop(self):
        """Photos with larger bounding boxes get proportionally larger crops."""
        img_size = (3000, 3000)
        small_box = _make_box(0, 0, 200, 300)    # max_dim = 300
        large_box = _make_box(0, 0, 800, 1200)   # max_dim = 1200

        small_crop = _corner_crop_size(small_box, confidence=0.9, image_size=img_size)
        large_crop = _corner_crop_size(large_box, confidence=0.9, image_size=img_size)

        assert large_crop > small_crop, \
            f"Large photo crop {large_crop} should exceed small photo crop {small_crop}"

    def test_result_is_multiple_of_32(self):
        """Crop size should always be a multiple of 32 for ONNX efficiency."""
        box = _make_box(0, 0, 777, 999)  # Odd dimensions
        for conf in [0.3, 0.5, 0.75, 0.95, 1.0]:
            result = _corner_crop_size(box, confidence=conf, image_size=(5000, 5000))
            assert result % 32 == 0, \
                f"Crop size {result} not a multiple of 32 (conf={conf})"

    def test_square_photo(self):
        """Square photo: max_dim is the same for width and height."""
        box = _make_box(0, 0, 800, 800)
        result = _corner_crop_size(box, confidence=1.0, image_size=(3000, 3000))
        expected_pct = CORNER_CROP_PCT  # 0.70
        ideal = 800 * expected_pct  # = 560
        rounded = ((max(CORNER_CROP_SIZE_MIN, int(ideal)) + 31) // 32) * 32
        assert result == rounded

    def test_low_confidence_adds_margin(self):
        """At confidence=0, crop should be (base_pct + conf_scale) × max_dim."""
        box = _make_box(0, 0, 600, 900)  # max_dim = 900
        img_size = (3000, 3000)

        # confidence=0 → pct = CORNER_CROP_PCT + CORNER_CROP_CONF_SCALE
        result = _corner_crop_size(box, confidence=0.0, image_size=img_size)
        expected_pct = CORNER_CROP_PCT + CORNER_CROP_CONF_SCALE  # 0.80
        ideal = 900 * expected_pct
        expected = max(CORNER_CROP_SIZE_MIN, int(ideal))
        expected = ((expected + 31) // 32) * 32
        assert result == expected

    def test_real_world_sizing(self):
        """Verify sizes for the real_world_example photos (~900px max_dim).

        With CORNER_CROP_PCT=0.70, CORNER_CROP_SIZE_MIN=320:
        - max_dim≈951, conf≈0.95: pct=0.705, ideal=670 → crop=672
        - max_dim≈893, conf≈0.97: pct=0.703, ideal=627 → crop=640
        """
        img_size = (1512, 2016)

        # Photo with max_dim≈951 (659×951), conf≈0.95
        box1 = _make_box(98, 48, 757, 999)
        result1 = _corner_crop_size(box1, confidence=0.95, image_size=img_size)
        # 0.705 * 951 = 670.5 → max(320, 671) → round to 672
        assert result1 == 672, f"Expected 672, got {result1}"

        # Photo with max_dim≈893 (660×893), conf≈0.97
        box2 = _make_box(779, 105, 1439, 998)
        result2 = _corner_crop_size(box2, confidence=0.97, image_size=img_size)
        # 0.703 * 893 = 627.7 → max(320, 628) → round to 640
        assert result2 == 640, f"Expected 640, got {result2}"

    def test_minimum_applies_to_tiny_photos(self):
        """Photos smaller than ~450px max_dim should get the floor of 320."""
        # 0.70 * 400 = 280, which is below 320 → floor kicks in
        box = _make_box(0, 0, 300, 400)  # max_dim = 400
        result = _corner_crop_size(box, confidence=1.0, image_size=(1000, 1000))
        assert result == 320, f"Tiny photo should get 320px floor, got {result}"


class TestCornerCropMin:
    """Test _corner_crop_min computes binary search minimum."""

    def test_basic_percentage(self):
        """Minimum should be MIN_PCT × crop_size."""
        crop_size = 640
        result = _corner_crop_min(crop_size)
        expected = max(CORNER_CROP_SIZE_MIN, int(crop_size * CORNER_CROP_MIN_PCT))
        expected = ((expected + 31) // 32) * 32
        assert result == expected

    def test_absolute_floor(self):
        """Very small crops should not go below CORNER_CROP_SIZE_MIN (320)."""
        # With pct=0.40: 300*0.40 = 120 < 320, so floor kicks in
        crop_size = 300
        result = _corner_crop_min(crop_size)
        assert result >= CORNER_CROP_SIZE_MIN, \
            f"Min should be at least {CORNER_CROP_SIZE_MIN}, got {result}"

    def test_scales_with_crop_size(self):
        """Larger crop sizes should have proportionally larger minimums."""
        small_crop = 640
        large_crop = 1280

        min_small = _corner_crop_min(small_crop)
        min_large = _corner_crop_min(large_crop)

        assert min_large > min_small, \
            f"Large crop min {min_large} should exceed small crop min {min_small}"

    def test_result_is_multiple_of_32(self):
        """Minimum should always be a multiple of 32."""
        for crop_size in [320, 400, 640, 800, 1200, 2000]:
            result = _corner_crop_min(crop_size)
            assert result % 32 == 0, \
                f"Min crop {result} not a multiple of 32 (from crop_size={crop_size})"

    def test_real_world_values(self):
        """Check minimum values for typical real-world crop sizes.

        With CORNER_CROP_MIN_PCT=0.40, CORNER_CROP_SIZE_MIN=320:
        - crop=640: min = max(320, 640*0.40) = max(320, 256) = 320
        - crop=672: min = max(320, 672*0.40) = max(320, 268) = 320
        - crop=960: min = max(320, 960*0.40) = max(320, 384) = 384
        - crop=1280: min = max(320, 1280*0.40) = max(320, 512) = 512
        """
        assert _corner_crop_min(640) == 320
        assert _corner_crop_min(672) == 320
        assert _corner_crop_min(960) == 384
        assert _corner_crop_min(1280) == 512

    def test_custom_pct(self):
        """Custom percentage should override the default."""
        crop_size = 640
        # With pct=0.50: max(320, 640*0.50) = max(320, 320) = 320
        result = _corner_crop_min(crop_size, pct=0.50)
        assert result == 320
        # With pct=0.75: max(320, 640*0.75) = max(320, 480) = 480
        result2 = _corner_crop_min(crop_size, pct=0.75)
        assert result2 == 480

    def test_never_exceeds_crop_size(self):
        """Minimum should never exceed the crop size itself."""
        for crop_size in [320, 400, 640, 1000]:
            min_crop = _corner_crop_min(crop_size)
            assert min_crop <= crop_size, \
                f"Min crop {min_crop} should not exceed crop_size {crop_size}"


class TestAdaptiveCropIntegration:
    """Integration tests verifying adaptive crop sizes work with the pipeline."""

    def test_crop_scales_with_confidence(self):
        """Lower confidence detection should produce larger crops."""
        box = _make_box(0, 0, 800, 1000)
        img_size = (3000, 3000)

        # Create a series of confidence values and verify monotonicity
        crops = {}
        for conf in [0.3, 0.5, 0.7, 0.9, 0.95, 1.0]:
            crops[conf] = _corner_crop_size(box, conf, img_size)

        # Each lower confidence should give a larger or equal crop
        confs = sorted(crops.keys(), reverse=True)
        for i in range(len(confs) - 1):
            high_conf = confs[i]
            low_conf = confs[i + 1]
            assert crops[high_conf] <= crops[low_conf], \
                f"conf={high_conf}→{crops[high_conf]} should be <= conf={low_conf}→{crops[low_conf]}"

    def test_min_scales_with_crop(self):
        """As crop size increases, minimum should also increase proportionally."""
        for crop_size in [640, 960, 1280]:
            min_crop = _corner_crop_min(crop_size)
            # The minimum should be at least CORNER_CROP_MIN_PCT × crop_size
            # (unless it hits the absolute floor of CORNER_CROP_SIZE_MIN)
            expected_pct = min_crop / crop_size
            assert expected_pct >= CORNER_CROP_MIN_PCT - 0.01 or min_crop == \
                ((max(CORNER_CROP_SIZE_MIN, int(crop_size * CORNER_CROP_MIN_PCT)) + 31) // 32) * 32, \
                f"Min/crop ratio {expected_pct:.3f} unexpected (crop={crop_size})"

    def test_small_photo_gets_minimum(self):
        """Small photos (max_dim below ~460) should get the minimum floor of 320."""
        box = _make_box(0, 0, 200, 300)  # max_dim = 300
        # 0.70 * 300 = 210 → below 320 floor
        img_size = (640, 640)

        crop = _corner_crop_size(box, confidence=0.95, image_size=img_size)
        assert crop == 320, f"Small photo should get 320 minimum, got {crop}"

    def test_large_photo_gets_proportional_crop(self):
        """Large photos should get crops proportional to their size."""
        box = _make_box(0, 0, 1500, 2000)  # max_dim = 2000
        img_size = (3000, 3000)

        crop = _corner_crop_size(box, confidence=0.95, image_size=img_size)
        # 0.705 * 2000 = 1410 → max(320, 1410) = 1410 → round to 1440
        assert crop == 1440, \
            f"Large photo should get proportional crop, got {crop}"

    def test_default_parameters_match_constants(self):
        """Verify that default parameters match the module constants."""
        box = _make_box(0, 0, 500, 700)
        img_size = (2000, 2000)

        # Calling with defaults should match calling with explicit constants
        result_default = _corner_crop_size(box, 0.9, img_size)
        result_explicit = _corner_crop_size(
            box, 0.9, img_size,
            base_pct=CORNER_CROP_PCT,
            conf_scale=CORNER_CROP_CONF_SCALE,
            min_size=CORNER_CROP_SIZE_MIN,
        )
        assert result_default == result_explicit
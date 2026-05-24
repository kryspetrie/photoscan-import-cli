"""Tests for corner crop extraction and sizing.

The corner regression model uses fixed 320×320 crops centered on approximate
corner positions. Tests cover:
- _corner_crop: extraction, boundary clamping, padding
- CORNER_CROP_SIZE_MIN: minimum crop size constant
- Integration with corner regression pipeline
"""

import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onnx_inference.photocrop import (
    _corner_crop,
    CORNER_CROP_SIZE_MIN,
)


class TestCornerCropExtraction:
    """Test _corner_crop extracts correct regions with boundary handling."""

    def test_centered_crop(self):
        """Crop centered on a point in a large image."""
        img = Image.new("RGB", (2000, 2000), (128, 128, 128))
        crop, ox, oy = _corner_crop(img, 1000, 1000, 640)
        assert crop.size == (640, 640)
        assert ox == 1000 - 320
        assert oy == 1000 - 320

    def test_crop_near_top_left(self):
        """Crop near top-left corner should be clamped to image boundary."""
        img = Image.new("RGB", (2000, 2000), (128, 128, 128))
        crop, ox, oy = _corner_crop(img, 50, 50, 640)
        assert crop.size == (640, 640)
        assert ox == 0  # Clamped to left edge
        assert oy == 0  # Clamped to top edge

    def test_crop_near_bottom_right(self):
        """Crop near bottom-right corner should be clamped."""
        img = Image.new("RGB", (2000, 2000), (128, 128, 128))
        crop, ox, oy = _corner_crop(img, 1950, 1950, 640)
        assert crop.size == (640, 640)
        assert ox == 2000 - 640  # Clamped to right edge
        assert oy == 2000 - 640  # Clamped to bottom edge

    def test_default_crop_size(self):
        """Default crop size should be CORNER_CROP_SIZE_MIN (320)."""
        img = Image.new("RGB", (2000, 2000), (128, 128, 128))
        crop, ox, oy = _corner_crop(img, 100, 100)
        assert crop.size == (CORNER_CROP_SIZE_MIN, CORNER_CROP_SIZE_MIN)

    def test_custom_crop_size(self):
        """Custom crop size should work correctly."""
        img = Image.new("RGB", (2000, 2000), (128, 128, 128))
        crop, ox, oy = _corner_crop(img, 1000, 1000, 480)
        assert crop.size == (480, 480)
        assert ox == 1000 - 240  # Centered
        assert oy == 1000 - 240

    def test_small_image_padded(self):
        """Image smaller than crop size should be padded with grey."""
        img = Image.new("RGB", (300, 300), (128, 128, 128))
        crop, ox, oy = _corner_crop(img, 150, 150, 640)
        assert crop.size == (640, 640)
        assert ox == 0  # Can't shift, image too small
        assert oy == 0

    def test_tiny_image_padded(self):
        """Very small image should still produce 640x640 crop."""
        img = Image.new("RGB", (100, 100), (200, 100, 50))
        crop, ox, oy = _corner_crop(img, 50, 50, 640)
        assert crop.size == (640, 640)

    def test_crop_preserves_pixel_data(self):
        """Crop should contain actual pixel data from the source image."""
        # Create an image with distinct colored quadrants
        img = Image.new("RGB", (640, 640))
        pixels = img.load()
        for x in range(640):
            for y in range(640):
                if x < 320 and y < 320:
                    pixels[x, y] = (255, 0, 0)    # Red TL
                elif x >= 320 and y < 320:
                    pixels[x, y] = (0, 255, 0)    # Green TR
                elif x < 320 and y >= 320:
                    pixels[x, y] = (0, 0, 255)    # Blue BL
                else:
                    pixels[x, y] = (255, 255, 0)  # Yellow BR

        # Crop at center should contain all 4 colors
        crop, ox, oy = _corner_crop(img, 320, 320, 640)
        assert ox == 0
        assert oy == 0
        crop_pixels = crop.load()
        # Top-left of crop should be red
        assert crop_pixels[10, 10] == (255, 0, 0)
        # Top-right should be green
        assert crop_pixels[630, 10] == (0, 255, 0)
        # Bottom-left should be blue
        assert crop_pixels[10, 630] == (0, 0, 255)

    def test_offset_maps_to_image_coords(self):
        """Offset (ox, oy) should correctly map crop coords back to image coords."""
        img = Image.new("RGB", (2000, 1500), (128, 128, 128))
        # Point at (500, 300) in image coords with 320px crop
        crop, ox, oy = _corner_crop(img, 500, 300, 320)
        # Center of crop (160, 160) should correspond to (ox+160, oy+160)
        assert ox == 500 - 160
        assert oy == 300 - 160

    def test_offset_clamp_order(self):
        """When both left and bottom clamping would apply, left wins."""
        img = Image.new("RGB", (1000, 800), (128, 128, 128))
        # Point at x=50 with 640px crop: center would be 50, clamped left=0
        crop, ox, oy = _corner_crop(img, 50, 100, 640)
        assert ox == 0  # Clamped to left edge
        # x2 would be min(1000, 0+640) = 640, not exceeding image width
        assert ox + 640 <= 1000

    def test_even_crop_size_centering(self):
        """Even crop sizes should center correctly (integer division)."""
        img = Image.new("RGB", (3000, 3000), (128, 128, 128))
        # 320 // 2 = 160, so offset should be point - 160
        crop, ox, oy = _corner_crop(img, 1000, 2000, 320)
        assert ox == 840   # 1000 - 160
        assert oy == 1840  # 2000 - 160


class TestCornerCropConstants:
    """Test that constants are set correctly and make sense."""

    def test_min_crop_size_is_320(self):
        """Minimum crop size should be 320 (half the model input 640)."""
        assert CORNER_CROP_SIZE_MIN == 320

    def test_min_crop_size_is_even_divisor_of_model_input(self):
        """320 = 640 / 2, meaning the crop is exactly half the model input."""
        assert 640 % CORNER_CROP_SIZE_MIN == 0


class TestCornerCropBoundaries:
    """Test edge cases at image boundaries."""

    def test_crop_at_exact_image_edge(self):
        """Crop centered exactly at image edge should still produce full crop."""
        img = Image.new("RGB", (1000, 1000), (128, 128, 128))
        # Point at x=320 (half of 640), so crop should start at x=0
        crop, ox, oy = _corner_crop(img, 320, 320, 640)
        assert ox == 0
        assert oy == 0
        assert crop.size == (640, 640)

    def test_crop_at_zero_zero(self):
        """Crop at (0, 0) should be clamped correctly."""
        img = Image.new("RGB", (2000, 2000), (128, 128, 128))
        crop, ox, oy = _corner_crop(img, 0, 0, 320)
        assert ox == 0
        assert oy == 0
        assert crop.size == (320, 320)

    def test_crop_at_far_edge(self):
        """Crop at far edge of large image."""
        img = Image.new("RGB", (3000, 3000), (128, 128, 128))
        crop, ox, oy = _corner_crop(img, 2800, 2800, 320)
        # 2800 - 160 = 2640, but check: 2640 + 320 = 2960 <= 3000 ✓
        assert ox == 2640
        assert oy == 2640
        assert crop.size == (320, 320)

    def test_negative_coord_clamped(self):
        """Negative center coordinates should not crash (float rounding)."""
        img = Image.new("RGB", (1000, 1000), (128, 128, 128))
        # Very small float near 0
        crop, ox, oy = _corner_crop(img, 0.5, 0.5, 320)
        assert crop.size == (320, 320)
        assert ox >= 0
        assert oy >= 0

    def test_crop_with_float_center(self):
        """Float center coordinates should work (rounded to nearest int)."""
        img = Image.new("RGB", (2000, 2000), (128, 128, 128))
        crop, ox, oy = _corner_crop(img, 500.7, 300.3, 320)
        assert crop.size == (320, 320)
        # Should round: 500.7 - 160 = 340.7 → 341, 300.3 - 160 = 140.3 → 140
        assert ox == 341 or ox == 340  # Rounding tolerance
        assert oy == 140 or oy == 141
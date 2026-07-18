"""Tests for preset/crop/border-fill separation in photocrop.py.

Presets control detection/refinement only.
--crop controls output crop method separately.
--border-fill controls warp fill color (default: edge-extend).
Both preset and crop have CLI defaults (corner_refine, warp-stretch).

Tests cover:
- _validate_preset_crop(): warning logic for questionable combinations
- _apply_preset(): preset sets detection args, user overrides take precedence
- Preset dicts contain NO crop-related keys
- parse_border_fill(): color and edge-extend parsing
- CLI integration: preset + crop + border_fill combinations via argparse
"""

import sys
import argparse
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from com.kryspetrie.photoscan_import_cli.photocrop import (
    _PRESETS,
    _PRESET_CROP_DEFAULTS,
    _validate_preset_crop,
    _apply_preset,
    parse_border_fill,
    DEFAULT_PRESET,
)


# ---------------------------------------------------------------------------
# Structural tests: presets are detection-only
# ---------------------------------------------------------------------------


class TestPresetStructure:
    """Verify presets contain only detection/refinement settings, no crop."""

    CROP_RELATED_KEYS = {"crop", "crop_margin", "border_fill", "crop_transparent"}

    def test_preset_names(self):
        assert set(_PRESETS.keys()) == {"fast", "pose_refine", "corner_refine"}

    def test_default_preset_exists(self):
        assert DEFAULT_PRESET in _PRESETS

    def test_default_preset_is_corner_refine(self):
        assert DEFAULT_PRESET == "corner_refine"

    def test_presets_have_no_crop_keys(self):
        for name, preset in _PRESETS.items():
            args = preset["args"]
            for key in args:
                assert key not in self.CROP_RELATED_KEYS, (
                    f"Preset '{name}' contains crop-related key '{key}'. "
                    f"Presets should be detection-only."
                )

    def test_fast_preset_has_only_warp_recover(self):
        """Fast preset only has warp_recover (enabled by default across all presets)."""
        assert _PRESETS["fast"]["args"] == {"warp_recover": True}

    def test_pose_refine_preset_has_pose_refine_and_adaptive(self):
        args = _PRESETS["pose_refine"]["args"]
        assert args.get("pose_refine") is True
        assert args.get("adaptive_margin") is True
        assert args.get("corner_refine") is False or "corner_refine" not in args

    def test_corner_refine_preset_has_all_refinement(self):
        args = _PRESETS["corner_refine"]["args"]
        assert args.get("pose_refine") is True
        assert args.get("corner_refine") is True
        assert args.get("adaptive_margin") is True
        assert args.get("corner_refine_iterations") == 2
        assert args.get("corner_refine_model") == "regression"

    def test_no_preset_has_cv_refine(self):
        """No preset includes cv_refine -- Sobel rescue is always automatic."""
        for name, preset in _PRESETS.items():
            args = preset["args"]
            assert args.get("cv_refine") is False or "cv_refine" not in args

    def test_all_presets_have_descriptions(self):
        for name, preset in _PRESETS.items():
            assert "description" in preset
            assert len(preset["description"]) > 0

    def test_crop_defaults_exist_for_all_presets(self):
        for name in _PRESETS:
            assert name in _PRESET_CROP_DEFAULTS, (
                f"Missing crop defaults for preset '{name}'"
            )

    def test_crop_defaults_contain_no_detection_keys(self):
        detection_keys = {
            "pose_refine", "cv_refine", "corner_refine",
            "corner_refine_iterations", "corner_refine_model",
            "adaptive_margin",
        }
        for name, defaults in _PRESET_CROP_DEFAULTS.items():
            for key in defaults:
                assert key not in detection_keys, (
                    f"Crop defaults for '{name}' contains detection key '{key}'"
                )


# ---------------------------------------------------------------------------
# Validation tests: _validate_preset_crop()
# ---------------------------------------------------------------------------


class TestValidatePresetCrop:

    # --- Valid combinations (no errors, no warnings) ---

    def test_fast_simple_corners_ok(self):
        errors, warnings = _validate_preset_crop("fast", "simple-corners")
        assert errors == [] and warnings == []

    def test_fast_simple_ok(self):
        errors, warnings = _validate_preset_crop("fast", "simple")
        assert errors == [] and warnings == []

    def test_pose_refine_warp_stretch_ok(self):
        errors, warnings = _validate_preset_crop("pose_refine", "warp-stretch")
        assert errors == [] and warnings == []

    def test_corner_refine_warp_stretch_ok(self):
        errors, warnings = _validate_preset_crop("corner_refine", "warp-stretch")
        assert errors == [] and warnings == []

    def test_corner_refine_warp_ok(self):
        errors, warnings = _validate_preset_crop("corner_refine", "warp")
        assert errors == [] and warnings == []

    def test_no_preset_with_crop_ok(self):
        errors, warnings = _validate_preset_crop(None, "simple-corners")
        assert errors == [] and warnings == []

    def test_no_preset_no_crop_ok(self):
        errors, warnings = _validate_preset_crop(None, None)
        assert errors == [] and warnings == []

    # --- Questionable combinations (warnings, no errors) ---

    def test_fast_warp_warns(self):
        errors, warnings = _validate_preset_crop("fast", "warp")
        assert errors == []
        assert len(warnings) == 1
        assert "fast" in warnings[0].lower()
        assert "warp" in warnings[0].lower()

    def test_fast_warp_stretch_warns(self):
        errors, warnings = _validate_preset_crop("fast", "warp-stretch")
        assert errors == []
        assert len(warnings) == 1
        assert "fast" in warnings[0].lower()

    # --- None-type safety: crop_mode=None should not crash ---

    def test_preset_with_none_crop_ok(self):
        errors, warnings = _validate_preset_crop("fast", None)
        assert errors == [] and warnings == []

    def test_corner_refine_with_none_crop_ok(self):
        errors, warnings = _validate_preset_crop("corner_refine", None)
        assert errors == [] and warnings == []


# ---------------------------------------------------------------------------
# parse_border_fill tests
# ---------------------------------------------------------------------------


class TestParseBorderFill:

    def test_edge_extend(self):
        assert parse_border_fill("edge-extend") == "edge-extend"

    def test_edge_extend_underscore(self):
        assert parse_border_fill("edge_extend") == "edge-extend"

    def test_edge_extend_case_insensitive(self):
        assert parse_border_fill("Edge-Extend") == "edge-extend"

    def test_white(self):
        assert parse_border_fill("white") == (255, 255, 255)

    def test_black(self):
        assert parse_border_fill("black") == (0, 0, 0)

    def test_grey(self):
        assert parse_border_fill("grey") == (114, 114, 114)

    def test_rgb_tuple(self):
        assert parse_border_fill("100,150,200") == (100, 150, 200)

    def test_hex_six(self):
        assert parse_border_fill("#FF8000") == (255, 128, 0)

    def test_hex_three(self):
        assert parse_border_fill("#F80") == (255, 136, 0)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_border_fill("magenta")


# ---------------------------------------------------------------------------
# _apply_preset() integration tests
# ---------------------------------------------------------------------------


def _make_minimal_parser():
    """Build a minimal argparse parser that matches the real CLI defaults:
    --preset corner_refine, --crop warp-stretch, --crop-margin 0.02,
    --border-fill edge-extend."""
    p = argparse.ArgumentParser()
    p.add_argument("--preset", type=str, default=DEFAULT_PRESET)
    p.add_argument("--crop", type=str, default="warp-stretch")
    p.add_argument("--crop-margin", type=float, default=0.02)
    p.add_argument("--border-fill", type=str, default="edge-extend")
    p.add_argument("--pose-refine", action="store_true", default=False)
    p.add_argument("--adaptive-margin", action="store_true", default=False)
    p.add_argument("--corner-refine", action="store_true", default=False)
    p.add_argument("--corner-refine-iterations", type=int, default=2)
    p.add_argument("--corner-refine-conf", type=float, default=0.3)
    p.add_argument("--corner-refine-model", type=str, default="regression")
    p.add_argument("--cv-refine", action="store_true", default=False)
    p.add_argument("--pose-refine-expand", type=float, default=0.05)
    return p


class TestApplyPreset:

    def test_default_preset_applies_corner_refine(self):
        """Default preset (corner_refine) applies pose_refine, corner_refine, adaptive_margin."""
        p = _make_minimal_parser()
        args = p.parse_args([])  # defaults: corner_refine preset, warp-stretch crop
        result = _apply_preset(p, args)
        assert result.pose_refine is True
        assert result.adaptive_margin is True
        assert result.corner_refine is True

    def test_fast_preset_applies_no_detection_args(self):
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "fast"])
        result = _apply_preset(p, args)
        assert result.pose_refine is False
        assert result.corner_refine is False
        assert result.cv_refine is False
        assert result.adaptive_margin is False

    def test_pose_refine_preset(self):
        """pose_refine = pose_refine + adaptive_margin, no corner_refine."""
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "pose_refine"])
        result = _apply_preset(p, args)
        assert result.pose_refine is True
        assert result.corner_refine is False
        assert result.adaptive_margin is True

    def test_corner_refine_preset(self):
        """corner_refine = pose_refine + corner_refine + adaptive_margin."""
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "corner_refine"])
        result = _apply_preset(p, args)
        assert result.pose_refine is True
        assert result.corner_refine is True
        assert result.adaptive_margin is True
        assert result.corner_refine_iterations == 2
        assert result.corner_refine_model == "regression"

    def test_presets_do_not_set_crop(self):
        """Presets never override --crop. Crop defaults come from parser."""
        p = _make_minimal_parser()
        for preset_name in _PRESETS:
            args = p.parse_args(["--preset", preset_name])
            result = _apply_preset(p, args)
            assert result.crop == "warp-stretch"

    def test_default_crop_margin_and_border(self):
        """Parser defaults: crop_margin=0.02, border_fill=edge-extend."""
        p = _make_minimal_parser()
        args = p.parse_args([])  # all defaults
        result = _apply_preset(p, args)
        assert result.crop_margin == 0.02
        assert result.border_fill == "edge-extend"

    def test_user_override_of_crop_margin(self):
        """User explicitly set --crop-margin; preset should NOT override."""
        p = _make_minimal_parser()
        args = p.parse_args([
            "--preset", "corner_refine",
            "--crop-margin", "0.05",
        ])
        result = _apply_preset(p, args)
        assert result.crop_margin == 0.05

    def test_user_override_of_border_fill(self):
        p = _make_minimal_parser()
        args = p.parse_args([
            "--preset", "corner_refine",
            "--border-fill", "black",
        ])
        result = _apply_preset(p, args)
        assert result.border_fill == "black"

    def test_user_override_of_preset(self):
        """User explicitly set --preset fast; should override default."""
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "fast"])
        result = _apply_preset(p, args)
        assert result.pose_refine is False
        assert result.corner_refine is False

    def test_unknown_preset_errors(self):
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "nonexistent"])
        with pytest.raises(SystemExit):
            _apply_preset(p, args)

    def test_fast_with_warp_warns(self, capsys):
        """fast + warp-stretch should produce a warning."""
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "fast"])
        _apply_preset(p, args)
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "fast" in captured.err.lower()

    def test_valid_combination_no_warning(self, capsys):
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "corner_refine"])
        _apply_preset(p, args)
        captured = capsys.readouterr()
        assert "Warning" not in captured.err


# ---------------------------------------------------------------------------
# CLI-level integration: --crop and --border-fill work with defaults
# ---------------------------------------------------------------------------


class TestCropWithoutPresetOverride:

    def test_default_crop_is_warp_stretch(self):
        """Default crop mode is warp-stretch."""
        p = _make_minimal_parser()
        args = p.parse_args([])
        result = _apply_preset(p, args)
        assert result.crop == "warp-stretch"

    def test_explicit_crop_override(self):
        p = _make_minimal_parser()
        args = p.parse_args(["--crop", "simple-corners"])
        result = _apply_preset(p, args)
        assert result.crop == "simple-corners"
        assert result.pose_refine is True  # still gets corner_refine preset

    def test_fast_preset_no_refinement(self):
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "fast"])
        result = _apply_preset(p, args)
        assert result.pose_refine is False
        assert result.corner_refine is False
        assert result.cv_refine is False

    def test_edge_extend_is_default_border_fill(self):
        p = _make_minimal_parser()
        args = p.parse_args([])
        assert args.border_fill == "edge-extend"
"""Tests for preset/crop separation in photocrop.py.

Presets control detection/refinement only.
--crop controls output crop method separately.
Preset without --crop is an error. Questionable combinations produce warnings.

Tests cover:
- _validate_preset_crop(): error/warning logic for each combination
- _apply_preset(): preset sets detection args, crop defaults applied when --crop present
- Preset dicts contain NO crop-related keys
- CLI integration: preset + crop combinations via argparse
"""

import sys
import argparse
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onnx_inference.photocrop import (
    _PRESETS,
    _PRESET_CROP_DEFAULTS,
    _validate_preset_crop,
    _apply_preset,
)


# ---------------------------------------------------------------------------
# Structural tests: presets are detection-only
# ---------------------------------------------------------------------------


class TestPresetStructure:
    """Verify presets contain only detection/refinement settings, no crop."""

    CROP_RELATED_KEYS = {"crop", "crop_margin", "border_fill", "crop_transparent"}

    def test_preset_names(self):
        assert set(_PRESETS.keys()) == {"quick", "standard", "thorough"}

    def test_presets_have_no_crop_keys(self):
        for name, preset in _PRESETS.items():
            args = preset["args"]
            for key in args:
                assert key not in self.CROP_RELATED_KEYS, (
                    f"Preset '{name}' contains crop-related key '{key}'. "
                    f"Presets should be detection-only."
                )

    def test_quick_preset_has_no_args(self):
        assert _PRESETS["quick"]["args"] == {}

    def test_standard_preset_has_refinement(self):
        args = _PRESETS["standard"]["args"]
        assert args.get("pose_refine") is True
        assert args.get("adaptive_margin") is True

    def test_thorough_preset_has_full_refinement(self):
        args = _PRESETS["thorough"]["args"]
        assert args.get("corner_refine") is True
        assert args.get("cv_refine") is True
        assert args.get("pose_refine") is True
        assert args.get("adaptive_margin") is True

    def test_thorough_preset_corner_refine_iterations(self):
        assert _PRESETS["thorough"]["args"]["corner_refine_iterations"] == 2

    def test_thorough_preset_corner_refine_model(self):
        assert _PRESETS["thorough"]["args"]["corner_refine_model"] == "regression"

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

    def test_quick_simple_corners_ok(self):
        errors, warnings = _validate_preset_crop("quick", "simple-corners")
        assert errors == [] and warnings == []

    def test_quick_simple_ok(self):
        errors, warnings = _validate_preset_crop("quick", "simple")
        assert errors == [] and warnings == []

    def test_standard_simple_corners_ok(self):
        errors, warnings = _validate_preset_crop("standard", "simple-corners")
        assert errors == [] and warnings == []

    def test_standard_warp_stretch_ok(self):
        errors, warnings = _validate_preset_crop("standard", "warp-stretch")
        assert errors == [] and warnings == []

    def test_standard_warp_ok(self):
        errors, warnings = _validate_preset_crop("standard", "warp")
        assert errors == [] and warnings == []

    def test_thorough_simple_corners_ok(self):
        errors, warnings = _validate_preset_crop("thorough", "simple-corners")
        assert errors == [] and warnings == []

    def test_thorough_warp_stretch_ok(self):
        errors, warnings = _validate_preset_crop("thorough", "warp-stretch")
        assert errors == [] and warnings == []

    def test_thorough_warp_ok(self):
        errors, warnings = _validate_preset_crop("thorough", "warp")
        assert errors == [] and warnings == []

    def test_no_preset_with_crop_ok(self):
        errors, warnings = _validate_preset_crop(None, "simple-corners")
        assert errors == [] and warnings == []

    def test_no_preset_no_crop_ok(self):
        errors, warnings = _validate_preset_crop(None, None)
        assert errors == [] and warnings == []

    # --- Questionable combinations (warnings, no errors) ---

    def test_quick_warp_warns(self):
        errors, warnings = _validate_preset_crop("quick", "warp")
        assert errors == []
        assert len(warnings) == 1
        assert "quick" in warnings[0].lower()
        assert "warp" in warnings[0].lower()

    def test_quick_warp_stretch_warns(self):
        errors, warnings = _validate_preset_crop("quick", "warp-stretch")
        assert errors == []
        assert len(warnings) == 1
        assert "quick" in warnings[0].lower()

    def test_thorough_simple_warns(self):
        errors, warnings = _validate_preset_crop("thorough", "simple")
        assert errors == []
        assert len(warnings) == 1
        assert "thorough" in warnings[0].lower()
        assert "simple" in warnings[0].lower()

    # --- Invalid combinations (errors) ---

    def test_preset_without_crop_errors(self):
        errors, warnings = _validate_preset_crop("quick", None)
        assert len(errors) == 1
        assert "--crop" in errors[0]
        assert warnings == []

    def test_standard_without_crop_errors(self):
        errors, warnings = _validate_preset_crop("standard", None)
        assert len(errors) == 1
        assert "--crop" in errors[0]
        assert warnings == []

    def test_thorough_without_crop_errors(self):
        errors, warnings = _validate_preset_crop("thorough", None)
        assert len(errors) == 1
        assert "--crop" in errors[0]
        assert warnings == []

    # --- Boundary: crop IS specified, so no "missing crop" error ---

    def test_no_double_issue_for_thorough_simple(self):
        """thorough + simple: only 1 warning, NOT also a 'missing --crop' error."""
        errors, warnings = _validate_preset_crop("thorough", "simple")
        assert errors == []
        assert len(warnings) == 1

    def test_no_double_issue_for_quick_warp(self):
        """quick + warp: only 1 warning, NOT also a 'missing --crop' error."""
        errors, warnings = _validate_preset_crop("quick", "warp")
        assert errors == []
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# _apply_preset() integration tests
# ---------------------------------------------------------------------------


def _make_minimal_parser():
    """Build a minimal argparse parser that has just the args needed
    to test _apply_preset. This avoids loading ONNX models."""
    p = argparse.ArgumentParser()
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--crop", type=str, default=None)
    p.add_argument("--crop-margin", type=float, default=0)
    p.add_argument("--border-fill", type=str, default="grey")
    p.add_argument("--pose-refine", action="store_true", default=False)
    p.add_argument("--adaptive-margin", action="store_true", default=False)
    p.add_argument("--corner-refine", action="store_true", default=False)
    p.add_argument("--corner-refine-iterations", type=int, default=1)
    p.add_argument("--corner-refine-conf", type=float, default=0.3)
    p.add_argument("--corner-refine-model", type=str, default="regression")
    p.add_argument("--cv-refine", action="store_true", default=False)
    p.add_argument("--pose-refine-expand", type=float, default=0.05)
    return p


class TestApplyPreset:

    def test_no_preset_returns_args_unchanged(self):
        p = _make_minimal_parser()
        args = p.parse_args([])
        args.preset = None
        original_crop = args.crop
        result = _apply_preset(p, args)
        assert result.crop == original_crop

    def test_quick_preset_applies_no_detection_args(self):
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "quick", "--crop", "simple-corners"])
        result = _apply_preset(p, args)
        assert result.pose_refine is False
        assert result.corner_refine is False
        assert result.cv_refine is False
        assert result.adaptive_margin is False

    def test_standard_preset_applies_auto_refine(self):
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "standard", "--crop", "simple-corners"])
        result = _apply_preset(p, args)
        assert result.pose_refine is True
        assert result.adaptive_margin is True

    def test_thorough_preset_applies_all_refinement(self):
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "thorough", "--crop", "warp-stretch"])
        result = _apply_preset(p, args)
        assert result.pose_refine is True
        assert result.corner_refine is True
        assert result.cv_refine is True
        assert result.adaptive_margin is True
        assert result.corner_refine_iterations == 2
        assert result.corner_refine_model == "regression"

    def test_preset_does_not_set_crop(self):
        """Presets never set --crop. It must be specified separately.
        Even with a preset, crop=None means the user must provide --crop."""
        p = _make_minimal_parser()
        # Without --crop, all presets should error
        for preset_name in _PRESETS:
            args = p.parse_args(["--preset", preset_name])
            with pytest.raises(SystemExit):
                _apply_preset(p, args)

    def test_crop_defaults_applied_when_crop_specified(self):
        """When --crop is used alongside a preset, crop defaults (margin,
        border_fill) should be applied."""
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "thorough", "--crop", "warp-stretch"])
        result = _apply_preset(p, args)
        # thorough crop defaults: crop_margin=0.02, border_fill=white
        assert result.crop_margin == 0.02
        assert result.border_fill == "white"

    def test_crop_defaults_for_quick(self):
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "quick", "--crop", "simple-corners"])
        result = _apply_preset(p, args)
        assert result.crop_margin == 0.02

    def test_crop_defaults_for_standard(self):
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "standard", "--crop", "simple-corners"])
        result = _apply_preset(p, args)
        assert result.crop_margin == 0.02

    def test_preset_without_crop_errors_before_defaults_applied(self):
        """Without --crop, _apply_preset errors before crop defaults are applied."""
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "thorough"])
        with pytest.raises(SystemExit):
            _apply_preset(p, args)

    def test_user_override_of_crop_margin(self):
        """User explicitly set --crop-margin; preset default should NOT override."""
        p = _make_minimal_parser()
        args = p.parse_args([
            "--preset", "thorough",
            "--crop", "warp-stretch",
            "--crop-margin", "0.05",
        ])
        result = _apply_preset(p, args)
        assert result.crop_margin == 0.05

    def test_user_override_of_border_fill(self):
        p = _make_minimal_parser()
        args = p.parse_args([
            "--preset", "thorough",
            "--crop", "warp-stretch",
            "--border-fill", "black",
        ])
        result = _apply_preset(p, args)
        assert result.border_fill == "black"

    def test_unknown_preset_errors(self):
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "nonexistent"])
        with pytest.raises(SystemExit):
            _apply_preset(p, args)

    def test_preset_with_crop_warp_issues_warning(self, capsys):
        """quick + warp should produce a warning on stderr."""
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "quick", "--crop", "warp"])
        # _apply_preset calls _log which writes to stderr
        _apply_preset(p, args)
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "quick" in captured.err.lower()

    def test_preset_without_crop_errors(self, capsys):
        """Preset without --crop should produce an error (not a warning)."""
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "standard"])
        with pytest.raises(SystemExit):
            _apply_preset(p, args)
        captured = capsys.readouterr()
        assert "--crop" in captured.err

    def test_valid_combination_no_warning(self, capsys):
        p = _make_minimal_parser()
        args = p.parse_args(["--preset", "standard", "--crop", "simple-corners"])
        _apply_preset(p, args)
        captured = capsys.readouterr()
        assert "Warning" not in captured.err


# ---------------------------------------------------------------------------
# CLI-level integration: --crop works without any preset
# ---------------------------------------------------------------------------


class TestCropWithoutPreset:

    def test_crop_alone_no_warning(self, capsys):
        """Using --crop without --preset should work silently."""
        p = _make_minimal_parser()
        args = p.parse_args(["--crop", "simple-corners"])
        result = _apply_preset(p, args)
        assert result.crop == "simple-corners"
        captured = capsys.readouterr()
        assert "Warning" not in captured.err

    def test_crop_alone_no_detection_args(self):
        p = _make_minimal_parser()
        args = p.parse_args(["--crop", "simple-corners"])
        result = _apply_preset(p, args)
        assert result.pose_refine is False
        assert result.corner_refine is False
        assert result.cv_refine is False
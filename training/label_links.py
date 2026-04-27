#!/usr/bin/env python3
"""
Label Symlink Manager
=====================

Manages the `data/labels` symlink that Ultralytics uses to find label files.

WHY THIS EXISTS
---------------
Ultralytics resolves label paths by replacing `/images/` with `/labels/` in
the image path. Since both dataset YAMLs (detection and pose) specify:

    path: /path/to/data
    train: images/train
    val: images/val

...Ultralytics will look for labels at:
    data/labels/train/*.txt
    data/labels/val/*.txt

But the actual label directories are:
    data/detection/labels/train/*.txt  (5-column bounding box labels)
    data/pose/labels/train/*.txt      (13-column keypoint labels)

The `data/labels` symlink must point to the correct label directory before
training begins. This module ensures the symlink is created correctly.

CRITICAL: Only one label type can be active at a time, because the two
datasets share the same image directory and Ultralytics uses path-based
resolution. The symlink must be switched between detection and pose training.

FUTURE ALTERNATIVE
------------------
Instead of symlinks, the data generator could output labels directly into
`data/labels/train/` and `data/labels/val/`. This would require restructuring
the generator to write detection and pose datasets into separate image+
label directory pairs, or using separate `path` roots for each dataset YAML.

For example, detection YAML could specify:
    path: /path/to/data/detection
    train: images/train
    val: images/val

With detection images symlinked (or copied) into data/detection/images/
and detection labels in data/detection/labels/. This would eliminate the
need for a shared `data/labels` symlink entirely. See PITFALLS.md for details.

Author: Photo Pose Detector Project
"""

import os
import shutil
from pathlib import Path


# Label directory paths relative to data root
DETECTION_LABELS = "detection/labels"
POSE_LABELS = "pose/labels"

# The shared symlink path (relative to data root)
LABELS_SYMLINK = "labels"


def ensure_labels_symlink(data_root: str | Path, label_type: str) -> Path:
    """
    Ensure the `data/labels` symlink points to the correct label directory.

    This MUST be called before any Ultralytics training to prevent the
    zero-loss bug where all images are treated as backgrounds.

    Args:
        data_root: Path to the data directory (e.g., /path/to/data)
        label_type: Either "detection" or "pose"

    Returns:
        The resolved path that data/labels now points to.

    Raises:
        ValueError: If label_type is not "detection" or "pose"
        FileNotFoundError: If the target label directory doesn't exist
    """
    data_root = Path(data_root).resolve()
    symlink_path = data_root / LABELS_SYMLINK

    if label_type == "detection":
        target_rel = DETECTION_LABELS
    elif label_type == "pose":
        target_rel = POSE_LABELS
    else:
        raise ValueError(f"label_type must be 'detection' or 'pose', got '{label_type}'")

    target_path = data_root / target_rel

    # Verify target directory exists
    if not target_path.exists():
        raise FileNotFoundError(
            f"Label directory not found: {target_path}\n"
            f"Run the data generator first: python generate.py --mode batch"
        )

    # Verify target has train/val subdirectories
    for split in ("train", "val"):
        if not (target_path / split).exists():
            raise FileNotFoundError(
                f"Label split directory not found: {target_path / split}\n"
                f"Expected {target_rel}/{split}/ with .txt label files"
            )

    # Check if symlink already exists and points to the correct target
    if symlink_path.is_symlink():
        current_target = os.readlink(str(symlink_path))
        current_resolved = Path(current_target)
        if not current_resolved.is_absolute():
            current_resolved = (symlink_path.parent / current_resolved).resolve()

        if current_resolved == target_path.resolve():
            # Already correct — nothing to do
            return symlink_path

        # Symlink exists but points to wrong target — switch it
        print(f"[label_links] Switching symlink: {symlink_path}")
        print(f"  from: {current_resolved}")
        print(f"  to:   {target_path}")
        symlink_path.unlink()

    elif symlink_path.exists() and symlink_path.is_dir():
        # A real directory exists (not a symlink) — this shouldn't happen
        # in normal operation. Back it up rather than removing it.
        backup_path = symlink_path.parent / f"{LABELS_SYMLINK}_backup_{label_type}"
        print(f"[label_links] WARNING: {symlink_path} is a real directory, not a symlink!")
        print(f"  Backing up to: {backup_path}")
        shutil.move(str(symlink_path), str(backup_path))

    # Create the symlink with absolute path for reliability
    target_absolute = target_path.resolve()
    symlink_path.symlink_to(target_absolute)
    print(f"[label_links] Created symlink: {symlink_path} -> {target_absolute}")

    # Verify the symlink works
    if not symlink_path.exists():
        raise RuntimeError(f"Symlink creation failed: {symlink_path} -> {target_absolute}")

    # Clear stale cache files — they may contain data for the wrong label type
    _clear_stale_caches(symlink_path)

    return symlink_path


def _clear_stale_caches(symlink_path: Path) -> None:
    """
    Remove .cache files from the resolved label directory.

    Ultralytics creates cache files alongside label .txt files. If the
    symlink previously pointed to a different label type (e.g., pose labels
    when detection is needed), the cache will contain wrong data and cause
    zero-loss training. Deleting cache forces Ultralytics to re-scan labels.
    """
    resolved = symlink_path.resolve()

    # Check the symlink-resolved path (where Ultralytics looks for cache)
    for cache_file in resolved.glob("*.cache"):
        try:
            cache_file.unlink()
            print(f"[label_links] Removed stale cache: {cache_file}")
        except OSError as e:
            print(f"[label_links] WARNING: Could not remove cache {cache_file}: {e}")

    # Also check if cache exists at the symlink path itself
    # (Ultralytics may resolve the symlink before looking for cache)
    for cache_file in symlink_path.glob("*.cache"):
        try:
            cache_file.unlink()
            print(f"[label_links] Removed stale cache (symlink path): {cache_file}")
        except OSError as e:
            print(f"[label_links] WARNING: Could not remove cache {cache_file}: {e}")


def get_data_root_from_yaml(yaml_path: str | Path) -> Path:
    """
    Extract the data root path from a dataset YAML file.

    Args:
        yaml_path: Path to dataset YAML (e.g., dataset_detection.yaml)

    Returns:
        The resolved data root directory path.
    """
    yaml_path = Path(yaml_path).resolve()

    # Simple YAML parsing — we only need the 'path' field
    with open(yaml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("path:"):
                data_path = line.split(":", 1)[1].strip()
                return Path(data_path).resolve()

    raise ValueError(f"No 'path' field found in {yaml_path}")


def verify_label_integrity(data_root: str | Path, label_type: str) -> dict:
    """
    Verify label files exist and have the expected column count.

    Args:
        data_root: Path to the data directory
        label_type: "detection" or "pose"

    Returns:
        Dict with 'train_count', 'val_count', 'expected_columns', 'errors'
    """
    data_root = Path(data_root).resolve()

    if label_type == "detection":
        label_dir = data_root / DETECTION_LABELS
        expected_cols = 5
    else:
        label_dir = data_root / POSE_LABELS
        expected_cols = 17  # 5 bbox + 4 keypoints × 3 values

    result = {
        "label_type": label_type,
        "label_dir": str(label_dir),
        "expected_columns": expected_cols,
        "train_count": 0,
        "val_count": 0,
        "errors": [],
    }

    for split in ("train", "val"):
        split_dir = label_dir / split
        if not split_dir.exists():
            result["errors"].append(f"Missing {split} directory: {split_dir}")
            continue

        label_files = list(split_dir.glob("*.txt"))
        count = len(label_files)
        result[f"{split}_count"] = count

        # Spot-check first and last label files
        for label_file in [label_files[0], label_files[-1]] if label_files else []:
            with open(label_file) as f:
                first_line = f.readline().strip()
                if first_line:
                    cols = len(first_line.split())
                    if cols != expected_cols:
                        result["errors"].append(
                            f"{label_file.name}: expected {expected_cols} columns, "
                            f"got {cols} (line: '{first_line[:80]}')"
                        )

    return result


if __name__ == "__main__":
    """CLI for label link management."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Manage label symlinks for Ultralytics training")
    parser.add_argument(
        "action",
        choices=["link", "verify", "status"],
        help="Action: 'link' to create symlink, 'verify' to check labels, 'status' to check current state"
    )
    parser.add_argument(
        "--type", choices=["detection", "pose"],
        help="Label type for 'link' and 'verify' actions"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override data root path (auto-detected from YAML if not given)"
    )
    parser.add_argument(
        "--yaml", type=str, default=None,
        help="Dataset YAML file to extract data root from"
    )

    args = parser.parse_args()

    if args.action == "status":
        data_root = Path(args.data_root) if args.data_root else None
        if not data_root:
            if args.yaml:
                data_root = get_data_root_from_yaml(args.yaml)
            else:
                # Try to auto-detect
                script_dir = Path(__file__).parent
                candidates = [
                    script_dir / ".." / "data",
                    Path("/Users/krys.petrie/dev/photo-pose-detector/data"),
                ]
                for c in candidates:
                    if c.exists():
                        data_root = c.resolve()
                        break

        if not data_root or not data_root.exists():
            print("Error: Cannot find data root. Use --data-root or --yaml")
            sys.exit(1)

        symlink = data_root / LABELS_SYMLINK
        if symlink.is_symlink():
            target = os.readlink(str(symlink))
            # Determine which type it points to
            resolved = symlink.resolve()
            det_path = (data_root / DETECTION_LABELS).resolve()
            pose_path = (data_root / POSE_LABELS).resolve()
            if resolved == det_path:
                label_type = "detection"
            elif resolved == pose_path:
                label_type = "pose"
            else:
                label_type = f"UNKNOWN (points to {resolved})"
            print(f"data/labels -> {target}")
            print(f"Currently pointing to: {label_type} labels")
        elif symlink.exists():
            print(f"data/labels exists as a real directory (not a symlink!)")
        else:
            print(f"data/labels does not exist")
            print("Training will FAIL — run 'label_links.py link --type detection' first")

    elif args.action == "link":
        if not args.type:
            print("Error: --type is required for 'link' action")
            sys.exit(1)

        if args.data_root:
            data_root = Path(args.data_root)
        elif args.yaml:
            data_root = get_data_root_from_yaml(args.yaml)
        else:
            # Auto-detect from dataset YAML
            script_dir = Path(__file__).parent
            yaml_name = "dataset_detection.yaml" if args.type == "detection" else "dataset_pose.yaml"
            yaml_path = script_dir / yaml_name
            if yaml_path.exists():
                data_root = get_data_root_from_yaml(yaml_path)
            else:
                print("Error: Cannot find dataset YAML. Use --data-root or --yaml")
                sys.exit(1)

        ensure_labels_symlink(data_root, args.type)

    elif args.action == "verify":
        if not args.type:
            print("Error: --type is required for 'verify' action")
            sys.exit(1)

        if args.data_root:
            data_root = Path(args.data_root)
        elif args.yaml:
            data_root = get_data_root_from_yaml(args.yaml)
        else:
            script_dir = Path(__file__).parent
            data_root = (script_dir / ".." / "data").resolve()

        result = verify_label_integrity(data_root, args.type)
        print(f"Label type: {result['label_type']}")
        print(f"Directory: {result['label_dir']}")
        print(f"Expected columns: {result['expected_columns']}")
        print(f"Train labels: {result['train_count']}")
        print(f"Val labels: {result['val_count']}")
        if result["errors"]:
            print(f"\nErrors ({len(result['errors'])}):")
            for err in result["errors"]:
                print(f"  ❌ {err}")
        else:
            print("\n✅ All checks passed")
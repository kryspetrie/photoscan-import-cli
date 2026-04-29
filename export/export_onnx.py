#!/usr/bin/env python3
"""
Photo Pose Detector - ONNX Export

Exports trained detection and pose models to ONNX format
for cross-platform deployment outside Python/Ultralytics.

Usage:
    # Export both models
    python export_onnx.py --all

    # Export detection model only
    python export_onnx.py --model runs/detection/photo-detector/weights/best.pt

    # Export pose model only
    python export_onnx.py --model runs/pose/photo-corner-detector/weights/best.pt

    # Export with specific options
    python export_onnx.py --all --imgsz 640 --opset 17 --test
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# MPS fallback for Mac
if sys.platform == "darwin" and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Check dependencies
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed.")
    print("Install with: pip install ultralytics")
    sys.exit(1)


# Default model paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"

DEFAULT_MODELS = {
    "detection": TRAINING_DIR / "runs" / "detection" / "photo-detector" / "weights" / "best.pt",
    "pose": TRAINING_DIR / "runs" / "pose" / "photo-corner-detector" / "weights" / "best.pt",
}


def export_to_onnx(
    model_path: str,
    output_dir: str = None,
    format: str = "onnx",
    dynamic: bool = True,
    simplify: bool = True,
    opset: int = 17,
    imgsz: int = 640,
    half: bool = False,
    batch: int = 1,
):
    """
    Export trained model to ONNX format.

    Args:
        model_path: Path to trained model (.pt file)
        output_dir: Output directory for ONNX model
        format: Export format (onnx, torchscript, etc.)
        dynamic: Enable dynamic input sizes
        simplify: Simplify ONNX graph
        opset: ONNX opset version
        imgsz: Input image size
        half: Export with FP16 precision
        batch: Batch size for inference
    """

    model_path = Path(model_path)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Determine model type from path for output naming
    model_type = "unknown"
    for key, default_path in DEFAULT_MODELS.items():
        if model_path.resolve() == default_path.resolve():
            model_type = key
            break

    # Determine output directory
    if output_dir is None:
        output_dir = PROJECT_ROOT / "models"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting to {format.upper()}...")
    print(f"  Model type: {model_type}")
    print(f"  Source: {model_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Dynamic input: {dynamic}")
    print(f"  Simplify graph: {simplify}")
    print(f"  Opset version: {opset}")
    print(f"  Image size: {imgsz}")

    # Export
    try:
        success_path = model.export(
            format=format,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            imgsz=imgsz,
            half=half,
            batch=batch,
        )

        print(f"\n✓ Export successful!")
        print(f"  Exported to: {success_path}")

        # Move to output directory with descriptive name
        src = Path(success_path)
        if model_type != "unknown":
            dest = output_dir / f"{model_type}_model.onnx"
        else:
            dest = output_dir / src.name

        if src.resolve() != dest.resolve():
            if dest.exists():
                dest.unlink()
            shutil.move(str(src), str(dest))
            success_path = dest
            print(f"  Moved to: {success_path}")

        # Print file info
        file_size = success_path.stat().st_size
        if file_size > 1024 * 1024:
            print(f"  File size: {file_size / (1024 * 1024):.2f} MB")
        else:
            print(f"  File size: {file_size / 1024:.2f} KB")

        print(f"\nModel ready for deployment!")
        print(f"  Path: {success_path}")

        # Verify ONNX model
        if format == "onnx":
            verify_onnx(success_path)

        return success_path

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        raise


def verify_onnx(model_path):
    """Verify ONNX model can be loaded and print I/O info."""
    try:
        import onnx
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
        print(f"  ONNX model verified ✓")

        # Print model info
        print(f"\n  ONNX Model Info:")
        for input_tensor in onnx_model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic'
                    for dim in input_tensor.type.tensor_type.shape.dim]
            dtype = input_tensor.type.tensor_type.elem_type
            print(f"    Input: {input_tensor.name} shape=({', '.join(map(str, shape))}) dtype={dtype}")

        for output_tensor in onnx_model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic'
                    for dim in output_tensor.type.tensor_type.shape.dim]
            dtype = output_tensor.type.tensor_type.elem_type
            print(f"    Output: {output_tensor.name} shape=({', '.join(map(str, shape))}) dtype={dtype}")

    except ImportError:
        print("  (Install onnx to verify model: pip install onnx)")
    except Exception as e:
        print(f"  Warning: ONNX verification failed: {e}")


def test_inference(model_path):
    """Test ONNX model inference with a dummy input."""
    try:
        import onnxruntime as ort
        import numpy as np

        print("\nTesting inference...")

        session = ort.InferenceSession(str(model_path))

        # Get input info
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape

        # Create dummy input matching expected shape
        # Replace dynamic dims with concrete values
        test_shape = []
        for dim in input_shape:
            if isinstance(dim, int) and dim > 0:
                test_shape.append(dim)
            else:
                test_shape.append(1 if len(test_shape) == 0 else 640)

        dummy_input = np.random.randn(*test_shape).astype(np.float32)

        # Run inference
        outputs = session.run(None, {input_name: dummy_input})

        print(f"  ✓ Inference test passed")
        print(f"    Input shape: {dummy_input.shape}")
        for i, output in enumerate(outputs):
            print(f"    Output {i}: {output.shape}")

    except ImportError:
        print("\n  (Install onnxruntime to test inference: pip install onnxruntime)")
    except Exception as e:
        print(f"\n  ✗ Inference test failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export detection and/or pose models to ONNX format"
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to a specific trained model (.pt file)"
    )
    model_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Export all trained models (detection + pose)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for ONNX models (default: project/models/)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "openvino"],
        help="Export format"
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic input sizes (fixed batch=1)"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX graph simplification"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export with FP16 precision"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test inference after export"
    )

    args = parser.parse_args()

    # Determine which models to export
    if args.all:
        models_to_export = []
        for name, path in DEFAULT_MODELS.items():
            if path.exists():
                models_to_export.append((name, path))
            else:
                print(f"⚠ Skipping {name} model: not found at {path}")
        if not models_to_export:
            print("Error: No models found to export.")
            sys.exit(1)
    elif args.model:
        models_to_export = [("custom", Path(args.model))]
    else:
        # Default: export all available
        models_to_export = []
        for name, path in DEFAULT_MODELS.items():
            if path.exists():
                models_to_export.append((name, path))
        if not models_to_export:
            print("Error: No models found. Specify --model or ensure defaults exist.")
            print("\nExpected model locations:")
            for name, path in DEFAULT_MODELS.items():
                exists = "✓" if path.exists() else "✗"
                print(f"  {exists} {name}: {path}")
            sys.exit(1)

    # Export each model
    exported = []
    for name, path in models_to_export:
        print("\n" + "=" * 60)
        print(f"Exporting {name} model")
        print("=" * 60)

        try:
            output_path = export_to_onnx(
                model_path=path,
                output_dir=args.output,
                format=args.format,
                dynamic=not args.no_dynamic,
                simplify=not args.no_simplify,
                opset=args.opset,
                imgsz=args.imgsz,
                half=args.half,
            )
            exported.append((name, output_path))

            if args.test and args.format == "onnx":
                test_inference(output_path)

        except Exception as e:
            print(f"Failed to export {name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    for name, path in exported:
        file_size = path.stat().st_size / (1024 * 1024)
        print(f"  {name}: {path} ({file_size:.1f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
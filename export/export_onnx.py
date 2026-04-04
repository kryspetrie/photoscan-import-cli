#!/usr/bin/env python3
"""
Photo Pose Detector - ONNX Export

Exports the trained model to ONNX format for cross-platform deployment.

Usage:
    python export_onnx.py --model runs/pose/photo-corner-detector/weights/best.pt
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Check dependencies
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed.")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def export_to_onnx(
    model_path: str,
    output_dir: str = None,
    format: str = "onnx",
    dynamic: bool = True,
    simplify: bool = True,
    opset: int = 12,
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
        print("\nAvailable paths to check:")
        # Look for common paths
        common_paths = [
            Path("runs/pose/photo-corner-detector/weights/best.pt"),
            Path("runs/pose/photo-corner-detector/weights/last.pt"),
        ]
        for p in common_paths:
            status = "✓" if p.exists() else "✗"
            print(f"  {status} {p}")
        sys.exit(1)
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "models" / "photo-corner-detector"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting to {format.upper()}...")
    print(f"  Model: {model_path}")
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
        
        # Move to output directory if different
        if Path(success_path).parent != output_dir:
            dest = output_dir / Path(success_path).name
            shutil.move(success_path, dest)
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
    """Verify ONNX model can be loaded."""
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
            print(f"    Input: {input_tensor.name} ({', '.join(map(str, shape))})")
        
        for output_tensor in onnx_model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic'
                    for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"    Output: {output_tensor.name} ({', '.join(map(str, shape))})")
            
    except ImportError:
        print("  (Install onnxruntime to verify model)")
    except Exception as e:
        print(f"  Warning: ONNX verification failed: {e}")


def test_inference(model_path):
    """Test ONNX model inference."""
    try:
        import onnxruntime as ort
        
        print("\nTesting inference...")
        
        # Create session
        session = ort.InferenceSession(str(model_path))
        
        # Get input name
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Create dummy input
        import numpy as np
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # Run inference
        output = session.run([output_name], {input_name: dummy_input})
        
        print(f"  ✓ Inference test passed")
        print(f"    Input shape: {dummy_input.shape}")
        print(f"    Output shape: {output[0].shape}")
        
    except ImportError:
        print("\n  (Install onnxruntime to test inference)")
        print("    pip install onnxruntime")
    except Exception as e:
        print(f"\n  ✗ Inference test failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export YOLO26-pose model to ONNX format"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="runs/pose/photo-corner-detector/weights/best.pt",
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for ONNX model"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "tflite", "coreml", "engine"],
        help="Export format"
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic input sizes"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX graph simplification"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version"
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
    
    # Export
    output_path = export_to_onnx(
        model_path=args.model,
        output_dir=args.output,
        format=args.format,
        dynamic=not args.no_dynamic,
        simplify=not args.no_simplify,
        opset=args.opset,
        imgsz=args.imgsz,
        half=args.half,
    )
    
    # Test inference if requested
    if args.test and args.format == "onnx":
        test_inference(output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

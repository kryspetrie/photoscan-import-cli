"""Allow running as ``python -m onnx_inference`` (delegates to photocrop CLI)."""
from onnx_inference.photocrop import main

if __name__ == "__main__":
    main()
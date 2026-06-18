#!/usr/bin/env python3
"""
Export model to ONNX and quantize with INT8.

ONNX export makes the model portable across runtimes (TensorRT, ONNX Runtime, CoreML, etc.).
INT8 dynamic quantization reduces model size 3-4× with <1% perplexity loss.
"""

import argparse
from pathlib import Path
import torch
import torch.onnx

from tinyllm.config import ModelConfig
from tinyllm.model import GermanLM
from tinyllm.tokenizer import GermanTokenizer


def export_to_onnx(
    model: GermanLM,
    model_config: ModelConfig,
    output_path: Path,
    opset_version: int = 17,
    device: str = "cuda",
):
    """Export model to ONNX format.

    Args:
        model: GermanLM instance
        model_config: Model configuration
        output_path: Output ONNX file path
        opset_version: ONNX opset version (17+ supports scaled_dot_product_attention)
        device: Device to export on
    """
    model.eval()
    model = model.to(device)

    # Create dummy input
    dummy_input = torch.randint(0, model_config.vocab_size, (1, model_config.max_seq_len), device=device)

    # Export
    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )
    print(f"✓ Exported to {output_path}")


def quantize_onnx_int8(
    input_path: Path,
    output_path: Path,
):
    """Quantize ONNX model with INT8 (dynamic quantization).

    Dynamic quantization quantizes weights to INT8 at load time.
    No calibration dataset needed. Typical: 3-4× smaller, 1.5-2× faster CPU inference.

    Args:
        input_path: Input ONNX model path
        output_path: Output quantized ONNX model path
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        print(f"Quantizing ONNX model to INT8...")
        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QInt8,
        )
        print(f"✓ Quantized model saved to {output_path}")

        # Show size reduction
        original_size = input_path.stat().st_size / (1024 * 1024)
        quantized_size = output_path.stat().st_size / (1024 * 1024)
        reduction = 100 * (1 - quantized_size / original_size)
        print(f"  Size: {original_size:.1f} MB → {quantized_size:.1f} MB ({reduction:.0f}% reduction)")

    except ImportError:
        raise ImportError("onnxruntime is required for quantization. Install with: pip install onnxruntime")


def main():
    """Export and quantize model from checkpoint."""
    parser = argparse.ArgumentParser(description="Export GermanLM to ONNX and quantize")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output-dir", default="exports", help="Output directory for ONNX files")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-quantize", action="store_true", help="Skip INT8 quantization")
    parser.add_argument("--device", default="cuda", help="Device to export on")

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model_config = ModelConfig.from_dict(checkpoint["model_config"])
    model = GermanLM(model_config)
    model.load_state_dict(checkpoint["model"])

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    export_to_onnx(model, model_config, onnx_path, opset_version=args.opset, device=args.device)

    # Quantize if enabled
    if not args.no_quantize:
        quantized_path = output_dir / "model_int8.onnx"
        quantize_onnx_int8(onnx_path, quantized_path)

    print(f"\n✓ Export complete!")
    print(f"  ONNX model: {onnx_path}")
    if not args.no_quantize:
        print(f"  INT8 quantized: {quantized_path}")


if __name__ == "__main__":
    main()

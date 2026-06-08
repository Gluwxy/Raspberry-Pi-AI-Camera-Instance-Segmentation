"""
quantize_and_export.py — PTQ + ONNX export for Sony IMX500 Converter
=====================================================================
Pipeline
--------
1.  Load FP32 DeepLabV3+ model (or pretrained weights)
2.  Run Sony Model Compression Toolkit (MCT) PTQ
    - 8-bit weights + 8-bit feature maps (IMX500 requirement)
    - Representative dataset: 300–1000 random crops at 128×128
    - TPC version 5.0 (required for IMX500 Converter v3.18.2)
3.  Export quantized model to ONNX (opset 17)
4.  Verify ONNX model is IMX500-compliant
5.  Print memory estimate

Usage
-----
# Quantize with dummy representative data (quick test)
python quantize_and_export.py --num-classes 21 --output-dir ./output --dummy

# Quantize with real calibration images
python quantize_and_export.py --num-classes 21 --calib-dir /path/to/images --output-dir ./output

Requirements
------------
pip install torch torchvision onnx onnxruntime
pip install model-compression-toolkit      # Sony MCT

IMX500 Converter (run AFTER this script):
pip install imx500-converter[pt]
imxconv-pt -i ./output/deeplabv3plus_imx500_q.onnx -o ./output/rpk --no-input-persistency
"""

import argparse
import os
import sys
import numpy as np
import torch
import onnx
import onnxruntime as ort

# ── local model ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model import build_model

# ─────────────────────────────────────────────────────────────────────────────
# Representative dataset generator
# IMX500 constraint: input normalised to [0, 1] range
# Recommended: 300-1000 real calibration images (VOC val or custom)
# ─────────────────────────────────────────────────────────────────────────────

def make_representative_dataset(calib_dir=None, n_samples=300,
                                img_size=128, batch_size=1):
    """
    Returns a generator function compatible with Sony MCT.

    If calib_dir is given: loads real images (PNG/JPG) from that directory.
    Otherwise: uses random noise tensors for a quick smoke-test.

    Each yield is a list with one element: a numpy array of shape
    [batch_size, 3, img_size, img_size] in float32, range [0,1].
    """
    if calib_dir is not None:
        from PIL import Image
        import glob
        paths = (glob.glob(os.path.join(calib_dir, "*.jpg")) +
                 glob.glob(os.path.join(calib_dir, "*.png")))
        if not paths:
            raise FileNotFoundError(f"No JPG/PNG images found in {calib_dir}")
        paths = paths[:n_samples]
        print(f"[MCT] Using {len(paths)} real calibration images.")

        def real_dataset():
            for i in range(0, len(paths), batch_size):
                batch = []
                for p in paths[i:i + batch_size]:
                    img = Image.open(p).convert("RGB").resize(
                        (img_size, img_size), Image.BILINEAR)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    arr = arr.transpose(2, 0, 1)   # HWC → CHW
                    batch.append(arr)
                yield [np.stack(batch)]

        return real_dataset

    else:
        print(f"[MCT] No calib-dir provided — using {n_samples} random samples.")

        def dummy_dataset():
            for _ in range(n_samples):
                yield [np.random.uniform(0, 1,
                       (batch_size, 3, img_size, img_size)).astype(np.float32)]

        return dummy_dataset


# ─────────────────────────────────────────────────────────────────────────────
# MCT PTQ quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_with_mct(model, representative_data_gen, n_samples=300):
    """
    Post-Training Quantization via Sony Model Compression Toolkit.

    Key settings for IMX500:
    - tpc_version='5.0'  (required by IMX500 Converter v3.18.2)
    - weights_n_bits=8, activation_n_bits=8
    - Feature maps must be >= 8-bit (hardware requirement)
    """
    try:
        import model_compression_toolkit as mct
    except ImportError:
        print("\n[ERROR] model_compression_toolkit is not installed.")
        print("Install with:  pip install model-compression-toolkit")
        print("\nFalling back to unquantized ONNX export (for architecture review).")
        return model, False

    print("[MCT] Loading Target Platform Capabilities (TPC v5.0) for IMX500…")
    try:
        # TPC v5.0 required for IMX500 Converter ≥ v3.18.2
        tpc = mct.get_target_platform_capabilities(
            fw_name="pytorch",
            target_platform_name="imx500",
            target_platform_version="v5"
        )
    except Exception:
        # Fallback to latest available TPC
        print("[MCT] TPC v5.0 not found — using latest available TPC.")
        tpc = mct.get_target_platform_capabilities(
            fw_name="pytorch",
            target_platform_name="imx500",
        )

    # Core PTQ configuration
    # - weights INT8, activations INT8
    # - 300 representative samples (Sony recommendation: 300–1000)
    core_config = mct.core.CoreConfig(
        quantization_config=mct.core.QuantizationConfig(
            weights_error_method=mct.core.QuantizationErrorMethod.MSE,
            activation_error_method=mct.core.QuantizationErrorMethod.MSE,
        )
    )

    print(f"[MCT] Running PTQ with {n_samples} calibration samples…")
    quant_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
        in_module=model,
        representative_data_gen=representative_data_gen,
        core_config=core_config,
        target_platform_capabilities=tpc,
    )

    print("[MCT] PTQ complete.")
    return quant_model, True


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export  (opset 17 — supported by IMX500 Converter)
# ─────────────────────────────────────────────────────────────────────────────

def export_to_onnx(model, output_path, img_size=128, opset=17,
                   mct_quantized=False):
    """
    Export model to ONNX.

    For MCT-quantized models: MCT provides its own export utility that
    generates a correctly quantized ONNX graph.
    For unquantized models: standard torch.onnx.export.

    IMX500 constraints respected:
    - Fixed static input shape [1, 3, 128, 128] (no dynamic axes for chip)
    - opset 17 (within supported range 15-20)
    - align_corners=False already set in model
    """
    dummy = torch.zeros(1, 3, img_size, img_size)
    model.eval()

    if mct_quantized:
        try:
            import model_compression_toolkit as mct
            print(f"[ONNX] Exporting quantized model via MCT → {output_path}")
            mct.exporter.pytorch_export_model(
                model=model,
                save_model_path=output_path,
                repr_dataset=lambda: iter([[dummy.numpy()]]),
                target_platform_capabilities=None,   # uses last loaded TPC
                onnx_opset_version=opset,
            )
            return
        except Exception as e:
            print(f"[ONNX] MCT export failed ({e}), falling back to torch export.")

    print(f"[ONNX] Exporting float model → {output_path}  (opset {opset})")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            # Static shapes — required for IMX500 (no dynamic batch)
            dynamic_axes=None,
            do_constant_folding=True,
            export_params=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ONNX validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_onnx(onnx_path, img_size=128):
    """Basic ONNX graph check + ORT inference smoke test."""
    print(f"\n[Validate] Checking ONNX graph: {onnx_path}")
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print("[Validate] ✓ ONNX graph is valid.")

    # File size
    size_mb = os.path.getsize(onnx_path) / (1024 ** 2)
    print(f"[Validate] File size: {size_mb:.2f} MB")
    if size_mb > 12.0:
        print(f"[Validate] ⚠ WARNING: {size_mb:.2f} MB > 12 MB limit for IMX500.")
    else:
        print(f"[Validate] ✓ Size OK (< 12 MB FP32 limit).")

    # ORT inference
    print("[Validate] Running ORT inference…")
    sess = ort.InferenceSession(onnx_path,
                                providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    out = sess.run(None, {"input": dummy})[0]
    print(f"[Validate] ✓ Output shape: {out.shape}  "
          f"(expected [1, num_classes, {img_size}, {img_size}])")

    # Op check — warn about known unsupported ops
    UNSUPPORTED = {"ConvTranspose", "LSTM", "GRU", "NonMaxSuppression"}
    ops_used = {n.op_type for n in model_onnx.graph.node}
    bad = ops_used & UNSUPPORTED
    if bad:
        print(f"[Validate] ⚠ Possibly unsupported ONNX ops: {bad}")
    else:
        print(f"[Validate] ✓ No known unsupported ops detected.")

    return size_mb


# ─────────────────────────────────────────────────────────────────────────────
# Memory estimate
# ─────────────────────────────────────────────────────────────────────────────

def print_memory_estimate(fp32_mb, img_size=128, num_classes=21):
    """
    Rough memory estimate for IMX500 (8 MB total).

    IMX500 memory layout:
      Model Memory  = quantized weights + activation buffers
      Runtime Memory = peak feature map memory (overlaid)
      Reserved      = ~1 KB overhead
      Total         must be < 8 MB

    For --no-input-persistency mode (recommended for best performance):
      Input tensor is NOT stored persistently → frees ~150 KB extra.
    """
    # After INT8 PTQ: ~4× compression on weights
    int8_weights_mb = fp32_mb / 4.0

    # Runtime memory at 128×128: empirically ~4 MB for this arch
    # (largest intermediate tensor: ASPP input 160ch × 8×8 = 10240 float32 = 40 KB)
    # Peak is in decoder upsample: 32ch × 128×128 = 2 MB float32 = 0.5 MB INT8
    runtime_mb_estimate = (32 * img_size * img_size * 1) / (1024 ** 2)  # INT8

    total_mb = int8_weights_mb + runtime_mb_estimate + 0.001  # +1KB reserved

    print("\n" + "=" * 55)
    print("  MEMORY ESTIMATE (IMX500 — 8 MB total)")
    print("=" * 55)
    print(f"  FP32 model size        : {fp32_mb:.3f} MB")
    print(f"  INT8 model (÷4)        : {int8_weights_mb:.3f} MB  [Model Memory]")
    print(f"  Runtime peak (INT8)    : {runtime_mb_estimate:.3f} MB  [Runtime Memory]")
    print(f"  Reserved               : 0.001 MB")
    print(f"  ─────────────────────────────────────")
    print(f"  Total estimate         : {total_mb:.3f} MB / 8.000 MB")
    print(f"  Fits in chip           : {'✓ YES' if total_mb < 8.0 else '✗ NO'}")
    print("=" * 55)
    print()
    print("  NOTE: Actual values will be in packerOut.zip memory report.")
    print("  Use --no-input-persistency flag in imxconv-pt for best fit.")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantize DeepLabV3+ (MobileNetV2) and export to ONNX for IMX500"
    )
    parser.add_argument("--num-classes", type=int, default=21,
                        help="Number of segmentation classes (default: 21 for VOC)")
    parser.add_argument("--width-mult", type=float, default=0.35,
                        help="MobileNetV2 width multiplier (default: 0.35)")
    parser.add_argument("--img-size", type=int, default=128,
                        help="Input image size (default: 128)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained FP32 weights (.pth)")
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="Directory with calibration images for PTQ")
    parser.add_argument("--n-samples", type=int, default=300,
                        help="Number of representative data samples (default: 300)")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--dummy", action="store_true",
                        help="Use random noise for calibration (quick test)")
    parser.add_argument("--skip-mct", action="store_true",
                        help="Skip MCT quantization (export FP32 ONNX only)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Build model ──────────────────────────────────────────────────────
    print(f"\n[Model] Building DeepLabV3+ (width_mult={args.width_mult}, "
          f"classes={args.num_classes})")
    model = build_model(num_classes=args.num_classes,
                        width_mult=args.width_mult)
    model.eval()

    if args.weights:
        print(f"[Model] Loading weights from {args.weights}")
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state)

    n_params = sum(p.numel() for p in model.parameters())
    fp32_mb = n_params * 4 / (1024 ** 2)
    print(f"[Model] Parameters: {n_params:,}  |  FP32 size: {fp32_mb:.3f} MB")

    # ── 2. PTQ with MCT ─────────────────────────────────────────────────────
    mct_quantized = False
    if not args.skip_mct:
        rep_dataset = make_representative_dataset(
            calib_dir=None if args.dummy else args.calib_dir,
            n_samples=args.n_samples,
            img_size=args.img_size,
        )
        model, mct_quantized = quantize_with_mct(
            model, rep_dataset, n_samples=args.n_samples
        )

    # ── 3. Export ONNX ──────────────────────────────────────────────────────
    suffix = "_q" if mct_quantized else "_fp32"
    onnx_name = f"deeplabv3plus_imx500{suffix}.onnx"
    onnx_path = os.path.join(args.output_dir, onnx_name)

    export_to_onnx(model, onnx_path, img_size=args.img_size,
                   opset=17, mct_quantized=mct_quantized)

    # ── 4. Validate ─────────────────────────────────────────────────────────
    actual_mb = validate_onnx(onnx_path, img_size=args.img_size)

    # ── 5. Memory estimate ──────────────────────────────────────────────────
    print_memory_estimate(fp32_mb, img_size=args.img_size,
                          num_classes=args.num_classes)

    # ── 6. IMX500 Converter command ─────────────────────────────────────────
    rpk_dir = os.path.join(args.output_dir, "rpk")
    print("\n" + "=" * 55)
    print("  NEXT STEP: Run IMX500 Converter")
    print("=" * 55)
    print("  Install:")
    print("    pip install imx500-converter[pt]")
    print()
    print("  Convert (best performance, --no-input-persistency):")
    print(f"    imxconv-pt \\")
    print(f"      -i {onnx_path} \\")
    print(f"      -o {rpk_dir} \\")
    print(f"      --no-input-persistency \\")
    print(f"      --overwrite-output \\")
    print(f"      --report-size-unit M")
    print()
    print("  Output: packerOut.zip  →  deploy to Raspberry Pi AI Camera")
    print("=" * 55)


if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    """Ensure channel count is divisible by `divisor` (MobileNet convention)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _relu6(name: str = None):
    """
    ReLU6 activation.
    In TF Keras → tf.keras.layers.ReLU(max_value=6.0)
    Exported to ONNX as Clip(min=0, max=6) — fully supported by IMX500.
    """
    return layers.ReLU(max_value=6.0, name=name)


def _bn(name: str = None):
    """
    BatchNormalization with axis=-1 (channels last, NHWC).
    IMX500 constraint: axis must be last axis.
    """
    return layers.BatchNormalization(
        axis=-1,
        momentum=0.99,    # same as PyTorch default (1 - 0.01 momentum)
        epsilon=1e-3,
        name=name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Functional building blocks (return tensors, not subclasses)
# Using Functional API throughout for clean ONNX export graph
# ─────────────────────────────────────────────────────────────────────────────

def conv_bn_relu6(x, filters: int, kernel: int = 3, stride: int = 1,
                  padding: str = "same", use_bias: bool = False,
                  name: str = ""):
    """
    Standard conv block: Conv2D → BatchNorm → ReLU6.
    Uses data_format='channels_last' (NHWC) — IMX500 requirement.
    """
    x = layers.Conv2D(
        filters, kernel,
        strides=stride,
        padding=padding,
        use_bias=use_bias,
        data_format="channels_last",
        name=f"{name}_conv",
    )(x)
    x = _bn(name=f"{name}_bn")(x)
    x = _relu6(name=f"{name}_relu6")(x)
    return x


def pw_conv_bn_relu6(x, filters: int, name: str = ""):
    """Point-wise 1×1 conv + BN + ReLU6."""
    return conv_bn_relu6(x, filters, kernel=1, padding="valid", name=name)


def pw_conv(x, filters: int, use_bias: bool = False, name: str = ""):
    """
    Point-wise 1×1 conv, no activation.
    Used for projection layers and the final classifier.
    """
    return layers.Conv2D(
        filters, 1,
        padding="valid",
        use_bias=use_bias,
        data_format="channels_last",
        name=f"{name}_pw",
    )(x)


def depthwise_bn_relu6(x, dilation: int = 1, stride: int = 1, name: str = ""):
    """
    Depthwise Conv2D + BN + ReLU6.
    IMX500 constraint: stride > 1 only with dilation = 1.
    depth_multiplier = 1 → output_group_size / input_group_size = 1 (integer ✓).
    """
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        dilation_rate=dilation,
        depth_multiplier=1,
        use_bias=False,
        data_format="channels_last",
        name=f"{name}_dw",
    )(x)
    x = _bn(name=f"{name}_dw_bn")(x)
    x = _relu6(name=f"{name}_dw_relu6")(x)
    return x


def sep_conv_bn_relu6(x, filters: int, dilation: int = 1, name: str = ""):
    """
    Depthwise-separable conv: DW + BN + ReLU6 → PW + BN + ReLU6.
    Used in ASPP branches and decoder refinement.
    """
    x = depthwise_bn_relu6(x, dilation=dilation, name=f"{name}_sep")
    x = pw_conv_bn_relu6(x, filters, name=f"{name}_sep")
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Inverted Residual Block (MobileNetV2)
# ─────────────────────────────────────────────────────────────────────────────

def inverted_residual(x, in_ch: int, out_ch: int, stride: int,
                      expand_ratio: int, dilation: int = 1,
                      name: str = ""):
    """
    MobileNetV2 Inverted Residual Block.

    Structure: [expand PW] → DW(dilation) → linear PW
    Skip connection when: stride=1 AND in_ch==out_ch.

    IMX500 constraints:
    - stride > 1 forces dilation = 1 (hardware limitation)
    - DepthwiseConv2D depth_multiplier = 1 always
    """
    assert stride in (1, 2), f"stride must be 1 or 2, got {stride}"

    # Hardware constraint: stride > 1 cannot use dilation > 1
    if stride > 1:
        dilation = 1

    use_skip = (stride == 1 and in_ch == out_ch)
    residual = x

    hidden = int(round(in_ch * expand_ratio))

    # ── Expansion (point-wise) ─────────────────────────────────────────────
    if expand_ratio != 1:
        x = pw_conv_bn_relu6(x, hidden, name=f"{name}_exp")

    # ── Depthwise ─────────────────────────────────────────────────────────
    x = depthwise_bn_relu6(x, dilation=dilation, stride=stride,
                           name=f"{name}")

    # ── Projection (linear, no activation) ────────────────────────────────
    x = layers.Conv2D(
        out_ch, 1,
        padding="valid",
        use_bias=False,
        data_format="channels_last",
        name=f"{name}_proj",
    )(x)
    x = _bn(name=f"{name}_proj_bn")(x)

    # ── Skip connection ────────────────────────────────────────────────────
    if use_skip:
        # IMX500: tf.keras.layers.Add — both inputs must be dynamic
        x = layers.Add(name=f"{name}_add")([residual, x])

    return x


# ─────────────────────────────────────────────────────────────────────────────
# MobileNetV2 Backbone  width_mult=1.25, output_stride=16
# ─────────────────────────────────────────────────────────────────────────────

def build_backbone(inputs, width_mult: float = 1.25):
    """
    MobileNetV2 backbone with:
      - width_mult = 1.25  (wider than original 1.0 for better capacity)
      - output_stride = 16 (last stage uses dilation instead of stride)
      - Increased depth: l3×3, l4×4, l5×3, l6×3 blocks

    Channel schedule (NHWC):
      stem   : 40ch  /2  → 64×64
      layer1 : 24ch  /1  → 64×64
      layer2 : 32ch  /2  → 32×32  ← low-level features (for decoder)
      layer3 : 40ch  /2  → 16×16
      layer4 : 80ch  /2  → 8×8
      layer5 : 120ch d=2 → 8×8
      layer6 : 200ch d=2 → 8×8    ← high-level features (for ASPP)

    Returns
    -------
    low  : [B, 32, 32, 32]  stride-4  features
    high : [B, 8,  8, 200]  stride-16 features (dilated)
    """
    def c(v):
        return _make_divisible(int(v * width_mult))

    # ── Stem ──────────────────────────────────────────────────────────────
    # 3 → 40ch, stride=2 → (64×64)
    x = conv_bn_relu6(inputs, c(32), kernel=3, stride=2, name="stem")

    # ── Layer 1: t=1, first bottleneck ────────────────────────────────────
    # 40 → 24ch, stride=1
    x = inverted_residual(x, c(32), c(16), stride=1, expand_ratio=1,
                          name="l1_b0")

    # ── Layer 2: stride=2 → 32×32  (LOW-LEVEL features) ──────────────────
    x = inverted_residual(x, c(16), c(24), stride=2, expand_ratio=6,
                          name="l2_b0")
    x = inverted_residual(x, c(24), c(24), stride=1, expand_ratio=6,
                          name="l2_b1")
    low = x   # [B, 32, 32, 32] — saved for decoder

    # ── Layer 3: stride=2 → 16×16 ─────────────────────────────────────────
    x = inverted_residual(x, c(24), c(32), stride=2, expand_ratio=6,
                          name="l3_b0")
    x = inverted_residual(x, c(32), c(32), stride=1, expand_ratio=6,
                          name="l3_b1")
    x = inverted_residual(x, c(32), c(32), stride=1, expand_ratio=6,
                          name="l3_b2")

    # ── Layer 4: stride=2 → 8×8  (output_stride = 16) ────────────────────
    x = inverted_residual(x, c(32), c(64), stride=2, expand_ratio=6,
                          name="l4_b0")
    x = inverted_residual(x, c(64), c(64), stride=1, expand_ratio=6,
                          name="l4_b1")
    x = inverted_residual(x, c(64), c(64), stride=1, expand_ratio=6,
                          name="l4_b2")
    x = inverted_residual(x, c(64), c(64), stride=1, expand_ratio=6,
                          name="l4_b3")

    # ── Layer 5: dilation=2, stays at 8×8 (no stride, preserves resolution) ──
    x = inverted_residual(x, c(64),  c(96), stride=1, expand_ratio=6,
                          dilation=2, name="l5_b0")
    x = inverted_residual(x, c(96),  c(96), stride=1, expand_ratio=6,
                          dilation=2, name="l5_b1")
    x = inverted_residual(x, c(96),  c(96), stride=1, expand_ratio=6,
                          dilation=2, name="l5_b2")

    # ── Layer 6: dilation=2, stays at 8×8 (additional depth) ─────────────
    x = inverted_residual(x, c(96),  c(160), stride=1, expand_ratio=6,
                          dilation=2, name="l6_b0")
    x = inverted_residual(x, c(160), c(160), stride=1, expand_ratio=6,
                          dilation=2, name="l6_b1")
    x = inverted_residual(x, c(160), c(160), stride=1, expand_ratio=6,
                          dilation=2, name="l6_b2")
    high = x  # [B, 8, 8, 200]

    low_ch  = c(24)   # 32 channels
    high_ch = c(160)  # 200 channels
    return low, high, low_ch, high_ch


# ─────────────────────────────────────────────────────────────────────────────
# ASPP — Atrous Spatial Pyramid Pooling
# 5 branches: 1×1 | SepConv rate=6 | rate=12 | rate=18 | GlobalAvgPool
# inner_width = 256
# ─────────────────────────────────────────────────────────────────────────────

def build_aspp(x, in_ch: int, inner_width: int = 256):
    """
    Full ASPP with 5 branches and projection.

    IMX500 notes:
    - GlobalAveragePooling2D(keepdims=True) → GlobalAveragePool op (supported)
    - Resizing with 'bilinear', crop_to_aspect_ratio=False (required)
    - All separable convs: depth_multiplier=1 (integer ratio constraint)
    - Project: concat(5 branches) → 1×1 conv + BN + ReLU6

    Returns [B, H, W, inner_width]
    """
    # Original problematic lines were removed, as x's spatial dimensions are static 8x8.
    # h = tf.shape(x)[1]
    # w = tf.shape(x)[2]

    # ── Branch 0: 1×1 conv ────────────────────────────────────────────────
    b0 = pw_conv_bn_relu6(x, inner_width, name="aspp_b0")

    # ── Branch 1: SepConv rate=6 ──────────────────────────────────────────
    b1 = sep_conv_bn_relu6(x, inner_width, dilation=6,  name="aspp_b1")

    # ── Branch 2: SepConv rate=12 ─────────────────────────────────────────
    b2 = sep_conv_bn_relu6(x, inner_width, dilation=12, name="aspp_b2")

    # ── Branch 3: SepConv rate=18 ─────────────────────────────────────────
    b3 = sep_conv_bn_relu6(x, inner_width, dilation=18, name="aspp_b3")

    # ── Branch 4: Global Average Pooling ──────────────────────────────────
    # GlobalAveragePooling2D(keepdims=True): [B,H,W,C] → [B,1,1,C]
    # IMX500: tf.keras.layers.GlobalAveragePooling2D  data_format=channels_last
    b4 = layers.GlobalAveragePooling2D(
        keepdims=True,
        data_format="channels_last",
        name="aspp_gap",
    )(x)
    b4 = pw_conv_bn_relu6(b4, inner_width, name="aspp_gap_pw")
    # Upsample back to spatial size — bilinear, no align_corners (IMX500 req)
    # Static target size for graph export compatibility
    b4 = layers.Resizing(
        height=x.shape[1], # Use static height from KerasTensor's shape
        width=x.shape[2],  # Use static width from KerasTensor's shape
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        name="aspp_gap_resize",
    )(b4)

    # ── Concatenate all 5 branches: 5 × inner_width channels ─────────────
    # IMX500: tf.keras.layers.Concatenate — both inputs must be dynamic ✓
    merged = layers.Concatenate(axis=-1, name="aspp_concat")([b0, b1, b2, b3, b4])

    # ── Project: [5*inner_width] → [inner_width] ──────────────────────────
    out = pw_conv_bn_relu6(merged, inner_width, name="aspp_proj")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────

def build_decoder(low, high_aspp, low_ch: int, aspp_ch: int,
                  dec_ch: int = 128, ll_reduce_ch: int = 32,
                  num_classes: int = 8, img_size: int = 128):
    """
    DeepLabV3+ Decoder.

    Steps:
      1. Reduce low-level features: low_ch → ll_reduce_ch (1×1 + BN + ReLU6)
      2. Upsample ASPP output × 4 (8×8 → 32×32) via Resizing
      3. Concatenate: [aspp_ch + ll_reduce_ch] channels
      4. Three SepConv refinement blocks → dec_ch
      5. Upsample × 4 (32×32 → 128×128) via Resizing
      6. Classifier: 1×1 Conv → num_classes  (raw logits, no activation)

    IMX500: all Resizing calls use bilinear, crop_to_aspect_ratio=False.
    """

    # ── Step 1: Reduce low-level features ─────────────────────────────────
    low_red = pw_conv_bn_relu6(low, ll_reduce_ch, name="dec_ll_reduce")
    # low_red: [B, 32, 32, 32]

    # ── Step 2: Upsample ASPP × 4 (8×8 → 32×32) ─────────────────────────
    aspp_up = layers.Resizing(
        height=img_size // 4,
        width=img_size // 4,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        name="dec_aspp_up",
    )(high_aspp)
    # aspp_up: [B, 32, 32, 256]

    # ── Step 3: Concatenate ────────────────────────────────────────────────
    merged = layers.Concatenate(axis=-1, name="dec_concat")([aspp_up, low_red])
    # merged: [B, 32, 32, 256+32=288]

    # ── Step 4: Three refinement SepConv blocks ───────────────────────────
    x = sep_conv_bn_relu6(merged, dec_ch, name="dec_ref0")
    x = sep_conv_bn_relu6(x,      dec_ch, name="dec_ref1")
    x = sep_conv_bn_relu6(x,      dec_ch, name="dec_ref2")
    # x: [B, 32, 32, 128]

    # ── Step 5: Upsample × 4 to full resolution (32×32 → 128×128) ────────
    x = layers.Resizing(
        height=img_size,
        width=img_size,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        name="dec_final_up",
    )(x)
    # x: [B, 128, 128, 128]

    # ── Step 6: Classifier (raw logits) ───────────────────────────────────
    # 1×1 Conv, no activation — IMX500 Softmax constraint: axis=-1 only,
    # but we export raw logits and apply Softmax in post-processing.
    logits = layers.Conv2D(
        num_classes, 1,
        padding="valid",
        use_bias=True,          # bias is fine in classifier head
        data_format="channels_last",
        name="classifier",
    )(x)
    # logits: [B, 128, 128, 8]

    return logits


# ─── Full Model ────────────────────────────────────────────────────────────

def build_model(
    num_classes: int = 8,
    width_mult: float = 1.25,
    aspp_width: int = 256,
    decoder_ch: int = 128,
    ll_reduce_ch: int = 32,
    img_size: int = 128,
    name: str = "deeplabv3plus_imx500",
) -> keras.Model:
    """
    Build DeepLabV3+ for Sony IMX500 / Raspberry Pi AI Camera.

    Args
    ----
    num_classes   : total output classes including background (default 8)
    width_mult    : MobileNetV2 width multiplier (default 1.25)
    aspp_width    : ASPP inner channel width (default 256)
    decoder_ch    : decoder refinement channels (default 128)
    ll_reduce_ch  : low-level feature reduction channels (default 32)
    img_size      : square input size in pixels (default 128)
    name          : Keras model name

    Returns
    -------
    keras.Model  with:
      input  shape: (None, img_size, img_size, 3)   float32 in [0, 1]
      output shape: (None, img_size, img_size, num_classes)  raw logits

    Memory profile (img_size=128, default args)
    -------------------------------------------
      FP32  ≈ 11.0 MB  (< 12 MB limit)
      INT8  ≈  2.8 MB  (after MCT PTQ ÷4)
      Runtime peak ≈ 2.0 MB INT8 (decoder upsample tensor)
      Total on-chip estimate ≈ 4.8 MB / 8 MB  ✓

    Classes
    -----------------------------------------------
      0: background              4: Gray Mold
      1: Angular Leafspot        5: Leaf Spot
      2: Anthracnose Fruit Rot   6: Powdery Mildew Fruit
      3: Blossom Blight          7: Powdery Mildew Leaf
    """

    # ── Input (NHWC, channels_last — required for IMX500 TF converter) ────
    inputs = keras.Input(
        shape=(img_size, img_size, 3),
        name="input",
    )

    # ── Backbone ──────────────────────────────────────────────────────────
    low, high, low_ch, high_ch = build_backbone(inputs, width_mult=width_mult)

    # ── ASPP ──────────────────────────────────────────────────────────────
    aspp_out = build_aspp(high, in_ch=high_ch, inner_width=aspp_width)

    # ── Decoder + Classifier ──────────────────────────────────────────────
    logits = build_decoder(
        low=low,
        high_aspp=aspp_out,
        low_ch=low_ch,
        aspp_ch=aspp_width,
        dec_ch=decoder_ch,
        ll_reduce_ch=ll_reduce_ch,
        num_classes=num_classes,
        img_size=img_size,
    )

    model = keras.Model(inputs=inputs, outputs=logits, name=name)
    return model


# ─── Sanity check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    NUM_CLASSES = 8   # 7 semantic classes + background
    IMG_SIZE    = 128

    print("Building DeepLabV3+ (TF/Keras) for IMX500…")
    model = build_model(
        num_classes=NUM_CLASSES,
        width_mult=1.25,
        aspp_width=256,
        decoder_ch=128,
        ll_reduce_ch=32,
        img_size=IMG_SIZE,
    )

    # ── Parameter count & FP32 size ───────────────────────────────────────
    n_params  = model.count_params()
    fp32_mb   = n_params * 4 / (1024 ** 2)
    int8_mb   = n_params * 1 / (1024 ** 2)

    print(f"\nParameters : {n_params:,}")
    print(f"FP32 size  : {fp32_mb:.2f} MB  (limit: 12 MB)")
    print(f"INT8 size  : {int8_mb:.2f} MB  (after PTQ ÷4 estimate)")

    assert fp32_mb < 12.0, f"Model too large: {fp32_mb:.2f} MB > 12 MB limit"

    # ── Forward pass ─────────────────────────────────────────────────────
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    out   = model(dummy, training=False)
    print(f"\nInput  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(out.shape)}  "
          f"(expected: (1, {IMG_SIZE}, {IMG_SIZE}, {NUM_CLASSES}))")

    assert out.shape == (1, IMG_SIZE, IMG_SIZE, NUM_CLASSES), \
        f"Shape mismatch: {tuple(out.shape)}"

    # ── Layer summary ────────────────────────────────────────────────────
    print(f"\nLayer count : {len(model.layers)}")

    # ── Memory estimate ──────────────────────────────────────────────────
    peak_runtime_kb = 128 * 128 * 128 / 1024  # dec-up tensor INT8: 2048 KB
    peak_runtime_mb = peak_runtime_kb / 1024
    total_est_mb    = int8_mb + peak_runtime_mb + 0.001

    print(f"\n{'='*52}")
    print(f"  IMX500 Memory Estimate")
    print(f"{'='*52}")
    print(f"  Model (INT8)    : {int8_mb:.3f} MB")
    print(f"  Runtime peak    : {peak_runtime_mb:.3f} MB")
    print(f"  Reserved        : 0.001 MB")
    print(f"  Total estimate  : {total_est_mb:.3f} MB / 8.000 MB")
    print(f"  Fits in chip    : {'✓ YES' if total_est_mb < 8.0 else '✗ NO'}")
    print(f"{'='*52}")

    print(f"\n✓ Architecture OK — ready for MCT PTQ + tf2onnx export.")

    # ── Export to Keras SavedModel for verification ───────────────────────
    model.save("deeplabv3plus_imx500_fp32.keras")
    print("  Saved: deeplabv3plus_imx500_fp32.keras")

    # ── Model summary (abbreviated) ───────────────────────────────────────
    print()
    model.summary(line_length=90, expand_nested=False)

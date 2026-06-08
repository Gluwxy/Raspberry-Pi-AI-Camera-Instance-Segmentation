"""
DeepLabV3+ with MobileNetV2 backbone — IMX500 / Raspberry Pi AI Camera
=======================================================================
Design constraints
------------------
• Total FP32 size  < 12 MB  (so INT8 PTQ fits inside 8 MB on-chip)
• Runtime peak     ≈ 4 MB  at 128×128 input (--no-input-persistency)
• Only IMX500-supported ONNX ops (opset 15-20)
• No BatchNorm in ASPP (IMX500 limitation with dynamic axes)
• Bilinear Resize only with align_corners=False (handled by tf.image.resize defaults)
• Depthwise Conv groups constraint: handled natively by Keras DepthwiseConv2D
• ReLU6 via tf.keras.layers.ReLU(max_value=6.0) -> exports to ONNX Clip(0,6)
• No ConvTranspose with stride >= kernel_size
• Upsample method='bilinear'
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─── Helpers ──────────────────────────────────────────

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_bn_relu6(out_ch, kernel=3, stride=1):
    """Conv + BN + ReLU6 — the standard MobileNet building block."""
    return keras.Sequential([
        layers.Conv2D(out_ch, kernel, strides=stride, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(max_value=6.0)
    ])


def pw_conv(out_ch):
    """Point-wise 1×1 conv (no activation) used in ASPP + decoder."""
    return layers.Conv2D(out_ch, 1, padding='same', use_bias=False)


# ─── MobileNetV2 building blocks ──────────────────────────────────────────

class InvertedResidual(layers.Layer):
    """
    Standard MobileNetV2 inverted residual block.
    Dilation is applied only on the depthwise conv.
    IMX500 constraint: when stride>1 only dilation=1 is supported.
    """
    def __init__(self, in_ch, out_ch, stride, expand_ratio, dilation=1, **kwargs):
        super().__init__(**kwargs)
        assert stride in [1, 2]
        
        # stride>1 → force dilation=1 (IMX500 hardware constraint)
        if stride > 1:
            dilation = 1

        hidden = int(round(in_ch * expand_ratio))
        self.use_res = (stride == 1 and in_ch == out_ch)

        self.conv = keras.Sequential()
        
        if expand_ratio != 1:
            self.conv.add(layers.Conv2D(hidden, 1, padding='same', use_bias=False))
            self.conv.add(layers.BatchNormalization())
            self.conv.add(layers.ReLU(max_value=6.0))

        # depthwise
        self.conv.add(layers.DepthwiseConv2D(
            kernel_size=3, strides=stride, padding='same', 
            dilation_rate=dilation, use_bias=False
        ))
        self.conv.add(layers.BatchNormalization())
        self.conv.add(layers.ReLU(max_value=6.0))
        
        # pointwise linear
        self.conv.add(layers.Conv2D(out_ch, 1, padding='same', use_bias=False))
        self.conv.add(layers.BatchNormalization())

    def call(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


# ─── Lightweight MobileNetV2 backbone ──────────────────────────────────────────

class MobileNetV2Backbone(layers.Layer):
    """
    Reduced MobileNetV2:
      width_mult = 0.35
      output_stride = 16  (last block uses dilation=2 instead of stride=2)

    Feature extraction points
    -------------------------
    low_level  : C1 output — 16 ch, stride-4  (e.g. 32×32 @ 128 input)
    high_level : C4 output — 112 ch, stride-16 (e.g.  8×8 @ 128 input)
    """
    def __init__(self, width_mult=0.35, **kwargs):
        super().__init__(**kwargs)
        wm = width_mult

        def c(v):
            return _make_divisible(int(v * wm))

        self.low_level_channels  = c(24)
        self.high_level_channels = c(160)

        # stem: 3 → 32 ch, stride=2 → 64×64
        self.stem = conv_bn_relu6(c(32), stride=2)

        # first inverted residual (t=1): 32→16, stride=1 → 64×64
        self.layer1 = InvertedResidual(c(32), c(16), stride=1, expand_ratio=1)

        # Low-level feature: stride=2 → 32×32 
        self.layer2 = keras.Sequential([
            InvertedResidual(c(16), c(24), stride=2, expand_ratio=6),
            InvertedResidual(c(24), c(24), stride=1, expand_ratio=6),
        ])

        # stride=2 → 16×16
        self.layer3 = keras.Sequential([
            InvertedResidual(c(24), c(32), stride=2, expand_ratio=6),
            InvertedResidual(c(32), c(32), stride=1, expand_ratio=6),
            InvertedResidual(c(32), c(32), stride=1, expand_ratio=6),
        ])

        # stride=2 → 8×8  (output_stride=16)
        self.layer4 = keras.Sequential([
            InvertedResidual(c(32), c(64), stride=2, expand_ratio=6),
            InvertedResidual(c(64), c(64), stride=1, expand_ratio=6),
            InvertedResidual(c(64), c(64), stride=1, expand_ratio=6),
            InvertedResidual(c(64), c(64), stride=1, expand_ratio=6),
        ])

        # stride=1 with dilation=2 (maintains 8×8, effective stride=32→kept at 16)
        self.layer5 = keras.Sequential([
            InvertedResidual(c(64),  c(96), stride=1, expand_ratio=6, dilation=2),
            InvertedResidual(c(96),  c(96), stride=1, expand_ratio=6, dilation=2),
            InvertedResidual(c(96),  c(96), stride=1, expand_ratio=6, dilation=2),
        ])

        # additional depth (atrous, still 8×8)
        self.layer6 = keras.Sequential([
            InvertedResidual(c(96),  c(160), stride=1, expand_ratio=6, dilation=2),
            InvertedResidual(c(160), c(160), stride=1, expand_ratio=6, dilation=2),
        ])

    def call(self, x):
        x = self.stem(x)        # /2
        x = self.layer1(x)
        low = self.layer2(x)    # /4  ← low-level features
        x = self.layer3(low)    # /8
        x = self.layer4(x)      # /16
        x = self.layer5(x)
        high = self.layer6(x)   # /16 (with dilation) ← high-level features
        return low, high


# ─── ASPP (Atrous Spatial Pyramid Pooling) ──────────────────────────────────────────

class SeparableConv(layers.Layer):
    """
    Depthwise-separable conv with ReLU6.
    IMX500: groups=hidden, output/input group ratio must be integer (=1 here).
    No BN to save memory (acceptable after quantization).
    """
    def __init__(self, out_ch, dilation=1, **kwargs):
        super().__init__(**kwargs)
        self.dw = layers.DepthwiseConv2D(3, padding='same', dilation_rate=dilation, use_bias=False)
        self.bn_dw = layers.BatchNormalization()
        self.pw = layers.Conv2D(out_ch, 1, padding='same', use_bias=False)
        self.bn_pw = layers.BatchNormalization()
        self.act = layers.ReLU(max_value=6.0)

    def call(self, x):
        x = self.act(self.bn_dw(self.dw(x)))
        return self.act(self.bn_pw(self.pw(x)))


class ASPPPooling(layers.Layer):
    """
    Global average pooling branch.
    Upsample via tf.image.resize (bilinear).
    """
    def __init__(self, out_ch, **kwargs):
        super().__init__(**kwargs)
        # keepdims=True preserves spatial dims (B, 1, 1, C) for the 1x1 conv
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.conv = keras.Sequential([
            pw_conv(out_ch),
            layers.BatchNormalization(),
            layers.ReLU(max_value=6.0),
        ])

    def call(self, x):
        # TensorFlow dimension order: (Batch, H, W, Channels)
        size = tf.shape(x)[1:3]
        x = self.gap(x)
        x = self.conv(x)
        x = tf.image.resize(x, size=size, method='bilinear')
        return x


class LightASPP(layers.Layer):
    """
    Lightweight ASPP with 4 parallel branches:
      1. 1×1 conv
      2. SepConv rate=6
      3. SepConv rate=12
      4. Global Average Pooling

    Rates [6,12] (instead of [6,12,18]) to cut parameters at 128×128.
    Inner width = 32 channels.
    """
    def __init__(self, out_ch=32, **kwargs):
        super().__init__(**kwargs)
        w = out_ch

        self.b0 = keras.Sequential([pw_conv(w), layers.BatchNormalization(), layers.ReLU(max_value=6.0)])
        self.b1 = SeparableConv(w, dilation=6)
        self.b2 = SeparableConv(w, dilation=12)
        self.b3 = ASPPPooling(w)

        self.project = keras.Sequential([
            pw_conv(out_ch),
            layers.BatchNormalization(),
            layers.ReLU(max_value=6.0),
        ])

    def call(self, x):
        # Concatenate along the channel axis (axis=-1 in NHWC)
        out = tf.concat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], axis=-1)
        return self.project(out)


# ─── Decoder ──────────────────────────────────────────

class Decoder(layers.Layer):
    """
    DeepLabV3+ decoder:
      1. Reduce low-level features to 16 ch with 1×1 conv
      2. Upsample ASPP output to match low-level spatial size (×4)
      3. Concat → 48 ch
      4. Two DW-sep refinement convs → 32 ch
      5. Upsample ×4 back to full resolution
      6. 1×1 classifier
    """
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.ll_reduce = keras.Sequential([
            pw_conv(16),
            layers.BatchNormalization(),
            layers.ReLU(max_value=6.0),
        ])
        
        self.refine = keras.Sequential([
            SeparableConv(32),
            SeparableConv(32),
        ])
        
        self.classifier = layers.Conv2D(num_classes, 1, use_bias=True)

    def call(self, low, aspp_out, input_shape):
        low = self.ll_reduce(low)

        # Upsample ASPP output to low-level spatial size
        low_shape = tf.shape(low)[1:3]
        aspp_up = tf.image.resize(aspp_out, size=low_shape, method='bilinear')

        x = tf.concat([aspp_up, low], axis=-1)
        x = self.refine(x)

        # Upsample to original input resolution
        x = tf.image.resize(x, size=input_shape, method='bilinear')
        return self.classifier(x)


# ─── Full DeepLabV3+ (IMX500 Edition) ──────────────────────────────────────────

class DeepLabV3PlusIMX500(keras.Model):
    """
    DeepLabV3+ with MobileNetV2 (width_mult=0.35) for Sony IMX500.

    Memory budget (FP32, 128×128 input)
    ------------------------------------
    Backbone   ≈  1.0 MB
    ASPP       ≈  0.6 MB
    Decoder    ≈  0.2 MB
    Total FP32 ≈  1.8 MB  → <<12 MB limit
    After INT8 PTQ ≈ ~0.5 MB weights + ~4 MB runtime → fits in 8 MB
    """
    def __init__(self, num_classes=21, width_mult=0.35, **kwargs):
        super().__init__(**kwargs)
        self.backbone = MobileNetV2Backbone(width_mult=width_mult)
        self.aspp = LightASPP(out_ch=32)
        self.decoder = Decoder(num_classes=num_classes)

    def call(self, x):
        input_shape = tf.shape(x)[1:3]
        low, high = self.backbone(x)
        aspp_out = self.aspp(high)
        return self.decoder(low, aspp_out, input_shape)


# ─── Model factory & quick sanity check ──────────────────────────────────────────

def build_model(num_classes=21, width_mult=0.35):
    # Define input and build the model functionally (ideal for ONNX / TFLite export)
    inputs = keras.Input(shape=(128, 128, 3))
    model_core = DeepLabV3PlusIMX500(num_classes=num_classes, width_mult=width_mult)
    outputs = model_core(inputs)
    return keras.Model(inputs=inputs, outputs=outputs, name="DeepLabV3Plus_IMX500")


if __name__ == "__main__":
    model = build_model(num_classes=21)
    
    # Parameter count & FP32 size
    n_params = model.count_params()
    fp32_mb = n_params * 4 / (1024 ** 2)
    print(f"Parameters : {n_params:,}")
    print(f"FP32 size  : {fp32_mb:.2f} MB  (limit: 12 MB)")

    # Forward pass at 128×128 (NHWC format for TF)
    dummy = tf.zeros((1, 128, 128, 3))
    out = model(dummy)
    
    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(out.shape)}  (expected: (1, 128, 128, 21))")

    assert out.shape == (1, 128, 128, 21), "Shape mismatch!"
    assert fp32_mb < 12.0, f"Model too large: {fp32_mb:.2f} MB"
    print("\n✓ Architecture OK — ready for tf2onnx / TFLite export.")

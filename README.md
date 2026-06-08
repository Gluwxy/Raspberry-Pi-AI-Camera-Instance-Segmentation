# Raspberry-Pi-AI-Camera-Instance-Segmentation
Fine-tuned DeepLabV3+ Raspberry Pi AI Camera Instance Segmentation

### Model Architecture

```python
model.summary()
```

```text
Building DeepLabV3+ (TF/Keras) for IMX500…

Parameters : 2,781,872
FP32 size  : 10.61 MB  (limit: 12 MB)
INT8 size  : 2.65 MB  (after PTQ ÷4 estimate)

Input  : (1, 128, 128, 3)
Output : (1, 128, 128, 8)  (expected: (1, 128, 128, 8))

Layer count : 194

====================================================
  IMX500 Memory Estimate
====================================================
  Model (INT8)    : 2.653 MB
  Runtime peak    : 2.000 MB
  Reserved        : 0.001 MB
  Total estimate  : 4.654 MB / 8.000 MB
  Fits in chip    : ✓ YES
====================================================
```

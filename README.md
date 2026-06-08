# Raspberry-Pi-AI-Camera-Instance-Segmentation
Fine-tuned DeepLabV3+ Raspberry Pi AI Camera Instance Segmentation

### Model Architecture
´´´
model.summary()
´´´


```text
Model: "DeepLabV3Plus_IMX500"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_27 (InputLayer)     │ (None, 128, 128, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ deep_lab_v3_plus_imx500_1       │ (None, 128, 128, 21)   │       173,653 │
│ (DeepLabV3PlusIMX500)           │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 173,653 (678.33 KB)
 Trainable params: 164,293 (641.77 KB)
 Non-trainable params: 9,360 (36.56 KB)

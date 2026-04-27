# Korea Mango: Hybrid VLA Model Architecture

This repository documents the architecture of **Korea Mango**, a small Vision-Language-Action (VLA) model with a BiMamba2-Transformer hybrid backbone.

> **Note**: This repository is shared for double-blind review purposes. It contains architectural documentation only. Full source code and pretrained checkpoints will be released after acceptance.

## Overview

Korea Mango is a compact VLA model designed for efficient robot action prediction. It combines:

- **BiMamba2-Transformer hybrid backbone** with learnable per-layer gates
- **256-bin discrete denoising action head** (8-step absorbing-mask)
- **Optional frozen DINOv2 vision encoder**

## Model Sizes

| Variant | Parameters | Backbone Layers | Embed Dim |
|---|---:|---:|---:|
| Small | 29.5M | 4 | 384 |
| Base | 93.6M | 8 | 512 |
| Large | 304.1M | 12 | 768 |
| Small-D (DINOv2) | 44.1M | 4 | 384 |

## Architecture

Pipeline: `Vision Encoder` → `Language Encoder` → `Fusion` → `Hybrid Backbone` → `Action Head`

See [docs/architecture.md](docs/architecture.md) for module-level details.

## Training Data

- Dataset: BridgeData V2 (WidowX 250)
- Action: 7-DoF delta action (xyz + rotation + gripper)
- Sampling: 5 Hz
- Supports full rotation range (±π rad)

## License

MIT License (see [LICENSE](LICENSE)).

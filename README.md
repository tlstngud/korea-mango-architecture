# Korea Mango: Hybrid VLA Model Architecture

Korea Mango는 BiMamba2-Transformer 하이브리드 백본을 가진 소형 Vision-Language-Action(VLA) 모델입니다.
This repository documents the architecture of **Korea Mango**, a small Vision-Language-Action (VLA) model with a BiMamba2-Transformer hybrid backbone.

> **알림 / Note**: 본 저장소는 이중 블라인드 심사를 위해 공개되었습니다. 구조 설명만 포함되어 있으며, 전체 코드와 학습된 가중치는 심사 통과 후 공개될 예정입니다.
> This repository is shared for double-blind review purposes. It contains architectural documentation only. Full source code and pretrained checkpoints will be released after acceptance.

---

## 개요 / Overview

Korea Mango는 효율적인 로봇 행동 예측을 위해 설계된 경량 VLA 모델로, 다음 세 가지 핵심 설계를 결합합니다.
Korea Mango is a compact VLA model designed for efficient robot action prediction. It combines:

- **레이어별 학습 게이트가 적용된 BiMamba2-Transformer 하이브리드 백본**
  BiMamba2-Transformer hybrid backbone with learnable per-layer gates
- **256-bin 이산 디노이징 액션 헤드 (8-step absorbing-mask)**
  256-bin discrete denoising action head (8-step absorbing-mask)
- **선택적 frozen DINOv2 비전 인코더**
  Optional frozen DINOv2 vision encoder

---

## 모델 크기 / Model Sizes

| Variant | Parameters | Backbone Layers | Embed Dim |
|---|---:|---:|---:|
| Small | 29.5M | 4 | 384 |
| Base | 93.6M | 8 | 512 |
| Large | 304.1M | 12 | 768 |
| Small-D (DINOv2) | 44.1M | 4 | 384 |

---

## 구조 / Architecture

전체 파이프라인 / Full pipeline:
`Vision Encoder` → `Language Encoder` → `Fusion` → `Hybrid Backbone` → `Action Head`

자세한 모듈 설명은 [docs/architecture.md](docs/architecture.md)를 참조하세요.
See [docs/architecture.md](docs/architecture.md) for module-level details.

---

## 학습 데이터 / Training Data

- **데이터셋 / Dataset**: BridgeData V2 (WidowX 250 로봇)
- **행동 차원 / Action**: 7-DoF delta action (xyz + rotation + gripper)
- **샘플링 / Sampling**: 5 Hz
- **회전 범위 / Rotation range**: 전체 회전 ±π rad 지원 (다른 로봇/데이터셋 호환을 위함)
  Supports full rotation range (±π rad) for cross-robot compatibility

---

## 라이선스 / License

MIT License (see [LICENSE](LICENSE)).

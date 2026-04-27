# Korea Mango: Hybrid VLA Model Architecture

Korea Mango는 BiMamba2-Transformer 하이브리드 백본을 가진 소형 Vision-Language-Action(VLA) 모델입니다.
This repository documents the architecture of **Korea Mango**, a small Vision-Language-Action (VLA) model with a BiMamba2-Transformer hybrid backbone.

> **알림 / Note**: 본 저장소는 이중 블라인드 심사를 위해 공개되었습니다. 구조 설명만 포함되어 있으며, 전체 코드와 학습된 가중치는 심사 통과 후 공개될 예정입니다.
> This repository is shared for double-blind review purposes. It contains architectural documentation only. Full source code and pretrained checkpoints will be released after acceptance.

---

## 1. 개요 / Overview

Korea Mango는 효율적인 로봇 행동 예측을 위해 설계된 경량 VLA 모델로, 다음 세 가지 핵심 설계를 결합합니다.
Korea Mango is a compact VLA model designed for efficient robot action prediction. It combines:

- **레이어별 학습 게이트가 적용된 BiMamba2-Transformer 하이브리드 백본**
  BiMamba2-Transformer hybrid backbone with learnable per-layer gates
- **256-bin 이산 디노이징 액션 헤드 (8-step absorbing-mask)**
  256-bin discrete denoising action head (8-step absorbing-mask)
- **선택적 frozen DINOv2 비전 인코더**
  Optional frozen DINOv2 vision encoder

---

## 2. 모델 크기 / Model Sizes

| Variant | Parameters | Backbone Layers | Embed Dim |
|---|---:|---:|---:|
| Small | 29.5M | 4 | 384 |
| Base | 93.6M | 8 | 512 |
| Large | 304.1M | 12 | 768 |
| Small-D (DINOv2) | 44.1M | 4 | 384 |

---

## 3. 전체 파이프라인 / Pipeline Overview

```
[RGB Image (256×256)]              [Language Instruction]
         ↓                                  ↓
  Vision Encoder                    Language Encoder
  (Scratch ViT or                   (Transformer 2/4/6 layers)
   frozen DINOv2-S)                          ↓
         ↓                            goal tokens
  visual tokens                             ↓
         └─────────→ Fusion (concat) ←──────┘
                          ↓
                  context tokens
                          ↓
                 Hybrid Backbone
              (BiMamba2 ⊕ Attention) × N
                          ↓
                    Action Head
              (Discrete Denoising 8 → 1 step)
                          ↓
              16-step × 7-DoF action chunk
```

---

## 4. 모듈 상세 / Module Details

### 4.1 Vision Encoder / 비전 인코더

두 가지 옵션 중 선택할 수 있습니다.
Two interchangeable options:

- **Scratch ViT** — 패치 크기 16×16, 4/6/8 레이어 (모델 크기에 따라)
  patch size 16×16, 4/6/8 layers (depending on model size)
- **DINOv2-ViT-S/14 (frozen)** — 사전학습된 표현, 그래디언트 업데이트 없음
  pretrained representation, no gradient updates

**출력 / Output**: 시각 토큰 시퀀스 / visual token sequence shape `[B, N_v, D]`

### 4.2 Language Encoder / 언어 인코더

표준 Transformer (2/4/6 레이어). 자연어 명령(예: "put the spoon on the towel")을 토큰화한 뒤 goal token으로 변환합니다.
Standard Transformer with 2/4/6 layers. Processes tokenized natural language instruction (e.g., "put the spoon on the towel") into goal tokens.

**출력 / Output**: `[B, N_l, D]`

### 4.3 Fusion / 퓨전

시각 토큰과 goal 토큰을 시퀀스 차원에서 concat한 뒤 선형 사상합니다. Cross-attention은 사용하지 않습니다.
Visual tokens and goal tokens are concatenated along the sequence dimension and projected linearly. No cross-attention.

**출력 / Output**: context tokens `[B, N_v + N_l, D]`

### 4.4 Hybrid Backbone / 하이브리드 백본

Korea Mango의 핵심 모듈입니다. 각 레이어는 BiMamba2 분기와 Multi-Head Attention 분기를 **레이어별 학습 가능한 스칼라 게이트** g_l로 결합합니다.
The core of Korea Mango. Each of N layers combines a BiMamba2 branch and a Multi-Head Attention branch via a **learnable scalar gate** g_l per layer:

```
h_m = BiMamba2(RMSNorm(x_l))             # O(n)
h_a = MultiHeadAttention(RMSNorm(x_l))   # O(n²)
x_{l+1} = x_l + σ(g_l) · h_m + (1 - σ(g_l)) · h_a + FFN
```

- **BiMamba2**: 양방향 Mamba2 (forward + reverse selective scans)
- **Attention**: 표준 MHA, pre-norm
- **σ**: sigmoid; g_l은 레이어별 학습 가능한 스칼라
  per-layer trainable scalar

**학습된 게이트의 행동 / Observed gate behavior**: 게이트 값 σ(g_l)은 레이어 깊이가 깊어질수록 증가하는 경향이 있으며, 이는 깊은 레이어에서 Mamba2의 비중이 커진다는 것을 의미합니다. 사전학습된 DINOv2 변형에서는 이 증가 폭이 더 큰 경향을 보여, 사전학습 시각 표현이 시퀀스 모델링 쪽으로 작업을 이동시킬 수 있음을 시사합니다.
Gate values σ(g_l) tend to **increase with layer depth**, indicating Mamba2 dominance in deeper layers. The slope is sharper for DINOv2-based variants, suggesting that pretrained vision shifts workload toward sequence modeling.

### 4.5 Action Head / 액션 헤드

8-step absorbing-mask 이산 디노이저 / 8-step absorbing-mask discrete denoiser:

- 7-DoF action을 차원당 **256개 bin으로 이산화**
  7-DoF action is discretized into **256 bins per dimension**
- 매 스텝마다 일부 bin을 fully masked 초기 상태로부터 예측
  Each step predicts a subset of bins from a fully masked initial state
- 16-step × 7-DoF 행동 청크 출력
  16-step action chunk × 7-DoF output
- 예측 후 bin 인덱스를 물리 단위(미터, 라디안, 그리퍼 open/close)로 역정규화
  After prediction, bin indices are denormalized to physical units (meters, radians, gripper open/close)

**행동 이산화 / Action discretization**: 연속적인 action 값을 차원별 전체 범위를 256개 균등 구간으로 분할하여 매핑합니다. 회전(rotation)에 대해서는 다른 로봇/데이터셋 호환성을 위해 **±π rad 전체 범위**를 지원하며, 이때 bin당 해상도는 약 0.025 rad (1.4°)입니다.
Continuous action values are mapped to 256 uniform bins covering each dimension's full range. For rotation, the model supports the **full ±π rad range** for cross-dataset compatibility, giving a per-bin resolution of ≈ 0.025 rad (1.4°).

---

## 5. 주요 하이퍼파라미터 / Key Hyperparameters

| Variant | Layers (V/L/Bb) | Embed Dim | Heads | Mamba2 d_state |
|---|---|---:|---:|---:|
| Small | 4/2/4 | 384 | 6 | 48 |
| Base | 6/4/8 | 512 | 8 | 64 |
| Large | 8/6/12 | 768 | 12 | 96 |

V/L/Bb = Vision / Language / Backbone 레이어 수 / number of layers.

---

## 6. 학습 데이터 / Training Data

- **데이터셋 / Dataset**: BridgeData V2 (WidowX 250 로봇)
- **행동 차원 / Action**: 7-DoF delta action (xyz + rotation + gripper)
- **샘플링 / Sampling**: 5 Hz
- **회전 범위 / Rotation range**: 전체 회전 ±π rad 지원 (다른 로봇/데이터셋 호환을 위함)
  Supports full rotation range (±π rad) for cross-robot compatibility

---

## 7. 라이선스 / License

MIT License (see [LICENSE](LICENSE)).

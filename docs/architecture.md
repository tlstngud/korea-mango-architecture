# Korea Mango: Architecture Details

## 1. Pipeline Overview

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

## 2. Vision Encoder

Two interchangeable options:

- **Scratch ViT** — patch size 16×16, 4/6/8 layers (depending on model size)
- **DINOv2-ViT-S/14 (frozen)** — pretrained representation, no gradient updates

Output: visual token sequence shape `[B, N_v, D]`.

## 3. Language Encoder

Standard Transformer with 2/4/6 layers. Processes tokenized natural language instruction (e.g., "put the spoon on the towel") into goal tokens.

Output: `[B, N_l, D]`.

## 4. Fusion

Visual tokens and goal tokens are concatenated along the sequence dimension and projected linearly. No cross-attention.

Output: context tokens `[B, N_v + N_l, D]`.

## 5. Hybrid Backbone

The core of Korea Mango. Each of N layers combines a BiMamba2 branch and a Multi-Head Attention branch via a **learnable scalar gate** g_l per layer:

```
h_m = BiMamba2(RMSNorm(x_l))             # O(n)
h_a = MultiHeadAttention(RMSNorm(x_l))   # O(n²)
x_{l+1} = x_l + σ(g_l) · h_m + (1 - σ(g_l)) · h_a + FFN
```

- BiMamba2: bidirectional Mamba2 (forward + reverse selective scans)
- Attention: standard MHA, pre-norm
- σ: sigmoid; g_l is a per-layer trainable scalar

### Observed gate behavior (post-training)

Gate values σ(g_l) tend to **increase with layer depth**, indicating Mamba2 dominance in deeper layers. The slope of this increase is sharper for DINOv2-based variants, suggesting that pretrained vision shifts workload toward sequence modeling.

## 6. Action Head

8-step absorbing-mask discrete denoiser:

- 7-DoF action is discretized into **256 bins per dimension**
- Each step predicts a subset of bins from a fully masked initial state
- 16-step action chunk × 7-DoF output
- After prediction, bin indices are denormalized to physical units (meters, radians, gripper open/close)

### Action discretization

Continuous action values are mapped to 256 uniform bins covering each dimension's full range. For rotation, the model supports the **full ±π rad range** for cross-dataset compatibility, giving a per-bin resolution of ≈ 0.025 rad (1.4°).

## 7. Key Hyperparameters (per variant)

| Variant | Layers (V/L/Bb) | Embed Dim | Heads | Mamba2 d_state |
|---|---|---:|---:|---:|
| Small | 4/2/4 | 384 | 6 | 48 |
| Base | 6/4/8 | 512 | 8 | 64 |
| Large | 8/6/12 | 768 | 12 | 96 |

V/L/Bb = Vision / Language / Backbone layers.

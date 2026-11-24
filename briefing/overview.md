# Overview

- **Paper ID:** 2505.23158
- **Title:** LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering
- **Domain:** Computer Vision / Neural Rendering
- **TL;DR:** LODGE makes large 3D Gaussian Splatting scenes faster and lighter by building smoothed, pruned and fine-tuned LOD levels and precomputed chunk-local active Gaussian sets, rendered with two-chunk opacity blending.

## Short Summary

Distant fine detail in large 3DGS scenes consumes memory and rasterisation time. LODGE builds lower-detail levels with depth-aware smoothing, importance pruning and fine-tuning. It clusters camera space into chunks with fixed active sets and blends the two nearest chunks to avoid popping.

This gym substitutes a synthetic street canyon for the paper's scenes. The trained full model is public; target images are supersampled renders of a hidden, denser true scene. The evaluator renders submissions itself, under a device budget for memory and rasterisation work.

## Key Results

**Paper, Table 3 (SmallCity):**

| Representation | PSNR (dB) | Loaded | Visible |
|---|---:|---:|---:|
| Full representation | 26.62 | 2639K | 925K |
| LODGE | 26.57 | 877K | 268K |

**This gym, hidden test, reference seed 0:**

| Views | Full representation | LODGE | Loaded vs full | Visible vs full |
|---|---:|---:|---:|---:|
| Trajectory | 41.18 dB | 40.83 dB | −44% | −39% |
| Flythrough | 37.50 dB | 36.92 dB | — | — |
| Off-path | 33.57 dB | 31.33 dB | — | — |

- Transition PSNR is 33.23 dB for LODGE and 33.25 dB for the full representation.
- Off-path views are where the reference falls short and where the gym has headroom.

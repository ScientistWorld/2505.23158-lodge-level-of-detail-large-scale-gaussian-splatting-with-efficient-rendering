# Paper's Method

## Key Contribution

LODGE introduces a level-of-detail representation for 3D Gaussian Splatting that reduces rendering work and resident memory for large scenes. It differs from prior LOD splatting methods by precomputing fixed active Gaussian sets for spatial chunks, avoiding per-frame selection overhead and enabling deployment on memory-constrained devices.

## Approach

The method starts from a trained full 3DGS representation. It builds lower-detail Gaussian levels by copying the full representation, applying a depth-aware 3D smoothing filter inspired by Mip-Splatting, pruning low-importance Gaussians using RadSplat-style maximum alpha-compositing contribution, and fine-tuning the surviving Gaussians. Depth thresholds are selected greedily by estimating average Gaussians per 16x16 rasterization tile on training views.

For rendering, active Gaussians are selected by camera distance: nearby regions use finer levels and distant regions use coarser levels. To reduce memory, training camera positions are clustered into chunks. Each chunk stores a precomputed active set with thresholds offset by chunk radius, then visibility filtering removes Gaussians with low per-chunk contribution. At render time, the two nearest chunks are rendered with opacity interpolation to avoid sharp transitions at chunk boundaries.

## Main Claims

The central claim is that a chunked LOD Gaussian representation preserves visual quality while reducing visible Gaussians, loaded Gaussians, and render cost. Secondary claims are that automatic threshold selection improves the quality-speed trade-off over manual thresholds, visibility filtering further reduces memory and cost, and two-chunk opacity blending removes chunk-boundary artifacts.

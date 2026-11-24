# Paper's Method (LODGE) and This Reproduction

## Key Contribution

LODGE is a level-of-detail representation for 3D Gaussian Splatting. It reduces rendering work and resident memory for large scenes by precomputing fixed active Gaussian sets for spatial chunks. This avoids per-frame selection and keeps only two chunks in memory.

## Approach (as implemented in `method/lodge.py`)

1. **Threshold selection.** Depth thresholds d_l are chosen greedily (d_1 first, then d_2 > d_1, …). Each step minimises the average number of Gaussians per 16×16 tile, measured by LOD-rendering a subset of training views (candidates 4–64 m).
2. **LOD levels.** Each level G_l is built from G0:
   - The Mip-Splatting 3D smoothing filter (Eq. 3) is added to every Gaussian. Its variance is s·(d_l/f)², and opacity is scaled by sqrt(det Σ / det(Σ + εI)).
   - Three rounds of RadSplat importance pruning follow, at thresholds 0.2γ, 0.6γ and γ. Importance is the maximum blending weight over training pixels.
   - Each pruning round is followed by 1000 fine-tuning steps:
     - loss: 0.8·L1 + 0.2·D-SSIM;
     - rendering: LOD rendering over levels ≤ l, with d_l drawn from U(0.7 d_l, 1.3 d_l);
     - gradients flow only into G_l.
3. **Chunks.** K-means on training camera positions (best of 10, fixed seed) gives N = 4/d_1 · max‖c − c̄‖ chunks (Eq. 5). Each chunk stores the Eq. (2) active set at its centre, with thresholds increased by the chunk radius (distance to the nearest other centre). Visibility filtering then keeps Gaussians whose importance, rendered from the chunk's training positions with their original and at least 32 random orientations, is at least 0.02.
4. **Rendering.** The two nearest chunks are blended with the opacity interpolation of Eq. (4). Ablation exports from the same build are the nearest chunk only and per-frame dynamic LOD.

The free hyperparameters were chosen on the public validation views (`notes/hardening/tuning.json`): γ = 0.03, s = 0.2, two levels; the search selected thresholds [27, 36] m. The result is 9 chunks with a pool of 277K Gaussians.

## Main Claims

- **Core:** a chunked LOD representation keeps quality near the full model while greatly reducing visible and loaded Gaussians.
- **Secondary:**
  - per-frame dynamic LOD reduces work but keeps all levels resident (memory above the full model);
  - chunking with visibility filtering reduces memory;
  - opacity blending removes chunk-boundary popping at a moderate memory cost;
  - automatic thresholds beat manual ones on the quality/efficiency trade-off;
  - fine-tuning after pruning is required for quality.

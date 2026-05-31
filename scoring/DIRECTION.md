# Research Direction

## Research Problem
This benchmark is about efficient rendering of large Gaussian-splatting scenes. The goal is to preserve novel-view image quality while reducing the amount of Gaussian data processed and loaded for each viewpoint.

## Core Contribution
The paper's contribution is a level-of-detail and spatial caching strategy for 3D Gaussian primitives. Future improvements should work on how Gaussian primitives are smoothed, selected, pruned, grouped, cached, or blended for efficient rendering.

## Approach Scope
Work should remain within Gaussian-splatting representation management. A scientist may change detail-level construction, threshold selection, importance scoring, chunk assignment, visibility filtering, or blending, provided the output remains a rendered view from Gaussian primitives and the method reports the same cost constraints.

## Out of Bounds
- Replacing the task with a neural image synthesizer that directly maps camera poses to pixels.
- Using a mesh, voxel, NeRF, or image-warping renderer instead of Gaussian splatting.
- Improving scores by changing the scene, camera split, image resolution, or metric implementation.
- Optimizing wall-clock runtime without reducing loaded/visible Gaussian counts or tile work.

# Overview

- **Paper ID:** 2505.23158
- **Title:** LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering
- **Domain:** Computer Vision / Neural Rendering
- **TL;DR:** LODGE makes large 3D Gaussian Splatting scenes faster and lighter by building smoothed/pruned LOD levels and precomputed chunk-local active Gaussian sets.

## Short Summary

The paper addresses the cost of rendering large Gaussian-splatting scenes where distant fine detail consumes memory and rasterization time. It builds lower-detail Gaussian levels using depth-aware smoothing, importance pruning, and fine-tuning, then clusters camera space into chunks with fixed active Gaussian sets. During rendering it blends the two nearest chunks to avoid visible chunk-boundary transitions. The reported results show similar or better quality than large-scene baselines with substantially fewer loaded Gaussians and higher rendering speed.

## Key Results

- On SmallCity, the paper reports 26.57 PSNR while reducing loaded Gaussians from 2.639M in the full representation to 877K in the final chunked/blended representation.
- In the SmallCity ablation, final opacity blending uses 268K visible Gaussians versus 925K for the full representation.
- On Hierarchical 3DGS and Zip-NeRF scenes, the method reports real-time FPS while keeping quality competitive with or better than other real-time Gaussian-splatting baselines.

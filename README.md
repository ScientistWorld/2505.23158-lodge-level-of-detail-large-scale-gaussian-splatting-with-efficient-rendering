# LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2505.23158"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2505.23158-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2505.23158](https://arxiv.org/abs/2505.23158) |
|:---|:---|
| **Domain** | Computer Vision / Neural Rendering |
| **TL;DR** | LODGE makes large 3D Gaussian Splatting scenes faster and lighter by building smoothed/pruned LOD levels and precomputed chunk-local active Gaussian sets. |

## Summary

The paper addresses the cost of rendering large Gaussian-splatting scenes where distant fine detail consumes memory and rasterization time. It builds lower-detail Gaussian levels using depth-aware smoothing, importance pruning, and fine-tuning, then clusters camera space into chunks with fixed active Gaussian sets. During rendering it blends the two nearest chunks to avoid visible chunk-boundary transitions. The reported results show similar or better quality than large-scene baselines with substantially fewer loaded Gaussians and higher rendering speed.

## Problem

How can a large 3D Gaussian Splatting scene be rendered from novel viewpoints with high image quality while reducing the number of Gaussian primitives that must be processed and resident in memory?

## The paper's method

LODGE introduces a level-of-detail representation for 3D Gaussian Splatting that reduces rendering work and resident memory for large scenes. It differs from prior LOD splatting methods by precomputing fixed active Gaussian sets for spatial chunks, avoiding per-frame selection overhead and enabling deployment on memory-constrained devices.

## Key results of the paper

- On SmallCity, the paper reports 26.57 PSNR while reducing loaded Gaussians from 2.639M in the full representation to 877K in the final chunked/blended representation.
- In the SmallCity ablation, final opacity blending uses 268K visible Gaussians versus 925K for the full representation.
- On Hierarchical 3DGS and Zip-NeRF scenes, the method reports real-time FPS while keeping quality competitive with or better than other real-time Gaussian-splatting baselines.

## Run it

```bash
pip install sciworld
sw research 2505.23158 --mode innovate --agent claude-code
sw evaluate 2505.23158
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).


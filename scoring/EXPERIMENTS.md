# Experiment Overview

## synthetic_lod_quality_efficiency
This experiment tests the paper's central trade-off on a compact deterministic Gaussian-splatting scene: a method should render held-out views with quality close to a full dense representation while reducing memory and rasterization-work proxies. It is designed to catch methods that improve quality only by processing the entire scene, and methods that reduce cost only by destroying view quality.

## synthetic_chunk_blending_ablation
This experiment tests whether different efficiency-oriented operating points expose a meaningful quality/cost trade-off on the same compact scene. A useful method should reduce loaded and visible Gaussian counts while maintaining held-out image quality, and should avoid localized view-quality failures when the camera moves between regions of the scene.

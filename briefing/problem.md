# Problem Definition

## Research Question

Given a trained, very large 3D Gaussian Splatting reconstruction of a scene, how can it be stored and rendered so that any camera position needs only a small resident subset of Gaussians and a small amount of rasterisation work, while novel views stay as close to the real images as the full model's renders?

## Why It Matters

Large scenes need many fine Gaussians. Rendering every one for every frame wastes work on distant, sub-pixel detail and needs more memory than mobile or low-power devices have. A good solution has to cut both memory (what is resident) and work (what the rasteriser processes per frame), keep quality close to the full model near and far from where the scene was captured, and avoid visible popping when the camera moves through the scene.

## The Benchmark Scene

A long synthetic street canyon (about 130 m of facades, road, sidewalks, cars, posts and trees). The public data in `data/lodge_city/` contains:

- `base_gaussians.npz`: the trained full reconstruction (about 247K anisotropic Gaussians with RGB colour). It was optimised on the training images and plays the role of the paper's full 3DGS model.
- `cameras.json`: intrinsics plus the camera sets `train` (209 capture views), `val_on_path`, `val_flythrough` and `val_off_path`.
- `images.npz`: target images for every public camera set.

The hidden test uses held-out cameras of the same three kinds and never-released target images.

## What a Solution Produces

A file `checkpoints/<method>/representation.npz` in the format documented at the top of `data/gsrender.py`:

- a pool of Gaussians;
- precomputed resident sets ("chunks") anchored at 3D positions;
- an optional per-Gaussian camera-distance band;
- a flag that selects one of the two built-in transition rules between neighbouring chunks.

The evaluator decides which Gaussians are resident and drawn for each camera, renders them with its own renderer, and measures everything. The scored entry is `checkpoints/lodge/`. Other directories are scored too, except names starting with `_`.

## Success Criteria

- Stay within the device budget:
  - peak resident Gaussians at most 0.65× the full model;
  - mean visible Gaussians at most 0.70× the full model's, per experiment.
- Keep PSNR within the stated tolerance of the full model in every experiment.
- Do not introduce popping along a dense camera path.
- Within those constraints, maximise image quality and minimise resident and visible Gaussians.

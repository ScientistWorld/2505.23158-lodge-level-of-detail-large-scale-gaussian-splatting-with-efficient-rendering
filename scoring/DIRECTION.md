# Research Direction

## Research Problem
Efficient rendering of large Gaussian-splatting scenes under a memory and per-frame work budget. A good solution keeps novel-view quality close to the full trained model, including away from the capture trajectory, while only a fraction of the Gaussians is resident and processed for any camera.

## Core Contribution
The paper's contribution is level-of-detail construction plus spatially cached resident sets for 3D Gaussian primitives. Improvements should change:
- how reduced-detail Gaussians are derived, optimised and pruned;
- how resident sets are chosen and assigned to regions of space;
- how transitions between resident sets are handled.

## Approach Scope
- **In scope:** any procedure that produces a Gaussian pool plus resident sets in the representation format. Examples:
  - better smoothing, merging or pruning;
  - better fine-tuning (objectives, viewpoints, distillation from the full model);
  - more or differently placed levels;
  - better chunk placement or per-Gaussian distance bands;
  - smarter visibility filtering.
- **Trade-offs:** methods may trade memory against quality only within the budget. Quality gains bought by exceeding the budget fail.

## Out of Bounds
- Replacing Gaussian splatting with another renderer (neural image synthesis, meshes, image warping).
- Using the validation or hidden cameras to fit the representation, or editing the evaluators, renderer copy, scene, cameras or targets.
- Reporting wall-clock speed instead of reducing loaded/visible Gaussians.

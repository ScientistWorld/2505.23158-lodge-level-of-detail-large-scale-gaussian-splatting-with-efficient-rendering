# Experiment Overview

All three experiments render the submitted representation (`checkpoints/<method>/representation.npz`) with the evaluator's own renderer on a large synthetic street scene. They compare it with target images that are never released. The evaluator also renders the full trained model as a fixed reference point (`full_representation`).

Each experiment has two kinds of check:
- **Gates:** a run fails, and gets the worst value of every metric, if it exceeds the device budget, drops more than a fixed tolerance below the full model's quality, or pops along the path.
- **Scored metrics:** within the gates, quality and cost are scored.

## trajectory_views

Held-out views taken from the same capture trajectory as the training images (every 8th frame). This experiment establishes the central claim: a reduced representation renders the same viewpoints with quality close to the full model, while far fewer Gaussians are resident in memory and processed per frame.
- **Scored:** PSNR, mean loaded Gaussians and mean visible Gaussians.

## flythrough

A dense camera path driven through the street. It establishes that switching between resident sets as the camera moves does not cause popping or flicker that the real scene does not have.
- **Scored:** per-frame PSNR, and the temporal residual of the worst frame-to-frame transitions (`transition_psnr`).
- **Gate:** transitions much worse than the full model's fail the run.

## off_path_views

Views away from the capture trajectory: lateral offsets, raised heights and arbitrary headings. This tests whether quality holds where the training cameras did not go. The reference method loses about 2 dB here relative to the full model, so this is the main room for improvement.
- **Scored:** PSNR, mean loaded Gaussians and mean visible Gaussians.

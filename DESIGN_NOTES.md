# Design Notes (current gym)

- **Submission format.** The submission is a Gaussian representation, not images or numbers. The evaluator chooses the resident set per camera, renders it, and counts the costs. A method can only improve scores by producing a better representation.
- **Hidden truth.** Target images are supersampled renders of a hidden, denser true scene. The public base model is a trained, coarser reconstruction, and its generator is not public. Copying or regenerating targets is therefore not possible.
- **Device budget.** Peak resident ≤ 0.65·N0 and mean visible ≤ 0.70× full. This encodes the paper's memory-constrained setting. Without it, normalising each metric by the spread of attempts let the full model, or a less aggressive LOD, outrank the reference on small PSNR gains.
- **Transition gate.** This encodes the paper's opacity-blending claim: popping is a failure, not a trade-off.
- **Experiments.** `trajectory_views` covers the paper's protocol (every 8th capture frame). `flythrough` covers chunk transitions. `off_path_views` is the headroom experiment.
- **Comparable slices.** The public validation sets mirror the hidden ones: same capture-frame rule, a comparable flythrough stretch, and stratified off-path poses.

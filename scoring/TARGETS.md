# Evaluation Targets

## Primary: synthetic_lod_quality_efficiency
Tests whether held-out rendered views remain close to the reference images while the representation uses less rendering work.
- **Metric**: psnr (higher is better)
- **How to run**: `bash scripts/reproduce.sh`

## Constraint: synthetic_lod_quality_efficiency
Tests whether quality gains are not obtained by loading or processing the full dense representation.
- **Metric**: loaded_gaussians (lower is better)
- **Metric**: visible_gaussians (lower is better)
- **How to run**: `bash scripts/reproduce.sh`

## Secondary: synthetic_chunk_blending_ablation
Tests the paper's design-validation axis: dynamic LOD, nearest cached chunks, and opacity-blended chunks should expose the quality/cost trade-offs of chunk caching and transition blending.
- **Metric**: psnr (higher is better)
- **Metric**: loaded_gaussians (lower is better)
- **Metric**: visible_gaussians (lower is better)
- **How to run**: `bash scripts/reproduce.sh`

# Explicit Relative Depth Modeling Motivation

## Claim

AMS already captures useful depth-layout information, but it remains implicit in feature differences and is not organized as explicit root-relative or pairwise joint-depth relations. Therefore an AMS-after relation module is motivated independently of UDE denoising.

## Evidence

| Dimension | Metric | Value | Interpretation |
|---|---:|---:|---|
| Implicit vs explicit relation | raw F_d distance vs |ΔZ| Spearman | -0.063 | Raw AMS feature geometry does not organize joints by true relative depth. |
| Implicit vs explicit relation | linear probe R2 for ΔZ from F_i-F_j | 0.965 | Relative depth is present but latent; a lightweight relation transform can expose it. |
| Ordinal depth | coarse μ all-pair ordinal acc | 0.623 | The scalar depth head alone does not preserve joint ordering reliably. |
| Ambiguous limb pairs | linear probe ordinal acc on wrists/elbows/legs | 0.884 | Hard self-occlusion-related pairs remain weaker than all-pair average. |
| Global component | F_d body-mean energy fraction | 0.544 | A large shared component suggests relation-specific residuals are not isolated. |
| Feature decoupling | off-diagonal joint-token cosine | 0.611 | High inter-joint similarity means vanilla token fusion may blur depth-layout distinctions. |
| Feature decoupling | top-3 residual PCA variance | 0.645 | Depth residual structure is low-dimensional, suitable for explicit relation/layout modeling. |
| Body-part structure | 3-part within-between cosine gap | 0.065 | AMS F_d already has body-part grouping; a module can exploit skeleton part priors. |
| Body-part structure | 5-part positive gap ratio | 0.863 | Fine limb grouping is present but not perfect, leaving room for structured refinement. |
| Pose awareness | F_d distance vs 3D pose distance Spearman | 0.854 | AMS output is pose-aware, so the missing piece is not generic pose information. |

## Module Direction

- Add the module after AMS and before multimodal token fusion.
- Convert `F_d` into root-relative depth tokens and pairwise/part-wise depth-layout tokens.
- Supervise or regularize with ordinal/pairwise depth relations from GT Z during training.
- This module addresses representation conversion and skeletal depth geometry, not noise suppression.

## Candidate Tests For A New Module

- Pairwise ordinal depth accuracy on all joint pairs, bones, and ambiguous limb pairs.
- Raw relation-token geometry correlation with `|Z_i-Z_j|` after the module.
- MPJPE change under self-occlusion-like pairs or large wrist/elbow depth gaps.
- Whether relation tokens reduce body-mean dominance while preserving pose-awareness.

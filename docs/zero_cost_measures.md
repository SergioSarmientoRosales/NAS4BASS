# Zero-Cost Measures For BASS SRIR

This repository includes TensorFlow/Keras zero-cost measures inspired by pruning-oriented predictors implemented in NASLib. They are adapted here as architecture-level scoring functions for super-resolution image restoration (SRIR) models generated from the BASS search space.

The goal is not to reproduce NASLib pruning behavior line by line. The goal is to obtain efficient, reproducible, and comparable proxy signals for candidate architecture evaluation.

## Reference Relationship

NASLib implements several measures in PyTorch for pruning and zero-cost prediction workflows, including GradNorm, SNIP, GraSP, Fisher, Plain, SynFlow, Zen, L2 norm, NWOT, and Jacobian covariance. NAS4BASS provides TensorFlow/Keras adaptations for the corresponding architecture-level SRIR setting:

| NASLib-style measure | NAS4BASS implementation |
| --- | --- |
| GradNorm | `evaluators/metrics/grad_norm.py` |
| SNIP | `evaluators/metrics/snip.py` |
| GraSP | `evaluators/metrics/grasp.py` |
| Fisher | `evaluators/metrics/fisher.py` |
| Plain | `evaluators/metrics/plain.py` |
| SynFlow | `evaluators/metrics/synflow.py` |
| Zen | `evaluators/metrics/zen.py` |
| L2 norm | `evaluators/metrics/l2_norm.py` |
| NWOT | `evaluators/metrics/nwot.py` |
| Jacobian covariance | `evaluators/metrics/jacob_cov.py` |

NAS4BASS also includes `params_score` and `zico` variants. These are project-level zero-cost baselines rather than direct adaptations from the NASLib measures directory. `params_score` returns the negative parameter count, so smaller models receive larger raw scores.

## Adaptation Philosophy

The TensorFlow implementations intentionally differ from NASLib where the task or framework requires it:

- PyTorch hooks are replaced by `tf.GradientTape` or Keras intermediate-output models.
- Classification logits are replaced by SRIR image-to-image outputs.
- Supervised proxy signals use LR inputs and HR targets with an SR loss, usually mean squared error.
- BASS branch-structured models are scored through the instantiated Keras graph, not through a hand-written branch traversal.
- Per-layer or per-parameter signals are aggregated into one scalar architecture score.

These deviations are acceptable when they preserve a coherent mathematical signal for SRIR architecture ranking.

## BASS Traversal

The BASS model builder instantiates three branches and a reconstruction path. The searchable operations include standard convolution, dilated convolution, depthwise separable convolution, inverted bottleneck convolution, transposed convolution, and identity.

The zero-cost measures traverse the instantiated Keras model:

- `Conv2D`, `Conv2DTranspose`, `DepthwiseConv2D`, and `Dense` kernels are included by the parameter- and gradient-based measures.
- Identity operations are intentionally excluded from direct parameter scoring because they do not own trainable weights.
- Branch fusion through `Add` and upsampling through `PixelShuffle` affect forward activations and gradients but do not contribute direct parameter tensors.
- Bias tensors are generally not included in the NASLib-style parameter arrays. This follows the weight-focused spirit of the original measures, but it should be treated as a documented limitation.
- Reconstruction layers are part of the Keras graph and can contribute to parameter- and gradient-based scores.

## TensorFlow Gradient Behavior

Measures that need gradients use eager-mode `tf.GradientTape`. The implementation handles `None` gradients defensively by skipping them or replacing them with zero tensors, depending on the measure. Fisher uses a persistent tape to query gradients of multiple intermediate activations from a single loss.

The current model space does not use BatchNorm or Dropout in the modular builder. If future BASS variants add stateful layers, each zero-cost measure should be revisited for `training=True` or `training=False` behavior.

## SRIR Inputs And Outputs

Zero-cost evaluation uses image-to-image tensors. The default zero-cost evaluator builds synthetic LR inputs with shape `(batch_size, 64, 64, 3)` and synthetic HR targets with shape `(batch_size, 128, 128, 3)`, corresponding to the default x2 SRIR setting. The modular model builder derives the pre-PixelShuffle channel count from `3 * upscale_factor**2`, so non-default scale factors can be represented consistently by callers that set `upscale_factor`.

This synthetic-batch choice is intentional for fast architecture scoring, but it means the scores estimate architectural signals under random inputs rather than dataset-specific restoration performance. Dataset-based batches can be supplied to the lower-level metric functions when a benchmark script provides explicit tensors.

## Score Aggregation And Direction

Most measures aggregate layer-level, channel-level, or parameter-level values by summing them into one architecture-level scalar. This is simple and traceable, but it can favor larger models because larger networks often own more parameters or activations.

In the main NAS pipeline, the transformed zero-cost score is treated as higher-is-better and converted to a minimization objective as `-score`. The CLI option `--zc-score-transform` makes score direction and size normalization explicit. Available transforms are `raw`, `div_params`, `neg_raw`, and `neg_div_params`.

By default, zero-cost evaluation uses `synflow` with the `raw` transform. `param_score` remains available as a size-only baseline and is best interpreted as a sanity check rather than a performance predictor.

## Determinism

The zero-cost evaluator creates a shared synthetic LR/HR batch from the run seed. It also assigns a stable TensorFlow seed to each decoded architecture before model construction and metric evaluation. This makes initialization- and synthetic-noise-dependent scores less sensitive to the order in which the search algorithm visits candidate architectures.

## Known Limitations

- Raw score aggregation can be size-biased.
- Several measures depend on random model initialization or synthetic random inputs.
- Per-architecture TensorFlow seeding improves order stability but does not remove the need to report run seeds.
- The default zero-cost evaluator is configured for the x2 SRIR setting unless callers pass another scale factor.
- `params_score` should be interpreted with care because it is a size baseline, not a performance proxy.
- The TensorFlow adaptations are not intended as pruning algorithms during training.

## Validation

Lightweight smoke tests live under `tests/`. They check that representative BASS architectures can be instantiated, that core searchable operations appear in the Keras graph, that collectors see trainable convolutional parameters, and that selected low-cost metrics return scalar numeric values in a TensorFlow environment.

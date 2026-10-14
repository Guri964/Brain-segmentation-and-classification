# Algorithm and novelty

This document states the baseline, the modifications, and the experiments that would
have to be run before any of them could be called novel. It is written this way on
purpose: a modification is a claim, and a claim needs an experiment with a control.

**No novelty claim in this repository is currently supported by evidence**, because no
labelled dataset has been supplied to this deployment. What follows is therefore a
design record and a protocol, not a result.

---

## 1. Baseline (reproduced from the original project)

| Component | Baseline |
|---|---|
| Four-channel segmentation | SegNet, 4→1, 256×256, logits, global z-score per modality |
| Single-channel segmentation | DynamicUNet `[16, 32, 64, 128, 256]`, 1→1, 512×512, internal sigmoid, image/255 |
| Training | Per-batch optimisation with a shared "best validation loss" checkpoint rule in the original notebook (see `docs/PROJECT_AUDIT.md`) |
| Ensemble | Equal-weight probability averaging, `P = (P₁ + P₂)/2`, threshold 0.5 |
| Explainability | Grad-CAM on SegNet |
| Features | A radiomics table (11 columns) and an unused 21→128→64→2 classifier |
| Clinical output | None |

The baseline is reproducible in the sense that the first two rows are implemented and
verified in this repository. It is *not* reproducible as a result, because the training
run, the splits and the evaluation that produced the shipped checkpoints were not
recorded — which is precisely the defect this platform was built to eliminate.

## 2. Modifications implemented here

### 2.1 SegNet and U-Net explained through their own paths

Grad-CAM is anchored on the same tensor the model actually consumed, and the target
layer is chosen per architecture (last encoder convolution for SegNet, last decoder
convolution for both U-Nets). The original implementation explained a model that was
not the model producing the mask.

*Test that would support it:* agreement between the heatmap and the predicted mask
(measured as the mean activation inside versus outside the region), plus a
sanity check that the heatmap is invariant to the explained model's parameters being
frozen. `tests/test_explainability_uncertainty.py` implements both as guard-rails; a
paper would need the same measurement across a cohort.

### 2.2 Ensemble with a documented threshold and an explicit refusal rule

The fusion is `P = Σ wᵢ·Pᵢ` with the weights, their source and the threshold recorded in
the analysis. Two additions over the baseline:

* **compatibility gating** — fusion runs only between models with an identical input
  specification, because averaging maps that do not describe the same pixels is
  meaningless;
* **health gating** — a checkpoint that fails the deterministic reference check is
  excluded, with the exclusion written into the analysis.

*Test that would support it:* equal versus validation-calibrated weights on the
validation split, then one evaluation on the test split, reported with confidence
intervals. `scripts/calibrate_ensemble.py` and `scripts/run_ablations.py` produce
exactly those two rows.

### 2.3 Confidence-aware fusion

Instead of a single global weight vector, each model's per-pixel weight is modulated by
its own predictive dispersion measured from the real input tensor:

```
reliabilityᵢ(p) = wᵢ / (1 + spreadᵢ(p) / scale)      P(p) = Σ reliabilityᵢ(p)·Pᵢ(p) / Σ reliabilityᵢ(p)
```

The dispersion comes from test-time augmentation passes on the input, never from the
probability map (which would be circular).

*Test that would support it:* confidence-aware versus equal versus calibrated weights,
identical members, identical threshold, validation split for the three-way comparison
and one test-set run. The `confidence_aware` row is implemented; the numbers are not
available until the dataset is.

### 2.4 Radiomics–deep feature fusion

The classifier can be trained on three representations from the same cases: radiomics
only, pooled encoder activations only, or their concatenation
(`--architecture feature_fusion`, features extracted by
`scripts/extract_deep_features.py`).

*Test that would support it:* the three representations on identical patient-level
splits with identical seeds, reported as a table with confidence intervals rather than
as a single number. The known risk is that the fusion representation needs
substantially more data before it beats radiomics alone, and the honest outcome may
well be "radiomics is competitive at this sample size" — which is itself a useful
result.

### 2.5 Attention U-Net (RAUNet) as a four-channel model

Attention gates on the skip connections concentrate decoder features on the region,
which is the architectural modification the ablation is designed to measure.

*Test that would support it:* RAUNet versus the U-Net without attention gates, same
filters, same data, same schedule, differing only in the gates. That is a
one-component ablation, and `scripts/run_ablations.py` is wired for it.

### 2.6 Measurement of cost, not just quality

Inference time, parameter count and peak memory are recorded per model, and the
evaluation output labels them as inference cost rather than training cost. For a
clinical research deployment, a 2 % Dice gain that triples latency is a different
proposal from the same gain for free.

## 3. What would make a claim publishable

1. A dataset with patient-level identifiers, one scan-selection rule and reference
   masks (or a labelled target with a recorded meaning).
2. Baseline metrics on the reserved test split, with confidence intervals.
3. One-component ablations, run on the validation split, with the seeds and
   configurations recorded.
4. A single test-set evaluation after selection, never before.
5. External validation on a second cohort or scanner, reported even if it is worse.
6. A literature review establishing that the combination has not been published
   already — without it, any of the modifications above may be a rediscovery.

## 4. Claim discipline

| Statement | Status |
|---|---|
| "The platform records everything needed to reproduce a run" | Supported by the implementation and the tests |
| "U-Net is better than SegNet here" | **Unsupported** — the SegNet checkpoint produces an empty mask and no test split exists |
| "The ensemble improves Dice" | **Unsupported** — no compatible pair and no test split |
| "Confidence-aware fusion is better than equal weights" | **Unsupported** — implemented, not measured |
| "Radiomics–deep fusion is more accurate" | **Unsupported** — implemented, not measured |
| "Attention gates improve boundary accuracy" | **Unsupported** — no trained attention model |

Every row marked unsupported becomes supportable by running the scripts in section 3
on real data, and by nothing else.

# Fixed-P0 residual SoH30 outer extension

This post hoc extension evaluates a TCN that predicts the residual over the causal Trend forecast. The final SoH30 prediction is clipped to [0, 1.5]. All 12 sources are held out in turn, with seed bases 2026 to 2030 and the same SoH test support as the original point comparison. The result is an additional evaluated system, not a replacement for the original TCN or an independent confirmatory test.

| Metric | Equal-source mean | Sample SD across five seed scores |
| --- | ---: | ---: |
| SoH30 MAE | 0.022371865766392718 | 0.0030039641993232515 |
| SoH30 RMSE | 0.030728762046474428 | 0.0024445320112840706 |
| SoH30 signed bias | +0.007832661490792639 | 0.004442236827014149 |

Units are normalised capacity fraction. Each metric was computed within each cell, then averaged equally over cells within a source and over sources within a seed. RMSE takes the square root within each cell before those averages. Bias is prediction minus observation. Sample SD describes the five outer seed scores conditional on the retained inner choices, not population uncertainty.

## Files

- `source_seed_metrics.csv`: 60 source/seed records with MAE, RMSE, bias and aggregate cell/window counts.
- `source_summary.csv`: twelve seed-averaged source summaries for the SoH panel of Figure 2.
- `summary.json`: aggregate scores, five seed means, Skill and descriptive paired source comparisons.
- `provenance.json`: aggregate input hashes, published runtime hash and bindings for the sixty retained predictions. The predictions themselves are not distributed.
- `../../configs/residual_outer/training_contract.json`: fixed P0 loss, learning rate, epoch rules, numerical profile and selected epochs for each held-out source.

## Reconstruction

From the repository root, run `python -B recompute_residual_outer.py`. It uses Python 3.11+ and the standard library. To additionally recalculate four source-bootstrap bands, run `python -B recompute_residual_outer.py --bootstrap` with NumPy. The resampling uses 10,000 paired draws, seed 20260914 and alphabetically ordered, seed-averaged source scores. No model is fitted, selected or rescored by these commands.

## Recipe and reuse

The released analysis fixes P0 with learning rate 0.001. For each held-out source, the outer epoch count is the median of three SoH-selected, source-blocked inner epochs with seed 2026. Six valid P0 inner folds were retained and thirty were added. Five matching CALB outer fits were retained and fifty-five outer fits were added. Initial sessions also produced legacy grid artifacts, but only P0 artifacts satisfying the final recipe and epoch rules enter this extension.

The runtime `residual_outer_followup.py` is provided as scientific implementation code and supports both legacy grid manifests and fixed-P0 continuation. The contract released here specifies fixed P0. A full execution still needs the authorised external inputs and original hash-bound deployment manifests. This release does not provision Kaggle credentials, notebook sessions, data, weights or prediction archives.

## Interpretation

Skill against Persistence is −6.930466668680912%, computed from unrounded MAEs. Residual MAE is lower than the original TCN on 12/12 sources, Extra Trees on 6/12, Trend on 1/12 and Persistence on 10/12. Its equal-source mean still exceeds Persistence, Extra Trees and Trend, so a majority source-win count alone does not establish a lower overall error. The original TCN was selected among three recipes, unlike this fixed-P0 system. This comparison therefore does not isolate the causal effect of residual learning.

Only SoH30 point outputs are evaluated here. Empty RUL fields in reconstructed tables mean not assessed. No residual RUL score, quantile interval, coverage or width is claimed. The earlier one-seed residual inner control on eleven validation sources, MAE 0.012974, remains under `results/submission/` and is not the same evaluation as the new outer MAE 0.022372.

# Additional diagnostic results

These aggregate results supplement the 21 October 2026 release. They are transcribed from saved research tables and the 29 October numeric review, not newly scored predictions. Values are rounded as shown in their sources; differences may not equal subtraction of the displayed rounded values. Empty fields mean not applicable, not zero.

| File | Source and interpretation |
| --- | --- |
| `leakage_control.csv` | Saved supplementary leakage-control table: nine target/seed pairs for fixed test points and training row counts, with training composition changed. SoH30 is MAE; RUL is RMSLE. `difference_reported` uses unrounded underlying values. This is a deliberately shared-history negative control, not the primary evaluation. |
| `stanford_components.csv` | Saved supplementary joint-domain component table: 17 reference configuration/target pairs. Separate versus joint refers to holding out the two related Stanford sources individually versus together. The fixed Extra Trees reference is not the selected five-seed primary system. These are post-hoc sensitivities, not proof of leakage. |
| `residual_soh_validation.csv` | Saved residual diagnostic summary: SoH30 MAE on 11 inner-validation sources, one seed and three folds excluding CALB. These are not outer-test results. |
| `selected_interval_summary.csv` | Saved historical quantile protocol summary at nominal 80% coverage. Inner validation selected Trend for SoH30 and Extra Trees for RUL. Width is averaged by point within source, then equally across sources; SoH is a fraction and RUL is in cycles. The selected-classical values come from the saved table; the TCN results are also represented in `results/coverage/`. |
| `bootstrap_reported.csv` | 29 October numeric review using the full-precision `results/primary/source_summary.csv`. Paired resampling of source blocks, 10,000 draws, seed 20260914; contrast is Extra Trees minus Trend. The intervals are descriptive conditional on these fitted models and sources, not population confidence intervals or independent validation. |

The leakage and Stanford values were checked against saved supplementary tables, rather than independently rescored from predictions. The residual summary was checked against the saved validation summary. No new training or prediction scoring is claimed. Raw data, cell-level metrics, predictions, weights and the paper are not included.

Recalculate the bootstrap with `python -B recompute_paired_bootstrap.py` using NumPy 2.3.5. The script resets the RNG for each endpoint, pairs the models by source, and orders sources alphabetically. The resampled statistics use the seed-averaged source scores, not separate seed resampling.

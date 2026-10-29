# Evidence for the 14-page submission

The target is *Pointwise Coverage Overstates the Reliability of Lithium-Ion Battery Forecasts on Unseen Data Sources*. Its PDF SHA-256 is recorded in `results/submission/provenance.json`. This map covers that submission, which contains Tables 1-5 and Figures 1-3. Earlier S-numbered tables are not the scope of this map.

Run from the repository root:

```sh
python -B verify_release.py
python -B recompute_summary.py
python -B recompute_common_input.py
python -B build_submission_results.py
```

These commands require Python 3.11+ and its standard library. The last command writes numeric CSVs for Tables 2-5, Figure 2 and Figure 3, plus assertion results, under `build/submission/`. Table 3 includes full-precision values, sample seed SDs and recomputed skills. Table 5 retains full-precision ranges rather than reproducing the manuscript's shortened display.

To generate fresh plots of the released values:

```sh
python -B build_submission_results.py --figures
```

Plotting requires Matplotlib (tested locally with 3.11.2 and NumPy 2.5.3). The plots reproduce numerical content; typography and layout differ from the submitted PDF. The separate `recompute_paired_bootstrap.py` was tested with NumPy 2.3.5 and reconstructs paired percentile bands from seed-averaged source scores.

| Submission item | Released evidence | Reconstruction |
| --- | --- | --- |
| Table 1; Figure 1 | Conceptual findings and pipeline, not measured arrays | No numeric reconstruction needed; protocol implementation entry points are in `REPRODUCIBILITY.md` |
| Table 2 | `results/primary/support.csv`; `results/censoring_and_snl/censoring_by_source.csv` | `table2.csv`; includes totals and censoring percentages |
| Table 3 | `results/primary/source_seed_metrics.csv`, `aggregate_metrics.csv`, `seed_macro_metrics.csv` | `recompute_summary.py`; `table3.csv` |
| Figure 2 | `results/primary/source_summary.csv` | `figure2.csv`, optional `figure2.pdf/png` |
| Section 5.1 paired bands | `results/diagnostics/bootstrap_reported.csv`; primary source scores | `recompute_paired_bootstrap.py` |
| Table 4; Figure 3 | `results/submission/selected_classical_by_source.csv`; `results/coverage/` and `results/hcp/` | `table4.csv`, `figure3.csv`, optional `figure3.pdf/png` |
| Section 5.2 other quantile models and late-life widths | `results/coverage/source_balanced_summary.csv`, `interval_metrics_by_source.csv`, `relative_width_support.csv`; select `last20` for late life | Includes cell-source width and relative-width estimators, distinct from Table 4's point-source width |
| Section 5.2 zero thresholds | `results/submission/width_decomposition_by_source.csv` | Check selects TCN.2026/per-origin: 19 zero thresholds over 35 scored source/target contexts |
| Section 5.2 XJTU example and swaps | `results/submission/xjtu_group_score_summary.csv`, `xjtu_rul90_fixed_prediction_qhat_swaps.csv`, `width_decomposition_by_source.csv` | Check verifies original thresholds and covered-history counts; swap rows are post-hoc diagnostics |
| Table 5 SNL checks | `results/censoring_and_snl/source_seed_sensitivity.csv`, `aggregate_sensitivity.csv` | First two rows of `table5.csv` |
| Table 5 shared histories | `results/submission/leakage_source_metrics.csv`, `leakage_overall_metrics.csv` | Source means checked for all target/mode/seed contexts; Table 5 shows SoH30 and RUL90 ranges |
| Table 5 joint Stanford | `results/submission/stanford_components_full_precision.csv` | Matching fixed-reference components in `table5.csv` |
| Section 5.3 common inputs | `results/common_input/selected_source_metrics.csv`, `summaries.json` | `recompute_common_input.py` (54 fits) |
| Section 5.3 residual TCN | `results/submission/residual_rows.csv`, `residual_paired_deltas.csv` | Checks the 11-source means and 11/11 versus direct, 3/11 versus Trend counts |
| Section 5.3 checkpoint sensitivity | `results/common_input/epoch_source_metrics.csv`, `checkpoint_sensitivity.json` | `recompute_common_input.py`; submission check verifies 17 later-better runs |

## Interpretation and provenance

The primary point comparison, earlier quantile comparison and post-hoc inner diagnostics use different contracts. Their inputs and model configurations must not be treated as matched simply because a family name is the same. In particular, joint Stanford uses fixed-reference Extra Trees, not the selected five-seed primary model.

`results/submission/provenance.json` records original relative artifact paths, byte hashes, export notes and row counts. These paths identify retained research artifacts; they are not additional downloadable files in this public repository. The additions export source-level or higher aggregates, not predictions, cell records or group identifiers. The XJTU group-score file contains only summary extrema and counts.

This release supports arithmetic inspection of reported aggregates. Its checks do not retrain models or independently rescore predictions. Full training still requires the authorized external data and sealed deployment artifacts described in `REPRODUCIBILITY.md`. The paper, raw cycle data, trained weights and per-cell predictions are not distributed here.

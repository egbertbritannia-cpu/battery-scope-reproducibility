# Scope of the point and quantile protocol comparison

The retained processed neural inputs for the primary point comparison, corrected TCN and GRU inner/outer deliveries were freshly compared on 29 October 2026. Each bundle contained 1,574 files totalling 471,431,735 bytes, with no differing, missing or additional file. Across TCN/GRU and twelve outer holdouts, 255 retained neural role/scaler records matched the corresponding quantile-system development/calibration/test role hashes, training-row counts, input mean/std values and target mean/std/transforms. These checks establish parity of those retained inputs and states, not independent physical validation of the upstream measurements.

The core Python, NumPy, pandas, PyArrow, scikit-learn, PyTorch, CUDA, cuDNN and Tesla T4 versions also matched in the inspected records. This is not a claim that all installed packages or historical classical-model environments match. Primary neural fits use float32 with AMP/TF32 disabled, whereas quantile-system code uses mixed precision. The retained historical classical prediction tables are not equivalent to fully replayable fitted tree objects.

| Component | Original primary point comparison | Quantile/interval systems |
| --- | --- | --- |
| Neural recipes | Q0 pinball and P0/P1 point-loss candidates | Joint pinball recipe |
| Epoch selection | Separate by target | RUL90-selected epoch shared across three targets |
| Extra Trees minimum leaf size | Candidates 5, 10, 20 | Fixed 10 |
| HGB maximum leaf nodes | Candidates 15, 31, 63 | Fixed 31 |
| Seed scope | Five outer seeds | Three seeds in historical neural point comparisons, with TCN 2026/2027 and GRU 2026 used for the interval extension |
| Calibration | Not read for primary point scoring | Required for the respective interval rules |

The new residual system uses the primary SoH support with a fixed P0 recipe. Equal inputs, roles and scaler values do not make its training trajectory identical to a selected Q0/P0/P1 system, and do not isolate an architectural effect. See `configs/residual_outer/input_parity.json` for the bounded audit summary and limitations.

## Coverage and width interpretation

For the two additional quantile systems GRU.2026 and TCN.2027, group-max all-history coverage across targets ranges from 49.27% to 62.48%. The 68.46% value belongs to TCN.2026/RUL80 and is the upper bound when all three quantile systems are considered. Both statements can be reconstructed from `results/coverage/source_balanced_summary.csv`, but their model scopes must not be mixed.

Table 4 widths use a mean over forecast origins within each source and then equal source weights. The XJTU widths of 1,935 and 3,907 cycles use cell-averaged widths and are different estimators. The approximately 82-fold late-life width/RUL value is 81.62624 from `cell_macro_positive_ratio_median` for TCN.2026/RUL90/group-max/last20. The subgroup uses t/τ ≥ 0.8, where τ = t + RUL, and the ratio excludes zero RUL. It is a cell/source average of median ratios, not a pooled median or the last 20% of forecast-origin counts.

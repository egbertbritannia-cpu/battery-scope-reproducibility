# BatteryScope reproducibility materials

Research code, configurations and source-level aggregate results for BatteryScope.

Snapshot: **21 October 2026**. This release is limited to code, results and supporting reproduction documentation. The manuscript, appendix text and paper figures are not published in this repository.

## Contents

| Directory | Contents |
|---|---|
| `protocol_and_provenance/` | Project training, selection, scoring, calibration and diagnostic code |
| `phase2-ouput/` | Small inherited runtime/configuration dependencies, not battery data; original spelling retained for code compatibility |
| `configs/` | Configurations, recorded environment and expected external-input hashes |
| `results/primary/` | Primary refit source/seed metrics and aggregate comparisons |
| `results/censoring_and_snl/` | Source censoring and evaluation-only SNL exclusions |
| `results/coverage/`, `results/hcp/` | Earlier quantile-model intervals and hierarchical calibration results |
| `results/inner_train_validation/` | Source-level inner training/validation diagnostic summaries |
| `results/common_input/` | Completed 54-fit common-input SoH diagnostic and validation histories by source |
| `docs/references.bib` | Research and data-source attribution |

## Reconstruct aggregate results

Python **3.11 or later**, standard library only:

```bash
python -B verify_release.py
python -B recompute_summary.py
python -B recompute_common_input.py
```

The commands check file hashes, reconstruct primary source/seed means and sample standard deviations, and reconstruct common-input and checkpoint-sensitivity summaries. They do not read a manuscript, train models, require a GPU, read raw battery data or use account tokens. These checks establish aggregate consistency, not independent prediction reproduction.

## Full training

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Full execution needs authorized external data, original deployment manifests and the recorded numerical environment. Raw measurements, cycle-profile shards, model weights, predictions and cell-level metrics are not distributed.

## Interpretation

- Primary point results cover 12 sources and five outer seeds; inner selection uses seed 2026. Model families do not have fully matched input representations or tuning budgets.
- Coverage results use separate, earlier quantile models; they are not intervals of the primary refit systems.
- The common-input SoH diagnostic has 54 fits: three families × three folds × three seeds × two input variants. CALB is excluded from training and validation. This is a post hoc inner diagnostic, not a new outer-test comparison or evidence that CALB data are invalid.
- RUL is scored only where the threshold event is observed. CALB/RUL90 is not evaluable; unavailable errors are not zero.
- Best-in-30-epochs summaries are exploratory and do not replace selected checkpoints or primary results.

## Citation and permissions

See [CITATION.cff](CITATION.cff) and [NOTICE.md](NOTICE.md). No publication, conference acceptance, independent replication or repository-wide MIT/Apache license is claimed.

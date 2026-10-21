# Reproducibility scope

## Aggregate reconstruction

Run the three standard-library commands in the README from the repository root. `release_manifest.json` binds the released bytes. This verifies file identity and aggregate consistency, not the correctness of raw measurements or independent prediction replay.

The primary script uses equal source weights within each seed and then averages seed-level scores. Standard deviations use the sample definition. The common-input script uses 11 validation sources per seed, reports three-seed means/SDs and checks the selected/final/best epoch summaries. Historical, primary and diagnostic results remain separate evidence.

## Implementation entry points

- Primary inner training: `protocol_and_provenance/neural_classical_fairness_v1/runtime/multisource_inner_kaggle.py`.
- Outer training and explicit continuation: `runtime/multisource_outer_kaggle.py`, `runtime/multisource_outer_lease.py` under the same directory.
- Point scoring: `protocol_and_provenance/neural_classical_fairness_v1/score_multisource_outer.py`.
- Interval scoring and hierarchical calibration: `score_coverage_extension.py` and `hcp_family_window.py` in that directory.
- Common-input SoH diagnostic: `runtime/common_sequence_scale_inner.py` and `runtime/common_sequence_split_inner.py`.

The source tree preserves inherited imports and paths. Some helper dependencies are under `tests/`; they are included as source dependencies, not as a claim that tests were run for this release. Operational guards reject invalid environments, mismatched manifests and overwriting accepted outputs.

## External requirements for full reruns

Obtain the authorized BatteryLife processed inputs matching `configs/external_inputs.json`, including the actual cycle-profile shards referenced by the index. A matching index alone is insufficient. Data access and redistribution follow the upstream providers' terms; this repository grants no data rights.

Provide frozen source-role selections, deployment manifests, target construction and the recorded environment. Configurations in this repository support inspection but do not provision a Kaggle/Colab runtime or all original deployment archives. Primary expected/observed versions and common-input CPU/GPU profiles are recorded separately. Do not substitute the local scoring environment for the GPU training environment.

Some account and machine locations were removed from public copies. This changes source-file hashes, so original sealed manifests must be reconciled through an authorized rerun workflow. Do not bypass integrity guards to treat edited source bytes as original frozen execution artifacts.

No raw records, profile shards, weights, prediction archives, cell-level metrics, account credentials, notebook outputs or internal operational logs are included. No independent end-to-end reproduction is claimed.

## Release boundary

The manuscript, appendix text and paper figures are intentionally omitted. Both aggregate reconstruction scripts operate without them. Original technical paths in training code are retained for inspection and do not imply that all execution data are distributed here.

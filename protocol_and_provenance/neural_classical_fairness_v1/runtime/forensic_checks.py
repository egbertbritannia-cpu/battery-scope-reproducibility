"""F1 checks on bound functions and synthetic data; never open outer labels.

These tests establish local functional invariants, not adequate tuning or a
cause of the historical CALB failure. No scientific definition is patched.
"""
from __future__ import annotations

import hashlib
import json


TARGETS = ("soh30", "rul90", "rul80")


def require(ok, message):
    if not ok:
        raise RuntimeError("F1_FUNCTIONAL_CHECK:" + message)


def run_checks(ns):
    import numpy as np
    import pandas as pd
    import torch

    passed = []
    def check(ok, name):
        require(bool(ok), name)
        passed.append(name)

    # No RNG consumption: running this suite cannot change a fit's initialization.
    y = torch.tensor([[.2, .3, .4], [.5, .6, .7]])
    mask = torch.tensor([[True, True, False], [True, False, True]])
    pred = {t: torch.tensor([[-.3, .1, .8], [-.2, .2, .9]]) for t in TARGETS}
    loss, tasks = ns["multitask_pinball_loss"](pred, y, mask)
    altered = y.clone(); altered[~mask] = float("nan")
    other, _ = ns["multitask_pinball_loss"](pred, altered, mask)
    check(torch.equal(loss, other), "masked_target_nan_does_not_change_loss")
    absent = mask.clone(); absent[:, 2] = False
    _, available = ns["multitask_pinball_loss"](pred, y, absent)
    check("rul80" not in available, "missing_head_not_reported_as_zero_loss")
    try:
        ns["multitask_pinball_loss"](pred, y, torch.zeros_like(mask))
    except RuntimeError:
        passed.append("no_observed_target_rejected")
    else:
        require(False, "no_observed_target_rejected")

    # Observed-only target scaler with identity and log1p endpoints.
    frame = pd.DataFrame(dict(canonical_cell_id=["synthetic.train"] * 3,
                              t_cycle=[30, 31, 32], sample_id=["a", "b", "c"]))
    for t in TARGETS:
        spec = ns["TARGET_SPECS"][t]
        frame[spec["y_col"]] = [.8, 1., 999999.] if t == "soh30" else [1., 9., 999999.]
        frame[spec["mask_col"]] = [True, True, False]
    scaler = ns["fit_target_scaler"](frame)
    changed = frame.copy()
    for t in TARGETS:
        changed.loc[2, ns["TARGET_SPECS"][t]["y_col"]] = -123456.
    check(scaler == ns["fit_target_scaler"](changed), "target_scaler_observed_only")
    check(scaler["fit_cell_hash"] == ns["sha256_json"](["synthetic.train"]),
          "target_scaler_records_train_id_hash")
    encoded, encoded_mask = ns["encode_multitask_targets"](frame.iloc[:2], scaler)
    for i, t in enumerate(TARGETS):
        restored = ns["inverse_scaled_target"](encoded[:, i], t, scaler)
        check(np.allclose(restored, frame.iloc[:2][ns["TARGET_SPECS"][t]["y_col"]],
                          rtol=1e-5, atol=1e-5), "target_roundtrip_" + t)
    check(encoded_mask.all(), "encoded_mask_alignment")
    try:
        ns["encode_multitask_targets"](frame.iloc[2:], scaler)
    except RuntimeError:
        passed.append("row_without_observed_head_rejected")
    else:
        require(False, "row_without_observed_head_rejected")

    values, masks = ns["VALUE_CHANNELS"], ns["VALUE_MASK_COLUMNS"]
    check(len(values) == len(masks) == 48, "bound_layout_48_values_48_masks")
    check(len(set(values)) == len(set(masks)) == 48 and not set(values) & set(masks),
          "bound_layout_channels_unique_and_disjoint")
    # cycle_index is both a lookup coordinate and a value channel in the real
    # contract. Perturb sensor values, never destroy the coordinate under test.
    mutable_values = [v for v in values if v != "cycle_index"]
    check(bool(mutable_values), "noncoordinate_value_channels_available")
    cycles = pd.DataFrame({"cycle_index": np.arange(1, 36)})
    for v, m in zip(values, masks, strict=True):
        if v != "cycle_index":
            cycles[v] = np.arange(1, 36, dtype=float)
        cycles[m] = np.ones(35, dtype=np.uint8)
    cycle_state = dict(mean=[0.] * 48, std=[1.] * 48)
    samples = pd.DataFrame({"t_cycle": [30, 31]})
    x = ns["make_window_inputs"](cycles, samples, cycle_state)
    future = cycles.copy(); future.loc[future.cycle_index > 31, mutable_values] = 1e9
    check(np.array_equal(future.cycle_index, cycles.cycle_index),
          "future_perturbation_preserves_cycle_coordinates")
    check(np.array_equal(x, ns["make_window_inputs"](future, samples, cycle_state)),
          "future_cycles_do_not_change_causal_windows")
    masked = cycles.copy(); masked.loc[0, masks] = 0
    baseline = ns["make_window_inputs"](masked, samples, cycle_state)
    masked.loc[0, mutable_values] = np.nan
    check(np.array_equal(masked.cycle_index, cycles.cycle_index),
          "masked_perturbation_preserves_cycle_coordinates")
    check(np.array_equal(baseline, ns["make_window_inputs"](masked, samples, cycle_state)),
          "masked_values_do_not_change_inputs")
    if "cycle_index" in values:
        coordinate_channel = values.index("cycle_index")
        check(baseline[0, 0, coordinate_channel] == 0
              and baseline[0, 0, coordinate_channel + 48] == 0,
              "masked_coordinate_feature_zero_lookup_coordinate_retained")
    check(np.all(x[0, :, 0] == np.arange(1, 31)) and
          np.all(x[1, :, 0] == np.arange(2, 32)), "window_row_endpoint_alignment")
    try:
        ns["make_window_inputs"](cycles.loc[cycles.cycle_index != 30], samples.iloc[:1], cycle_state)
    except RuntimeError:
        passed.append("missing_causal_endpoint_rejected")
    else:
        require(False, "missing_causal_endpoint_rejected")
    for name, invalid in (("empty_cycle_index_rejected", cycles.iloc[:0]),
                          ("descending_cycle_index_rejected", cycles.iloc[::-1]),
                          ("duplicate_cycle_index_rejected", pd.concat([cycles.iloc[:1], cycles]))):
        try:
            ns["make_window_inputs"](invalid, samples, cycle_state)
        except RuntimeError:
            passed.append(name)
        else:
            require(False, name)

    # Synthetic cycle scaler: do not call the bound disk/profile reader.
    original_load = ns["load_cycle_shard"]
    reads = []
    def load(cell):
        require(cell == "synthetic.train", "cycle_scaler_read_nontrain_cell")
        reads.append(cell)
        return cycles.copy()
    ns["load_cycle_shard"] = load
    try:
        state = ns["fit_cycle_scaler"]({"synthetic.train"})
        check(reads == ["synthetic.train"] and state["fit_cell_count"] == 1 and
              state["fit_cell_hash"] == ns["sha256_json"](["synthetic.train"]),
              "cycle_scaler_train_only_and_ids_bound")
    finally:
        ns["load_cycle_shard"] = original_load
    check(float(ns["inverse_scaled_target"](np.array([-100.]), "soh30", scaler)[0]) == 0.,
          "inverse_before_lower_bound")
    check(float(ns["inverse_scaled_target"](np.array([100.]), "soh30", scaler)[0]) == 1.5,
          "soh_upper_bound_unchanged")
    return dict(status="F1_SYNTHETIC_BOUND_FUNCTION_CHECKS_PASS", checks=passed,
                feature_layout=dict(value_channels=list(values), value_mask_columns=list(masks),
                                    coordinate_column="cycle_index",
                                    perturbed_value_channels=mutable_values),
                outer_labels_accessed=False, real_profiles_loaded=False,
                generalization_or_root_cause_established=False)


def raw_inverse(z, target, scaler):
    """Expose all three stages; preserve existing bounds, never choose new ones."""
    import numpy as np
    state = scaler["targets"][target]
    transformed = np.asarray(z, dtype=np.float64) * state["std"] + state["mean"]
    physical = np.expm1(transformed) if state["transform"] == "log1p" else transformed.copy()
    bounded = np.clip(physical, 0., 1.5) if target == "soh30" else np.maximum(physical, 0.)
    require(np.isfinite(physical).all(), "nonfinite_unbounded_inverse")
    return transformed, physical, bounded


def gradient_probe(ns, model, x, y, masks):
    """First development batch, dropout off; descriptive, no optimizer step/RNG use."""
    import math
    import torch
    original_mode = model.training
    model.eval()
    try:
        prediction = model(x)
        params = [p for name, p in model.named_parameters() if p.requires_grad and "heads." not in name]
        vectors, contributions, counts = {}, {}, {}
        for i, t in enumerate(TARGETS):
            counts[t] = int(masks[:, i].sum())
            if counts[t] == 0:
                contributions[t] = None
                continue
            single = torch.zeros_like(masks); single[:, i] = masks[:, i]
            loss, _ = ns["multitask_pinball_loss"](prediction, y, single)
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            vector = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().reshape(-1)
                                for g, p in zip(grads, params, strict=True)]).double()
            require(bool(torch.isfinite(vector).all()), "gradient_probe_nonfinite")
            vectors[t] = vector
            contributions[t] = float(torch.linalg.vector_norm(vector))
        pairs = {}
        for i, a in enumerate(TARGETS):
            for b in TARGETS[i + 1:]:
                if a not in vectors or b not in vectors:
                    pairs[a + "|" + b] = None
                    continue
                norm = contributions[a] * contributions[b]
                cosine = float(torch.dot(vectors[a], vectors[b]) / norm) if norm > 0 else None
                require(cosine is None or math.isfinite(cosine), "gradient_probe_cosine")
                pairs[a + "|" + b] = cosine
        return dict(observed_counts=counts, trunk_gradient_norm=contributions,
                    pairwise_cosine=pairs, optimizer_updates=0, dropout_enabled=False,
                    role="development_batch_descriptive_not_negative_transfer_proof")
    finally:
        model.train(original_mode)


def parameter_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seal_output(root, config, runtime_path, protocol_path):
    """Seal pilot and campaign evidence, including failures; no data copied."""
    from pathlib import Path
    root = Path(root)
    records = {}
    # A ZIP-mounted input may be extracted inside workspace/input_extracted.
    # Never export it or caches. Explicit allowlist keeps the return code/output-only.
    allowed = ("scripts/", "protocol_and_provenance/", "paper/audits/")
    for p in sorted(root.rglob("*")):
        relative = p.relative_to(root).as_posix()
        if (p.is_file() and (relative.startswith(allowed) or "/" not in relative)
                and p.name != "output_manifest.json" and "__pycache__" not in p.parts):
            records[relative] = dict(bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt = dict(schema="nfr1-diagnostic-output-v2", shard_id=config["shard_id"],
                   jobs=config["jobs"], protocol_sha256=sha(protocol_path),
                   runtime_sha256=sha(runtime_path), files=records, outer_scoring=False,
                   diagnostic_only=True, actual_GPU_acceptance_pending=True)
    path = root / "output_manifest.json"
    require(not path.exists(), "refuse_overwrite_output_seal")
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipt

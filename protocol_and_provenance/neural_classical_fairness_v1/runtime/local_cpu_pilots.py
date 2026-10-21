"""Locked, inner-only CPU development pilots. Never a production/scoring API.

One epoch neural pilots are NOT selection-effect estimates. Classical uses
declared outer preprocessing under an explicit new inner reconstruction role.
Each job runs in a new process; only locally generated hash-sealed models reload.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / "_review_tmp/nfr1_local_cpu_cache"))
from protocol_and_provenance.neural_classical_fairness_v1.runtime import selection_controls as selection

BUNDLE = ROOT / "phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2"
OUTER = ROOT / "phase2-ouput/batteryscope_stage7_programmatic_v6/stage8_outer_model_executor.py"
GRU = ROOT / "protocol_and_provenance/data_integrity/e06_gru_model.py"
TARGETS = selection.TARGETS
CLASSICAL_PREDICTION_POLICIES = {
    "extra_trees": "sklearn_predict_single_worker_fixed_tree_order_v1",
    "ridge": "native_predict_v1",
    "hist_gradient_boosting": "native_predict_v1",
}


def require(ok, message):
    if not ok:
        raise RuntimeError("NFR1_CPU_DEV:" + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def grid():
    configs = {}
    for alpha in (1, 10, 100):
        configs[f"ridge_{alpha}"] = dict(family="ridge", params=dict(alpha=alpha, solver="lsqr",
            tol=1e-6, max_iter=10000, fit_intercept=True))
    for leaves in (31, 15, 63):
        configs[f"hgb_{leaves}"] = dict(family="hist_gradient_boosting", params=dict(max_leaf_nodes=leaves,
            max_iter=200, learning_rate=.05, l2_regularization=.01, early_stopping=False, random_state=2026))
    for leaves in (10, 5, 20):
        configs[f"et_{leaves}"] = dict(family="extra_trees", params=dict(n_estimators=128, max_depth=16,
            min_samples_leaf=leaves, max_features=.7, bootstrap=False, n_jobs=2, random_state=2026))
    return configs


def jobs():
    neural = [dict(kind="neural", source="CALB", architecture=a, candidate=c, inner_fold=0, seed=2026)
              for a in ("TCN", "GRU") for c in selection.PRIORITY]
    neural += [dict(kind="neural", source="CALCE", architecture=a, candidate="Q0", inner_fold=0, seed=2026)
               for a in ("TCN", "GRU")]
    classical = [dict(kind="classical", source=s, candidate=c, target=t, inner_fold=0, seed=2026)
                 for s, configs in (("CALB", tuple(grid())), ("CALCE", ("hgb_31", "et_10")))
                 for c in configs for t in TARGETS]
    references = [dict(kind="references", source=s, target=t, inner_fold=0, seed=2026)
                  for s in ("CALB", "CALCE") for t in TARGETS]
    # Fixed operational ordering, no score-dependent queue decisions.
    first_tree = next(j for j in classical if j["source"] == "CALB" and j["candidate"] == "et_10" and j["target"] == "soh30")
    ordered = [neural[0], first_tree] + neural[1:] + [j for j in classical if j != first_tree] + references
    for job in ordered:
        job["id"] = ".".join(str(job[k]) for k in ("kind", "source", "inner_fold")) + "." + ".".join(
            str(job[k]) for k in ("architecture", "candidate", "target") if k in job)
    return ordered


class Guard:
    def __init__(self, run, directory, seconds=3600):
        import psutil
        self.run, self.directory = Path(run), Path(directory)
        self.proc, self.psutil = psutil.Process(), psutil
        self.deadline = time.monotonic() + seconds
        self.peak = 0
        self.last = 0.
        self.phase = "setup"

    def check(self):
        now = time.monotonic()
        rss = self.proc.memory_info().rss
        available = self.psutil.virtual_memory().available
        self.peak = max(self.peak, rss)
        if now - self.last >= 5:
            write(self.directory / "heartbeat.json", dict(phase=self.phase, pid=os.getpid(),
                rss_bytes=rss, peak_rss_bytes=self.peak, available_bytes=available,
                remaining_seconds=max(0, self.deadline - now)))
            self.last = now
        require(not (self.run / "STOP").exists(), "AUTHOR_STOP")
        require(now < self.deadline, "JOB_TIME_CAP")
        require(available >= 1.5 * 2**30 and rss <= 4.5 * 2**30, "RAM_GUARD")
        require(shutil.disk_usage(self.run).free >= 8 * 2**30, "DISK_GUARD")


class CellMetrics:
    """Streaming sufficient statistics, macro within cell then source, never pooled RMSE."""
    def __init__(self, target):
        require(target in TARGETS, "UNKNOWN_TARGET")
        self.target, self.cells = target, {}

    def add(self, source, cell, truth, raw, bounded):
        import numpy as np
        truth, raw, bounded = [np.asarray(v, dtype=float) for v in (truth, raw, bounded)]
        require(truth.ndim == 1 and truth.shape == raw.shape == bounded.shape and len(truth) > 0, "METRIC_SHAPE")
        require(all(np.isfinite(v).all() for v in (truth, raw, bounded)), "NONFINITE_METRIC")
        require((truth >= 0).all() and (bounded >= 0).all(), "NEGATIVE_METRIC_INPUT")
        error = bounded - truth
        squared = (np.log1p(bounded) - np.log1p(truth))**2 if self.target.startswith("rul") else error**2
        values = dict(n=len(truth), absolute=float(np.abs(error).sum()), signed=float(error.sum()),
            squared=float(squared.sum()), lower=int((raw < 0).sum()),
            upper=int((raw > 1.5).sum()) if self.target == "soh30" else 0,
            raw_min=float(raw.min()), raw_max=float(raw.max()), bounded_min=float(bounded.min()), bounded_max=float(bounded.max()))
        key = (str(source), str(cell))
        if key not in self.cells:
            self.cells[key] = values
        else:
            old = self.cells[key]
            for k, value in values.items():
                old[k] = min(old[k], value) if k.endswith("_min") else max(old[k], value) if k.endswith("_max") else old[k] + value

    def result(self):
        if not self.cells:
            return dict(status="NOT_EVALUABLE", anchors=0, cells=0)
        groups = {}
        for (source, cell), values in self.cells.items():
            n = values["n"]
            groups.setdefault(source, []).append(dict(mae=values["absolute"]/n, bias=values["signed"]/n,
                rmse_or_rmsle=math.sqrt(values["squared"]/n), lower_clip=values["lower"]/n, upper_clip=values["upper"]/n))
        by_source = {s: {k: math.fsum(v[k] for v in records)/len(records) for k in records[0]} for s, records in groups.items()}
        balanced = {k: math.fsum(v[k] for v in by_source.values())/len(by_source) for k in next(iter(by_source.values()))}
        metric = "mae" if self.target == "soh30" else "rmse_or_rmsle"
        return dict(status="EVALUABLE", anchors=sum(v["n"] for v in self.cells.values()), cells=len(self.cells),
            sources=len(groups), source_balanced=balanced, by_source=by_source, selection_score=balanced[metric],
            ranges={k: (min if k.endswith("_min") else max)(v[k] for v in self.cells.values())
                    for k in ("raw_min", "raw_max", "bounded_min", "bounded_max")},
            clipping_definition="strict raw<0/raw>1.5, source-balanced mean cell fractions")


def verified_namespace(job, guard, directory, *, cycles=True):
    import numpy as np
    c = json.loads((BUNDLE / "inner_config.frozen.json").read_text())
    bindings = json.loads((ROOT / "protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json").read_text())
    require(sha(BUNDLE / "inner_config.frozen.json") == bindings["config"]["sha256"], "CONFIG_HASH")
    for record in bindings["inputs"]:
        guard.check()
        require(sha(ROOT / record["path"]) == record["sha256"], "INPUT_HASH")
    recipe = json.loads((BUNDLE / "code/inner_recipe.json").read_text())
    # Bind the exact source files also via the sealed launch manifest before worker start.
    for record in recipe["definitions"]:
        require(hashlib.sha256(record["source"].encode()).hexdigest() == record["source_sha256"], "RECIPE_HASH")
    sys.path.insert(0, str(BUNDLE / "code"))
    reference = load_module("nfr1_local_inner_reference", BUNDLE / "code/smoke_reference_runtime.py")
    ns = reference.make_namespace(BUNDLE, recipe, c, metadata_only=not cycles)
    expected = next(r for r in c["inner_contexts"] if r["source"] == job["source"] and r["inner_fold"] == job["inner_fold"])
    roles = reference.validate_inner_roles(ns, job["source"], job["inner_fold"], expected)
    train, val = set(roles["selection_train"]), set(roles["early_validation"])
    require(train and val and not train & val, "ROLE_OVERLAP")
    membership = ns["fold_membership"]
    held = membership.loc[membership.evaluation_mode.eq("LODO") & membership.outer_entity.eq(job["source"])]
    forbidden = set(held.loc[~held.split.eq("development"), "canonical_cell_id"].astype(str))
    require(not (train | val) & forbidden, "OUTER_CALIBRATION_ROLE")
    require(not held.loc[held.canonical_cell_id.isin(train | val), "source"].eq(job["source"]).any(), "HELD_SOURCE_ACCESS")
    log = []
    def read(path, columns, cells):
        import pyarrow.dataset as ds
        guard.check()
        cells = set(cells)
        require(cells and (cells <= train or cells <= val), "FORBIDDEN_OR_MIXED_ROLE_READ")
        path = Path(path).resolve()
        require(path in {(BUNDLE / "phase2" / n).resolve() for n in
                         ("sample_target_index.parquet", "window_index.parquet", "tabular_feature_index.parquet")}, "FORBIDDEN_TABLE")
        frame = ds.dataset(path, format="parquet").to_table(columns=list(columns),
            filter=ds.field("canonical_cell_id").isin(sorted(cells))).to_pandas()
        require(set(frame.canonical_cell_id.astype(str)) <= cells, "READ_FILTER_ESCAPE")
        log.append(dict(table=path.name, columns=list(columns), cells=len(cells), cell_hash=digest(sorted(cells)), rows=len(frame)))
        write(directory / "access_log.json", log)
        return frame
    ns["read_for_cells"] = read
    cycle_stage = {"train_only": True}
    checked = {}
    def resolve(cell):
        guard.check()
        require(cell in (train if cycle_stage["train_only"] else train | val), "PROFILE_ROLE")
        record = ns["cycle_store_index"].loc[cell]
        path = (BUNDLE / "phase2" / str(record["shard_path"])).resolve()
        require(path.is_relative_to((BUNDLE / "phase2").resolve()), "SHARD_PATH_ESCAPE")
        if cell not in checked:
            require(sha(path) == str(record["sha256"]).lower(), "PROFILE_HASH")
            checked[cell] = str(record["sha256"]).lower()
            write(directory / "profile_hashes.json", checked)
        return path
    ns["resolve_cycle_shard"] = resolve
    rows_train = ns["load_multitask_window_targets"](train)
    rows_val = ns["load_multitask_window_targets"](val)
    require(rows_train.sample_id.is_unique and rows_val.sample_id.is_unique and
            not set(rows_train.sample_id) & set(rows_val.sample_id), "SAMPLE_ALIGNMENT")
    state = ns["fit_target_scaler"](rows_train)
    cycle = ns["fit_cycle_scaler"](train) if cycles else None
    cycle_stage["train_only"] = False
    write(directory / "roles.json", dict(train_cells=sorted(train), validation_cells=sorted(val),
        role_hashes=expected, train_rows=len(rows_train), full_validation_rows=len(rows_val),
        validation_sample_hash=digest(sorted(rows_val.sample_id.astype(str))), target_scaler=state,
        cycle_scaler=cycle, validation_scope="full_eligible_inner", calibration_labels_accessed=False,
        outer_labels_accessed=False, official_fit=False))
    return ns, rows_train, rows_val, cycle, state, read, train, val


def initialize(ns, job):
    import numpy as np
    import torch
    ns["SEED"] = job["seed"]
    effective = ns["deterministic_fold_seed"]("LODO", job["source"], job["inner_fold"])
    ns["SEED"] = 2026
    random.seed(effective); np.random.seed(effective); torch.manual_seed(effective)
    model = ns["MultitaskQuantileTCN"]() if job["architecture"] == "TCN" else load_module("nfr1_cpu_gru", GRU).make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=selection.candidate_recipes()[job["candidate"]]["learning_rate"],
                                 weight_decay=ns["TCN_WEIGHT_DECAY"])
    return model.float().cpu(), optimizer, effective


def neural_metrics(ns, model, iterator, scaler, guard):
    import numpy as np
    import torch
    metrics = {t: CellMetrics(t) for t in TARGETS}
    expected = {t: 0 for t in TARGETS}
    sample_hash = hashlib.sha256()
    first_batch = None
    model.eval()
    with torch.no_grad():
        for x, y, masks, rows in iterator.iter_prediction_batches():
            guard.check()
            if first_batch is None:
                first_batch = x.copy()
            pred = model(torch.from_numpy(x))
            for sample in rows.sample_id.astype(str):
                sample_hash.update((sample + "\n").encode())
            for i, target in enumerate(TARGETS):
                active = masks[:, i].astype(bool)
                expected[target] += int(active.sum())
                if not active.any():
                    continue
                z = pred[target].detach().numpy()[active, 1].astype(float)
                st = scaler["targets"][target]
                raw = z * st["std"] + st["mean"]
                raw = np.expm1(raw) if st["transform"] == "log1p" else raw
                bounded = np.clip(raw, 0., 1.5) if target == "soh30" else np.maximum(raw, 0.)
                selected = rows.loc[active]
                truth = selected[ns["TARGET_SPECS"][target]["y_col"]].to_numpy(float)
                for (source, cell), indices in selected.groupby(["source", "canonical_cell_id"], sort=True).indices.items():
                    metrics[target].add(source, cell, truth[indices], raw[indices], bounded[indices])
    result = {t: accumulator.result() for t, accumulator in metrics.items()}
    require(all(result[t]["anchors"] == expected[t] for t in TARGETS), "VALIDATION_SUPPORT_LOST")
    return result, sample_hash.hexdigest(), first_batch


def neural_job(job, run, directory, guard):
    import numpy as np
    import torch
    started = time.monotonic()
    ns, train, val, cycle, target, *_ = verified_namespace(job, guard, directory)
    preprocessing = time.monotonic() - started
    require(all(target["targets"][t]["active"] for t in TARGETS), "INACTIVE_TRAIN_HEAD")
    model, optimizer, effective = initialize(ns, job)
    iterator = ns["MultitaskCycleWindowIterator"](train, cycle, target)
    valid = ns["MultitaskCycleWindowIterator"](val, cycle, target)
    guard.phase = "train_epoch_1"
    model.train(); optimizer.zero_grad(set_to_none=True)
    batches, losses, norms = 0, [], []
    head_gradients = {t: [] for t in ("trunk", *TARGETS)}
    exposure = {t: 0 for t in TARGETS}
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    train_start = time.monotonic()
    for x, y, masks, _ in iterator.iter_cell_balanced_microbatches(epoch=1):
        guard.check()
        outputs = model(torch.from_numpy(x))
        loss = selection.training_loss(outputs, torch.from_numpy(y), torch.from_numpy(masks), job["candidate"])
        require(bool(torch.isfinite(loss)), "NONFINITE_TRAIN_LOSS")
        (loss / 8).backward(); batches += 1; losses.append(float(loss.detach()))
        for i, t in enumerate(TARGETS):
            exposure[t] += int(masks[:, i].sum())
        if batches % 8 == 0:
            sums = {t: 0. for t in head_gradients}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    key = next((t for t in TARGETS if t in name), "trunk")
                    sums[key] += float(parameter.grad.detach().double().square().sum())
            for key, value in sums.items(): head_gradients[key].append(math.sqrt(value))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            norms.append(float(norm)); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            require(all(bool(torch.isfinite(v).all()) for st in optimizer.state.values()
                        for v in st.values() if torch.is_tensor(v)), "NONFINITE_OPTIMIZER")
    require(batches > 0 and batches % 8 == 0, "INCOMPLETE_ACCUMULATION")
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "NONFINITE_PARAMETERS")
    require(any(not torch.equal(v, before[k]) for k, v in model.named_parameters()), "NO_PARAMETER_UPDATES")
    train_seconds = time.monotonic() - train_start
    guard.phase = "full_inner_validation"
    validation_start = time.monotonic()
    metrics, order_hash, batch = neural_metrics(ns, model, valid, target, guard)
    validation_seconds = time.monotonic() - validation_start
    for t in TARGETS:
        spec = ns["TARGET_SPECS"][t]
        observed = ns["normalize_bool"](val[spec["mask_col"]], spec["mask_col"])
        expected = int((observed & np.isfinite(val[spec["y_col"]].to_numpy(float))).sum())
        require(metrics[t]["anchors"] == expected, "FULL_SUPPORT_MISMATCH:" + t)
    torch.save(model.state_dict(), directory / "model.pt")
    np.save(directory / "replay_x.npy", batch, allow_pickle=False)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(batch))
    np.savez(directory / "replay_expected.npz", **{t: prediction[t].numpy() for t in TARGETS})
    write(directory / "model.integrity.json", dict(sha256=sha(directory / "model.pt"),
        replay_x_sha256=sha(directory / "replay_x.npy"), replay_expected_sha256=sha(directory / "replay_expected.npz"),
        effective_seed=effective, epoch=1, trained_quantiles=job["candidate"] == "Q0",
        production_artifact=False, replay_scope="first_full_validation_batch_only"))
    return dict(status="PILOT_TRAIN_COMPLETE_RELOAD_PENDING", kind="neural", metrics=metrics,
        validation_order_sha256=order_hash, full_validation=True, one_epoch_only=True,
        preprocessing_seconds=preprocessing, training_seconds=train_seconds, validation_seconds=validation_seconds,
        optimizer_updates=batches//8, microbatches=batches, observed_target_exposures=exposure,
        loss_mean_microbatch=float(np.mean(losses)), gradient_norm_before_clip_mean=float(np.mean(norms)),
        gradient_norm_before_clip_by_head={t: float(np.mean(v)) for t,v in head_gradients.items()},
        effective_seed=effective, selection_efficacy_established=False, checkpoint_epoch=1)


def classical_adapter():
    module = load_module("nfr1_classical_reference", OUTER)
    executor = object.__new__(module.BatteryScopeFoldExecutor)
    feature = json.loads((BUNDLE / "phase2/feature_spec.json").read_text())
    executor.feature_columns = feature["tabular_feature_columns"]
    require(len(executor.feature_columns) == len(set(executor.feature_columns)) == 112, "OFFICIAL_FEATURE_SET")
    return module, executor


def classical_restore(raw, family, target, state):
    import numpy as np
    raw = np.asarray(raw, float)
    if family == "ridge":
        raw = np.clip(raw, state["scaled_min"], state["scaled_max"])*state["scale"] + state["centre"]
    natural = np.expm1(raw) if target.startswith("rul") else raw
    require(np.isfinite(natural).all(), "NONFINITE_CLASSICAL_INVERSE")
    return natural, np.clip(natural, 0, 1.5) if target == "soh30" else np.maximum(natural, 0)


def classical_predict(model, x, family):
    """Canonical inference, including the full validation and reload fixture.

    Forest predict accumulates tree outputs in worker completion order when
    parallel. A fixed seed does not fix floating-point addition order. Keep
    parallel fitting parameters, but use sklearn's ordered single-worker
    prediction path. Restore n_jobs even on failure; no tolerance relaxation.
    Only used by sequential, process-isolated CPU development jobs.
    """
    require(family in CLASSICAL_PREDICTION_POLICIES, "UNKNOWN_PREDICTION_POLICY")
    if family != "extra_trees":
        return model.predict(x)
    fit_n_jobs = model.get_params()["n_jobs"]
    try:
        model.set_params(n_jobs=1)
        return model.predict(x)
    finally:
        model.set_params(n_jobs=fit_n_jobs)


def frame_metrics(frame, truth, raw, bounded, target):
    accumulator = CellMetrics(target)
    for (source, cell), indices in frame.groupby(["source", "canonical_cell_id"], sort=True).indices.items():
        accumulator.add(source, cell, truth[indices], raw[indices], bounded[indices])
    return accumulator.result()


def classical_job(job, run, directory, guard):
    import joblib
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
    module, executor = classical_adapter()
    ns, train, val, _, _, read, train_ids, val_ids = verified_namespace(job, guard, directory, cycles=False)
    target = job["target"]; spec = ns["TARGET_SPECS"][target]
    # Target frames use the same observed window/sample universe as the neural pool.
    def join(rows, cells):
        cols = list(dict.fromkeys(["sample_id", "canonical_cell_id"] + executor.feature_columns))
        predictors = read(BUNDLE / "phase2/tabular_feature_index.parquet", cols, cells)
        require(predictors.sample_id.is_unique, "DUPLICATE_TABULAR_SAMPLE")
        mask = ns["normalize_bool"](rows[spec["mask_col"]], spec["mask_col"])
        selected = rows.loc[mask & np.isfinite(rows[spec["y_col"]].to_numpy(float))].copy()
        # Classical historical targets are float64 from the source index. Keep
        # their precision instead of reusing the neural float32 target encoding.
        labels = read(BUNDLE / "phase2/sample_target_index.parquet",
                      ["sample_id", "canonical_cell_id", spec["y_col"]], cells)
        selected = selected.drop(columns=spec["y_col"]).merge(labels,
            on=["sample_id", "canonical_cell_id"], how="left", validate="one_to_one")
        require(np.isfinite(selected[spec["y_col"]].to_numpy(float)).all(), "RAW_TARGET_ALIGNMENT")
        f = selected.merge(predictors, on=["sample_id", "canonical_cell_id"], how="left", validate="one_to_one", indicator=True)
        require(f._merge.eq("both").all(), "TABULAR_ELIGIBLE_SAMPLE_MISSING")
        return f.drop(columns="_merge").reset_index(drop=True)
    train, val = join(train, train_ids), join(val, val_ids)
    if train.empty or val.empty:
        return dict(status="NOT_EVALUABLE", training_rows=len(train), validation_rows=len(val),
                    reason="endpoint lacks eligible train/validation support", learned_fits=0)
    truth = val[spec["y_col"]].to_numpy(float)
    y = train[spec["y_col"]].to_numpy(float)
    require((y >= 0).all() and (truth >= 0).all(), "NEGATIVE_TARGET")
    if job["kind"] == "references":
        names = ("soh_persistence", "causal_linear_trend") if target == "soh30" else (
            "rul_train_median", "rul_train_log_mean", "causal_linear_trend")
        result = {}
        for name in names:
            guard.check()
            pred = executor._naive_predict(name, target, train, val)
            result[name] = frame_metrics(val, truth, pred, pred, target)
            # Reference functions return bounded predictions, so raw clipping
            # statistics cannot be recovered and must not masquerade as zero.
            result[name]["raw_clipping_unavailable"] = True
            for key in ("lower_clip", "upper_clip"):
                result[name]["source_balanced"].pop(key, None)
                for source_metrics in result[name]["by_source"].values(): source_metrics.pop(key, None)
        write(directory / "reference_metrics.json", result)
        return dict(status="REFERENCE_EXECUTIONS_COMPLETE", reference_executions=len(names), learned_fits=0,
                    metrics=result, raw_clipping_unavailable=True)
    config = grid()[job["candidate"]]; family = config["family"]
    start = time.monotonic(); guard.phase = "classical_preprocess"
    state = executor._fit_preprocessor(train, family)
    state["fit_role"] = "inner_train_only_new_runtime_reconstruction"
    x = executor._transform_features(train, state, family)
    xv = executor._transform_features(val, state, family)
    transformed = np.log1p(y) if target.startswith("rul") else y.copy()
    target_state = dict(transform=spec["transform"])
    if family == "ridge":
        centre, scale = float(transformed.mean()), float(transformed.std())
        scale = scale if math.isfinite(scale) and scale > 1e-12 else 1.
        transformed = (transformed-centre)/scale
        target_state.update(centre=centre, scale=scale, scaled_min=float(transformed.min()), scaled_max=float(transformed.max()))
    constructors = dict(ridge=Ridge, hist_gradient_boosting=HistGradientBoostingRegressor, extra_trees=ExtraTreesRegressor)
    model = constructors[family](**config["params"])
    prep_seconds = time.monotonic() - start
    guard.check(); guard.phase = "classical_fit"
    start = time.monotonic(); model.fit(x, transformed); fit_seconds = time.monotonic() - start
    guard.check(); guard.phase = "classical_predict"
    raw, bounded = classical_restore(classical_predict(model, xv, family), family, target, target_state)
    metrics = frame_metrics(val, truth, raw, bounded, target)
    artifact = dict(estimator=model, feature_state=state, target_state=target_state, family=family, target=target,
        candidate=job["candidate"], train_sample_hash=digest(sorted(train.sample_id.astype(str))),
        train_cell_hash=digest(sorted(set(train.canonical_cell_id.astype(str)))), params=model.get_params(),
        provenance="NEW_RUNTIME_INNER_RECONSTRUCTION_NOT_HISTORICAL_REPLAY",
        prediction_policy=CLASSICAL_PREDICTION_POLICIES[family])
    joblib.dump(artifact, directory / "pipeline.joblib", compress=3)
    # Bounded forward-reload fixture: preprocessing included, raw features preserved.
    fixture = val.iloc[:min(len(val), 256)]
    fixture[executor.feature_columns].to_parquet(directory / "replay_features.parquet", index=False)
    np.save(directory / "replay_expected.npy", bounded[:len(fixture)], allow_pickle=False)
    write(directory / "model.integrity.json", dict(sha256=sha(directory / "pipeline.joblib"),
        feature_sha256=sha(directory / "replay_features.parquet"), expected_sha256=sha(directory / "replay_expected.npy"),
        replay_scope="first_256_full_validation_rows_only", feature_state=state, target_state=target_state,
        estimator_params=model.get_params(), training_sample_hash=artifact["train_sample_hash"],
        training_cell_hash=artifact["train_cell_hash"], provenance=artifact["provenance"],
        prediction_policy=artifact["prediction_policy"]))
    return dict(status="PILOT_TRAIN_COMPLETE_RELOAD_PENDING", kind="classical", training_rows=len(train),
        validation_rows=len(val), validation_sample_hash=digest(sorted(val.sample_id.astype(str))), metrics=metrics,
        preprocessing_seconds=prep_seconds, training_seconds=fit_seconds, learned_fits=1,
        provenance=artifact["provenance"], prediction_policy=artifact["prediction_policy"], full_validation=True)


def replay(job, directory):
    import numpy as np
    integrity = json.loads((directory / "model.integrity.json").read_text())
    if job["kind"] == "neural":
        import torch
        require(sha(directory / "model.pt") == integrity["sha256"] and
                sha(directory / "replay_x.npy") == integrity["replay_x_sha256"] and
                sha(directory / "replay_expected.npz") == integrity["replay_expected_sha256"], "REPLAY_HASH")
        # Namespace on synthetic bound definitions, no label/profile reads for reload.
        from tests.test_neural_fairness_f1 import synthetic_namespace
        ns = synthetic_namespace()
        model, _, effective = initialize(ns, job)
        require(effective == integrity["effective_seed"], "REPLAY_SEED")
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True), strict=True)
        model.eval()
        x = np.load(directory / "replay_x.npy", allow_pickle=False)
        with torch.no_grad():
            actual = model(torch.from_numpy(x))
        with np.load(directory / "replay_expected.npz", allow_pickle=False) as expected:
            require(all(np.array_equal(actual[t].numpy(), expected[t]) for t in TARGETS), "REPLAY_PREDICTION_DRIFT")
    else:
        import joblib
        import pandas as pd
        require(sha(directory / "pipeline.joblib") == integrity["sha256"] and
                sha(directory / "replay_features.parquet") == integrity["feature_sha256"] and
                sha(directory / "replay_expected.npy") == integrity["expected_sha256"], "REPLAY_HASH")
        artifact = joblib.load(directory / "pipeline.joblib")  # Only own artifact after sealed hash gate.
        policy = CLASSICAL_PREDICTION_POLICIES.get(artifact["family"])
        require(policy is not None and artifact.get("prediction_policy") == integrity.get("prediction_policy") == policy,
                "REPLAY_PREDICTION_POLICY_MISMATCH")
        require(artifact["target"] == job["target"] and
                artifact["estimator"].get_params() == integrity["estimator_params"] and
                artifact["feature_state"] == integrity["feature_state"] and
                artifact["target_state"] == integrity["target_state"], "REPLAY_PIPELINE_STATE_DRIFT")
        _, executor = classical_adapter()
        f = pd.read_parquet(directory / "replay_features.parquet")
        x = executor._transform_features(f, artifact["feature_state"], artifact["family"])
        _, pred = classical_restore(classical_predict(artifact["estimator"], x, artifact["family"]),
                                   artifact["family"], job["target"], artifact["target_state"])
        require(np.array_equal(pred, np.load(directory / "replay_expected.npy", allow_pickle=False)), "REPLAY_PREDICTION_DRIFT")
    write(directory / "replay.json", dict(status="FRESH_PROCESS_FORWARD_FIXTURE_EXACT_PASS",
        scope=integrity["replay_scope"], full_validation_replay=False, labels_read=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("job", "replay"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    require(run.is_relative_to(ROOT / "paper/audits/neural_classical_fairness_2026-10-08_v1"), "OUTPUT_PATH")
    manifest = json.loads((run / "launch_manifest.json").read_text())
    from protocol_and_provenance.neural_classical_fairness_v1.runtime.environment_identity import collect
    environment = json.loads((run / "environment.lock.json").read_text())
    require(collect() == environment["identity"], "WORKER_ENVIRONMENT_DRIFT")
    for record in manifest["bindings"]:
        require(sha(ROOT / record["path"]) == record["sha256"], "SEALED_CODE_INPUT_DRIFT:"+record["path"])
    require(manifest["jobs"] == jobs(), "JOB_MANIFEST_DRIFT")
    job = next(j for j in manifest["jobs"] if j["id"] == args.job_id)
    directory = run / "jobs" / job["id"]
    import torch
    torch.set_num_threads(4 if job["kind"] == "neural" else 2)
    torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    guard = Guard(run, directory)
    started = time.monotonic()
    try:
        if args.mode == "replay":
            replay(job, directory)
            return
        directory.mkdir(parents=True, exist_ok=False)
        write(directory / "started.json", dict(job=job, pid=os.getpid(), manifest_sha256=sha(run / "launch_manifest.json")))
        result = neural_job(job, run, directory, guard) if job["kind"] == "neural" else classical_job(job, run, directory, guard)
        result.update(wall_seconds=time.monotonic()-started, peak_sampled_rss_bytes=guard.peak,
                      production_fit=False, outer_scoring=False, fairness_closed=False)
        write(directory / "result.json", result)
        print(json.dumps(dict(job_id=job["id"], status=result["status"], seconds=result["wall_seconds"])), flush=True)
    except Exception as exc:
        write(directory / ("replay_failure.json" if args.mode == "replay" else "failure.json"),
              dict(status="FAILED_REVIEW_REQUIRED", error=str(exc), traceback=traceback.format_exc(),
                   wall_seconds=time.monotonic()-started, outer_scoring=False, fairness_closed=False))
        raise


if __name__ == "__main__":
    main()

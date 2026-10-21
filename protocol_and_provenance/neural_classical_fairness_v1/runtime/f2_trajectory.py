"""F2 trajectory engine: inner-only, full validation, checkpoint prefix QA.

Production execution remains gated outside this module. The pilot uses this same
engine for three epochs, never promotes its descriptive policy maps to selection.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import time

from . import local_cpu_pilots as local
from . import selection_controls as selection


def synchronize(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def rng_signature():
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return dict(python=repr(random.getstate()), numpy=(numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]),
        torch=torch.get_rng_state().numpy().tobytes(),
        cuda=[s.cpu().numpy().tobytes() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def model_digest(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize(ns, job, device):
    import torch
    model, _, effective = local.initialize(ns, job)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(effective)
    # Construct AdamW AFTER moving the model: device conversion may replace
    # Parameter objects on some runtimes. Never retain a CPU optimizer by assumption.
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=selection.candidate_recipes()[job["candidate"]]["learning_rate"], weight_decay=ns["TCN_WEIGHT_DECAY"])
    local.require(all(p.device == torch.device(device) for p in model.parameters()), "F2_DEVICE")
    local.require({id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in model.parameters()},
                  "F2_OPTIMIZER_PARAMETER_BINDING")
    return model, optimizer, effective


def validate(ns, model, iterator, scaler, guard, device, expected_support):
    import numpy as np
    import torch
    accumulators = {t:local.CellMetrics(t) for t in selection.TARGETS}
    sample_hash, seen = hashlib.sha256(), set()
    first = None
    before_rng, before_weights = rng_signature(), model_digest(model)
    model.eval()
    synchronize(device); start = time.monotonic()
    with torch.no_grad():
        for x, _, masks, rows in iterator.iter_prediction_batches():
            guard.check()
            local.require(len(rows) == len(x) == len(masks), "F2_VALIDATION_BATCH_ALIGNMENT")
            ids = rows.sample_id.astype(str).tolist()
            local.require(len(ids) == len(set(ids)) and not seen.intersection(ids), "F2_DUPLICATE_VALIDATION_SAMPLE")
            seen.update(ids)
            for sample in ids:
                sample_hash.update((sample + "\n").encode())
            if first is None:
                first = x.copy()
            predictions = model(torch.from_numpy(x).to(device))
            for i, target in enumerate(selection.TARGETS):
                active = masks[:, i].astype(bool)
                if not active.any():
                    continue
                values = predictions[target].detach().cpu().numpy()[active, 1].astype(float)
                state = scaler["targets"][target]
                raw = values * state["std"] + state["mean"]
                raw = np.expm1(raw) if state["transform"] == "log1p" else raw
                bounded = np.clip(raw, 0, 1.5) if target == "soh30" else np.maximum(raw, 0)
                frame = rows.loc[active].reset_index(drop=True)
                truth = frame[ns["TARGET_SPECS"][target]["y_col"]].to_numpy(float)
                for (source, cell), indices in frame.groupby(["source", "canonical_cell_id"], sort=True).indices.items():
                    accumulators[target].add(source, cell, truth[indices], raw[indices], bounded[indices])
    synchronize(device); elapsed = time.monotonic()-start
    local.require(before_rng == rng_signature() and before_weights == model_digest(model), "F2_VALIDATION_MUTATED_STATE_OR_RNG")
    metrics = {t:a.result() for t,a in accumulators.items()}
    local.require(first is not None and len(seen) == expected_support["all_rows"], "F2_FULL_VALIDATION_ROW_COUNT")
    for target in selection.TARGETS:
        local.require(metrics[target]["status"] == "EVALUABLE" and
                      metrics[target]["anchors"] == expected_support[target], "F2_ENDPOINT_SUPPORT:"+target)
    stats = {t:[dict(source=s, cell=c, **v) for (s,c),v in sorted(a.cells.items())] for t,a in accumulators.items()}
    for target in selection.TARGETS:
        local.require(reaggregate_statistics(stats[target], target) == metrics[target], "F2_STATISTICS_AGGREGATION_DRIFT")
    return metrics, stats, sample_hash.hexdigest(), first, elapsed


def reaggregate_statistics(records, target):
    """Independent arithmetic from exported per-cell sufficient statistics."""
    local.require(target in selection.TARGETS and records, "F2_STATISTICS_ENDPOINT_OR_EMPTY")
    ids=[(r["source"],r["cell"]) for r in records]
    local.require(len(ids) == len(set(ids)), "F2_STATISTICS_DUPLICATE_CELL")
    groups={}
    for r in records:
        n=r["n"]
        local.require(type(n) is int and n>0 and all(math.isfinite(r[k]) for k in
            ("absolute","signed","squared","raw_min","raw_max","bounded_min","bounded_max")) and
            r["absolute"]>=0 and r["squared"]>=0 and abs(r["signed"])<=r["absolute"]+1e-10,
            "F2_STATISTICS_INVALID")
        local.require(all(type(r[k]) is int and 0<=r[k]<=n for k in ("lower","upper")), "F2_STATISTICS_CLIP_COUNT")
        groups.setdefault(r["source"],[]).append(dict(mae=r["absolute"]/n,bias=r["signed"]/n,
            rmse_or_rmsle=math.sqrt(r["squared"]/n),lower_clip=r["lower"]/n,upper_clip=r["upper"]/n))
    by_source={s:{k:math.fsum(v[k] for v in rows)/len(rows) for k in rows[0]} for s,rows in groups.items()}
    balanced={k:math.fsum(v[k] for v in by_source.values())/len(by_source) for k in next(iter(by_source.values()))}
    return dict(status="EVALUABLE",anchors=sum(r["n"] for r in records),cells=len(records),sources=len(groups),
        source_balanced=balanced,by_source=by_source,
        selection_score=balanced["mae" if target=="soh30" else "rmse_or_rmsle"],
        ranges={k:(min if k.endswith("_min") else max)(r[k] for r in records)
                for k in ("raw_min","raw_max","bounded_min","bounded_max")},
        clipping_definition="strict raw<0/raw>1.5, source-balanced mean cell fractions")


def train_epoch(model, optimizer, iterator, candidate, epoch, guard, device):
    import torch
    synchronize(device); started = time.monotonic()
    model.train(); optimizer.zero_grad(set_to_none=True)
    exposures = {t:0 for t in selection.TARGETS}
    losses, norms, batches, clips = [], [], 0, 0
    gradients = {t:[] for t in ("trunk", *selection.TARGETS)}
    before = model_digest(model)
    for x, y, masks, _ in iterator.iter_cell_balanced_microbatches(epoch=epoch):
        guard.check()
        predictions = model(torch.from_numpy(x).to(device))
        loss = selection.training_loss(predictions, torch.from_numpy(y).to(device), torch.from_numpy(masks).to(device), candidate)
        local.require(bool(torch.isfinite(loss)), "F2_NONFINITE_LOSS")
        (loss/8).backward(); batches += 1; losses.append(float(loss.detach().cpu()))
        for i,t in enumerate(selection.TARGETS):
            exposures[t] += int(masks[:,i].sum())
        if batches % 8 == 0:
            sums = {t:0. for t in gradients}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    key = next((t for t in selection.TARGETS if t in name), "trunk")
                    sums[key] += float(parameter.grad.detach().double().square().sum().cpu())
            for key, value in sums.items():
                gradients[key].append(math.sqrt(value))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            value = float(norm.detach().cpu()); norms.append(value); clips += int(value > 1.)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            local.require(all(bool(torch.isfinite(v).all()) for st in optimizer.state.values()
                              for v in st.values() if torch.is_tensor(v)), "F2_NONFINITE_OPTIMIZER")
    local.require(batches > 0 and batches % 8 == 0, "F2_INCOMPLETE_ACCUMULATION")
    local.require(all(bool(torch.isfinite(p).all()) for p in model.parameters()) and model_digest(model) != before,
                  "F2_NO_UPDATE_OR_NONFINITE_PARAMETERS")
    synchronize(device)
    return dict(training_seconds=time.monotonic()-started, optimizer_updates=batches//8, microbatches=batches,
        observed_target_exposures=exposures, loss_mean_microbatch=math.fsum(losses)/len(losses),
        gradient_norm_before_clip_mean=math.fsum(norms)/len(norms),
        gradient_clipping_update_fraction=clips/len(norms),
        gradient_norm_before_clip_by_head={t:math.fsum(v)/len(v) for t,v in gradients.items()})


def policy_map(history, checkpoints, *, production=False):
    local.require(len(history) == len(checkpoints) and 0 < len(history) <= selection.MAX_EPOCHS, "F2_HISTORY_LENGTH")
    local.require([r["epoch"] for r in history] == list(range(1, len(history)+1)), "F2_EPOCH_ORDER")
    local.require(all(c["epoch"] == r["epoch"] and r["checkpoint"] == c
                      for r,c in zip(history,checkpoints)), "F2_CHECKPOINT_EPOCH_ALIGNMENT")
    local.require(not production or len(history) == selection.MAX_EPOCHS, "F2_PILOT_CANNOT_PROMOTE")
    scores = {t:[r["metrics"][t]["selection_score"] for r in history] for t in selection.TARGETS}
    policies = selection.checkpoint_policies(scores)
    for policy in ("S0", "S1"):
        for target in selection.TARGETS:
            epoch = policies[policy][target]["best_epoch"]
            policies[policy][target]["checkpoint"] = checkpoints[epoch-1]
    return dict(policies=policies, trajectory_epochs=len(history), full_validation=True,
                descriptive_only=not production, cross_fold_candidate_selection=False,
                production_ready=False, outer_scoring=False)


def run_trajectory(ns, job, train_iterator, valid_iterator, scaler, expected_support, guard, directory,
                   *, device="cpu", epochs=3):
    import numpy as np
    import torch
    local.require(type(epochs) is int and 1 <= epochs <= selection.MAX_EPOCHS, "F2_EPOCH_CAP")
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    model, optimizer, effective = initialize(ns, job, device)
    initial = model_digest(model)
    local.write(directory/"trajectory_started.json",dict(job=job,planned_epochs=epochs,
        learned_fit_initializations=1,effective_seed=effective,initial_parameter_sha256=initial,
        production_fit=False,outer_scoring=False))
    history, checkpoints, expected_order = [], [], None
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs+1):
        guard.phase = "train_epoch_"+str(epoch)
        train = train_epoch(model, optimizer, train_iterator, job["candidate"], epoch, guard, device)
        guard.phase = "full_validation_epoch_"+str(epoch)
        metrics, sufficient, order, batch, validation_seconds = validate(
            ns, model, valid_iterator, scaler, guard, device, expected_support)
        local.require(expected_order is None or expected_order == order, "F2_VALIDATION_ORDER_CHANGED")
        expected_order = order
        local.write(directory / f"cell_statistics_epoch_{epoch:02d}.json", sufficient)
        synchronize(device); t = time.monotonic()
        path = directory / f"checkpoint_epoch_{epoch:02d}.pt"
        torch.save(dict(state_dict=model.state_dict(), optimizer_state=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        epoch=epoch, effective_seed=effective, candidate=job["candidate"]), path)
        synchronize(device)
        checkpoints.append(dict(epoch=epoch, file=path.name, sha256=local.sha(path), bytes=path.stat().st_size))
        record = dict(epoch=epoch, **train, validation_seconds=validation_seconds,
                      checkpoint_io_seconds=time.monotonic()-t, metrics=metrics, validation_order_sha256=order,
                      checkpoint=checkpoints[-1], parameter_sha256=model_digest(model))
        history.append(record)
        local.write(directory / "history.json", history)
        local.write(directory / "policies.descriptive.json", policy_map(history, checkpoints))
        print("[F2 EPOCH]", job.get("id", job["architecture"]), epoch, round(train["training_seconds"],2), round(validation_seconds,2), flush=True)
    # A bounded exact reload fixture. Each stored checkpoint remains epoch-specific.
    np.save(directory / "replay_x.npy", batch, allow_pickle=False)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(batch).to(device))
    np.savez(directory / "replay_expected.npz", **{t:predictions[t].detach().cpu().numpy() for t in selection.TARGETS})
    result = dict(status="F2_TRAJECTORY_COMPLETE_REPLAY_PENDING", epochs=epochs,
        effective_seed=effective, initial_parameter_sha256=initial, history_sha256=local.sha(directory/"history.json"),
        checkpoints=checkpoints, validation_order_sha256=expected_order, full_validation=True,
        fixture=dict(x_sha256=local.sha(directory/"replay_x.npy"), expected_sha256=local.sha(directory/"replay_expected.npz")),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if str(device).startswith("cuda") else None,
        interval_eligible=job["candidate"] == "Q0", production_fit=False,
        cross_fold_candidate_selected=False, outer_scoring=False, fairness_closed=False)
    local.write(directory/"result.json", result)
    return result


def replay(ns, job, directory, guard, device):
    import numpy as np
    import torch
    directory = Path(directory)
    result = json.loads((directory/"result.json").read_text())
    local.require(local.sha(directory/"history.json") == result["history_sha256"], "F2_REPLAY_HISTORY_HASH")
    for checkpoint in result["checkpoints"]:
        path = (directory/checkpoint["file"]).resolve()
        local.require(path.parent == directory.resolve() and local.sha(path) == checkpoint["sha256"], "F2_REPLAY_CHECKPOINT_HASH")
        guard.check()
    local.require(local.sha(directory/"replay_x.npy") == result["fixture"]["x_sha256"] and
                  local.sha(directory/"replay_expected.npz") == result["fixture"]["expected_sha256"], "F2_REPLAY_FIXTURE_HASH")
    model, _, effective = initialize(ns, job, device)
    checkpoint = torch.load(directory/result["checkpoints"][-1]["file"], map_location=device, weights_only=True)
    local.require(checkpoint["effective_seed"] == result["effective_seed"] == effective and
                  checkpoint["candidate"] == job["candidate"] and checkpoint["epoch"] == result["epochs"], "F2_REPLAY_CONTEXT")
    model.load_state_dict(checkpoint["state_dict"], strict=True); model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(np.load(directory/"replay_x.npy", allow_pickle=False)).to(device))
    with np.load(directory/"replay_expected.npz", allow_pickle=False) as expected:
        local.require(all(np.array_equal(predictions[t].detach().cpu().numpy(), expected[t]) for t in selection.TARGETS),
                      "F2_REPLAY_EXACT_PREDICTION_DRIFT")
    local.write(directory/"replay.json", dict(status="F2_FRESH_PROCESS_BOUNDED_EXACT_PASS",
        scope="last_epoch_first_full_validation_batch_only", full_validation_replay=False, labels_read=False))


def synthetic_prefix_qa(ns, directory, device):
    """Same actual engine/models: short-2 vs long-3 exact epoch-2 checkpoints."""
    import numpy as np
    import pandas as pd
    import torch
    from types import SimpleNamespace
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    class Iterator:
        def iter_cell_balanced_microbatches(self, epoch):
            rng = np.random.default_rng(2026 + 1009*epoch)
            for _ in range(8):
                yield rng.normal(size=(2,30,96)).astype(np.float32), np.zeros((2,3), np.float32), np.ones((2,3), bool), None
        def iter_prediction_batches(self):
            rows = pd.DataFrame(dict(sample_id=["synthetic1", "synthetic2"], canonical_cell_id=["cell1","cell2"],
                source=["A", "B"], soh_t_plus_30=[.8,.8], rul90_cycles=[3.,3.], rul80_cycles=[3.,3.]))
            yield np.ones((2,30,96),np.float32), np.zeros((2,3),np.float32), np.ones((2,3),bool), rows
    guard = SimpleNamespace(check=lambda:None, phase="synthetic")
    scaler = dict(targets={t:dict(mean=.8 if t == "soh30" else math.log1p(3), std=.1,
                                 transform="identity" if t == "soh30" else "log1p") for t in selection.TARGETS})
    support = dict(all_rows=2, **{t:2 for t in selection.TARGETS})
    checks = []
    for architecture in ("TCN", "GRU"):
        for candidate in selection.PRIORITY:
            job = dict(id=f"synthetic.{architecture}.{candidate}", architecture=architecture, candidate=candidate,
                       source="CALB", inner_fold=0, seed=2026)
            a, b = directory/(architecture+candidate+"_long"), directory/(architecture+candidate+"_short")
            run_trajectory(ns, job, Iterator(), Iterator(), scaler, support, guard, a, device=device, epochs=3)
            run_trajectory(ns, job, Iterator(), Iterator(), scaler, support, guard, b, device=device, epochs=2)
            left = torch.load(a/"checkpoint_epoch_02.pt", map_location="cpu", weights_only=True)
            right = torch.load(b/"checkpoint_epoch_02.pt", map_location="cpu", weights_only=True)
            def same(x,y):
                if torch.is_tensor(x):
                    return torch.is_tensor(y) and torch.equal(x,y)
                if isinstance(x,dict):
                    return isinstance(y,dict) and x.keys() == y.keys() and all(same(x[k],y[k]) for k in x)
                if isinstance(x,(list,tuple)):
                    return type(x) is type(y) and len(x) == len(y) and all(same(u,v) for u,v in zip(x,y))
                return x == y
            local.require(same(left,right), "F2_SYNTHETIC_ENGINE_PREFIX_DRIFT")
            replay(ns,job,a,guard,device)
            checks.append(dict(architecture=architecture,candidate=candidate,prefix_epoch=2, exact=True))
    result = dict(status="ACTUAL_ENGINE_TCN_GRU_PREFIX_RELOAD_SYNTHETIC_PASS", checks=checks,
                  synthetic_fits=12, synthetic_epoch_passes=30, battery_fits=0, device=str(device),
                  does_not_establish_real_data_or_cross_device_prefix_parity=True)
    local.write(directory/"summary.json",result)
    return result

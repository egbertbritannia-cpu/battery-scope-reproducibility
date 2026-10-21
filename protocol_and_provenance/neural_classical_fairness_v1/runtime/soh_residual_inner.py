"""Exploratory, matched direct/residual SoH experiment; inner data only.

Uses the accepted TCN/P0 joint trainer. The only intervention is replacing the
constant SoH training-mean offset with a causal per-window Trend offset, keeping
the original training std, model initialization, inputs, sampler and losses.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_trajectory as engine
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_kaggle_pilot as pilot
from protocol_and_provenance.neural_classical_fairness_v1.runtime import kaggle_classical_remaining as inherited

MANIFEST = 'soh_residual_inner_delivery.json'
RUN_NAME = 'batteryscope_soh_residual_inner_v1'
TREND = 'research_trend_soh30'
CURRENT = 'deg__soh_trailing_med5__last'
SLOPE = 'deg__soh_trailing_med5__theil_sen_cycle'


def jobs():
    return [dict(id=f'residual.CALB.{fold}.TCN.P0.{arm}.2026', kind='neural',
                 source='CALB', inner_fold=fold, architecture='TCN', candidate='P0',
                 arm=arm, seed=2026) for fold in range(3) for arm in ('direct', 'residual')]


def manifest():
    m = json.loads((ROOT / MANIFEST).read_text())
    local.require(m['schema'] == 'soh-residual-inner-research-v1' and m['jobs'] == jobs(), 'RESIDUAL_QUEUE')
    local.require(m['epochs'] == 30 and m['head_weights'] == {'soh30': 1., 'rul90': 1., 'rul80': .5}
                  and m['candidate'] == 'P0' and m['learning_rate'] == .001, 'RESIDUAL_RECIPE')
    local.require(m['scientific_role'] == 'POST_OUTER_EXPLORATORY_INNER_ONLY'
                  and not any(m[k] for k in ('outer_scoring', 'calibration_scoring',
                       'replace_primary_comparison', 'automatic_outer_promotion', 'automatic_retry', 'automatic_resume')), 'RESIDUAL_SCOPE')
    for name, expected in m['files'].items():
        p = (ROOT / name).resolve()
        local.require(p.is_relative_to(ROOT) and p.is_file() and
                      dict(bytes=p.stat().st_size, sha256=local.sha(p)) == expected, 'RESIDUAL_CODE:' + name)
    return m


def attach_trend(ns, train, val, reader, train_cells, val_cells):
    """Two causal feature columns; fallback median from inner train only."""
    import numpy as np
    import pandas as pd
    spec = ns['TARGET_SPECS']['soh30']
    raw = reader(local.BUNDLE / 'phase2/sample_target_index.parquet',
                 ['sample_id', 'canonical_cell_id', spec['y_col']], train_cells)
    active = train.loc[train[spec['mask_col']], ['sample_id', 'canonical_cell_id']]
    raw = active.merge(raw, on=['sample_id', 'canonical_cell_id'], validate='one_to_one')
    values = raw[spec['y_col']].to_numpy(float)
    local.require(len(values) == len(active) and np.isfinite(values).all(), 'RESIDUAL_FALLBACK_TRAIN_SUPPORT')
    fallback = float(np.median(values))
    def join(frame, cells):
        features = reader(local.BUNDLE / 'phase2/tabular_feature_index.parquet',
                          ['sample_id', 'canonical_cell_id', CURRENT, SLOPE], cells)
        local.require(features.sample_id.is_unique, 'RESIDUAL_DUPLICATE_FEATURES')
        result = frame.merge(features, on=['sample_id', 'canonical_cell_id'], how='left',
                             validate='one_to_one', indicator=True)
        local.require(len(result) == len(frame) and result._merge.eq('both').all(), 'RESIDUAL_TREND_JOIN')
        current = result[CURRENT].to_numpy(float); slope = result[SLOPE].to_numpy(float)
        prediction = np.where(np.isfinite(slope) & (slope < 0), current + 30. * slope, current)
        result[TREND] = np.clip(np.where(np.isfinite(prediction), prediction, fallback), 0., 1.5)
        local.require(np.isfinite(result[TREND]).all(), 'RESIDUAL_TREND_FINITE')
        return result.drop(columns=['_merge', CURRENT, SLOPE]).sort_values(['canonical_cell_id', 't_cycle'], kind='mergesort').reset_index(drop=True)
    return join(train, train_cells), join(val, val_cells), dict(fallback_inner_train_median=fallback,
        feature_columns=[CURRENT, SLOPE], trend_rule='clip(current+30*negative_finite_slope; else current; missing=>train_median,0,1.5)')


def attach_encoder(ns, arm):
    original = ns['encode_multitask_targets']
    def encode(rows, scaler):
        import numpy as np
        y, masks = original(rows, scaler)
        if arm == 'residual':
            y = y.copy(); st = scaler['targets']['soh30']; active = masks[:, 0].astype(bool)
            truth = rows[ns['TARGET_SPECS']['soh30']['y_col']].to_numpy(float)
            offset = rows[TREND].to_numpy(float)
            # Preserve the direct arm's train-only scale; do not refit residual variance.
            y[active, 0] = ((truth[active] - offset[active]) / st['std']).astype(np.float32)
            restored = y[active, 0].astype(float) * st['std'] + offset[active]
            local.require(np.allclose(restored, truth[active], atol=2e-7, rtol=2e-7), 'RESIDUAL_ADD_BACK_ROUNDTRIP')
        return y, masks
    ns['encode_multitask_targets'] = encode


def validate(ns, model, iterator, scaler, guard, device, expected_support, *, arm):
    """Full eligible validation, truth remains original SoH; add-back before bounds."""
    import numpy as np
    import torch
    accumulators = {t: local.CellMetrics(t) for t in local.TARGETS}
    before_rng, before_weights = engine.rng_signature(), engine.model_digest(model)
    seen = set(); order = hashlib.sha256(); first = None
    model.eval(); engine.synchronize(device); start = time.monotonic()
    with torch.no_grad():
        for x, _, masks, rows in iterator.iter_prediction_batches():
            guard.check(); ids = rows.sample_id.astype(str).tolist()
            local.require(len(ids) == len(set(ids)) and not seen.intersection(ids), 'RESIDUAL_VALIDATION_DUPLICATE')
            seen.update(ids)
            for sample in ids: order.update((sample + '\n').encode())
            if first is None: first = x.copy()
            outputs = model(torch.from_numpy(x).to(device))
            for index, t in enumerate(local.TARGETS):
                active = masks[:, index].astype(bool)
                if not active.any(): continue
                z = outputs[t].detach().cpu().numpy()[active, 1].astype(float)
                state = scaler['targets'][t]
                offset = rows.loc[active, TREND].to_numpy(float) if arm == 'residual' and t == 'soh30' else state['mean']
                raw = z * state['std'] + offset
                if state['transform'] == 'log1p': raw = np.expm1(raw)
                local.require(np.isfinite(raw).all(), 'RESIDUAL_INVERSE_FINITE')
                bounded = np.clip(raw, 0, 1.5) if t == 'soh30' else np.maximum(raw, 0)
                frame = rows.loc[active].reset_index(drop=True)
                truth = frame[ns['TARGET_SPECS'][t]['y_col']].to_numpy(float)
                for (source, cell), positions in frame.groupby(['source', 'canonical_cell_id'], sort=True).indices.items():
                    accumulators[t].add(source, cell, truth[positions], raw[positions], bounded[positions])
    engine.synchronize(device)
    local.require(before_rng == engine.rng_signature() and before_weights == engine.model_digest(model), 'RESIDUAL_VALIDATION_MUTATED_STATE')
    metrics = {t: a.result() for t, a in accumulators.items()}
    statistics = {t: [dict(source=s, cell=c, **v) for (s, c), v in sorted(a.cells.items())] for t, a in accumulators.items()}
    local.require(len(seen) == expected_support['all_rows'] and first is not None, 'RESIDUAL_FULL_VALIDATION')
    for t in local.TARGETS:
        local.require(metrics[t]['anchors'] == expected_support[t] and
                      engine.reaggregate_statistics(statistics[t], t) == metrics[t], 'RESIDUAL_STATISTICS:' + t)
    return metrics, statistics, order.hexdigest(), first, time.monotonic() - start


def support(ns, rows):
    import numpy as np
    value = dict(all_rows=len(rows))
    for t in local.TARGETS:
        spec = ns['TARGET_SPECS'][t]
        value[t] = int((rows[spec['mask_col']].to_numpy(bool) & np.isfinite(rows[spec['y_col']].to_numpy(float))).sum())
    return value


def worker(mode, bundle, run, job_id):
    import torch
    m = manifest(); pilot.configure(bundle)
    actual, mismatch = pilot.environment(m)
    local.require(not mismatch and actual == json.loads((run / 'environment.lock.json').read_text()), 'RESIDUAL_ENVIRONMENT')
    inherited.input_checks(m, bundle)
    job = next(j for j in m['jobs'] if j['id'] == job_id)
    directory = run / 'jobs' / job_id
    if mode == 'replay':
        guard = pilot.Guard(run, directory, seconds=m['budget']['replay_seconds'])
        engine.replay(pilot.definitions(bundle), job, directory / 'trajectory', guard, 'cuda:0')
        return
    directory.mkdir(parents=True, exist_ok=False)
    guard = pilot.Guard(run, directory, seconds=m['budget']['job_seconds'])
    local.write(directory / 'started.json', dict(job=job, role=m['scientific_role'], delivery_sha256=local.sha(ROOT / MANIFEST)))
    ns, train, val, cycle, scaler, reader, train_cells, val_cells = local.verified_namespace(job, guard, directory, cycles=True)
    train, val, trend_state = attach_trend(ns, train, val, reader, train_cells, val_cells)
    local.write(directory / 'trend_state.json', trend_state)
    attach_encoder(ns, job['arm'])
    engine.validate = functools.partial(validate, arm=job['arm'])
    train_iterator = ns['MultitaskCycleWindowIterator'](train, cycle, scaler)
    val_iterator = ns['MultitaskCycleWindowIterator'](val, cycle, scaler)
    local.write(directory / 'expected_support.json', support(ns, val))
    result = engine.run_trajectory(ns, job, train_iterator, val_iterator, scaler, support(ns, val),
                                  guard, directory / 'trajectory', device='cuda:0', epochs=30)
    history = json.loads((directory / 'trajectory/history.json').read_text())
    # Endpoint selection remains S1; only SoH is the research primary endpoint.
    policies = engine.policy_map(history, result['checkpoints'], production=True)
    local.write(directory / 'inner_policy.json', policies)
    trend_metrics = local.CellMetrics('soh30')
    active = val[ns['TARGET_SPECS']['soh30']['mask_col']].to_numpy(bool)
    frame = val.loc[active].reset_index(drop=True)
    truth = frame[ns['TARGET_SPECS']['soh30']['y_col']].to_numpy(float)
    p = frame[TREND].to_numpy(float)
    for (source, cell), positions in frame.groupby(['source', 'canonical_cell_id'], sort=True).indices.items():
        trend_metrics.add(source, cell, truth[positions], p[positions], p[positions])
    local.write(directory / 'trend_validation.json', trend_metrics.result())
    # Fixed endpoint diagnostics after learning: no extra optimizer steps or epoch selection.
    selected = policies['policies']['S1']['soh30']['best_epoch']
    model, _, _ = engine.initialize(ns, job, 'cuda:0')
    diagnostics = []
    for epoch in sorted({1, 30, selected}):
        cp = result['checkpoints'][epoch - 1]
        state = torch.load(directory / 'trajectory' / cp['file'], map_location='cuda:0', weights_only=True)
        model.load_state_dict(state['state_dict'], strict=True)
        metrics, stats, order, _, seconds = engine.validate(ns, model, train_iterator, scaler, guard, 'cuda:0', support(ns, train))
        local.write(directory / f'train_cell_statistics_epoch_{epoch:02d}.json', stats)
        diagnostics.append(dict(epoch=epoch, metrics=metrics, order_sha256=order, seconds=seconds, checkpoint_sha256=cp['sha256']))
    local.write(directory / 'train_endpoint_diagnostics.json', dict(records=diagnostics, selection_from_train=False, optimizer_steps=0))
    result.update(status='EXPLORATORY_INNER_RESIDUAL_FIT_COMPLETE_REPLAY_PENDING',
                  arm=job['arm'], scientific_role=m['scientific_role'], primary_endpoint='soh30',
                  outer_scoring=False, replace_primary_comparison=False)
    local.write(directory / 'result.json', result)


def campaign(bundle):
    run = ROOT / 'output' / RUN_NAME
    local.require(not run.exists(), 'RESIDUAL_FRESH_SESSION')
    run.mkdir(parents=True); started = time.monotonic(); ledger = []; error = None
    try:
        m = manifest(); pilot.configure(bundle)
        actual, mismatch = pilot.environment(m)
        local.write(run / 'environment_check.json', dict(actual=actual, mismatches=mismatch))
        local.require(not mismatch, 'RESIDUAL_ENVIRONMENT_MISMATCH')
        local.write(run / 'environment.lock.json', actual)
        local.write(run / 'delivery_manifest.snapshot.json', m)
        local.write(run / 'input_mapping.json', inherited.input_checks(m, bundle))
        prefix = 'phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/'
        remapped = dict(config=m['source_bindings']['config'], inputs=[])
        for r in m['source_bindings']['inputs']:
            local.require(r['path'].startswith(prefix), 'RESIDUAL_INPUT_PREFIX')
            name = r['path'][len(prefix):]
            local.require(name in m['input_files'], 'RESIDUAL_INPUT_BOUND')
            remapped['inputs'].append(dict(r, path=str((Path(bundle) / name).resolve())))
        local.write(ROOT / 'protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json', remapped)
        deadline = started + m['budget']['session_seconds'] - m['budget']['reserve_seconds']
        for job in m['jobs']:
            local.require(time.monotonic() + m['budget']['job_seconds'] + m['budget']['replay_seconds'] < deadline,
                          'RESIDUAL_TIME_ADMISSION')
            entry = dict(job=job, state='STARTED'); ledger.append(entry)
            local.write(run / 'compute_ledger.json', ledger)
            for mode in ('job', 'replay'):
                seconds = m['budget']['job_seconds' if mode == 'job' else 'replay_seconds']
                command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__)), mode,
                           '--bundle', str(bundle), '--run', str(run), '--job-id', job['id']]
                with (run / (job['id'] + '.' + mode + '.log')).open('x', encoding='utf-8') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    import psutil
                    born = psutil.Process(child.pid).create_time()
                    try:
                        child.wait(timeout=seconds)
                        local.require(child.returncode == 0, 'RESIDUAL_WORKER_FAILED:' + job['id'] + ':' + mode)
                    finally:
                        inherited.terminate_owned(child, born)
            entry['state'] = 'COMPLETE'; local.write(run / 'compute_ledger.json', ledger)
        local.write(run / 'summary.json', dict(status='SIX_EXPLORATORY_INNER_FITS_COMPLETE_AUDIT_PENDING',
                    learned_fits=6, epochs=180, seed_bases=[2026], outer_scoring=False,
                    replace_primary_comparison=False, automatic_outer_promotion=False))
    except BaseException as exc:
        error = exc
        local.write(run / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc(),
                    completed_jobs=sum(r['state'] == 'COMPLETE' for r in ledger), automatic_retry=False))
    finally:
        files = sorted(p for p in run.rglob('*') if p.is_file())
        local.write(run / 'output_inventory.json', dict(files={p.relative_to(run).as_posix():
                    dict(bytes=p.stat().st_size, sha256=local.sha(p)) for p in files}))
        target = ROOT.parent / (RUN_NAME + '_results.zip')
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for p in files + [run / 'output_inventory.json']:
                archive.write(p, RUN_NAME + '/' + p.relative_to(run).as_posix())
        print('RESULT OR FAILURE ZIP:', target, flush=True)
    if error: raise error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['campaign', 'job', 'replay'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--job-id')
    args = parser.parse_args()
    campaign(args.bundle) if args.mode == 'campaign' else worker(args.mode, args.bundle, args.run, args.job_id)

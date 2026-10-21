"""Post hoc inner experiment: common sequence inputs and SoH MAE for three families.

No outer/calibration scoring or primary model replacement. Each subprocess owns
one fit. Immutable epoch/tree checkpoints support explicit, hash-bound resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
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

MANIFEST = 'common_sequence_scale_inner_delivery.json'
RUN_NAME = 'batteryscope_common_sequence_scale_inner_v1'
CAPACITY_CHANNELS = ('discharge_ah', 'charge_ah', 'discharge_wh', 'charge_wh',
    'cumulative_discharge_ah', 'cumulative_charge_ah', 'cumulative_absolute_energy_wh',
    'current_mean_a', 'current_std_a', 'current_rms_a', 'current_median_abs_a',
    'current_peak_abs_a', 'dqdv_peak1_magnitude', 'dqdv_peak2_magnitude', 'dqdv_area')


def jobs():
    return [dict(id=f'common.CALB.{fold}.{family}.{arm}.{seed}', source='CALB',
        inner_fold=fold, architecture=family, kind='classical' if family == 'ET' else 'neural',
        candidate='P0', seed=seed, arm=arm)
        for seed in (2026, 2027, 2028) for fold in range(3)
        for family in ('TCN', 'GRU', 'ET') for arm in ('original', 'cap15')]


def specification():
    return dict(schema='common-sequence-scale-inner-v1', source_context='CALB',
        role='POST_OUTER_EXPLORATORY_INNER_ONLY', outer_scoring=False, calibration_scoring=False,
        replace_primary_comparison=False, independent_confirmation=False,
        common_input=[30, 96], classical_input='same tensor flattened to 2880; no engineered extras',
        capacity_channels=list(CAPACITY_CHANNELS), cap=5., cap_scope='train and validation',
        train_sample='at most32 eligible SoH anchors/cell, rounded linspace positions; full cohort before fit',
        sample_weight='inverse selected count/cell, normalized to mean1; identical objective in all families',
        cycle_scaler='accepted train-only per-cell capped256 cycle reservoir, including full train profiles',
        target_scaler='accepted train-only scaler; only SoH transform contributes to learning',
        labels_read='existing inner multitask loader/scaler reads inner SoH/RUL; RUL has no loss or selection role',
        neural=dict(architectures='accepted TCN and GRU', loss='weighted absolute error in original SoH units',
            output='SoH q50 only; inverse train mean/std before loss, no prediction bounds in loss',
            learning_rate=.001, weight_decay=.00001, optimizer='AdamW', epochs=30,
            micro_batch=32, effective_batch=256, gradient_clip_norm=1., dtype='float32',
            amp=False, tf32=False, epoch_order='all selected anchors exactly once; per-epoch seeded permutation',
            stop='logical patience5 prefix; continue to epoch30, never select beyond prefix',
            selection='full validation MAE: mean cells within source then mean sources; min improvement1e-8'),
        extra_trees=dict(n_estimators=128, max_depth=16, min_samples_leaf=10,
            max_features=.7, criterion='absolute_error', bootstrap=False, n_jobs=2,
            checkpoint_trees=16, warm_start=True),
        validation='all eligible SoH windows; raw predictions and same bounds[0,1.5] for scoring',
        seeds=[2026, 2027, 2028], inner_folds=[0, 1, 2], planned_fits=54,
        interpretation='paired preprocessing effect primary; family ranking secondary; not equal HPO/compute',
        chronology='Designed after CALB outer inspection and fixed-model scale diagnosis; cap chosen post hoc.')


def manifest():
    m = json.loads((ROOT / MANIFEST).read_text())
    local.require(m['specification'] == specification() and m['jobs'] == jobs(), 'COMMON_RECIPE')
    local.require(m['science_sha256'] == local.digest(specification()), 'COMMON_SCIENCE_HASH')
    local.require(len(m['active_job_ids']) == len(set(m['active_job_ids'])) and
        set(m['active_job_ids']) <= {j['id'] for j in jobs()}, 'COMMON_ASSIGNED_QUEUE')
    for name, seal in m['files'].items():
        path = (ROOT / name).resolve()
        local.require(path.is_relative_to(ROOT) and path.is_file() and
            dict(bytes=path.stat().st_size, sha256=local.sha(path)) == seal, 'COMMON_CODE:' + name)
    return m


def tensor_hash(x):
    return hashlib.sha256(x.tobytes(order='C')).hexdigest()


def atomic_torch(path, value):
    import torch
    local.require(not path.exists(), 'CHECKPOINT_NO_OVERWRITE')
    temporary = path.with_suffix('.partial')
    torch.save(value, temporary)
    os.replace(temporary, path)
    return dict(file=path.name, bytes=path.stat().st_size, sha256=local.sha(path))


def checkpoint_read(directory, pointer, loader):
    record = json.loads((directory / pointer).read_text())
    path = (directory / record['file']).resolve()
    local.require(path.parent == directory.resolve() and path.stat().st_size == record['bytes']
        and local.sha(path) == record['sha256'], 'CHECKPOINT_HASH')
    return loader(path), record


def apply_arm(x, indices, arm):
    import numpy as np
    if arm == 'original':
        return x
    local.require(arm == 'cap15', 'UNKNOWN_ARM')
    result = x.copy()
    result[:, :, indices] = np.clip(result[:, :, indices], -5., 5.)
    local.require(np.array_equal(result[:, :, 48:], x[:, :, 48:]), 'MASK_CHANGED')
    return result


def prepare(job, run, directory, guard):
    import numpy as np
    import pandas as pd
    ns, train, val, cycle, target, *_ = local.verified_namespace(job, guard, directory, cycles=True)
    spec = ns['TARGET_SPECS']['soh30']
    def eligible(rows):
        active = ns['normalize_bool'](rows[spec['mask_col']], spec['mask_col']).to_numpy(bool)
        return rows.loc[active & np.isfinite(rows[spec['y_col']].to_numpy(float))].reset_index(drop=True)
    train, val = eligible(train), eligible(val)
    local.require(len(train) > 0 and len(val) > 0 and target['targets']['soh30']['active'], 'SOH_SUPPORT')
    groups = []
    for _, rows in train.groupby('canonical_cell_id', sort=True):
        positions = np.rint(np.linspace(0, len(rows) - 1, min(32, len(rows)))).astype(int)
        local.require(len(set(positions)) == len(positions), 'SAMPLE_POSITIONS')
        groups.append(rows.iloc[positions])
    selected = pd.concat(groups, ignore_index=True)
    n = len(selected); cells = selected.canonical_cell_id.nunique()
    counts = selected.groupby('canonical_cell_id').sample_id.transform('size').to_numpy(float)
    weights = n / (cells * counts)
    totals = pd.Series(weights).groupby(selected.canonical_cell_id).sum().to_numpy(float)
    local.require(np.allclose(totals, n / cells, atol=1e-10, rtol=1e-10)
        and abs(weights.mean() - 1.) < 1e-12, 'CELL_WEIGHT_PARITY')
    indices = [list(cycle['channels']).index(c) for c in CAPACITY_CHANNELS]
    x = np.empty((n, 30, 96), dtype=np.float32)
    for cell, positions in selected.groupby('canonical_cell_id', sort=True).indices.items():
        guard.check()
        x[positions] = ns['make_window_inputs'](ns['load_cycle_shard'](str(cell)), selected.iloc[positions], cycle)
    y = selected[spec['y_col']].to_numpy(np.float32)
    binding = dict(sample_ids=selected.sample_id.astype(str).tolist(),
        train_sample_sha256=local.digest(selected.sample_id.astype(str).tolist()),
        train_tensor_sha256=tensor_hash(x), train_truth_sha256=tensor_hash(y),
        train_weights_sha256=tensor_hash(weights), train_rows=n, train_cells=cells,
        validation_rows=len(val), validation_sample_sha256=local.digest(val.sample_id.astype(str).tolist()),
        cycle_scaler=cycle, target_scaler=target, capacity_channel_indices=indices,
        per_cell_weight_min=float(totals.min()), per_cell_weight_max=float(totals.max()))
    shared = run / 'fold_bindings' / f"fold_{job['inner_fold']}.json"
    if shared.exists():
        local.require(json.loads(shared.read_text()) == binding, 'CROSS_FAMILY_INPUT_PARITY')
    else:
        local.write(shared, binding)
    local.write(directory / 'sample_binding.json', dict(fold_binding_sha256=local.sha(shared),
        arm=job['arm'], original_tensor_sha256=binding['train_tensor_sha256'],
        train_samples=n, validation_samples=len(val), objective='cell-balanced SoH MAE'))
    x = apply_arm(x, indices, job['arm'])
    local.write(directory / 'arm_tensor.json', dict(sha256=tensor_hash(x), masks_sha256=tensor_hash(x[:, :, 48:])))
    return ns, x, y, weights, val, cycle, target, indices


def evaluate(ns, predictor, val, cycle, indices, arm, guard):
    import numpy as np
    metric = local.CellMetrics('soh30')
    order = hashlib.sha256(); first = None
    for cell, rows in val.groupby('canonical_cell_id', sort=True):
        profile = ns['load_cycle_shard'](str(cell))
        for start in range(0, len(rows), 256):
            guard.check()
            frame = rows.iloc[start:start + 256]
            x = apply_arm(ns['make_window_inputs'](profile, frame, cycle), indices, arm)
            raw = np.asarray(predictor(x), dtype=float)
            local.require(raw.shape == (len(frame),) and np.isfinite(raw).all(), 'VALIDATION_FINITE_SHAPE')
            truth = frame[ns['TARGET_SPECS']['soh30']['y_col']].to_numpy(float)
            local.require(frame.source.nunique() == 1, 'CELL_SOURCE')
            metric.add(str(frame.source.iloc[0]), str(cell), truth, raw, np.clip(raw, 0., 1.5))
            for sid in frame.sample_id.astype(str): order.update((sid + '\n').encode())
            if first is None: first = (x.copy(), raw.copy())
        del profile
    result = metric.result()
    local.require(result['anchors'] == len(val) and first is not None, 'FULL_VALIDATION_SUPPORT')
    stats = [dict(source=s, cell=c, **values) for (s, c), values in sorted(metric.cells.items())]
    local.require(engine.reaggregate_statistics(stats, 'soh30') == result, 'CELL_STATISTICS_REPLAY')
    return dict(metrics=result, cells=stats, order_sha256=order.hexdigest()), first


def neural(job, ns, x, y, weights, val, cycle, target, indices, guard, directory, m):
    import numpy as np
    import torch
    model, optimizer, effective = engine.initialize(ns, job, 'cuda:0')
    for name, parameter in model.named_parameters():
        if 'rul90' in name or 'rul80' in name: parameter.requires_grad_(False)
    initial = engine.model_digest(model)
    head_before = {k: v.detach().clone() for k, v in model.state_dict().items() if 'rul90' in k or 'rul80' in k}
    st = target['targets']['soh30']
    local.require(st['transform'] == 'identity' and st['std'] > 0, 'SOH_TARGET_TRANSFORM')
    # The checkpoint stores only tensors/native types so weights_only=True is sufficient.
    start_epoch = 1; history = []; best = None; stale = 0; closed = False
    # A completed checkpoint can precede its atomic pointer if export was interrupted.
    # Recover only native checkpoints already covered by the explicit resume archive.
    epochs_on_disk = sorted(directory.glob('epoch_[0-9][0-9].pt'))
    if epochs_on_disk:
        newest = epochs_on_disk[-1]
        local.write(directory / 'latest.json', dict(file=newest.name, bytes=newest.stat().st_size, sha256=local.sha(newest)))
    if (directory / 'latest.json').exists():
        cp, _ = checkpoint_read(directory, 'latest.json', lambda p: torch.load(p, map_location='cpu', weights_only=True))
        local.require(cp['job'] == job and cp['science_sha256'] == m['science_sha256']
            and cp['initial_parameter_sha256'] == initial and cp['effective_seed'] == effective, 'NEURAL_RESUME_BINDING')
        model.load_state_dict(cp['state_dict']); optimizer.load_state_dict(cp['optimizer_state'])
        torch.set_rng_state(cp['torch_rng']); torch.cuda.set_rng_state_all(cp['cuda_rng'])
        random.setstate(cp['python_rng'])
        history, best, stale, closed = cp['history'], cp['best'], cp['stale'], cp['closed']
        start_epoch = cp['epoch'] + 1
    def predict(a):
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(a).to('cuda:0'))['soh30'][:, 1].cpu().numpy().astype(float) * st['std'] + st['mean']
    local.write(directory / 'initialization.json', dict(effective_seed=effective, initial_parameter_sha256=initial,
        seed_pairing='same effective seed and per-epoch permutation for original/cap15 of same family/fold'))
    for epoch in range(start_epoch, 31):
        start = time.monotonic(); guard.phase = f'train_epoch_{epoch}'
        model.train()
        order = np.random.default_rng(effective + 1009 * epoch).permutation(len(x))
        epoch_absolute = 0.
        for begin in range(0, len(order), 256):
            batch = order[begin:begin + 256]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(batch), 32):
                guard.check(); positions = batch[offset:offset + 32]
                pred = model(torch.from_numpy(x[positions]).to('cuda:0'))['soh30'][:, 1]
                truth = torch.from_numpy(y[positions]).to('cuda:0')
                weight = torch.from_numpy(weights[positions].astype(np.float32)).to('cuda:0')
                loss_sum = (torch.abs(pred * st['std'] + st['mean'] - truth) * weight).sum()
                local.require(bool(torch.isfinite(loss_sum)), 'TRAIN_LOSS_FINITE')
                (loss_sum / len(batch)).backward()
                epoch_absolute += float(loss_sum.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        guard.phase = f'full_validation_epoch_{epoch}'
        before_rng = engine.rng_signature(); before = engine.model_digest(model)
        validation, _ = evaluate(ns, predict, val, cycle, indices, job['arm'], guard)
        local.require(before_rng == engine.rng_signature() and before == engine.model_digest(model), 'VALIDATION_STATE_CHANGED')
        score = validation['metrics']['selection_score']
        if not closed:
            if best is None or score < best['score'] - 1e-8:
                best = dict(epoch=epoch, score=score); stale = 0
            else:
                stale += 1
            if stale >= 5: closed = True
        record = dict(epoch=epoch, train_weighted_mae=epoch_absolute / len(x), validation=validation['metrics'],
            seconds=time.monotonic() - start, order_sha256=tensor_hash(order), prefix_closed=closed)
        history.append(record)
        cp = dict(job=job, science_sha256=m['science_sha256'], epoch=epoch, effective_seed=effective,
            initial_parameter_sha256=initial, state_dict=model.state_dict(), optimizer_state=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(),
            history=history, best=best, stale=stale, closed=closed)
        seal = atomic_torch(directory / f'epoch_{epoch:02d}.pt', cp)
        local.write(directory / 'latest.json', seal)
        local.write(directory / 'history.json', history)
        print(json.dumps(dict(job=job['id'], epoch=epoch, validation_mae=score, selected=best)), flush=True)
    local.require(all(torch.equal(v, model.state_dict()[k]) for k, v in head_before.items()), 'RUL_HEAD_CHANGED')
    selected = directory / f"epoch_{best['epoch']:02d}.pt"
    local.write(directory / 'selected.json', dict(file=selected.name, bytes=selected.stat().st_size, sha256=local.sha(selected)))
    cp, _ = checkpoint_read(directory, 'selected.json', lambda p: torch.load(p, map_location='cuda:0', weights_only=True))
    model.load_state_dict(cp['state_dict'])
    validation, fixture = evaluate(ns, predict, val, cycle, indices, job['arm'], guard)
    local.require(validation['metrics']['selection_score'] == best['score'], 'SELECTED_SCORE_REPLAY')
    np.savez(directory / 'replay_fixture.npz', x=fixture[0], raw=fixture[1])
    local.write(directory / 'validation.json', validation)
    return dict(selected_epoch=best['epoch'], epochs=30, effective_seed=effective,
        metrics=validation['metrics'], learned_fits=1)


def classical(job, ns, x, y, weights, val, cycle, target, indices, guard, directory, m):
    import joblib
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor
    params = dict(m['specification']['extra_trees']); step = params.pop('checkpoint_trees')
    params['random_state'] = job['seed']; params['n_estimators'] = step
    model = ExtraTreesRegressor(**params); flat = x.reshape(len(x), -1)
    start_trees = step
    trees_on_disk = sorted(directory.glob('trees_[0-9][0-9][0-9].joblib'))
    if trees_on_disk:
        newest = trees_on_disk[-1]
        local.write(directory / 'latest.json', dict(file=newest.name, bytes=newest.stat().st_size, sha256=local.sha(newest)))
    if (directory / 'latest.json').exists():
        model, _ = checkpoint_read(directory, 'latest.json', joblib.load)
        expected = {k: v for k, v in params.items() if k != 'n_estimators'}
        local.require(all(model.get_params()[k] == v for k, v in expected.items())
            and model.n_features_in_ == 2880 and len(model.estimators_) == model.n_estimators, 'ET_RESUME_BINDING')
        start_trees = model.n_estimators + step
    for count in range(start_trees, 129, step):
        guard.phase = f'fit_trees_{count}'; guard.check(); start = time.monotonic()
        model.set_params(n_estimators=count)
        model.fit(flat, y, sample_weight=weights)
        path = directory / f'trees_{count:03d}.joblib'; temporary = path.with_suffix('.partial')
        local.require(not path.exists(), 'ET_NO_OVERWRITE')
        joblib.dump(model, temporary, compress=0); os.replace(temporary, path)
        local.write(directory / 'latest.json', dict(file=path.name, bytes=path.stat().st_size, sha256=local.sha(path)))
        print(json.dumps(dict(job=job['id'], trees=count, seconds=time.monotonic() - start)), flush=True)
        guard.check()
    def predict(a):
        return local.classical_predict(model, a.reshape(len(a), -1), 'extra_trees')
    validation, fixture = evaluate(ns, predict, val, cycle, indices, job['arm'], guard)
    local.write(directory / 'selected.json', json.loads((directory / 'latest.json').read_text()))
    np.savez(directory / 'replay_fixture.npz', x=fixture[0], raw=fixture[1])
    local.write(directory / 'validation.json', validation)
    return dict(trees=128, metrics=validation['metrics'], learned_fits=1)


def worker(mode, bundle, run, job_id):
    import numpy as np
    import torch
    m = manifest(); pilot.configure(bundle)
    actual, mismatch = pilot.environment(m)
    local.require(not mismatch and actual == json.loads((run / 'environment.lock.json').read_text()), 'COMMON_ENVIRONMENT')
    inherited.input_checks(m, bundle)
    job = next(j for j in m['jobs'] if j['id'] == job_id)
    directory = run / 'jobs' / job_id
    guard = pilot.Guard(run, directory, seconds=m['budget']['job_seconds'])
    if mode == 'replay':
        fixture = np.load(directory / 'replay_fixture.npz')
        if job['architecture'] == 'ET':
            import joblib
            model, _ = checkpoint_read(directory, 'selected.json', joblib.load)
            raw = local.classical_predict(model, fixture['x'].reshape(len(fixture['x']), -1), 'extra_trees')
        else:
            ns = pilot.definitions(bundle)
            model, _, _ = engine.initialize(ns, job, 'cuda:0')
            cp, _ = checkpoint_read(directory, 'selected.json', lambda p: torch.load(p, map_location='cuda:0', weights_only=True))
            model.load_state_dict(cp['state_dict']); model.eval()
            st = json.loads((directory / 'roles.json').read_text())['target_scaler']['targets']['soh30']
            with torch.no_grad():
                raw = model(torch.from_numpy(fixture['x']).to('cuda:0'))['soh30'][:, 1].cpu().numpy().astype(float) * st['std'] + st['mean']
        local.require(np.array_equal(raw, fixture['raw']), 'FRESH_PROCESS_REPLAY_FAILED')
        local.write(directory / 'replay.json', dict(exact=True, examples=len(raw), checkpoint=json.loads((directory / 'selected.json').read_text()),
            scope='first full-validation batch only; not a full-validation independent replay'))
        return
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'started.json').exists():
        started = json.loads((directory / 'started.json').read_text())
        local.require(started['job'] == job and started['science_sha256'] == m['science_sha256'], 'JOB_RESUME_SCOPE')
    else:
        local.write(directory / 'started.json', dict(job=job, science_sha256=m['science_sha256']))
    start = time.monotonic(); data = prepare(job, run, directory, guard)
    result = (classical if job['architecture'] == 'ET' else neural)(job, *data, guard, directory, m)
    result.update(job=job, status='FIT_COMPLETE_REPLAY_PENDING', wall_seconds=time.monotonic() - start,
        science_sha256=m['science_sha256'], peak_rss_bytes=guard.peak, outer_scoring=False)
    local.write(directory / 'result.json', result)


def restore(archive, expected, run, m):
    local.require(local.sha(archive) == expected, 'RESUME_ARCHIVE_HASH')
    with zipfile.ZipFile(archive) as z:
        local.require(z.testzip() is None, 'RESUME_ARCHIVE_CRC')
        prefix = RUN_NAME + '/'
        local.require(all(n.startswith(prefix) and not n.endswith('/') for n in z.namelist()), 'RESUME_ARCHIVE_PREFIX')
        inventory = json.loads(z.read(prefix + 'output_inventory.json'))['files']
        local.require(set(z.namelist()) == {prefix + n for n in inventory} | {prefix + 'output_inventory.json'}, 'RESUME_ARCHIVE_CONTENTS')
        for name, seal in inventory.items():
            p = (run / name).resolve(); data = z.read(prefix + name)
            local.require(p.is_relative_to(run.resolve()) and len(data) == seal['bytes']
                and hashlib.sha256(data).hexdigest() == seal['sha256'], 'RESUME_FILE_HASH')
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    old = json.loads((run / 'delivery_manifest.snapshot.json').read_text())
    local.require(old['science_sha256'] == m['science_sha256'] and old['files'] == m['files']
        and old['input_files'] == m['input_files'], 'RESUME_SCIENCE_OR_CODE_DRIFT')
    # Preserve the previous failure/status without claiming a fresh initialization.
    for name in ('failure.json', 'summary.json'):
        p = run / name
        if p.exists(): p.rename(run / (name + f'.resumed_{time.time_ns()}'))
    if (run / 'STOP').exists(): (run / 'STOP').unlink()
    local.write(run / f'resume_{time.time_ns()}.json', dict(archive_sha256=expected, completed_fits_reused=True,
        continuation='epoch/tree boundary only; pending uncheckpointed work may repeat'))


def campaign(bundle, resume_zip=None, resume_sha256=None):
    run = ROOT / 'output' / RUN_NAME
    local.require(not run.exists(), 'FRESH_WORKSPACE_REQUIRED')
    run.mkdir(parents=True); start = time.monotonic(); ledger = []; error = None
    try:
        m = manifest(); pilot.configure(bundle)
        if resume_zip:
            local.require(resume_sha256 is not None, 'RESUME_EXPLICIT_HASH_REQUIRED')
            restore(resume_zip, resume_sha256, run, m)
        actual, mismatch = pilot.environment(m)
        local.write(run / 'environment_check.json', dict(actual=actual, mismatches=mismatch))
        local.require(not mismatch, 'COMMON_ENVIRONMENT_MISMATCH')
        if (run / 'environment.lock.json').exists():
            local.require(json.loads((run / 'environment.lock.json').read_text()) == actual, 'RESUME_ENVIRONMENT_DRIFT')
        local.write(run / 'environment.lock.json', actual)
        local.write(run / 'delivery_manifest.snapshot.json', m)
        local.write(run / 'input_mapping.json', inherited.input_checks(m, bundle))
        prefix = 'phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/'
        remapped = dict(config=m['source_bindings']['config'], inputs=[])
        for row in m['source_bindings']['inputs']:
            local.require(row['path'].startswith(prefix), 'COMMON_INPUT_PREFIX')
            name = row['path'][len(prefix):]
            local.require(name in m['input_files'], 'COMMON_INPUT_BINDING')
            remapped['inputs'].append(dict(row, path=str((Path(bundle) / name).resolve())))
        local.write(ROOT / 'protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json', remapped)
        deadline = start + m['budget']['session_seconds'] - m['budget']['reserve_seconds']
        for job in m['jobs']:
            if job['id'] not in m['active_job_ids']: continue
            directory = run / 'jobs' / job['id']
            if (directory / 'result.json').exists() and (directory / 'replay.json').exists():
                local.require(json.loads((directory / 'result.json').read_text())['science_sha256'] == m['science_sha256'], 'COMPLETE_REUSE_SEAL')
                ledger.append(dict(job=job, state='COMPLETE_REUSED')); continue
            if time.monotonic() + 300 >= deadline: break
            entry = dict(job=job, state='STARTED'); ledger.append(entry)
            local.write(run / 'compute_ledger.json', ledger)
            for mode in ('job', 'replay'):
                if mode == 'job' and (directory / 'result.json').exists(): continue
                remaining = min(m['budget']['job_seconds'], deadline - time.monotonic())
                local.require(remaining > 30, 'TIME_RESERVE')
                command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__)), mode,
                    '--bundle', str(bundle), '--run', str(run), '--job-id', job['id']]
                logname = job['id'] + '.' + mode + '.' + str(time.time_ns()) + '.log'
                with (run / logname).open('x', encoding='utf-8') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    import psutil
                    born = psutil.Process(child.pid).create_time()
                    try:
                        until = time.monotonic() + remaining
                        # Forward only our fit progress lines, keeping a durable full log.
                        with (run / logname).open('r', encoding='utf-8') as progress:
                            while child.poll() is None:
                                local.require(time.monotonic() < until, 'CHILD_TIME_CAP_CHECKPOINTED')
                                try: child.wait(timeout=min(5., until - time.monotonic()))
                                except subprocess.TimeoutExpired: pass
                                for line in progress:
                                    if line.startswith('{"job":'): print(line, end='', flush=True)
                            for line in progress:
                                if line.startswith('{"job":'): print(line, end='', flush=True)
                        local.require(child.returncode == 0, 'COMMON_WORKER_FAILED:' + job['id'] + ':' + mode)
                    finally:
                        inherited.terminate_owned(child, born)
                print(json.dumps(dict(job=job['id'], mode=mode, state='FINISHED')), flush=True)
            entry['state'] = 'COMPLETE'; local.write(run / 'compute_ledger.json', ledger)
        complete = sum(r['state'].startswith('COMPLETE') for r in ledger)
        local.write(run / 'summary.json', dict(status='PILOT_COMPLETE_AUDIT_PENDING' if complete == len(m['active_job_ids']) else 'SESSION_BUDGET_CHECKPOINTED',
            completed_active_fits=complete, assigned_fits=len(m['active_job_ids']), planned_campaign_fits=54,
            seconds=time.monotonic() - start, outer_scoring=False, primary_comparison_replaced=False))
    except BaseException as exc:
        error = exc
        local.write(run / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc(),
            complete_fits=sum(r['state'].startswith('COMPLETE') for r in ledger), automatic_retry=False))
    finally:
        files = sorted(p for p in run.rglob('*') if p.is_file())
        local.write(run / 'output_inventory.json', dict(files={p.relative_to(run).as_posix():
            dict(bytes=p.stat().st_size, sha256=local.sha(p)) for p in files if p.name != 'output_inventory.json'}))
        target = ROOT.parent / (RUN_NAME + '_results.zip')
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(p for p in run.rglob('*') if p.is_file()):
                z.write(path, RUN_NAME + '/' + path.relative_to(run).as_posix())
        print('RESULT OR FAILURE ZIP:', target, flush=True)
    if error: raise error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['campaign', 'job', 'replay'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--job-id')
    parser.add_argument('--resume-zip', type=Path)
    parser.add_argument('--resume-sha256')
    args = parser.parse_args()
    if args.mode == 'campaign': campaign(args.bundle, args.resume_zip, args.resume_sha256)
    else: worker(args.mode, args.bundle, args.run, args.job_id)

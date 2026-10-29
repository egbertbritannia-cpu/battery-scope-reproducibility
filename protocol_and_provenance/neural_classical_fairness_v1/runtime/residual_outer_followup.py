"""Post hoc residual SoH extension: nested selection, sealed outer predictions.

One source per fresh Kaggle session. Three candidates x three inner folds,
then five independent outer refits. Test labels are never read in this runtime.
This does not replace accepted primary results or claim fresh confirmatory data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_kaggle_pilot as pilot
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_trajectory as engine
from protocol_and_provenance.neural_classical_fairness_v1.runtime import selection_controls as selection
from protocol_and_provenance.neural_classical_fairness_v1.runtime import soh_residual_inner as residual
from protocol_and_provenance.neural_classical_fairness_v1.runtime import multisource_outer_core as outer
from protocol_and_provenance.neural_classical_fairness_v1.runtime import multisource_outer_contract as outer_contract
from protocol_and_provenance.neural_classical_fairness_v1.runtime import kaggle_classical_remaining as inherited

MANIFEST = 'residual_outer_followup_delivery.json'


def manifest():
    value = json.loads((ROOT/MANIFEST).read_text())
    local.require(value['schema'] == 'residual-outer-followup-20261029-v1', 'FOLLOWUP_SCHEMA')
    expected_candidates=['P0'] if value.get('fixed_p0') else list(selection.PRIORITY)
    local.require(value['candidates'] == expected_candidates and value['seeds'] == list(outer_contract.SEEDS), 'FOLLOWUP_GRID')
    local.require(value['primary_endpoint'] == 'soh30' and value['post_hoc'] is True, 'FOLLOWUP_ROLE')
    for name, seal in value['files'].items():
        path = (ROOT/name).resolve()
        local.require(path.is_relative_to(ROOT) and path.is_file() and outer.binding(path) == seal, 'FOLLOWUP_CODE:'+name)
    return value


def setup(bundle, directory):
    value = manifest()
    pilot.configure(bundle)
    actual, mismatches = pilot.environment(value)
    local.write(directory/'environment_check.json', dict(actual=actual, mismatches=mismatches))
    local.require(not mismatches, 'FOLLOWUP_ENVIRONMENT_MISMATCH')
    inherited.input_checks(value, bundle)
    prefix = 'phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/'
    bindings = dict(config=value['source_bindings']['config'], inputs=[])
    for record in value['source_bindings']['inputs']:
        local.require(record['path'].startswith(prefix), 'FOLLOWUP_INPUT_PREFIX')
        relative = record['path'][len(prefix):]
        local.require(relative in value['input_files'], 'FOLLOWUP_UNBOUND_INPUT')
        bindings['inputs'].append(dict(record, path=str((Path(bundle)/relative).resolve())))
    local.write(ROOT/'protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json', bindings)
    return value


def source_selection(records, candidates=None):
    """Equal validation-source weights, not equal-fold weights; no test scores."""
    candidates=list(selection.PRIORITY) if candidates is None else list(candidates)
    local.require(candidates in (list(selection.PRIORITY),['P0']), 'FOLLOWUP_CANDIDATE_SCOPE')
    means = {}
    for candidate in candidates:
        selected = [r for r in records if r['candidate'] == candidate]
        local.require(len(selected) == 3 and {r['fold'] for r in selected} == {0, 1, 2}, 'FOLLOWUP_INNER_GRID')
        values = {}
        for record in selected:
            for source, metrics in record['metrics']['by_source'].items():
                local.require(source not in values, 'FOLLOWUP_DUPLICATE_VALIDATION_SOURCE')
                values[source] = metrics['mae']
        local.require(len(values) == 11 and all(math.isfinite(v) for v in values.values()), 'FOLLOWUP_VALIDATION_SUPPORT')
        means[candidate] = statistics.mean(values.values())
    chosen = next(c for c in candidates if means[c] <= min(means.values()) + selection.TOLERANCE)
    epochs = [r['best_epoch'] for r in sorted(records, key=lambda r:r['fold']) if r['candidate'] == chosen]
    return dict(candidate=chosen, scores=means, inner_epochs=epochs, outer_epochs=int(statistics.median(epochs)),
                selection_from_test=False, selection_endpoint='soh30', selection_seed=2026)


def inner_job(bundle, run, job):
    import torch
    directory = run/'jobs'/job['id']
    directory.mkdir(parents=True, exist_ok=False)
    setup(bundle, directory)
    guard = pilot.Guard(run, directory, seconds=7200)
    ns, train, valid, cycle, target, reader, tcells, vcells = local.verified_namespace(job, guard, directory)
    train, valid, trend = residual.attach_trend(ns, train, valid, reader, tcells, vcells)
    residual.attach_encoder(ns, 'residual')
    train_it = ns['MultitaskCycleWindowIterator'](train, cycle, target)
    valid_it = ns['MultitaskCycleWindowIterator'](valid, cycle, target)
    model, optimizer, effective = engine.initialize(ns, job, 'cuda:0')
    initial = engine.model_digest(model)
    history = []
    for epoch in range(1, 31):
        trained = engine.train_epoch(model, optimizer, train_it, job['candidate'], epoch, guard, 'cuda:0')
        metrics, sufficient, order, _, seconds = residual.validate(ns, model, valid_it, target, guard, 'cuda:0',
                                                                  residual.support(ns, valid), arm='residual')
        history.append(dict(epoch=epoch, train=trained, metrics=metrics['soh30'], validation_order_sha256=order,
                            validation_seconds=seconds))
        selected = selection.select_logical_prefix([r['metrics']['selection_score'] for r in history])
        if epoch == selected['best_epoch']:
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, effective_seed=effective), directory/'best.pt')
            local.write(directory/'best_cell_statistics.json', sufficient['soh30'])
        local.write(directory/'history.json', history)
        print('[RESIDUAL INNER]', job['source'], job['candidate'], job['inner_fold'], epoch,
              metrics['soh30']['selection_score'], flush=True)
        if selected['logical_stop_epoch'] is not None:
            break
    best = history[selected['best_epoch']-1]
    local.write(directory/'result.json', dict(status='COMPLETE', candidate=job['candidate'], fold=job['inner_fold'],
        best_epoch=selected['best_epoch'], metrics=best['metrics'], effective_seed=effective,
        initial_parameter_sha256=initial, checkpoint=outer.binding(directory/'best.pt'), trend=trend,
        epochs_executed=len(history), outer_labels_accessed=False, full_validation=True))


def trend_for_rows(ctx, rows, cells, fallback):
    import numpy as np
    cols = ['sample_id', 'canonical_cell_id', residual.CURRENT, residual.SLOPE]
    features = ctx.read(ctx.bundle/'phase2/tabular_feature_index.parquet', cols, set(cells))
    merged = rows.merge(features, on=['sample_id','canonical_cell_id'], how='left', validate='one_to_one', indicator=True)
    local.require(len(merged) == len(rows) and merged._merge.eq('both').all(), 'FOLLOWUP_TREND_JOIN')
    current, slope = merged[residual.CURRENT].to_numpy(float), merged[residual.SLOPE].to_numpy(float)
    prediction = np.where(np.isfinite(slope) & (slope < 0), current+30*slope, current)
    merged[residual.TREND] = np.clip(np.where(np.isfinite(prediction), prediction, fallback), 0, 1.5)
    return merged.drop(columns=['_merge', residual.CURRENT, residual.SLOPE])


def outer_job(bundle, run, job):
    import numpy as np
    import pandas as pd
    import torch
    directory = run/'jobs'/job['id']
    directory.mkdir(parents=True, exist_ok=False)
    value = setup(bundle, directory)
    guard = pilot.Guard(run, directory, seconds=7200)
    ctx = outer.Context(bundle, dict(source=job['source'], outer_role_metadata=value['outer_role_metadata']), directory, guard)
    train, cycle, target = ctx.train(True)
    spec = ctx.ns['TARGET_SPECS']['soh30']
    valid = train.loc[train[spec['mask_col']], spec['y_col']].to_numpy(float)
    local.require(len(valid)>0 and np.isfinite(valid).all(), 'FOLLOWUP_TRAIN_FALLBACK')
    fallback = float(np.median(valid))
    train = trend_for_rows(ctx, train, ctx.roles['development']['cell_ids'], fallback)
    residual.attach_encoder(ctx.ns, 'residual')
    iterator = ctx.ns['MultitaskCycleWindowIterator'](train, cycle, target)
    model, optimizer, effective = outer.initialize(ctx.ns, job, 'cuda:0')
    history = []
    for epoch in range(1, job['fit_epochs']+1):
        history.append(dict(epoch=epoch, **engine.train_epoch(model, optimizer, iterator, job['candidate'], epoch, guard, 'cuda:0')))
        local.write(directory/'history.json', history)
        print('[RESIDUAL OUTER]', job['source'], job['seed'], epoch, '/', job['fit_epochs'], flush=True)
    checkpoint = directory/'model.pt'
    before_reload = engine.model_digest(model)
    torch.save(dict(state_dict=model.state_dict(), target_scaler=target, cycle_scaler=cycle,
                    effective_seed=effective, epochs=job['fit_epochs'], trend_fallback=fallback), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location='cuda:0', weights_only=False)['state_dict'], strict=True)
    local.require(engine.model_digest(model)==before_reload, 'FOLLOWUP_CHECKPOINT_RELOAD_CHANGED_WEIGHTS')
    test = ctx.test_rows()
    test = trend_for_rows(ctx, test, ctx.roles['test']['cell_ids'], fallback)
    frame_parts = []
    before = engine.model_digest(model)
    model.eval()
    with torch.no_grad():
        for x, rows in outer.prediction_batches(ctx.ns, test, cycle):
            guard.check()
            quantiles = model(torch.from_numpy(x).to('cuda:0'))['soh30'].detach().cpu().numpy().astype(float)
            raw = quantiles[:,1]*target['targets']['soh30']['std'] + rows[residual.TREND].to_numpy(float)
            local.require(np.isfinite(raw).all(), 'FOLLOWUP_PREDICTION_FINITE')
            frame = rows[list(outer_contract.METADATA)].copy()
            frame['endpoint']='soh30'; frame['y_pred_unbounded']=raw; frame['y_pred']=np.clip(raw,0,1.5)
            frame_parts.append(frame)
    local.require(engine.model_digest(model)==before, 'FOLLOWUP_PREDICTION_MUTATED_MODEL')
    frame = pd.concat(frame_parts, ignore_index=True)
    record = outer.save_prediction(directory/'predictions.parquet', frame, test, ['soh30'])
    local.write(directory/'result.json', dict(status='COMPLETE_PREDICTIONS_SEALED_NOT_SCORED', prediction=record,
        checkpoint=outer.binding(checkpoint), seed=job['seed'], epochs=job['fit_epochs'],
        calibration_labels_accessed=False, outer_labels_accessed=False, post_hoc=True))


def export(run):
    target = ROOT.parent/(run.name+'_results.zip')
    files = sorted(p for p in run.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='inventory.json')
    inventory = {p.relative_to(run).as_posix(): outer.binding(p) for p in files}
    local.write(run/'inventory.json', inventory)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files+[run/'inventory.json']:
            archive.write(path, run.name+'/'+path.relative_to(run).as_posix())
    print('OUTPUT', str(target), flush=True)


def campaign(bundle):
    value = manifest()
    source = value['source']
    run = ROOT/'output'/('batteryscope_residual_outer_20261029_'+source)
    run.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        setup(bundle, run)
        local.write(run/'protocol.snapshot.json', value)
        reused=value.get('reused_inner',[])
        local.require(all(r['candidate']=='P0' and r['source']==source and r['provenance']['sha256'] for r in reused), 'FOLLOWUP_REUSE_PROVENANCE')
        reused_folds={r['fold'] for r in reused}
        local.require(len(reused_folds)==len(reused), 'FOLLOWUP_REUSE_DUPLICATE_FOLD')
        local.write(run/'reused_inner.json',reused)
        queue = [dict(id=f'inner.{source}.{fold}.TCN.{candidate}.residual.2026', source=source, inner_fold=fold,
                      architecture='TCN', candidate=candidate, seed=2026, kind='inner')
                 for candidate in value['candidates'] for fold in range(3) if fold not in reused_folds]
        for phase in ('inner', 'outer'):
            if phase == 'outer':
                records = reused+[json.loads((run/'jobs'/j['id']/'result.json').read_text()) for j in queue]
                chosen = source_selection(records,value['candidates'])
                local.write(run/'selection.frozen.json', chosen)
                queue = [dict(id=f'outer.{source}.TCN.{chosen["candidate"]}.residual.{seed}', source=source,
                              architecture='TCN', candidate=chosen['candidate'], seed=seed, kind='outer',
                              effective_seed=outer_contract.effective_seed(seed,source), fit_epochs=chosen['outer_epochs'])
                         for seed in outer_contract.SEEDS]
            for job in queue:
                local.require(time.monotonic()-start < value['session_seconds']-1800, 'FOLLOWUP_SESSION_ADMISSION')
                with (run/(job['id']+'.log')).open('x', encoding='utf8') as log:
                    args = [sys.executable,'-B','-X','utf8',str(Path(__file__)),phase,'--bundle',str(bundle),
                            '--run',str(run),'--job',json.dumps(job)]
                    child = subprocess.Popen(args, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
                    try:
                        deadline=time.monotonic()+min(7500,max(1,value['session_seconds']-1800-(time.monotonic()-start)))
                        while child.poll() is None:
                            local.require(time.monotonic()<deadline, 'FOLLOWUP_CHILD_DEADLINE')
                            local.write(run/'heartbeat.json',dict(job=job['id'],phase=phase,at_unix=time.time()))
                            log.flush()
                            tail=(run/(job['id']+'.log')).read_text(encoding='utf8',errors='replace').splitlines()[-1:]
                            print('[FOLLOWUP PROGRESS]',job['id'],*tail,flush=True)
                            time.sleep(30)
                        result = child.returncode
                    except BaseException:
                        child.terminate()
                        child.wait(timeout=30)
                        raise
                local.require(result == 0, 'FOLLOWUP_CHILD_FAILED:'+job['id'])
                export(run)
        local.write(run/'summary.json', dict(status='FIXED_P0_OUTER_COMPLETE_AUDIT_PENDING' if value.get('fixed_p0') else 'NINE_INNER_FIVE_OUTER_COMPLETE_AUDIT_PENDING',source=source,
            reused_inner_fits=len(reused),new_inner_fits=len(value['candidates'])*3-len(reused),outer_fits=5,
            outer_labels_accessed=False, new_primary_results_promoted=False, post_hoc=True))
    except BaseException as error:
        local.write(run/'failure.json', dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        export(run)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['campaign','inner','outer'])
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--job')
    args=parser.parse_args()
    if args.mode=='campaign':
        campaign(args.bundle)
    else:
        job=json.loads(args.job)
        (inner_job if args.mode=='inner' else outer_job)(args.bundle,args.run,job)

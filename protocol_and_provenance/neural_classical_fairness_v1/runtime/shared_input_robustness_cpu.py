"""Paired raw/train-range-input experiment; no outer fits or automatic promotion."""
from pathlib import Path
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'robustness_results'
RELEASE = ROOT / 'delivery'
ARCHIVE = Path('/kaggle/working/batteryscope_shared_input_robustness_cpu_v1_results.zip')


def require(ok, message):
    if not ok:
        raise RuntimeError('SHARED_INPUT_ROBUSTNESS:' + message)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.partial')
    with temporary.open('w', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, p)


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_archive():
    temporary = ARCHIVE.with_suffix('.partial')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix != '.partial':
                compression = zipfile.ZIP_STORED if p.suffix in {'.parquet', '.joblib', '.pt'} else zipfile.ZIP_DEFLATED
                z.write(p, 'batteryscope_shared_input_robustness_cpu_v1/' + p.relative_to(OUT).as_posix(), compress_type=compression)
    os.replace(temporary, ARCHIVE)


def evaluate(data, state, predict, model, family, ident, rung, bounds, save_weights=True):
    import numpy as np
    import pandas as pd
    import torch
    import joblib
    start_cpu, start_wall = time.process_time(), time.monotonic()
    artifacts = []
    result = {}
    train_max = float(data['train']['y'].max())
    for role in ['validation', 'train']:
        r = data[role]
        standardized = np.asarray(predict(r['x']), dtype='float64')
        log_prediction = standardized * state['target_std'] + state['target_mean']
        require(np.isfinite(log_prediction).all(), 'NONFINITE_LOG_OUTPUT')
        nonnegative_log = np.maximum(log_prediction, 0.)
        with np.errstate(over='ignore', invalid='ignore'):
            prediction = np.expm1(nonnegative_log)
            absolute = np.abs(prediction - r['y'])
            squared = (prediction - r['y']) ** 2
        overflow = ~np.isfinite(prediction) | ~np.isfinite(squared)
        frame = r['meta'].copy()
        frame['truth_rul_cycles'] = r['y']
        frame['prediction_standardized'] = standardized
        frame['prediction_log1p_unclipped'] = log_prediction
        frame['prediction_rul_cycles'] = prediction
        frame['absolute_error_cycles'] = absolute
        frame['squared_error_cycles'] = squared
        frame['squared_log_error'] = (nonnegative_log - np.log1p(r['y'])) ** 2
        frame['lower_clipped'] = log_prediction < 0
        frame['cycle_metric_overflow'] = overflow
        frame['above_train_max'] = prediction > train_max
        frame['above_ten_times_train_max'] = prediction > 10 * train_max
        # Input counters are diagnostic, evaluated outside the fit CPU clock.
        frame['max_absolute_standardized_feature'] = np.max(np.abs(r['x']), axis=1)
        frame['outside_train_feature_dimensions'] = np.sum((r['x'] < bounds['lower']) | (r['x'] > bounds['upper']), axis=1)
        cell = frame.groupby(['source', 'canonical_cell_id'], sort=True).agg(
            mae=('absolute_error_cycles', 'mean'), mse=('squared_error_cycles', 'mean'),
            log_mse=('squared_log_error', 'mean'), windows=('sample_id', 'size'),
            lower_clip=('lower_clipped', 'mean'), above_train_max=('above_train_max', 'mean'),
            severe_fraction=('above_ten_times_train_max', 'mean'),
            overflow=('cycle_metric_overflow', 'any'))
        cell['rmsle'] = np.sqrt(cell.log_mse)
        cell['rmse'] = np.sqrt(cell.mse)
        # Do not silently omit overflow rows from mean cycle errors.
        cell.loc[cell.overflow, ['mae', 'mse', 'rmse']] = np.nan
        source = cell.groupby('source').mean(numeric_only=True)
        if cell.overflow.any():
            source.loc[cell[cell.overflow].index.get_level_values('source').unique(), ['mae', 'mse', 'rmse']] = np.nan
        by_source = {}
        for s, row in source.iterrows():
            by_source[str(s)] = dict(rmsle=float(row.rmsle), mae=float(row.mae) if np.isfinite(row.mae) else None,
                rmse=float(row.rmse) if np.isfinite(row.rmse) else None)
        metrics = dict(primary=float(source.rmsle.mean()), by_source=by_source, cells=len(cell), windows=len(frame),
            lower_clip=float(source.lower_clip.mean()), upper_clip=0.,
            max_log_prediction=float(log_prediction.max()), min_log_prediction=float(log_prediction.min()),
            max_prediction_cycles=float(prediction.max()) if np.isfinite(prediction).all() else None,
            train_truth_max=train_max, max_truth=float(r['y'].max()),
            cycle_metric_overflow_windows=int(overflow.sum()), severe_windows=int((prediction > 10 * train_max).sum()),
            above_train_max_windows=int((prediction > train_max).sum()),
            mae=float(source.mae.mean()) if not cell.overflow.any() else None)
        result[role] = metrics
        stem = f'{ident}_cpu{rung}_{role}'
        prediction_path = OUT / 'predictions' / (stem + '.parquet')
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(prediction_path, index=False, compression='zstd')
        cell_path = OUT / 'cell_metrics' / (stem + '.parquet')
        cell_path.parent.mkdir(parents=True, exist_ok=True)
        cell.reset_index().to_parquet(cell_path, index=False, compression='zstd')
        artifacts.extend([prediction_path, cell_path])
        if role == 'validation':
            worst = frame.sort_values('absolute_error_cycles', ascending=False).head(20)
            indices = worst.index.to_numpy()
            context = []
            for idx in indices:
                vector = r['x'][idx]
                excess = np.maximum(bounds['lower'] - vector, vector - bounds['upper'])
                order = np.argsort(-np.abs(vector))[:8]
                context.append(dict(sample_id=str(frame.loc[idx, 'sample_id']),
                    features=[dict(name=bounds['columns'][i], value=float(vector[i]), train_min=float(bounds['lower'][i]),
                        train_max=float(bounds['upper'][i]), range_excess=max(0., float(excess[i]))) for i in order]))
            context_path = OUT / 'extreme_context' / (stem + '.json')
            write(context_path, context)
            artifacts.append(context_path)
    # Preserve MLP weights at every rung and the final forest for replay.
    weight_path = OUT / 'weights' / (f'{ident}_cpu{rung}' + ('.pt' if family == 'MLP' else '.joblib'))
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    if not save_weights:
        weight_path = None
    elif family == 'MLP':
        torch.save(dict(state_dict=model.state_dict(), architecture=str(model), matrix_state=state), weight_path)
    elif rung == 45:
        joblib.dump(model, weight_path, compress=3)
    else:
        weight_path = None
    if weight_path is not None:
        artifacts.append(weight_path)
    result['artifacts'] = [dict(path=p.relative_to(OUT).as_posix(), sha256=sha(p), bytes=p.stat().st_size) for p in artifacts]
    result['evaluation_and_artifact_cpu_seconds'] = time.process_time() - start_cpu
    result['evaluation_and_artifact_wall_seconds'] = time.monotonic() - start_wall
    return result


def fit(run, data, state, family, config, seed, ident, bounds):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cpu = wall = 0.
    count = 0
    trajectory = []
    rungs = []
    if family == 'MLP':
        dimensions = [112, *config['widths'], 1]
        layers = []
        for i in range(len(dimensions)-1):
            layers.append(torch.nn.Linear(dimensions[i], dimensions[i+1]))
            if i < len(dimensions)-2:
                layers.append(torch.nn.ReLU())
        model = torch.nn.Sequential(*layers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        x = torch.from_numpy(data['train']['x']); y = torch.from_numpy(data['train']['z'])
        def advance():
            model.train(); order = torch.randperm(len(x)); total = 0.; n = 0
            for ids in order.split(1024):
                optimizer.zero_grad(set_to_none=True)
                loss = (model(x[ids]).flatten()-y[ids]).square().mean()
                require(bool(torch.isfinite(loss)), 'NONFINITE_MLP_LOSS')
                loss.backward(); optimizer.step(); total += float(loss.detach()) * len(ids); n += len(ids)
            return total / n
        def predict(a):
            model.eval()
            with torch.no_grad():
                return torch.cat([model(torch.from_numpy(v)).flatten() for v in np.array_split(a, max(1, math.ceil(len(a)/8192)))]).numpy()
    else:
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(**{k:v for k,v in config.items() if k!='id'}, n_estimators=0,
            criterion='squared_error', bootstrap=False, n_jobs=2, random_state=seed, warm_start=True)
        def advance():
            model.set_params(n_estimators=count+1, n_jobs=2); model.fit(data['train']['x'], data['train']['z'])
            return None
        def predict(a):
            model.set_params(n_jobs=1)
            return model.predict(a)
    for rung in [5, 15, 45]:
        while cpu < rung:
            run.guard()
            require(count < (100000 if family=='MLP' else 10000), 'SIZE_CAP')
            c, w = time.process_time(), time.monotonic()
            loss = advance()
            cpu += time.process_time()-c; wall += time.monotonic()-w; count += 1
            if family == 'MLP':
                trajectory.append(dict(epoch=count, fit_cpu_seconds=cpu, online_train_z_mse=loss))
        run.guard()
        views={}
        for arm in ['raw','train_range']:
            arm_data={key:dict(value) if isinstance(value,dict) else value for key,value in data.items()}
            if arm=='train_range':
                for role in ['train','validation','validation_sampled']:
                    arm_data[role]['x']=np.clip(data[role]['x'],bounds['lower'],bounds['upper'])
                require(np.array_equal(arm_data['train']['x'],data['train']['x']),'TRAIN_MATRIX_CHANGED')
            views[arm]=evaluate(arm_data,state,predict,model,family,ident+'_'+arm,rung,bounds,save_weights=arm=='raw')
            del arm_data
        result=dict(views=views,identical_weights_between_views=True)
        result.update(requested_cpu_seconds=rung, fit_cpu_seconds=cpu, fit_wall_seconds=wall,
            overshoot_cpu_seconds=cpu-rung, budget_matched=cpu-rung <= max(.5, .05*rung),
            epochs=count if family=='MLP' else None, trees=count if family=='Extra Trees' else None)
        write(OUT/'rungs'/f'{ident}_cpu{rung}.json', result)
        rungs.append(result)
    matched = all(r['budget_matched'] for r in rungs)
    warning = any(v['validation']['severe_windows'] or v['validation']['cycle_metric_overflow_windows'] for r in rungs for v in r['views'].values())
    return dict(status='COMPLETE' if matched else 'BUDGET_UNMATCHED', model_health='WARNING' if warning else 'NO_SEVERE_RANGE_WARNING',
        rungs=rungs, train_trajectory=trajectory, fit_cpu_seconds=cpu, fit_wall_seconds=wall)


def main():
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    require(not OUT.exists(), 'FRESH_OUTPUT_REQUIRED'); OUT.mkdir()
    try:
        import numpy as np
        import pandas as pd
        import pyarrow
        import platform
        import psutil
        import sklearn
        import torch
        p = read(RELEASE/'protocol.frozen.json')
        require(sha(RELEASE/'protocol.frozen.json')==read(RELEASE/'protocol_seal.json')['sha256'], 'PROTOCOL_SEAL')
        for name, row in p['embedded_files'].items():
            require(sha(ROOT/name)==row['sha256'], 'SOURCE_DRIFT:'+name)
        observed = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, pyarrow=pyarrow.__version__, scikit_learn=sklearn.__version__, torch=torch.__version__)
        write(OUT/'environment.json',dict(observed=observed,expected=p['expected_kaggle_cpu_environment'],platform=platform.platform(),gpu_used=False,clock='time.process_time'))
        require(observed==p['expected_kaggle_cpu_environment'], 'NUMERICAL_ENVIRONMENT_MISMATCH_NO_INSTALL')
        torch.set_num_threads(2); torch.set_num_interop_threads(1); pyarrow.set_cpu_count(2); pyarrow.set_io_thread_count(2)
        for name in ['protocol.frozen.json', 'protocol_seal.json']:
            shutil.copyfile(RELEASE/name, OUT/name)
        bootstrap = load('robustness_mount',ROOT/'kaggle_bootstrap.py')
        bundle = bootstrap.resolve_bundle('/kaggle/input',ROOT)
        for name,row in p['dataset_inputs'].items():
            require(sha(bundle/name)==row['sha256'],'DATA_DRIFT:'+name)
        old = load('robustness_reader',ROOT/'protocol_and_provenance/neural_classical_fairness_v1/run_shared_input_diagnostic.py')
        curve = load('robustness_curve',ROOT/'protocol_and_provenance/neural_classical_fairness_v1/run_shared_input_budget_curves_v3.py')
        old.BUNDLE=bundle;old.OUT=OUT;old.write=write;curve.OUT=OUT;curve.PARENT=ROOT/'parent_matrices';curve.write=write
        class Run(old.Run):
            def __init__(self):
                self.p=p;self.started=time.monotonic();self.access=[];self.phase='setup'
                members=pd.read_parquet(bundle/'phase2/battery_phase2_fold_membership.parquet')
                members=members[members.evaluation_mode.eq('LODO')&members.outer_entity.eq('CALB')].copy()
                require(members.canonical_cell_id.is_unique,'DUPLICATE_MEMBERSHIP')
                require((members.groupby('duplicate_family_id').split.nunique()==1).all(),'OUTER_FAMILY_OVERLAP')
                self.members=members;self.dev=members[members.split.eq('development')];self.roles={}
                config=read(bundle/'inner_config.frozen.json')
                for fold,block in enumerate(p['source_blocks']):
                    val=self.dev[self.dev.source.isin(block)];train=self.dev[~self.dev.source.isin(block)]
                    train=train[~train.duplicate_family_id.isin(val.duplicate_family_id)]
                    tr=set(train.canonical_cell_id);va=set(val.canonical_cell_id)
                    expected=next(r for r in config['inner_contexts'] if r['source']=='CALB' and r['inner_fold']==fold)
                    require(old.digest_ids(tr)==expected['train_cell_hash'] and old.digest_ids(va)==expected['validation_cell_hash'],'INNER_ROLE_HASH')
                    require(not tr&va and not set(train.duplicate_family_id)&set(val.duplicate_family_id),'INNER_OVERLAP')
                    require('CALB' not in set(train.source)|set(val.source),'CALB_IN_DEV')
                    self.roles[fold]=dict(train=tr,validation=va)
                write(OUT/'roles.json',{str(k):{r:sorted(v) for r,v in row.items()} for k,row in self.roles.items()})
            def check_protected(self):
                for name,row in p['dataset_inputs'].items():require(sha(bundle/name)==row['sha256'],'MOUNTED_INPUT_CHANGED')
                for name,row in p['embedded_files'].items():require(sha(ROOT/name)==row['sha256'],'SOURCE_CHANGED')
            def guard(self):
                begin=time.monotonic()
                while psutil.virtual_memory().available<.75*2**30:
                    require(time.monotonic()-begin<900,'RAM_WAIT_LIMIT')
                    require(time.monotonic()-self.started<p['resource']['max_total_seconds'],'TOTAL_TIME_CAP')
                    time.sleep(5)
                super().guard()
        run=Run()
        write(OUT/'started.json',dict(at_utc=old.now(),expected_jobs=6,outer_fits=0,post_hoc=True))
        data,state=curve.full_matrix(old,run,2,'rul90')
        lower=data['train']['x'].min(axis=0);upper=data['train']['x'].max(axis=0)
        bounds=dict(lower=lower,upper=upper,columns=p['feature_columns'])
        write(OUT/'train_feature_bounds.json',dict(lower=lower.tolist(),upper=upper.tolist(),columns=p['feature_columns'],fit_role='sampled_inner_train_only',train_sample_hash=state['train_sample_hash']))
        done=[]; checkpoint_archive()
        for seed in [2026,2027,2028]:
            families=['MLP','Extra Trees'] if seed%2==0 else ['Extra Trees','MLP']
            for family in families:
                config=p['experiment']['mlp_config' if family=='MLP' else 'et_config']
                ident=f'fold2_rul90_seed{seed}_{config["id"]}'
                run.phase=ident;run.guard()
                job=fit(run,data,state,family,config,seed,ident,bounds)
                job.update(id=ident,fold=2,target='rul90',seed=seed,family=family,candidate=config,
                    train_sample_hash=state['train_sample_hash'],validation_sample_hash=state['full_validation_sample_hash'],
                    identical_weights_between_views=True)
                path=OUT/'jobs'/f'{ident}.json';require(not path.exists(),'NO_IMPLICIT_RETRY');write(path,job)
                done.append(dict(id=ident,status=job['status'],model_health=job['model_health'],sha256=sha(path)))
                write(OUT/'completed_jobs.json',dict(completed=len(done),expected=6,jobs=done))
                checkpoint_archive();print(json.dumps(dict(id=ident,completed=len(done),expected=6,status=job['status'],model_health=job['model_health'])),flush=True)
                gc.collect()
        run.check_protected()
        summary=[]
        for family in ['MLP','Extra Trees']:
            for arm in ['raw','train_range']:
                rows=[read(OUT/'jobs'/f'{r["id"]}.json') for r in done if read(OUT/'jobs'/f'{r["id"]}.json')['family']==family]
                values=[r['rungs'][-1]['views'][arm]['validation']['primary'] for r in rows]
                mean=math.fsum(values)/3
                summary.append(dict(family=family,arm=arm,primary_rmsle=mean,seed_sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/2),
                    severe_windows_by_seed={str(r['seed']):r['rungs'][-1]['views'][arm]['validation']['severe_windows'] for r in rows},
                    cycle_metric_overflow_by_seed={str(r['seed']):r['rungs'][-1]['views'][arm]['validation']['cycle_metric_overflow_windows'] for r in rows},
                    max_prediction_by_seed={str(r['seed']):r['rungs'][-1]['views'][arm]['validation']['max_prediction_cycles'] for r in rows}))
        matched=all(r['status']=='COMPLETE' for r in done)
        result=dict(status=('BUDGET_UNMATCHED_NO_COMPARISON' if not matched else ('COMPLETE_WITH_MODEL_WARNINGS' if any(r['model_health']=='WARNING' for r in done) else 'COMPLETE_NO_SEVERE_RANGE_WARNINGS')),
            jobs=6,rungs=18,input_view_evaluations=36,budget_matched=matched,summary=summary,at_utc=old.now(),elapsed_seconds=time.monotonic()-run.started,
            fairness_resolved=False,automatic_promotion=False,new_outer_fits=0,independent_confirmation=False,
            interpretation='Paired post-hoc input-range intervention in one previously inspected fold. No guarantee that clipping resolves instability. No global family ranking.')
        write(OUT/'result.json',result)
        write(OUT/'artifact_manifest.json',dict(files=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix!='.partial' and p.name!='artifact_manifest.json']))
        print(json.dumps(result),flush=True)
    except Exception:
        write(OUT/'failure.json',dict(traceback=traceback.format_exc(),partial_artifacts_preserved=True))
        raise
    finally:
        checkpoint_archive()


if __name__=='__main__':
    main()

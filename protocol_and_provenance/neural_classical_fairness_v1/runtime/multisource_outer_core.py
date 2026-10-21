"""Development-only multisource outer fits; test predictor reads, never scoring or CQR."""
from __future__ import annotations
import json
import math
from pathlib import Path
import random
import time
from . import multisource_outer_contract as contract
from . import f2_kaggle_pilot as pilot
from . import f2_trajectory as engine
from . import local_cpu_pilots as local
from . import selection_controls as selection


def binding(path):
    return dict(bytes=Path(path).stat().st_size,sha256=local.sha(path))


class Context:
    """All table reads are role/column-filtered; profiles have a stage guard."""
    def __init__(self,bundle,sealed,directory,guard,*,replay=False):
        import pandas as pd
        self.bundle=Path(bundle).resolve();self.sealed=sealed;self.directory=Path(directory)
        self.guard=guard;self.stage='prediction' if replay else 'development_fit'
        self.log=[];self.profiles={};self.roles=sealed['outer_role_metadata']['roles']
        self.ns=pilot.definitions(self.bundle)
        self.ns['OUTER_SOURCE']=sealed['source']
        feature=json.loads((self.bundle/'phase2/feature_spec.json').read_text())
        self.feature_columns=feature['tabular_feature_columns']
        membership=self.bundle/'phase2/battery_phase2_fold_membership.parquet'
        local.require(binding(membership)==sealed['outer_role_metadata']['membership'], 'MULTI_OUTER_MEMBERSHIP_HASH')
        rows=self.ns['fold_membership']
        rows=rows.loc[rows.evaluation_mode.eq('LODO')&rows.outer_entity.eq(sealed['source'])]
        local.require(rows.canonical_cell_id.is_unique, 'MULTI_OUTER_DUPLICATE_MEMBERSHIP')
        families={}
        for r,expected in self.roles.items():
            f=rows.loc[rows.split.eq(r)]
            local.require(sorted(f.canonical_cell_id.astype(str))==expected['cell_ids'] and
                local.digest(sorted(set(f.duplicate_family_id.astype(str))))==expected['family_hash'], 'MULTI_OUTER_MEMBERSHIP_ROLE')
            families[r]=set(f.duplicate_family_id.astype(str))
        for r in families:
            for other in families:
                if r!=other:local.require(not families[r]&families[other], 'MULTI_OUTER_FAMILY_OVERLAP')
        local.require(not rows.loc[rows.split.eq('development'),'source'].eq(sealed['source']).any() and
            rows.loc[rows.split.eq('test'),'source'].eq(sealed['source']).all(), 'MULTI_OUTER_HELD_SOURCE')
        self.ns['read_for_cells']=self.read
        self.ns['resolve_cycle_shard']=self.resolve
        # Do not use a prediction iterator which encodes labels or filters masks.
        if replay:
            self.state=json.loads((self.directory/'roles_and_scalers.json').read_text())
            local.require(self.state['roles']==self.roles, 'MULTI_OUTER_REPLAY_ROLES')
        else:self.state=None

    def read(self,path,columns,cells):
        import pyarrow.dataset as ds
        self.guard.check();path=Path(path).resolve()
        local.require(path.parent==(self.bundle/'phase2').resolve(), 'MULTI_OUTER_TABLE_PATH')
        role=contract.check_read(path.name,columns,cells,self.roles,self.stage,self.feature_columns)
        cells=set(cells)
        frame=ds.dataset(path,format='parquet').to_table(columns=list(columns),
            filter=ds.field('canonical_cell_id').isin(sorted(cells))).to_pandas()
        local.require(set(frame.canonical_cell_id.astype(str))<=cells, 'MULTI_OUTER_READ_FILTER_ESCAPE')
        self.log.append(dict(table=path.name,columns=list(columns),role=role,stage=self.stage,
            cells=len(cells),cell_hash=local.digest(sorted(cells)),rows=len(frame)))
        local.write(self.directory/('replay_access_log.json' if self.state is not None else 'access_log.json'),self.log)
        return frame

    def resolve(self,cell):
        self.guard.check();cell=str(cell)
        role='development' if self.stage=='development_fit' else 'test' if self.stage=='prediction' else None
        local.require(role is not None and cell in self.roles[role]['cell_ids'], 'MULTI_OUTER_PROFILE_ROLE')
        r=self.ns['cycle_store_index'].loc[cell]
        path=(self.bundle/'phase2'/str(r['shard_path'])).resolve()
        local.require(path.is_relative_to((self.bundle/'phase2').resolve()), 'MULTI_OUTER_PROFILE_PATH')
        if cell not in self.profiles:
            local.require(local.sha(path)==str(r['sha256']).lower(), 'MULTI_OUTER_PROFILE_HASH')
            self.profiles[cell]=dict(sha256=str(r['sha256']).lower(),role=role)
            local.write(self.directory/('replay_profile_hashes.json' if self.state is not None else 'profile_hashes.json'),self.profiles)
        return path

    def train(self,cycles):
        import numpy as np
        local.require(self.stage=='development_fit' and self.state is None, 'MULTI_OUTER_REPLAY_CANNOT_FIT')
        train=set(self.roles['development']['cell_ids'])
        rows=self.ns['load_multitask_window_targets'](train)
        target=self.ns['fit_target_scaler'](rows)
        target['fit_role']='outer_development_observed_targets_only'
        local.require(all(target['targets'][t]['active'] for t in local.TARGETS), 'MULTI_OUTER_INACTIVE_TRAIN_HEAD')
        cycle=self.ns['fit_cycle_scaler'](train) if cycles else None
        if cycle is not None:cycle['fit_role']='outer_development_cells_only'
        state=dict(roles=self.roles,training_rows=len(rows),training_sample_hash=local.digest(sorted(rows.sample_id.astype(str))),
            target_scaler=target,cycle_scaler=cycle,calibration_labels_accessed=False,outer_labels_accessed=False)
        local.write(self.directory/'roles_and_scalers.json',state)
        return rows,cycle,target

    def test_rows(self):
        self.stage='prediction';cells=set(self.roles['test']['cell_ids'])
        windows=self.read(self.bundle/'phase2/window_index.parquet',contract.METADATA[:3],cells)
        meta=self.read(self.bundle/'phase2/sample_target_index.parquet',contract.METADATA,cells)
        local.require(windows.sample_id.is_unique and meta.sample_id.is_unique, 'MULTI_OUTER_DUPLICATE_TEST_SAMPLE')
        rows=windows.merge(meta,on=list(contract.METADATA[:3]),how='left',validate='one_to_one',indicator=True)
        local.require(not rows.empty and rows._merge.eq('both').all() and len(rows)==len(windows), 'MULTI_OUTER_ALL_TEST_WINDOWS_JOIN')
        rows=rows.drop(columns='_merge').sort_values(['canonical_cell_id','t_cycle'],kind='mergesort').reset_index(drop=True)
        local.require(set(rows.columns)==set(contract.METADATA), 'MULTI_OUTER_TEST_METADATA_ONLY')
        return rows

    def features(self,rows,cells):
        columns=['sample_id','canonical_cell_id',*self.feature_columns]
        x=self.read(self.bundle/'phase2/tabular_feature_index.parquet',columns,set(cells))
        local.require(x.sample_id.is_unique, 'MULTI_OUTER_DUPLICATE_TABULAR_SAMPLE')
        f=rows.merge(x,on=['sample_id','canonical_cell_id'],how='left',validate='one_to_one',indicator=True)
        local.require(len(f)==len(rows) and f._merge.eq('both').all(), 'MULTI_OUTER_FEATURE_ALIGNMENT')
        return f.drop(columns='_merge').reset_index(drop=True)


def initialize(ns,job,device):
    import numpy as np
    import torch
    effective=contract.effective_seed(job['seed'],job['source'])
    local.require(effective==job['effective_seed'], 'MULTI_OUTER_EFFECTIVE_SEED')
    random.seed(effective);np.random.seed(effective);torch.manual_seed(effective)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(effective)
    model=ns['MultitaskQuantileTCN']() if job['architecture']=='TCN' else local.load_module('nfr1_outer_gru',local.GRU).make_model()
    model=model.float().cpu().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=selection.candidate_recipes()[job['candidate']]['learning_rate'],
        weight_decay=ns['TCN_WEIGHT_DECAY'])
    local.require({id(p) for p in model.parameters()}=={id(p) for g in optimizer.param_groups for p in g['params']} and
        all(p.device==torch.device(device) for p in model.parameters()), 'MULTI_OUTER_MOVED_OPTIMIZER_BINDING')
    ns['SEED']=effective
    return model,optimizer,effective


def prediction_batches(ns,rows,cycle):
    """No target scaler, target encoder, active-mask selector or eligibility read."""
    for cell,group in rows.groupby('canonical_cell_id',sort=True):
        profile=ns['load_cycle_shard'](str(cell))
        for start in range(0,len(group),ns['TCN_PREDICTION_BATCH_SIZE']):
            batch=group.iloc[start:start+ns['TCN_PREDICTION_BATCH_SIZE']].reset_index(drop=True)
            yield ns['make_window_inputs'](profile,batch,cycle),batch


def predict(model,batches,target,guard,device):
    import numpy as np
    import pandas as pd
    import torch
    before_rng=engine.rng_signature();before_weights=engine.model_digest(model)
    model.eval();outputs=[]
    with torch.no_grad():
        for x,rows in batches:
            guard.check();values=model(torch.from_numpy(x).to(device))
            for t in local.TARGETS:
                z=values[t].detach().cpu().numpy()[:,1].astype(float)
                state=target['targets'][t];raw=z*state['std']+state['mean']
                raw=np.expm1(raw) if state['transform']=='log1p' else raw
                local.require(np.isfinite(raw).all(), 'MULTI_OUTER_NONFINITE_INVERSE')
                bounded=np.clip(raw,0,1.5) if t=='soh30' else np.maximum(raw,0)
                f=rows[list(contract.METADATA)].copy()
                f['endpoint']=t;f['y_pred_unbounded']=raw;f['y_pred']=bounded
                outputs.append(f)
    local.require(before_rng==engine.rng_signature() and before_weights==engine.model_digest(model), 'MULTI_OUTER_PREDICT_MUTATED_STATE')
    local.require(outputs, 'MULTI_OUTER_NO_PREDICTIONS')
    return pd.concat(outputs,ignore_index=True).sort_values(['endpoint','canonical_cell_id','t_cycle'],kind='mergesort').reset_index(drop=True)


def save_prediction(path,frame,rows,targets):
    import numpy as np
    local.require(set(frame.columns)==set(contract.METADATA)|{'endpoint','y_pred','y_pred_unbounded'}, 'MULTI_OUTER_PREDICTION_COLUMNS')
    for t in targets:
        f=frame.loc[frame.endpoint.eq(t)].reset_index(drop=True)
        local.require(f[list(contract.METADATA)].equals(rows[list(contract.METADATA)].reset_index(drop=True)) and
            np.isfinite(f.y_pred.to_numpy(float)).all(), 'MULTI_OUTER_PREDICTION_SUPPORT_ALIGNMENT')
    local.require(len(frame)==len(rows)*len(targets) and set(frame.endpoint)==set(targets), 'MULTI_OUTER_PREDICTION_COUNT')
    frame.to_parquet(path,index=False)
    return dict(file=Path(path).name,**binding(path),rows=len(frame),test_windows=len(rows),
        test_sample_hash=local.digest(rows.sample_id.astype(str).tolist()),endpoints=list(targets),
        point_only=True,outer_labels_accessed=False)


def train_trajectory(ns,job,iterator,guard,directory,predict_callback,device):
    """Fixed inner-selected epochs. Callback cannot mutate training RNG/weights."""
    import torch
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    model,optimizer,effective=initialize(ns,job,device)
    initial=engine.model_digest(model);history=[];exports=[]
    for epoch in range(1,job['fit_epochs']+1):
        guard.phase='outer_train_epoch_'+str(epoch)
        record=dict(epoch=epoch,**engine.train_epoch(model,optimizer,iterator,job['candidate'],epoch,guard,device),
            parameter_sha256=engine.model_digest(model))
        if epoch in contract.export_epochs(job):
            path=directory/f'checkpoint_epoch_{epoch:02d}.pt'
            torch.save(dict(state_dict=model.state_dict(),optimizer_state=optimizer.state_dict(),
                epoch=epoch,effective_seed=effective,candidate=job['candidate'],architecture=job['architecture'],
                parameter_sha256=engine.model_digest(model)),path)
            checkpoint=dict(file=path.name,epoch=epoch,**binding(path))
            before_rng=engine.rng_signature();before_weights=engine.model_digest(model)
            prediction=predict_callback(model,epoch)
            local.require(before_rng==engine.rng_signature() and before_weights==engine.model_digest(model),
                'MULTI_OUTER_EXPORT_MUTATED_TRAINING_PREFIX')
            exports.append(dict(checkpoint=checkpoint,prediction=prediction))
        history.append(record);local.write(directory/'history.json',history)
        print('[MULTI OUTER EPOCH]',job['id'],epoch,'/',job['fit_epochs'],flush=True)
    result=dict(status='MULTI_OUTER_NEURAL_FIT_COMPLETE_REPLAY_PENDING',epochs=job['fit_epochs'],learned_fits=1,
        effective_seed=effective,initial_parameter_sha256=initial,history=binding(directory/'history.json'),exports=exports,
        checkpoint_count=len(exports),point_only=True,outer_scoring=False,outer_labels_accessed=False,
        calibration_labels_accessed=False,selection_from_test=False)
    local.write(directory/'integrity.json',result)
    return result


def neural_job(bundle,sealed,job,directory,guard,device='cuda:0'):
    started=time.monotonic();ctx=Context(bundle,sealed,directory,guard)
    train,cycle,target=ctx.train(True)
    ns=ctx.ns;ns['SEED']=job['effective_seed']
    iterator=ns['MultitaskCycleWindowIterator'](train,cycle,target)
    rows_cache=[]
    def callback(model,epoch):
        if not rows_cache:rows_cache.append(ctx.test_rows())
        ctx.stage='prediction';guard.phase='outer_predict_epoch_'+str(epoch)
        frame=predict(model,prediction_batches(ns,rows_cache[0],cycle),target,guard,device)
        record=save_prediction(directory/f'prediction_epoch_{epoch:02d}.parquet',frame,rows_cache[0],local.TARGETS)
        ctx.stage='development_fit'
        return record
    result=train_trajectory(ns,job,iterator,guard,directory/'trajectory',callback,device)
    result.update(kind='neural',wall_seconds=time.monotonic()-started,peak_sampled_rss_bytes=guard.peak,
        training_rows=len(train),active_train_cells=len(iterator.groups),
        preprocessors=binding(directory/'roles_and_scalers.json'),test_windows=len(rows_cache[0]))
    local.write(directory/'result.json',result)
    return result


def classical_job(bundle,sealed,job,directory,guard):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
    started=time.monotonic();ctx=Context(bundle,sealed,directory,guard)
    train,_,_=ctx.train(False);ns=ctx.ns;t=job['target'];spec=ns['TARGET_SPECS'][t]
    eligible=ns['normalize_bool'](train[spec['mask_col']],spec['mask_col']) & np.isfinite(train[spec['y_col']].to_numpy(float))
    train=train.loc[eligible].drop(columns=spec['y_col'])
    cells=set(ctx.roles['development']['cell_ids'])
    labels=ctx.read(ctx.bundle/'phase2/sample_target_index.parquet',['sample_id','canonical_cell_id',spec['y_col']],cells)
    train=train.merge(labels,on=['sample_id','canonical_cell_id'],how='left',validate='one_to_one')
    train=ctx.features(train,cells)
    y=train[spec['y_col']].to_numpy(float)
    local.require(len(y)>0 and np.isfinite(y).all() and (y>=0).all(), 'MULTI_OUTER_CLASSICAL_TRAIN_SUPPORT')
    _,executor=local.classical_adapter()
    if job['kind']=='references':
        rows=ctx.test_rows();test=ctx.features(rows,ctx.roles['test']['cell_ids']);outputs=[]
        names=('soh_persistence','soh_train_mean','causal_linear_trend') if t=='soh30' else ('rul_train_median','rul_train_log_mean','causal_linear_trend')
        for name in names:
            guard.check();bounded=(np.full(len(test),np.clip(y.mean(),0,1.5)) if name=='soh_train_mean' else executor._naive_predict(name,t,train,test))
            f=rows.copy();f['endpoint']=t;f['y_pred_unbounded']=np.nan;f['y_pred']=bounded
            r=save_prediction(directory/(name+'.parquet'),f,rows,[t]);r.update(reference=name,raw_clipping_unavailable=True)
            outputs.append(r)
        result=dict(status='MULTI_OUTER_REFERENCES_COMPLETE',kind='references',learned_fits=0,
            reference_executions=len(outputs),predictions=outputs,raw_clipping_unavailable=True)
    else:
        family=job['family'];guard.phase='outer_classical_preprocess'
        state=executor._fit_preprocessor(train,family);state['fit_role']='outer_development_only_new_runtime_reconstruction'
        x=executor._transform_features(train,state,family)
        transformed=np.log1p(y) if t.startswith('rul') else y.copy()
        target_state=dict(transform=spec['transform'])
        if family=='ridge':
            centre=float(transformed.mean());scale=float(transformed.std())
            scale=scale if math.isfinite(scale) and scale>1e-12 else 1.
            transformed=(transformed-centre)/scale
            target_state.update(centre=centre,scale=scale,scaled_min=float(transformed.min()),scaled_max=float(transformed.max()))
        constructors=dict(ridge=Ridge,hist_gradient_boosting=HistGradientBoostingRegressor,extra_trees=ExtraTreesRegressor)
        estimator=constructors[family](**job['params']);guard.check();guard.phase='outer_classical_fit'
        fit_start=time.monotonic();estimator.fit(x,transformed);fit_seconds=time.monotonic()-fit_start
        guard.check();del x
        artifact=dict(estimator=estimator,feature_state=state,target_state=target_state,family=family,target=t,
            job=job,params=estimator.get_params(),train_sample_hash=local.digest(sorted(train.sample_id.astype(str))),
            train_cell_hash=local.digest(sorted(set(train.canonical_cell_id.astype(str)))),
            prediction_policy=local.CLASSICAL_PREDICTION_POLICIES[family],
            provenance='NEW_RUNTIME_OUTER_RECONSTRUCTION_NOT_HISTORICAL_REPLAY')
        joblib.dump(artifact,directory/'pipeline.joblib',compress=3)
        rows=ctx.test_rows();test=ctx.features(rows,ctx.roles['test']['cell_ids'])
        xv=executor._transform_features(test,state,family);guard.phase='outer_classical_predict'
        raw,bounded=local.classical_restore(local.classical_predict(estimator,xv,family),family,t,target_state)
        f=rows.copy();f['endpoint']=t;f['y_pred_unbounded']=raw;f['y_pred']=bounded
        prediction=save_prediction(directory/'prediction.parquet',f,rows,[t])
        integrity=dict(pipeline=binding(directory/'pipeline.joblib'),prediction=prediction,feature_state=state,
            target_state=target_state,params=estimator.get_params(),prediction_policy=artifact['prediction_policy'],
            preprocessors=binding(directory/'roles_and_scalers.json'),training_sample_hash=artifact['train_sample_hash'],
            training_cell_hash=artifact['train_cell_hash'])
        local.write(directory/'model.integrity.json',integrity)
        result=dict(status='MULTI_OUTER_CLASSICAL_FIT_COMPLETE_REPLAY_PENDING',kind='classical',learned_fits=1,
            training_seconds=fit_seconds,integrity=binding(directory/'model.integrity.json'),predictions=[prediction])
    result.update(training_rows=len(train),test_windows=len(rows),wall_seconds=time.monotonic()-started,
        peak_sampled_rss_bytes=guard.peak,point_only=True,outer_labels_accessed=False,
        calibration_labels_accessed=False,outer_scoring=False,selection_from_test=False)
    local.write(directory/'result.json',result)
    return result


def exact_prediction(actual,expected):
    import numpy as np
    local.require(list(actual.columns)==list(expected.columns) and actual.shape==expected.shape, 'MULTI_OUTER_REPLAY_SHAPE_COLUMNS')
    for k in actual.columns:
        if k in ('y_pred','y_pred_unbounded'):
            local.require(np.array_equal(actual[k].to_numpy(),expected[k].to_numpy(),equal_nan=True), 'MULTI_OUTER_FULL_REPLAY_DRIFT:'+k)
        else:local.require(actual[k].equals(expected[k]), 'MULTI_OUTER_FULL_REPLAY_METADATA:'+k)


def replay(bundle,sealed,job,directory,guard,device='cuda:0'):
    """Fresh process per job, NEW model for EACH epoch; full test predictions."""
    import pandas as pd
    result=json.loads((directory/'result.json').read_text());ctx=Context(bundle,sealed,directory,guard,replay=True)
    local.require(binding(directory/'roles_and_scalers.json')==result.get('preprocessors',
        json.loads((directory/'model.integrity.json').read_text())['preprocessors'] if job['kind']=='classical' else None),
        'MULTI_OUTER_REPLAY_PREPROCESSOR_HASH')
    rows=ctx.test_rows();checks=[]
    if job['kind']=='neural':
        import torch
        integrity=json.loads((directory/'trajectory/integrity.json').read_text())
        local.require(integrity['exports']==result['exports'] and integrity['epochs']==job['fit_epochs'], 'MULTI_OUTER_REPLAY_EXPORT_MAP')
        local.require([r['checkpoint']['epoch'] for r in result['exports']]==contract.export_epochs(job), 'MULTI_OUTER_REPLAY_ALL_EPOCHS')
        for r in result['exports']:
            cp=r['checkpoint'];prediction=r['prediction'];path=directory/'trajectory'/cp['file'];pp=directory/prediction['file']
            local.require(binding(path)=={k:cp[k] for k in ('bytes','sha256')} and
                binding(pp)=={k:prediction[k] for k in ('bytes','sha256')}, 'MULTI_OUTER_REPLAY_HASH_BEFORE_LOAD')
            model,_,effective=initialize(ctx.ns,job,device)
            state=torch.load(path,map_location=device,weights_only=True)
            local.require(state['epoch']==cp['epoch'] and state['effective_seed']==effective and
                state['candidate']==job['candidate'] and state['architecture']==job['architecture'], 'MULTI_OUTER_REPLAY_CHECKPOINT_ID')
            model.load_state_dict(state['state_dict'],strict=True)
            local.require(engine.model_digest(model)==state['parameter_sha256'], 'MULTI_OUTER_REPLAY_PARAMETER_HASH')
            actual=predict(model,prediction_batches(ctx.ns,rows,ctx.state['cycle_scaler']),ctx.state['target_scaler'],guard,device)
            exact_prediction(actual,pd.read_parquet(pp));checks.append(dict(epoch=cp['epoch'],full_rows=len(actual),exact=True))
            del state,model,actual
    else:
        import joblib
        integrity=json.loads((directory/'model.integrity.json').read_text())
        local.require(binding(directory/'model.integrity.json')==result['integrity'] and
            binding(directory/'pipeline.joblib')==integrity['pipeline'] and
            binding(directory/'prediction.parquet')=={k:integrity['prediction'][k] for k in ('bytes','sha256')}, 'MULTI_OUTER_REPLAY_HASH_BEFORE_LOAD')
        artifact=joblib.load(directory/'pipeline.joblib')  # Own hash-verified artifact, never arbitrary returned pickle.
        local.require(artifact['job']==job and artifact['params']==artifact['estimator'].get_params()==integrity['params'] and
            artifact['feature_state']==integrity['feature_state'] and artifact['target_state']==integrity['target_state'] and
            artifact['prediction_policy']==integrity['prediction_policy']==local.CLASSICAL_PREDICTION_POLICIES[job['family']],
            'MULTI_OUTER_REPLAY_PIPELINE_STATE')
        test=ctx.features(rows,ctx.roles['test']['cell_ids']);_,executor=local.classical_adapter()
        x=executor._transform_features(test,artifact['feature_state'],job['family'])
        raw,bounded=local.classical_restore(local.classical_predict(artifact['estimator'],x,job['family']),job['family'],job['target'],artifact['target_state'])
        actual=rows.copy();actual['endpoint']=job['target'];actual['y_pred_unbounded']=raw;actual['y_pred']=bounded
        exact_prediction(actual,pd.read_parquet(directory/'prediction.parquet'));checks.append(dict(full_rows=len(actual),exact=True))
    local.require(all(r['role']=='test' and r['stage']=='prediction' for r in ctx.log), 'MULTI_OUTER_REPLAY_LABEL_READ')
    receipt=dict(status='FRESH_PROCESS_FULL_PREDICTION_REPLAY_EXACT_PASS',job_id=job['id'],checks=checks,
        full_prediction_replay=True,all_exported_checkpoints=True,outer_labels_accessed=False,
        calibration_labels_accessed=False,outer_scoring=False)
    local.write(directory/'replay.json',receipt)
    return receipt


def synthetic_qa(ns,directory,device):
    """Actual outer loop and outer seed: prefix2 short vs3 long, all 6 recipes."""
    import numpy as np
    import torch
    from types import SimpleNamespace
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    checks=[];initial={}
    for a in ('TCN','GRU'):
        for c in selection.PRIORITY:
            effective=contract.effective_seed(2026,ns['OUTER_SOURCE'])
            class Iterator:
                def iter_cell_balanced_microbatches(self,epoch):
                    rng=np.random.default_rng(effective+1009*epoch)
                    for _ in range(8):yield rng.normal(size=(2,30,96)).astype(np.float32),np.zeros((2,3),np.float32),np.ones((2,3),bool),None
            guard=SimpleNamespace(check=lambda:None,phase='synthetic')
            captures={}
            for length in (3,2):
                job=dict(id=f'synthetic.{a}.{c}.{length}',source=ns['OUTER_SOURCE'],architecture=a,candidate=c,seed=2026,effective_seed=effective,
                    fit_epochs=length,checkpoint_exports=[dict(epoch=e) for e in range(1,length+1)])
                def callback(model,epoch):
                    model.eval()
                    with torch.no_grad():out=model(torch.ones((2,30,96),device=device))
                    captures[(length,epoch)]={t:v.detach().cpu().numpy().copy() for t,v in out.items()}
                    return dict(synthetic=True)
                r=train_trajectory(ns,job,Iterator(),guard,directory/f'{a}{c}_{length}',callback,device)
                initial.setdefault(a,set()).add(r['initial_parameter_sha256'])
            left=torch.load(directory/f'{a}{c}_3/checkpoint_epoch_02.pt',map_location='cpu',weights_only=True)
            right=torch.load(directory/f'{a}{c}_2/checkpoint_epoch_02.pt',map_location='cpu',weights_only=True)
            def same(x,y):
                if torch.is_tensor(x):return torch.is_tensor(y) and torch.equal(x,y)
                if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
                if isinstance(x,(tuple,list)):return type(x) is type(y) and len(x)==len(y) and all(same(u,v) for u,v in zip(x,y))
                return x==y
            local.require(same(left,right) and all(np.array_equal(captures[(3,2)][t],captures[(2,2)][t]) for t in local.TARGETS),
                'MULTI_OUTER_SYNTHETIC_PREFIX')
            # Epoch1 must actually reload epoch1, not silently use the final model.
            for epoch in (1,2,3):
                model,_,_=initialize(ns,job,device)
                cp=torch.load(directory/f'{a}{c}_3/checkpoint_epoch_{epoch:02d}.pt',map_location=device,weights_only=True)
                model.load_state_dict(cp['state_dict'],strict=True);model.eval()
                with torch.no_grad():out=model(torch.ones((2,30,96),device=device))
                local.require(all(np.array_equal(out[t].detach().cpu().numpy(),captures[(3,epoch)][t]) for t in local.TARGETS),
                    'MULTI_OUTER_SYNTHETIC_ALL_CHECKPOINT_RELOAD')
            checks.append(dict(architecture=a,candidate=c,prefix_epoch=2,all_checkpoint_reload=True,exact=True))
    local.require(all(len(v)==1 for v in initial.values()), 'MULTI_OUTER_SYNTHETIC_MATCHED_INITIALIZATION')
    r=dict(status='ACTUAL_OUTER_LOOP_PREFIX_ALL_CHECKPOINT_SYNTHETIC_PASS',checks=checks,synthetic_fits=12,
        synthetic_epoch_passes=30,battery_fits=0,device=str(device))
    local.write(directory/'summary.json',r);return r

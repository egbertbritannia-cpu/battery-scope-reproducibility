"""Four-arm train-only preprocessing diagnostic, refit on one Kaggle CPU host."""
from pathlib import Path
import gc,hashlib,importlib.util,json,math,os,random,shutil,sys,time,traceback,zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'quantile_support_results'
RELEASE=ROOT/'delivery'
ARCHIVE=Path('/kaggle/working/batteryscope_shared_input_quantile_support_cpu_v1_results.zip')
PREFIX='batteryscope_shared_input_quantile_support_cpu_v1/'


def require(ok,message):
    if not ok:raise RuntimeError('QUANTILE_SUPPORT:'+message)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial')
    with q.open('w',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    os.replace(q,p)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def checkpoint():
    q=ARCHIVE.with_suffix('.partial')
    with zipfile.ZipFile(q,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix!='.partial':z.write(p,PREFIX+p.relative_to(OUT).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix in {'.parquet','.pt','.joblib'} else zipfile.ZIP_DEFLATED)
    os.replace(q,ARCHIVE)


def prepare_arm(raw_data,raw_features,parent_state,columns,arm,rule):
    import numpy as np
    import warnings
    slope_indices=[i for i,c in enumerate(columns) if c.endswith(('__theil_sen_cycle','__slope_efc'))]
    require(len(slope_indices)==20,'SLOPE_COLUMN_COUNT')
    observation_index=columns.index('w30_observed_fraction')
    arrays={role:raw_features[role].copy() for role in ['train','validation']}
    counts={};short={}
    for role,a in arrays.items():
        fraction=a[:,observation_index]
        require(np.isfinite(fraction).all() and ((fraction>=0)&(fraction<=1)).all(),'WINDOW_OBSERVATION_FRACTION')
        n=np.rint(fraction*30).astype('int64')
        require(np.max(np.abs(n-fraction*30))<1e-4,'NONINTEGER_WINDOW_SUPPORT')
        short[role]=n<rule['minimum_observations']
        counts[role]=dict(rows=len(a),short_window_rows=int(short[role].sum()))
        a[~np.isfinite(a)]=np.nan
        if arm in ['support','combined']:
            a[np.ix_(short[role],slope_indices)]=np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(arrays['train'],axis=0)
        quantiles=np.nanquantile(arrays['train'],rule['quantiles'],axis=0,method='linear')
    median=np.where(np.isfinite(median),median,0.)
    lo=np.where(np.isfinite(quantiles[0]),quantiles[0],median)
    hi=np.where(np.isfinite(quantiles[1]),quantiles[1],median)
    require((lo<=hi).all(),'QUANTILE_ORDER')
    imputed={role:np.where(np.isfinite(a),a,median) for role,a in arrays.items()}
    quantile_enabled=arm in ['quantile','combined']
    for role,a in imputed.items():
        counts[role]['quantile_tail_entries']=int(((a<lo)|(a>hi)).sum()) if quantile_enabled else 0
        counts[role]['quantile_tail_rows']=int(np.any((a<lo)|(a>hi),axis=1).sum()) if quantile_enabled else 0
        if quantile_enabled:np.clip(a,lo,hi,out=a)
    mean=imputed['train'].mean(0);std=imputed['train'].std(0);std=np.where(std>1e-8,std,1.)
    if arm=='raw':
        require(np.allclose(median,parent_state['feature_median'],rtol=1e-12,atol=1e-12),'RAW_MEDIAN_PARITY')
        require(np.allclose(mean,parent_state['feature_mean'],rtol=1e-12,atol=1e-12),'RAW_MEAN_PARITY')
        require(np.allclose(std,parent_state['feature_std'],rtol=1e-12,atol=1e-12),'RAW_STD_PARITY')
        # Retain byte-identical baseline transforms despite reduction-layout rounding.
        mean=np.asarray(parent_state['feature_mean']);std=np.asarray(parent_state['feature_std'])
    data={role:dict(raw_data[role]) for role in ['train','validation']}
    for role,a in imputed.items():
        data[role]['x']=((a-mean)/std).astype('float32')
        require(np.isfinite(data[role]['x']).all(),'NONFINITE_TRANSFORM')
    if arm=='raw':
        require(np.array_equal(data['train']['x'],raw_data['train']['x']),'RAW_TRAIN_PARENT_PARITY')
        require(np.array_equal(data['validation']['x'],raw_data['validation']['x']),'RAW_VALIDATION_PARENT_PARITY')
    state=dict(parent_state,feature_median=median.tolist(),feature_mean=mean.tolist(),feature_std=std.tolist(),
        preprocessing_arm=arm,quantile_enabled=quantile_enabled,support_guard_enabled=arm in ['support','combined'],
        quantile_probabilities=rule['quantiles'],quantile_raw_lower=lo.tolist(),quantile_raw_upper=hi.tolist(),
        minimum_observations=rule['minimum_observations'],slope_columns=[columns[i] for i in slope_indices],counts=counts,
        fit_role='sampled_inner_train_only',target_scaler_unchanged=True,
        transform_hashes={role:hashlib.sha256(data[role]['x'].tobytes()).hexdigest() for role in ['train','validation']})
    bounds=dict(lower=data['train']['x'].min(0),upper=data['train']['x'].max(0),columns=columns)
    write(OUT/'matrices'/f'{arm}_fold2_rul90.json',state)
    return data,state,bounds


def fit(run,common,data,state,family,config,seed,ident,bounds):
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    cpu=wall=0.;count=0;trace=[];rungs=[]
    if family=='MLP':
        dimensions=[112,*config['widths'],1];layers=[]
        for i in range(len(dimensions)-1):
            layers.append(torch.nn.Linear(dimensions[i],dimensions[i+1]))
            if i<len(dimensions)-2:layers.append(torch.nn.ReLU())
        model=torch.nn.Sequential(*layers);optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        x=torch.from_numpy(data['train']['x']);y=torch.from_numpy(data['train']['z'])
        def advance():
            model.train();order=torch.randperm(len(x));total=0.;n=0
            for ids in order.split(1024):
                optimizer.zero_grad(set_to_none=True);loss=(model(x[ids]).flatten()-y[ids]).square().mean()
                require(bool(torch.isfinite(loss)),'NONFINITE_MLP_LOSS');loss.backward();optimizer.step();total+=float(loss.detach())*len(ids);n+=len(ids)
            return total/n
        def predict(a):
            model.eval()
            with torch.no_grad():return torch.cat([model(torch.from_numpy(v)).flatten() for v in np.array_split(a,max(1,math.ceil(len(a)/8192)))]).numpy()
    else:
        from sklearn.ensemble import ExtraTreesRegressor
        model=ExtraTreesRegressor(**{k:v for k,v in config.items() if k!='id'},n_estimators=0,criterion='squared_error',bootstrap=False,n_jobs=2,random_state=seed,warm_start=True)
        def advance():model.set_params(n_estimators=count+1,n_jobs=2);model.fit(data['train']['x'],data['train']['z'])
        def predict(a):model.set_params(n_jobs=1);return model.predict(a)
    for rung in [5,15,45]:
        while cpu<rung:
            run.guard();require(count<(100000 if family=='MLP' else 10000),'SIZE_CAP')
            c,w=time.process_time(),time.monotonic();loss=advance();cpu+=time.process_time()-c;wall+=time.monotonic()-w;count+=1
            if family=='MLP':trace.append(dict(epoch=count,fit_cpu_seconds=cpu,online_train_z_mse=loss))
        run.guard();result=common.evaluate(data,state,predict,model,family,ident,rung,bounds)
        result.update(requested_cpu_seconds=rung,fit_cpu_seconds=cpu,fit_wall_seconds=wall,overshoot_cpu_seconds=cpu-rung,
            budget_matched=cpu-rung<=max(.5,.05*rung),epochs=count if family=='MLP' else None,trees=count if family=='Extra Trees' else None)
        write(OUT/'rungs'/f'{ident}_cpu{rung}.json',result);rungs.append(result)
    matched=all(r['budget_matched'] for r in rungs)
    warning=any(r['validation']['severe_windows'] or r['validation']['cycle_metric_overflow_windows'] for r in rungs)
    return dict(status='COMPLETE' if matched else 'BUDGET_UNMATCHED',model_health='WARNING' if warning else 'NO_SEVERE_RANGE_WARNING',rungs=rungs,train_trajectory=trace,fit_cpu_seconds=cpu,fit_wall_seconds=wall)


def main():
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    require(not OUT.exists(),'FRESH_OUTPUT_REQUIRED');OUT.mkdir()
    try:
        import numpy as np
        import pandas as pd
        import platform,psutil,pyarrow,sklearn,torch
        p=read(RELEASE/'protocol.frozen.json');require(sha(RELEASE/'protocol.frozen.json')==read(RELEASE/'protocol_seal.json')['sha256'],'PROTOCOL_SEAL')
        for name,row in p['embedded_files'].items():require(sha(ROOT/name)==row['sha256'],'SOURCE_DRIFT:'+name)
        observed=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,pyarrow=pyarrow.__version__,scikit_learn=sklearn.__version__,torch=torch.__version__)
        write(OUT/'environment.json',dict(observed=observed,expected=p['expected_kaggle_cpu_environment'],platform=platform.platform(),gpu_used=False,clock='time.process_time'))
        require(observed==p['expected_kaggle_cpu_environment'],'ENVIRONMENT_MISMATCH_NO_INSTALL')
        torch.set_num_threads(2);torch.set_num_interop_threads(1);pyarrow.set_cpu_count(2);pyarrow.set_io_thread_count(2)
        for name in ['protocol.frozen.json','protocol_seal.json']:shutil.copyfile(RELEASE/name,OUT/name)
        mount=load('quantile_support_mount',ROOT/'kaggle_bootstrap.py');bundle=mount.resolve_bundle('/kaggle/input',ROOT)
        for name,row in p['dataset_inputs'].items():require(sha(bundle/name)==row['sha256'],'DATA_DRIFT:'+name)
        code=ROOT/'protocol_and_provenance/neural_classical_fairness_v1'
        old=load('quantile_support_reader',code/'run_shared_input_diagnostic.py')
        curve=load('quantile_support_parent_matrix',code/'run_shared_input_budget_curves_v3.py')
        common=load('quantile_support_artifacts',code/'runtime/shared_input_robustness_cpu.py')
        old.BUNDLE=bundle;old.OUT=OUT;old.write=write;curve.OUT=OUT;curve.PARENT=ROOT/'parent_matrices';curve.write=write;common.OUT=OUT
        class Run(old.Run):
            def __init__(self):
                self.p=p;self.started=time.monotonic();self.access=[];self.phase='setup';self.roles={}
                members=pd.read_parquet(bundle/'phase2/battery_phase2_fold_membership.parquet')
                members=members[members.evaluation_mode.eq('LODO')&members.outer_entity.eq('CALB')].copy()
                require(members.canonical_cell_id.is_unique,'MEMBERSHIP_DUPLICATES');require((members.groupby('duplicate_family_id').split.nunique()==1).all(),'OUTER_FAMILY_OVERLAP')
                self.members=members;self.dev=members[members.split.eq('development')]
                config=read(bundle/'inner_config.frozen.json')
                for fold,block in enumerate(p['source_blocks']):
                    val=self.dev[self.dev.source.isin(block)];train=self.dev[~self.dev.source.isin(block)];train=train[~train.duplicate_family_id.isin(val.duplicate_family_id)]
                    tr=set(train.canonical_cell_id);va=set(val.canonical_cell_id);expected=next(r for r in config['inner_contexts'] if r['source']=='CALB' and r['inner_fold']==fold)
                    require(old.digest_ids(tr)==expected['train_cell_hash'] and old.digest_ids(va)==expected['validation_cell_hash'],'ROLE_HASH')
                    require(not tr&va and not set(train.duplicate_family_id)&set(val.duplicate_family_id),'INNER_OVERLAP')
                    require('CALB' not in set(train.source)|set(val.source),'CALB_IN_DEV');self.roles[fold]=dict(train=tr,validation=va)
                write(OUT/'roles.json',{str(k):{r:sorted(v) for r,v in row.items()} for k,row in self.roles.items()})
            def check_protected(self):
                for name,row in p['dataset_inputs'].items():require(sha(bundle/name)==row['sha256'],'MOUNTED_INPUT_CHANGED')
                for name,row in p['embedded_files'].items():require(sha(ROOT/name)==row['sha256'],'SOURCE_CHANGED')
            def guard(self):
                begin=time.monotonic()
                while psutil.virtual_memory().available<.75*2**30:
                    require(time.monotonic()-begin<900,'RAM_WAIT_LIMIT');require(time.monotonic()-self.started<p['resource']['max_total_seconds'],'TOTAL_TIME_CAP');time.sleep(5)
                super().guard()
        run=Run();write(OUT/'started.json',dict(at_utc=old.now(),expected_jobs=24,expected_rungs=72,outer_fits=0,post_hoc=True))
        data,parent_state=curve.full_matrix(old,run,2,'rul90')
        # Reload original feature values instead of reversing rounded float32 parent transforms.
        features={}
        for role in ['train','validation']:
            meta=data[role]['meta'];f=run.table('tabular_feature_index.parquet',['sample_id','canonical_cell_id',*p['feature_columns']],set(meta.canonical_cell_id),2,role,meta.sample_id.tolist())
            merged=meta.merge(f,on=['sample_id','canonical_cell_id'],how='left',validate='one_to_one',indicator=True);require(merged._merge.eq('both').all(),'RAW_FEATURE_MISSING')
            features[role]=merged[p['feature_columns']].to_numpy('float64');del f,merged
        done=[];checkpoint()
        # Retain all four predefined arms; no best-arm or best-seed selection.
        for arm in p['experiment']['arms']:
            run.phase='prepare_'+arm;run.guard()
            arm_data,state,bounds=prepare_arm(data,features,parent_state,p['feature_columns'],arm,p['preprocessing'])
            for seed in [2026,2027,2028]:
                for family in (['MLP','Extra Trees'] if seed%2==0 else ['Extra Trees','MLP']):
                    config=p['experiment']['mlp_config' if family=='MLP' else 'et_config'];ident=f'fold2_rul90_seed{seed}_{config["id"]}_{arm}'
                    run.phase=ident;run.guard();job=fit(run,common,arm_data,state,family,config,seed,ident,bounds)
                    job.update(id=ident,seed=seed,family=family,arm=arm,candidate=config,fold=2,target='rul90',
                        train_sample_hash=state['train_sample_hash'],validation_sample_hash=state['full_validation_sample_hash'],
                        matrix_state_sha256=sha(OUT/'matrices'/f'{arm}_fold2_rul90.json'))
                    path=OUT/'jobs'/f'{ident}.json';require(not path.exists(),'NO_IMPLICIT_RETRY');write(path,job)
                    done.append(dict(id=ident,status=job['status'],model_health=job['model_health'],sha256=sha(path)))
                    write(OUT/'completed_jobs.json',dict(completed=len(done),expected=24,jobs=done));checkpoint()
                    print(json.dumps(dict(id=ident,completed=len(done),expected=24,status=job['status'],model_health=job['model_health'])),flush=True);gc.collect()
            del arm_data;gc.collect()
        run.check_protected();jobs=[read(OUT/'jobs'/f'{r["id"]}.json') for r in done];summary=[]
        for arm in p['experiment']['arms']:
            for family in ['MLP','Extra Trees']:
                rows=[r for r in jobs if r['arm']==arm and r['family']==family];require(len(rows)==3,'SUMMARY_SEED_COUNT')
                metrics=[r['rungs'][-1]['validation'] for r in rows];values=[r['primary'] for r in metrics];mean=math.fsum(values)/3
                summary.append(dict(arm=arm,family=family,primary_rmsle=mean,seed_sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/2),
                    severe_windows_by_seed={str(r['seed']):r['rungs'][-1]['validation']['severe_windows'] for r in rows},
                    max_prediction_by_seed={str(r['seed']):r['rungs'][-1]['validation']['max_prediction_cycles'] for r in rows},
                    cycle_metric_overflow_by_seed={str(r['seed']):r['rungs'][-1]['validation']['cycle_metric_overflow_windows'] for r in rows},
                    mae_by_seed={str(r['seed']):r['rungs'][-1]['validation']['mae'] for r in rows}))
        matched=all(r['status']=='COMPLETE' for r in done)
        result=dict(status='BUDGET_UNMATCHED_NO_COMPARISON' if not matched else ('COMPLETE_WITH_MODEL_WARNINGS' if any(r['model_health']=='WARNING' for r in done) else 'COMPLETE_NO_SEVERE_RANGE_WARNINGS'),
            jobs=24,rungs=72,budget_matched=matched,primary_arm='combined',primary_rung=45,summary=summary,at_utc=old.now(),elapsed_seconds=time.monotonic()-run.started,
            fairness_resolved=False,automatic_promotion=False,new_outer_fits=0,independent_confirmation=False,
            interpretation='Four fixed train-only preprocessing arms after inspected failures in one inner fold. Descriptive intervention, no best-arm selection or global model ranking.')
        write(OUT/'result.json',result)
        write(OUT/'artifact_manifest.json',dict(files=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix!='.partial' and p.name!='artifact_manifest.json']))
        print(json.dumps(result),flush=True)
    except Exception:
        write(OUT/'failure.json',dict(traceback=traceback.format_exc(),partial_artifacts_preserved=True));raise
    finally:checkpoint()


if __name__=='__main__':main()

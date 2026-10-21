"""Six matched new-representation GPU pilot fits. Inner only; no promotion."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1.runtime import capacity_scale_repair as repair
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_kaggle_pilot as pilot
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_trajectory as engine
from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
from protocol_and_provenance.neural_classical_fairness_v1.runtime import kaggle_classical_remaining as inherited
MANIFEST='capacity_repair_delivery.json'
RUN_NAME='batteryscope_nfr1_capacity_repair_pilot_v1'


def jobs():
    return [dict(id=f'repair.{a}.{r}.CALB.0.2026',kind='neural',source='CALB',inner_fold=0,
        seed=2026,architecture=a,candidate='P1',representation=r) for a in ('TCN','GRU') for r in repair.RECIPES]


def manifest():
    m=json.loads((ROOT/MANIFEST).read_text())
    local.require(m['schema']=='capacity-repair-pilot-v1' and m['jobs']==jobs() and m['epochs']==3 and
        m['normalized_channels']==list(repair.NORMALIZED) and m['standardized_cap']==5 and
        m['outer_scoring'] is m['candidate_selected'] is False,'CAPACITY_REPAIR_MANIFEST')
    for n,b in m['files'].items():
        p=(ROOT/n).resolve();local.require(p.is_relative_to(ROOT) and p.is_file() and
            dict(bytes=p.stat().st_size,sha256=local.sha(p))==b,'CAPACITY_REPAIR_CODE:'+n)
    return m


def synthetic_gate(ns,directory,device):
    import numpy as np
    import pandas as pd
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    channels=ns['VALUE_CHANNELS'];masks=ns['VALUE_MASK_COLUMNS']
    frame=pd.DataFrame({c:np.full(40,1.,np.float32) for c in channels}|
        {c:np.ones(40,np.uint8) for c in masks})
    frame['cycle_index']=np.arange(1,41,dtype=np.float32);frame['discharge_ah']=60.
    frame['discharge_ah_over_q_ref']=1.
    for c in repair.NORMALIZED:frame[c]=60.
    samples=pd.DataFrame(dict(t_cycle=[30,35],sample_id=['synthetic0','synthetic1'],canonical_cell_id=['c0','c1'],
        source=['S','S'],soh_t_plus_30=[.8,.8],rul90_cycles=[3.,3.],rul80_cycles=[3.,3.]))
    scaler=dict(mean=[0.]*48,std=[1.]*48)
    target=dict(targets={t:dict(mean=.8 if t=='soh30' else np.log(4),std=1.,transform='identity' if t=='soh30' else 'log1p') for t in local.TARGETS})
    checks=[]
    for architecture in ('TCN','GRU'):
        for representation in repair.RECIPES:
            f=repair.transform(frame,representation)
            x=repair.bound_inputs(ns['make_window_inputs'](f,samples,scaler),representation)
            class Iterator:
                def iter_cell_balanced_microbatches(self,epoch):
                    for _ in range(8):yield x,np.zeros((2,3),np.float32),np.ones((2,3),bool),samples
                def iter_prediction_batches(self):yield x,np.zeros((2,3),np.float32),np.ones((2,3),bool),samples
            guard=type('SyntheticGuard',(),dict(check=lambda self:None,phase='synthetic'))()
            job=dict(id=architecture+representation,architecture=architecture,candidate='P1',source='CALB',inner_fold=0,seed=2026)
            d=directory/job['id']
            result=engine.run_trajectory(ns,job,Iterator(),Iterator(),target,
                dict(all_rows=2,soh30=2,rul90=2,rul80=2),guard,d,device=device,epochs=2)
            engine.replay(ns,job,d,guard,device)
            checks.append(dict(architecture=architecture,representation=representation,epochs=result['epochs'],exact_fixture_replay=True))
    result=dict(status='CAPACITY_REPAIR_ACTUAL_ENGINE_SYNTHETIC_PASS',checks=checks,synthetic_fits=6,
                synthetic_epochs=12,battery_fits=0,device=device)
    local.write(directory/'summary.json',result);return result


def worker(mode,bundle,run,job_id):
    import numpy as np
    m=manifest();pilot.configure(bundle)
    actual,mismatches=pilot.environment(m)
    local.require(not mismatches and actual==json.loads((run/'environment.lock.json').read_text()),'CAPACITY_REPAIR_ENVIRONMENT')
    session=json.loads((run/'session_started.json').read_text())
    local.require(local.sha(ROOT/MANIFEST)==session['delivery_sha256'] and
        local.sha(ROOT/'protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json')==session['bindings_sha256'],'CAPACITY_REPAIR_SESSION_SEAL')
    inherited.input_checks(m,bundle)
    if mode=='qa':synthetic_gate(pilot.definitions(bundle),run/'synthetic_qa','cuda:0');return
    job=next(j for j in m['jobs'] if j['id']==job_id);directory=run/'jobs'/job_id
    guard=pilot.Guard(run,directory,seconds=300 if mode=='replay' else 3600)
    if mode=='replay':
        # Fixture inputs are already in the new representation; no scaler refit here.
        engine.replay(pilot.definitions(bundle),job,directory/'trajectory',guard,'cuda:0');return
    directory.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    local.write(directory/'started.json',dict(job=job,delivery_sha256=local.sha(ROOT/MANIFEST)))
    ns,train,val,_,target,_,train_cells,val_cells=local.verified_namespace(job,guard,directory,cycles=True)
    adapter=repair.attach(ns,job['representation'])
    # Independent explicit train-only read allowlist for the NEW scaler refit.
    transformed_loader=ns['load_cycle_shard']
    def fit_load(cell):
        local.require(cell in train_cells,'CAPACITY_REPAIR_NEW_SCALER_VALIDATION_READ')
        return transformed_loader(cell)
    ns['load_cycle_shard']=fit_load
    cycle=ns['fit_cycle_scaler'](train_cells)
    ns['load_cycle_shard']=transformed_loader
    cycle.update(fit_role='capacity_repair_inner_train_only',feature_contract=repair.FEATURE_CONTRACT,
                 representation=job['representation'])
    roles=json.loads((directory/'roles.json').read_text())
    roles.update(cycle_scaler=cycle,feature_adapter=adapter,old_preprocessing_pass_not_reused=True)
    local.write(directory/'roles.json',roles)
    support=dict(all_rows=len(val))
    for t in local.TARGETS:
        spec=ns['TARGET_SPECS'][t]
        support[t]=int((ns['normalize_bool'](val[spec['mask_col']],spec['mask_col'])&np.isfinite(val[spec['y_col']].to_numpy(float))).sum())
        local.require(support[t]>0 and target['targets'][t]['active'],'CAPACITY_REPAIR_EMPTY_INNER_HEAD')
    local.write(directory/'expected_support.json',support)
    preprocessing_seconds=time.monotonic()-start
    result=engine.run_trajectory(ns,job,ns['MultitaskCycleWindowIterator'](train,cycle,target),
        ns['MultitaskCycleWindowIterator'](val,cycle,target),target,support,guard,directory/'trajectory',device='cuda:0',epochs=3)
    result.update(representation=job['representation'],feature_adapter=adapter,preprocessing_seconds=preprocessing_seconds,
        wall_seconds=time.monotonic()-start,peak_sampled_rss_bytes=guard.peak,production=False,candidate_selected=False,
        outer_scoring=False,fairness_closed=False)
    local.write(directory/'result.json',result)


def child(mode,bundle,run,job_id,deadline):
    import psutil
    cap=900 if mode=='qa' else 300 if mode=='replay' else 3600
    started=time.monotonic();end=min(deadline,started+cap);peak=0
    with (run/((job_id or 'synthetic')+'.'+mode+'.log')).open('x',encoding='utf-8') as log:
        command=[sys.executable,'-B','-X','utf8',str(Path(__file__)),mode,'--bundle',str(bundle),'--run',str(run)]
        if job_id:command+=['--job-id',job_id]
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1'),
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        owner=psutil.Process(process.pid);birth=owner.create_time()
        try:
            while process.poll() is None:
                rss=0
                try:owned=[owner]+owner.children(recursive=True)
                except psutil.NoSuchProcess:process.wait(timeout=10);break
                for p in owned:
                    try:rss+=p.memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss);resources=inherited.resources(run)
                output_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
                local.write(run/'heartbeat.json',dict(mode=mode,job_id=job_id,worker_pid=process.pid,
                    peak_tree_rss_bytes=peak,resource=resources,output_bytes=output_bytes))
                local.require(not (run/'STOP').exists() and time.monotonic()<end,'CAPACITY_REPAIR_STOP_TIME')
                local.require(rss<=4.5*2**30 and resources['available_ram_bytes']>=1.5*2**30 and
                    resources['free_disk_bytes']>=8*2**30 and output_bytes<=8*2**30,'CAPACITY_REPAIR_RESOURCE')
                time.sleep(2)
            local.require(process.returncode==0,'CAPACITY_REPAIR_WORKER_FAILED:'+str(job_id)+':'+mode)
        finally:inherited.terminate_owned(process,birth);process.wait(timeout=10)
    local.require(peak*1.3<=4.5*2**30,'CAPACITY_REPAIR_RAM_MARGIN')
    return dict(mode=mode,seconds=time.monotonic()-started,peak_tree_rss_bytes=peak)


def run(bundle):
    run=ROOT/'output'/RUN_NAME;local.require(not run.exists(),'CAPACITY_REPAIR_FRESH_SESSION')
    run.mkdir(parents=True);start=time.monotonic();ledger=[dict(job=j,state='NOT_STARTED',processes=[]) for j in jobs()];error=None
    try:
        m=manifest();local.write(run/'delivery_manifest.snapshot.json',m)
        local.write(run/'compute_ledger.json',ledger)
        pilot.configure(bundle);actual,mismatches=pilot.environment(m)
        local.write(run/'environment_check.json',dict(actual=actual,mismatches=mismatches))
        local.require(not mismatches,'CAPACITY_REPAIR_ENVIRONMENT_MISMATCH:'+json.dumps(mismatches))
        local.write(run/'environment.lock.json',actual);inherited.input_checks(m,bundle)
        prefix='phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/'
        remapped=dict(config=m['source_bindings']['config'],inputs=[])
        for r in m['source_bindings']['inputs']:
            local.require(r['path'].startswith(prefix),'CAPACITY_REPAIR_INPUT_PREFIX');name=r['path'][len(prefix):]
            local.require(name in m['input_files'],'CAPACITY_REPAIR_UNBOUND_INPUT')
            remapped['inputs'].append(dict(r,path=str((Path(bundle)/name).resolve())))
        path=ROOT/'protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json';local.write(path,remapped)
        local.write(run/'session_started.json',dict(jobs=m['jobs'],delivery_sha256=local.sha(ROOT/MANIFEST),bindings_sha256=local.sha(path)))
        local.require(inherited.can_start(inherited.resources(run)),'CAPACITY_REPAIR_PREFLIGHT')
        deadline=start+21600-1800;child('qa',bundle,run,None,deadline)
        for job in m['jobs']:
            local.require(inherited.can_start(inherited.resources(run)) and time.monotonic()+1.3*3900<deadline,'CAPACITY_REPAIR_START_MARGIN')
            print('[CAPACITY REPAIR PILOT START]',job['id'],flush=True)
            entry=next(e for e in ledger if e['job']['id']==job['id'])
            entry['state']='STARTED';local.write(run/'compute_ledger.json',ledger)
            try:
                entry['processes'].append(child('job',bundle,run,job['id'],deadline))
                entry['processes'].append(child('replay',bundle,run,job['id'],deadline));entry['state']='COMPLETE'
            except BaseException as exc:
                entry.update(state='FAILED',error=str(exc));local.write(run/'compute_ledger.json',ledger);raise
            local.write(run/'compute_ledger.json',ledger)
        # Full models must start at exactly the same weights across representations.
        for a in ('TCN','GRU'):
            starts={json.loads((run/'jobs'/j['id']/'result.json').read_text())['initial_parameter_sha256'] for j in m['jobs'] if j['architecture']==a}
            local.require(len(starts)==1,'CAPACITY_REPAIR_MATCHED_INITIALIZATION')
    except BaseException as exc:
        error=str(exc);local.write(run/'failure.json',dict(error=error,traceback=traceback.format_exc()))
    finally:
        n=sum(e['state']=='COMPLETE' for e in ledger)
        summary=dict(status='CAPACITY_REPAIR_PILOT_COMPLETE_REVIEW_REQUIRED' if n==6 and error is None else 'CAPACITY_REPAIR_PILOT_STOPPED_REVIEW_REQUIRED',
            completed_jobs=n,planned_jobs=6,completed_battery_fits=n,completed_battery_epochs=3*n,
            completed_counts_require_training_and_replay=True,failed_or_partial_jobs=sum(e['state']=='FAILED' for e in ledger),
            not_started_jobs=sum(e['state']=='NOT_STARTED' for e in ledger),error=error,
            wall_seconds=time.monotonic()-start,production=False,candidate_selected=False,outer_scoring=False,fairness_closed=False)
        local.write(run/'summary.json',summary)
        files=sorted(p for p in run.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        local.write(run/'output_inventory.json',dict(files={p.relative_to(run).as_posix():dict(bytes=p.stat().st_size,sha256=local.sha(p)) for p in files},outer_scoring=False))
        target=ROOT.parent/(RUN_NAME+'_results.zip')
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in files+[run/'output_inventory.json']:z.write(p,RUN_NAME+'/'+p.relative_to(run).as_posix())
        print('DOWNLOAD CAPACITY REPAIR RESULTS / FAILURE:',target,flush=True);print(json.dumps(summary),flush=True)
    return 0 if error is None and n==6 else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['pilot','qa','job','replay']);parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--run',type=Path);parser.add_argument('--job-id');args=parser.parse_args()
    if args.mode=='pilot':sys.exit(run(args.bundle))
    worker(args.mode,args.bundle,args.run,args.job_id)

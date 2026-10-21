"""Four disjoint inner-only shards. No outer, scoring, resume or accuracy retries."""
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
from protocol_and_provenance.neural_classical_fairness_v1.runtime import full_inner_contract as contract
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_kaggle_pilot as pilot
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_trajectory as engine
from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
from protocol_and_provenance.neural_classical_fairness_v1.runtime import kaggle_classical_remaining as inherited

MANIFEST='full_inner_delivery.json'


def run_name(shard): return 'batteryscope_nfr1_calb_full_inner_v1_'+shard


def manifest():
    m=json.loads((ROOT/MANIFEST).read_text());contract.validate(m)
    for name,seal in m['files'].items():
        p=(ROOT/name).resolve()
        local.require(p.is_relative_to(ROOT) and p.is_file() and p.stat().st_size==seal['bytes'] and
            local.sha(p)==seal['sha256'],'FULL_INNER_CODE_HASH:'+name)
    return m


def environment(m):
    actual,mismatches=pilot.environment(m)
    actual.update(execution_role='NEURAL_CUDA_CLASSICAL_CPU_SAME_NEW_KAGGLE_SESSION_NOT_EQUAL_RISK',
        classical_prediction_policies=local.CLASSICAL_PREDICTION_POLICIES)
    return actual,mismatches


def resources(run):
    record=inherited.resources(run)
    record['output_bytes']=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    return record


def monitor_gate(snapshot,budget):
    local.require(snapshot['available_ram_bytes']>=1.5*2**30 and snapshot['free_disk_bytes']>=8*2**30 and
        snapshot['output_bytes']<=budget['output_bytes'],'FULL_INNER_RESOURCE_OUTPUT_GUARD')


def worker(mode,bundle,run,shard,job_id):
    m=manifest();pilot.configure(bundle)
    actual,mismatches=environment(m)
    session=json.loads((run/'session_started.json').read_text())
    local.require(not mismatches and actual==json.loads((run/'environment.lock.json').read_text()),'FULL_INNER_WORKER_ENVIRONMENT')
    local.require(session['shard']==shard and session['assigned_job_ids']==m['shards'][shard] and
        local.sha(ROOT/MANIFEST)==session['delivery_sha256'] and
        local.sha(ROOT/'protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json')==session['bindings_sha256'],
        'FULL_INNER_SESSION_SEAL')
    inherited.input_checks(m,bundle)
    if mode=='qa':
        ns=pilot.definitions(bundle)
        result=engine.synthetic_prefix_qa(ns,run/'synthetic_neural_qa','cuda:0')
        classical=inherited.synthetic_gate(local)
        local.write(run/'synthetic_classical_qa.json',classical)
        local.require(result['synthetic_fits']==12 and classical['synthetic_fits']==2,'FULL_INNER_QA_COUNTS')
        return
    local.require(job_id in m['shards'][shard],'FULL_INNER_UNASSIGNED_JOB')
    job=next(j for j in m['jobs'] if j['id']==job_id)
    directory=run/'jobs'/job_id
    guard=pilot.Guard(run,directory,seconds=300 if mode=='replay' else 3600)
    if mode=='replay':
        if job['kind']=='neural': engine.replay(pilot.definitions(bundle),job,directory/'trajectory',guard,'cuda:0')
        else: local.replay(job,directory)
        return
    directory.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    local.write(directory/'started.json',dict(job=job,delivery_sha256=local.sha(ROOT/MANIFEST),
        fit_role='FULL_INNER_NEW_RUNTIME_NOT_OUTER',pilot_artifact_imported=False))
    try:
        if job['kind']=='neural':
            import numpy as np
            ns,train,val,cycle,scaler,*_=local.verified_namespace(job,guard,directory,cycles=True)
            preprocessing=time.monotonic()-start
            support=dict(all_rows=len(val))
            for t in local.TARGETS:
                ts=ns['TARGET_SPECS'][t]
                support[t]=int((ns['normalize_bool'](val[ts['mask_col']],ts['mask_col']) &
                    np.isfinite(val[ts['y_col']].to_numpy(float))).sum())
                local.require(support[t]>0 and scaler['targets'][t]['active'],'FULL_INNER_NOT_EVALUABLE:'+t)
            local.write(directory/'expected_support.json',support)
            result=engine.run_trajectory(ns,job,ns['MultitaskCycleWindowIterator'](train,cycle,scaler),
                ns['MultitaskCycleWindowIterator'](val,cycle,scaler),scaler,support,guard,directory/'trajectory',
                device='cuda:0',epochs=30)
            h=json.loads((directory/'trajectory/history.json').read_text())
            policy=contract.accept_trajectory(h,result['checkpoints'])
            # Keep engine files byte-identical. This receipt declares the full-inner
            # role separately; this still is NOT an outer production predictor.
            local.write(directory/'inner_policy.json',policy)
            result.update(status='FULL_INNER_NEURAL_COMPLETE_REPLAY_PENDING',kind='neural',
                preprocessing_seconds=preprocessing,inner_selection_fit=True,full_trajectory_epochs=30,
                is_bounded_pilot=False,learned_fits=1,epoch_passes=30)
        else:
            result=local.classical_job(job,run,directory,guard)
            wanted='REFERENCE_EXECUTIONS_COMPLETE' if job['kind']=='references' else 'PILOT_TRAIN_COMPLETE_RELOAD_PENDING'
            local.require(result['status']==wanted,'FULL_INNER_NOT_EVALUABLE_OR_UNKNOWN_STATUS')
            result['underlying_helper_status']=result['status']
            result['status']='FULL_INNER_REFERENCES_COMPLETE' if job['kind']=='references' else 'FULL_INNER_CLASSICAL_COMPLETE_REPLAY_PENDING'
            result.update(inner_selection_fit=job['kind']=='classical',is_bounded_pilot=False,
                full_validation=True,epoch_passes=0)
        result.update(wall_seconds=time.monotonic()-start,peak_sampled_rss_bytes=guard.peak,
            fit_role='FULL_INNER_NEW_RUNTIME_NOT_OUTER',candidate_selected=False,
            outer_scoring=False,calibration_scoring=False,fairness_closed=False)
        local.write(directory/'result.json',result)
    except BaseException as exc:
        local.write(directory/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),outer_scoring=False))
        raise


def child(mode,bundle,run,shard,job,deadline,m):
    import psutil
    cap=m['budget']['synthetic_seconds'] if mode=='qa' else m['budget']['replay_seconds'] if mode=='replay' else m['budget']['job_seconds']
    deadline=min(deadline,time.monotonic()+cap);start=time.monotonic();peak=0;last_epoch=0
    name='synthetic_qa' if job is None else job['id']
    with (run/(name+'.'+mode+'.log')).open('x',encoding='utf-8') as log:
        args=[sys.executable,'-B','-X','utf8',str(Path(__file__)),mode,'--bundle',str(bundle),
            '--run',str(run),'--shard',shard]
        if job:args+=['--job-id',job['id']]
        process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1'),
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        owner=psutil.Process(process.pid);birth=owner.create_time()
        try:
            while process.poll() is None:
                try:owned=[owner]+owner.children(recursive=True)
                except psutil.NoSuchProcess:process.wait(timeout=10);break
                rss=0
                for p in owned:
                    try:rss+=p.memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss);snapshot=resources(run)
                local.write(run/'heartbeat.json',dict(state='RUNNING',shard=shard,mode=mode,job_id=name,
                    worker_pid=process.pid,worker_create_time=birth,resource=snapshot,peak_tree_rss_bytes=peak,
                    elapsed_seconds=time.monotonic()-start,wall_time_unix=time.time()))
                monitor_gate(snapshot,m['budget'])
                local.require(rss<=4.5*2**30 and time.monotonic()<deadline and not (run/'STOP').exists(),
                    'FULL_INNER_RSS_TIME_AUTHOR_STOP')
                if job and job['kind']=='neural' and mode=='job':
                    h=run/'jobs'/job['id']/'trajectory/history.json'
                    if h.is_file():
                        count=len(json.loads(h.read_text()))
                        if count!=last_epoch:print('[INNER EPOCH]',job['id'],count,'/30',flush=True);last_epoch=count
                time.sleep(2)
            local.require(process.returncode==0,'FULL_INNER_WORKER_FAILED:'+name+':'+mode)
        finally:
            inherited.terminate_owned(process,birth);process.wait(timeout=10)
    local.require(peak*1.3<=4.5*2**30,'FULL_INNER_MEMORY_MARGIN')
    return dict(mode=mode,seconds=time.monotonic()-start,peak_tree_rss_bytes=peak)


def export(run,shard):
    target=ROOT.parent/(run_name(shard)+'_results.zip')
    files=sorted(p for p in run.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    local.write(run/'output_inventory.json',dict(files={p.relative_to(run).as_posix():dict(bytes=p.stat().st_size,
        sha256=local.sha(p)) for p in files},outer_scoring=False,fairness_closed=False))
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[run/'output_inventory.json']:z.write(p,run_name(shard)+'/'+p.relative_to(run).as_posix())
    print('DOWNLOAD FULL INNER OUTPUT / FAILURE DIAGNOSTICS:',target,flush=True)


def run(bundle,shard):
    local.require(shard in contract.SHARDS,'FULL_INNER_SHARD')
    run=ROOT/'output'/run_name(shard);local.require(not run.exists(),'FULL_INNER_FRESH_SESSION_NO_RESUME')
    run.mkdir(parents=True);start=time.monotonic();m=None;error=None;ledger=[]
    try:
        m=manifest();pilot.configure(bundle);actual,mismatches=environment(m)
        local.write(run/'delivery_manifest.snapshot.json',m)
        local.write(run/'environment_check.json',dict(actual=actual,mismatches=mismatches))
        local.require(not mismatches,'FULL_INNER_ENVIRONMENT_MISMATCH:'+json.dumps(mismatches))
        local.write(run/'environment.lock.json',actual)
        mapped=inherited.input_checks(m,bundle);local.write(run/'input_mapping.json',mapped)
        remapped=dict(config=m['source_bindings']['config'],inputs=[])
        prefix='phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/'
        for record in m['source_bindings']['inputs']:
            local.require(record['path'].startswith(prefix),'FULL_INNER_INPUT_PREFIX')
            n=record['path'][len(prefix):];local.require(n in m['input_files'],'FULL_INNER_INPUT_UNBOUND')
            remapped['inputs'].append(dict(record,path=str((bundle/n).resolve())))
        bindings=ROOT/'protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json'
        local.write(bindings,remapped);local.write(run/'input_bindings.remapped.json',remapped)
        local.write(run/'session_started.json',dict(shard=shard,assigned_job_ids=m['shards'][shard],
            delivery_sha256=local.sha(ROOT/MANIFEST),bindings_sha256=local.sha(bindings),outer_scoring=False))
        snapshot=resources(run);local.write(run/'preflight.json',snapshot)
        local.require(inherited.can_start(snapshot),'FULL_INNER_RESOURCE_PREFLIGHT')
        deadline=start+m['budget']['wall_seconds']-m['budget']['reserve_seconds']
        qa=child('qa',bundle,run,shard,None,deadline,m);local.write(run/'synthetic_qa_process.json',qa)
        for job_id in m['shards'][shard]:
            j=next(j for j in m['jobs'] if j['id']==job_id)
            snapshot=resources(run);monitor_gate(snapshot,m['budget'])
            local.require(inherited.can_start(snapshot),'FULL_INNER_START_RESOURCE')
            replay_cap=0 if j['kind']=='references' else m['budget']['replay_seconds']
            local.require(time.monotonic()+1.3*(m['budget']['job_seconds']+replay_cap)<deadline,'FULL_INNER_ADMISSION_TIME_MARGIN')
            entry=dict(job=j,state='WORKER_STARTED',processes=[],accepted_learned_fits=0,accepted_epoch_passes=0,
                accepted_reference_executions=0)
            ledger.append(entry);local.write(run/'compute_ledger.json',ledger)
            local.write(run/'current_job.json',dict(state='RUNNING',job=j))
            print('[FULL INNER START]',j['id'],flush=True)
            try:
                entry['processes'].append(child('job',bundle,run,shard,j,deadline,m))
                if j['kind']!='references':entry['processes'].append(child('replay',bundle,run,shard,j,deadline,m))
                result=json.loads((run/'jobs'/j['id']/'result.json').read_text())
                entry.update(state='COMPLETE',accepted_learned_fits=int(j['kind']!='references'),
                    accepted_epoch_passes=30 if j['kind']=='neural' else 0,
                    accepted_reference_executions=result.get('reference_executions',0))
            except BaseException as exc:entry.update(state='FAILED_OR_PARTIAL',error=str(exc));raise
            local.write(run/'compute_ledger.json',ledger)
            print('[FULL INNER COMPLETE]',j['id'],flush=True)
    except BaseException as exc:
        error=str(exc) or type(exc).__name__
        local.write(run/'failure.json',dict(error=error,traceback=traceback.format_exc()))
    finally:
        assigned=m['shards'][shard] if m else []
        seen={r['job']['id'] for r in ledger}
        for jid in assigned:
            if jid not in seen:ledger.append(dict(job=next(j for j in m['jobs'] if j['id']==jid),
                state='NOT_ACCEPTED_AFTER_STOP',accepted_learned_fits=0,accepted_epoch_passes=0,accepted_reference_executions=0))
        local.write(run/'compute_ledger.json',ledger)
        n=sum(r['state']=='COMPLETE' for r in ledger)
        summary=dict(status='FULL_INNER_SHARD_COMPLETE_REVIEW_REQUIRED' if error is None and n==len(assigned) else
            'FULL_INNER_SHARD_PARTIAL_REVIEW_REQUIRED',shard=shard,completed_jobs=n,planned_jobs=len(assigned),error=error,
            accepted_learned_fits=sum(r['accepted_learned_fits'] for r in ledger),
            accepted_neural_epoch_passes=sum(r['accepted_epoch_passes'] for r in ledger),
            accepted_reference_executions=sum(r['accepted_reference_executions'] for r in ledger),
            saved_neural_epoch_checkpoints=len(list((run/'jobs').glob('*/trajectory/checkpoint_epoch_*.pt'))),
            synthetic_fits_separate=14 if (run/'synthetic_classical_qa.json').exists() else None,
            started_job_attempts=sum(r['state'] in ('COMPLETE','FAILED_OR_PARTIAL') for r in ledger),
            wall_seconds=time.monotonic()-start,candidate_selected=False,outer_scoring=False,fairness_closed=False)
        local.write(run/'summary.json',summary);local.write(run/'heartbeat.json',dict(state='SUPERVISOR_TERMINAL',**summary))
        local.write(run/'current_job.json',dict(state='SUPERVISOR_TERMINAL',status=summary['status']))
        export(run,shard);print(json.dumps(summary,indent=2),flush=True)
    return 0 if error is None and len(assigned)>0 and n==len(assigned) else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('campaign','qa','job','replay'));p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--shard',choices=contract.SHARDS,required=True);p.add_argument('--run',type=Path);p.add_argument('--job-id')
    args=p.parse_args()
    if args.mode=='campaign':sys.exit(run(args.bundle.resolve(),args.shard))
    local.require(args.run is not None and args.run.resolve()==ROOT/'output'/run_name(args.shard),'FULL_INNER_WORKER_OUTPUT')
    worker(args.mode,args.bundle.resolve(),args.run.resolve(),args.shard,args.job_id)

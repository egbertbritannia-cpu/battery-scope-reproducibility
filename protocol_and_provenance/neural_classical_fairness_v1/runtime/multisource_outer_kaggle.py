"""One fresh-session notebook per source; selected outer fits. No outer labels or scoring."""
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
from protocol_and_provenance.neural_classical_fairness_v1.runtime import multisource_outer_contract as contract
from protocol_and_provenance.neural_classical_fairness_v1.runtime import multisource_outer_core as core
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_kaggle_pilot as pilot
from protocol_and_provenance.neural_classical_fairness_v1.runtime import kaggle_classical_remaining as inherited
from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local

MANIFEST=contract.MANIFEST


def run_name(shard,source=None):
    source=source or json.loads((ROOT/MANIFEST).read_text())['source']
    return 'batteryscope_nfr1_outer_'+source+'_v1'


def manifest():
    m=json.loads((ROOT/MANIFEST).read_text());s=contract.read_selection(ROOT/'selection.frozen.json')
    contract.validate(m,s)
    for name,seal in m['files'].items():
        p=(ROOT/name).resolve()
        local.require(p.is_relative_to(ROOT) and p.is_file() and core.binding(p)==seal, 'MULTI_OUTER_CODE_HASH:'+name)
    return m,s


def resources(run):
    r=inherited.resources(run)
    r['output_bytes']=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    return r


def monitor_gate(snapshot,budget):
    local.require(snapshot['available_ram_bytes']>=1.5*2**30 and snapshot['free_disk_bytes']>=8*2**30 and
        snapshot['output_bytes']<=budget['output_bytes'], 'MULTI_OUTER_RESOURCE_OUTPUT_GUARD')


def worker(mode,bundle,run,shard,job_id):
    m,s=manifest();pilot.configure(bundle);actual,mismatches=pilot.environment(m)
    session=json.loads((run/'session_started.json').read_text())
    local.require(not mismatches and actual==json.loads((run/'environment.lock.json').read_text()) and
        session['shard']==shard and session['assigned_job_ids']==m['shards'][shard] and
        local.sha(ROOT/MANIFEST)==session['delivery_sha256'], 'MULTI_OUTER_WORKER_ENV_SESSION')
    inherited.input_checks(m,bundle)
    if mode=='qa':
        ns=pilot.definitions(bundle);ns['OUTER_SOURCE']=s['source']
        core.synthetic_qa(ns,run/'synthetic_neural_qa','cuda:0')
        local.write(run/'synthetic_classical_qa.json',inherited.synthetic_gate(local))
        return
    local.require(job_id in m['shards'][shard], 'MULTI_OUTER_UNASSIGNED_JOB')
    j=next(j for j in m['jobs'] if j['id']==job_id);directory=run/'jobs'/job_id
    guard=pilot.Guard(run,directory,seconds=m['budget']['replay_seconds'] if mode=='replay' else m['budget']['job_seconds'])
    if mode=='replay':
        core.replay(bundle,s,j,directory,guard)
        return
    directory.mkdir(parents=True,exist_ok=False)
    local.write(directory/'started.json',dict(job=j,delivery_sha256=local.sha(ROOT/MANIFEST),selection_sha256=m['selection_sha256'],
        fit_role='POST_HOC_MULTI_OUTER_POINT_ONLY',outer_labels_accessed=False,calibration_labels_accessed=False))
    try:
        if j['kind']=='neural':core.neural_job(bundle,s,j,directory,guard)
        else:core.classical_job(bundle,s,j,directory,guard)
    except BaseException as exc:
        local.write(directory/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),outer_scoring=False))
        raise


def child(mode,bundle,run,shard,job,deadline,m):
    import psutil
    cap=m['budget']['synthetic_seconds'] if mode=='qa' else m['budget']['replay_seconds'] if mode=='replay' else m['budget']['job_seconds']
    deadline=min(deadline,time.monotonic()+cap);started=time.monotonic();peak=0
    name='synthetic_qa' if job is None else job['id']
    with (run/(name+'.'+mode+'.log')).open('x',encoding='utf-8') as log:
        args=[sys.executable,'-B','-X','utf8',str(Path(__file__)),mode,'--bundle',str(bundle),'--run',str(run),'--shard',shard]
        if job is not None:args+=['--job-id',job['id']]
        p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        owner=psutil.Process(p.pid);birth=owner.create_time()
        try:
            while p.poll() is None:
                try:owned=[owner]+owner.children(recursive=True)
                except psutil.NoSuchProcess:p.wait(timeout=10);break
                rss=0
                for q in owned:
                    try:rss+=q.memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss);snapshot=resources(run)
                local.write(run/'heartbeat.json',dict(state='RUNNING',shard=shard,mode=mode,job_id=name,
                    worker_pid=p.pid,worker_create_time=birth,resource=snapshot,peak_tree_rss_bytes=peak,
                    elapsed_seconds=time.monotonic()-started,wall_time_unix=time.time()))
                monitor_gate(snapshot,m['budget'])
                local.require(rss<=4.5*2**30 and time.monotonic()<deadline and not (run/'STOP').exists(), 'MULTI_OUTER_RSS_TIME_AUTHOR_STOP')
                time.sleep(2)
            local.require(p.returncode==0,'MULTI_OUTER_WORKER_FAILED:'+name+':'+mode)
        finally:
            inherited.terminate_owned(p,birth);p.wait(timeout=10)
    local.require(peak*1.3<=4.5*2**30,'MULTI_OUTER_MEMORY_MARGIN')
    return dict(mode=mode,seconds=time.monotonic()-started,peak_tree_rss_bytes=peak,worker_pid=p.pid,worker_create_time=birth)


def export(run,shard):
    target=ROOT.parent/(run_name(shard)+'_results.zip')
    files=sorted(p for p in run.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='output_inventory.json')
    local.write(run/'output_inventory.json',dict(files={p.relative_to(run).as_posix():core.binding(p) for p in files},
        outer_scoring=False,outer_labels_accessed=False,fairness_closed=False))
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[run/'output_inventory.json']:z.write(p,run_name(shard)+'/'+p.relative_to(run).as_posix())
    print('DOWNLOAD MULTI OUTER OUTPUT / FAILURE DIAGNOSTICS:',target,flush=True)


def run(bundle,shard):
    local.require(shard in contract.SHARDS,'MULTI_OUTER_SHARD')
    run=ROOT/'output'/run_name(shard);local.require(not run.exists(),'MULTI_OUTER_FRESH_SESSION_NO_RESUME')
    run.mkdir(parents=True);started=time.monotonic();m=None;error=None;ledger=[]
    try:
        m,s=manifest();pilot.configure(bundle);actual,mismatches=pilot.environment(m)
        local.write(run/'delivery_manifest.snapshot.json',m);local.write(run/'selection.snapshot.json',s)
        local.write(run/'environment_check.json',dict(actual=actual,mismatches=mismatches))
        local.require(not mismatches,'MULTI_OUTER_ENVIRONMENT_MISMATCH:'+json.dumps(mismatches))
        local.write(run/'environment.lock.json',actual)
        local.write(run/'input_mapping.json',inherited.input_checks(m,bundle))
        local.write(run/'session_started.json',dict(shard=shard,assigned_job_ids=m['shards'][shard],
            delivery_sha256=local.sha(ROOT/MANIFEST),selection_sha256=m['selection_sha256'],outer_scoring=False))
        snapshot=resources(run);local.write(run/'preflight.json',snapshot)
        local.require(inherited.can_start(snapshot),'MULTI_OUTER_PREFLIGHT')
        deadline=started+m['budget']['wall_seconds']-m['budget']['reserve_seconds']
        local.write(run/'synthetic_qa_process.json',child('qa',bundle,run,shard,None,deadline,m))
        for jid in m['shards'][shard]:
            job=next(j for j in m['jobs'] if j['id']==jid)
            snapshot=resources(run);monitor_gate(snapshot,m['budget'])
            local.require(inherited.can_start(snapshot),'MULTI_OUTER_START_RESOURCE')
            replay_cap=0 if job['kind']=='references' else m['budget']['replay_seconds']
            local.require(time.monotonic()+1.3*(m['budget']['job_seconds']+replay_cap)<deadline,'MULTI_OUTER_ADMISSION_TIME_MARGIN')
            entry=dict(job=job,state='WORKER_STARTED',processes=[],accepted_learned_fits=0,
                accepted_epoch_passes=0,accepted_reference_executions=0)
            ledger.append(entry);local.write(run/'compute_ledger.json',ledger)
            print('[MULTI OUTER START]',jid,flush=True)
            try:
                entry['processes'].append(child('job',bundle,run,shard,job,deadline,m))
                if job['kind']!='references':entry['processes'].append(child('replay',bundle,run,shard,job,deadline,m))
                r=json.loads((run/'jobs'/jid/'result.json').read_text())
                expected={'neural':'MULTI_OUTER_NEURAL_FIT_COMPLETE_REPLAY_PENDING',
                    'classical':'MULTI_OUTER_CLASSICAL_FIT_COMPLETE_REPLAY_PENDING','references':'MULTI_OUTER_REFERENCES_COMPLETE'}[job['kind']]
                local.require(r['status']==expected,'MULTI_OUTER_UNKNOWN_COMPLETION_STATUS')
                entry.update(state='COMPLETE',accepted_learned_fits=int(job['kind']!='references'),
                    accepted_epoch_passes=job.get('fit_epochs',0),accepted_reference_executions=r.get('reference_executions',0))
            except BaseException as exc:entry.update(state='FAILED_OR_PARTIAL',error=str(exc));raise
            local.write(run/'compute_ledger.json',ledger);print('[MULTI OUTER COMPLETE]',jid,flush=True)
    except BaseException as exc:
        error=str(exc) or type(exc).__name__;local.write(run/'failure.json',dict(error=error,traceback=traceback.format_exc()))
    finally:
        assigned=m['shards'][shard] if m else [];seen={r['job']['id'] for r in ledger}
        for jid in assigned:
            if jid not in seen:ledger.append(dict(job=next(j for j in m['jobs'] if j['id']==jid),state='NOT_ACCEPTED_AFTER_STOP',
                accepted_learned_fits=0,accepted_epoch_passes=0,accepted_reference_executions=0))
        local.write(run/'compute_ledger.json',ledger);n=sum(r['state']=='COMPLETE' for r in ledger)
        summary=dict(status='MULTI_OUTER_SHARD_COMPLETE_REVIEW_REQUIRED' if error is None and n==len(assigned) else 'MULTI_OUTER_SHARD_PARTIAL_REVIEW_REQUIRED',
            shard=shard,completed_jobs=n,planned_jobs=len(assigned),error=error,
            accepted_learned_fits=sum(r['accepted_learned_fits'] for r in ledger),
            accepted_neural_epoch_passes=sum(r['accepted_epoch_passes'] for r in ledger),
            accepted_reference_executions=sum(r['accepted_reference_executions'] for r in ledger),
            wall_seconds=time.monotonic()-started,outer_scoring=False,outer_labels_accessed=False,
            calibration_labels_accessed=False,point_only=True,fairness_closed=False)
        local.write(run/'summary.json',summary);local.write(run/'heartbeat.json',dict(state='SUPERVISOR_TERMINAL',**summary))
        export(run,shard);print(json.dumps(summary,indent=2),flush=True)
    return 0 if error is None and len(assigned)>0 and n==len(assigned) else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('campaign','qa','job','replay'));p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--shard',choices=contract.SHARDS,required=True);p.add_argument('--run',type=Path);p.add_argument('--job-id')
    args=p.parse_args()
    if args.mode=='campaign':sys.exit(run(args.bundle.resolve(),args.shard))
    local.require(args.run is not None and args.run.resolve()==ROOT/'output'/run_name(args.shard),'MULTI_OUTER_WORKER_OUTPUT')
    worker(args.mode,args.bundle.resolve(),args.run.resolve(),args.shard,args.job_id)

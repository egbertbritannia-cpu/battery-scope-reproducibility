"""Matched capacity-repair full inner fits, six isolated Kaggle shards.

Three representations share one architecture/fold per shard. This stage writes
inner histories and checkpoint policies; it never selects across folds or scores
outer labels. A stopped fit is retained as partial, never silently resumed.
"""
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1.runtime import capacity_repair_pilot as pilot_repair
from protocol_and_provenance.neural_classical_fairness_v1.runtime import capacity_scale_repair as repair
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_kaggle_pilot as gpu
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_trajectory as engine
from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
from protocol_and_provenance.neural_classical_fairness_v1.runtime import kaggle_classical_remaining as inherited

MANIFEST = 'capacity_repair_full_inner_delivery.json'
SHARDS = ('A','B','C','D','E','F')
ASSIGNMENT = {shard:(architecture,fold) for shard,architecture,fold in zip(
    SHARDS,('TCN','TCN','TCN','GRU','GRU','GRU'),(0,1,2,0,1,2))}


def run_name(shard):
    local.require(shard in SHARDS,'REPAIR_FULL_INNER_SHARD')
    return 'batteryscope_nfr1_capacity_repair_full_inner_v1_' + shard


def jobs():
    return [dict(id=f'repair.full.{architecture}.{recipe}.CALB.{fold}.2026',kind='neural',
        source='CALB',inner_fold=fold,seed=2026,architecture=architecture,
        candidate='P1',representation=recipe,epochs=30)
        for architecture in ('TCN','GRU') for fold in range(3) for recipe in repair.RECIPES]


def assignments():
    all_jobs=jobs()
    return {s:[j['id'] for j in all_jobs if (j['architecture'],j['inner_fold'])==pair]
            for s,pair in ASSIGNMENT.items()}


def validate(m):
    local.require(m['schema']=='capacity-repair-full-inner-v1' and m['jobs']==jobs() and
        m['shards']==assignments() and m['epochs']==30 and
        m['normalized_channels']==list(repair.NORMALIZED) and
        m['feature_contract']==repair.FEATURE_CONTRACT and
        m['representations']==list(repair.RECIPES) and m['standardized_cap']==5 and
        m['objective']=='P1_point_q50_LR_0.0003' and
        m['head_weights']=={'soh30':1.,'rul90':1.,'rul80':.5} and
        m['selection']['policies']==['S0','S1'] and
        m['selection']['automatic_cross_fold_selection'] is False and
        m['selection']['all_three_folds_required'] is True and
        m['selection']['outer_labels_used'] is False and
        len(m['role_metadata'])==3 and
        [r['inner_fold'] for r in m['role_metadata']]==[0,1,2] and
        all(r['source']=='CALB' and r['train_cells'] and r['validation_cells'] and
            not set(r['train_cells']) & set(r['validation_cells']) and
            r['labels_or_profiles_read'] is False for r in m['role_metadata']) and
        m['budget']==dict(session_seconds=28800,reserve_seconds=1800,job_seconds=5400,
                          replay_seconds=300,synthetic_seconds=900,workers=1,
                          rss_bytes=int(4.5*2**30),output_bytes=8*2**30) and
        m['technical_pilot_review']['status']=='ARTIFACTS_ACCEPTED_PILOT_REPAIR_EFFICACY_UNPROVEN' and
        m['after_outer_observations_post_hoc'] is True and
        all(m[k] is False for k in ('outer_scoring','candidate_selected','fairness_closed',
                                    'automatic_retry','automatic_resume')),
        'REPAIR_FULL_INNER_CONTRACT')
    local.require(len(m['jobs'])==18 and len(m['shards'])==6 and
        all(len(v)==3 for v in m['shards'].values()) and
        len(set(sum(m['shards'].values(),[])))==18,'REPAIR_FULL_INNER_ASSIGNMENT')


def manifest():
    m=json.loads((ROOT/MANIFEST).read_text())
    validate(m)
    for name,seal in m['files'].items():
        path=(ROOT/name).resolve()
        local.require(path.is_relative_to(ROOT) and path.is_file() and
            dict(bytes=path.stat().st_size,sha256=local.sha(path))==seal,
            'REPAIR_FULL_INNER_CODE_HASH:'+name)
    return m


def resources(run):
    state=inherited.resources(run)
    state['output_bytes']=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    return state


def resource_gate(state,budget):
    local.require(state['available_ram_bytes']>=1.5*2**30 and
        state['free_disk_bytes']>=8*2**30 and
        state['output_bytes']<=budget['output_bytes'],'REPAIR_FULL_INNER_RESOURCE')


def worker(mode,bundle,run,shard,job_id):
    import numpy as np
    m=manifest();gpu.configure(bundle)
    actual,mismatches=gpu.environment(m)
    local.require(not mismatches and actual==json.loads((run/'environment.lock.json').read_text()),
                  'REPAIR_FULL_INNER_WORKER_ENVIRONMENT')
    session=json.loads((run/'session_started.json').read_text())
    bindings=ROOT/'protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json'
    local.require(session['shard']==shard and session['assigned_job_ids']==m['shards'][shard] and
        session['delivery_sha256']==local.sha(ROOT/MANIFEST) and
        session['bindings_sha256']==local.sha(bindings),'REPAIR_FULL_INNER_SESSION')
    inherited.input_checks(m,bundle)
    if mode=='qa':
        qa=pilot_repair.synthetic_gate(gpu.definitions(bundle),run/'synthetic_qa','cuda:0')
        local.require(qa['synthetic_fits']==6 and qa['synthetic_epochs']==12 and
                      qa['battery_fits']==0,'REPAIR_FULL_INNER_SYNTHETIC')
        return
    local.require(job_id in m['shards'][shard],'REPAIR_FULL_INNER_UNASSIGNED')
    job=next(j for j in m['jobs'] if j['id']==job_id)
    directory=run/'jobs'/job_id
    guard=gpu.Guard(run,directory,seconds=m['budget']['replay_seconds'] if mode=='replay'
                    else m['budget']['job_seconds'])
    if mode=='replay':
        engine.replay(gpu.definitions(bundle),job,directory/'trajectory',guard,'cuda:0')
        return
    directory.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    local.write(directory/'started.json',dict(job=job,delivery_sha256=local.sha(ROOT/MANIFEST),
        role='POST_HOC_INNER_SELECTION',pilot_checkpoint_imported=False,outer_scoring=False))
    try:
        ns,train,val,_,target,_,train_cells,_=local.verified_namespace(job,guard,directory,cycles=True)
        frozen_roles=m['role_metadata'][job['inner_fold']]
        roles=json.loads((directory/'roles.json').read_text())
        local.require(roles['train_cells']==frozen_roles['train_cells'] and
            roles['validation_cells']==frozen_roles['validation_cells'],
            'REPAIR_FULL_INNER_FROZEN_ROLE_METADATA')
        adapter=repair.attach(ns,job['representation'])
        transformed_loader=ns['load_cycle_shard']
        def train_only(cell):
            local.require(cell in train_cells,'REPAIR_FULL_INNER_SCALER_VALIDATION_READ')
            return transformed_loader(cell)
        ns['load_cycle_shard']=train_only
        cycle=ns['fit_cycle_scaler'](train_cells)
        ns['load_cycle_shard']=transformed_loader
        cycle.update(fit_role='capacity_repair_inner_train_only',feature_contract=repair.FEATURE_CONTRACT,
                     representation=job['representation'])
        roles.update(cycle_scaler=cycle,feature_adapter=adapter,old_preprocessing_pass_not_reused=True)
        local.write(directory/'roles.json',roles)
        support=dict(all_rows=len(val))
        for name in local.TARGETS:
            spec=ns['TARGET_SPECS'][name]
            support[name]=int((ns['normalize_bool'](val[spec['mask_col']],spec['mask_col']) &
                np.isfinite(val[spec['y_col']].to_numpy(float))).sum())
            local.require(support[name]>0 and target['targets'][name]['active'],
                'REPAIR_FULL_INNER_EMPTY_HEAD:'+name)
        local.write(directory/'expected_support.json',support)
        prep=time.monotonic()-start
        result=engine.run_trajectory(ns,job,ns['MultitaskCycleWindowIterator'](train,cycle,target),
            ns['MultitaskCycleWindowIterator'](val,cycle,target),target,support,guard,
            directory/'trajectory',device='cuda:0',epochs=30)
        history=json.loads((directory/'trajectory/history.json').read_text())
        policy=engine.policy_map(history,result['checkpoints'],production=True)
        local.write(directory/'inner_policy.json',policy)
        result.update(status='REPAIR_FULL_INNER_COMPLETE_REPLAY_PENDING',representation=job['representation'],
            feature_adapter=adapter,preprocessing_seconds=prep,wall_seconds=time.monotonic()-start,
            peak_sampled_rss_bytes=guard.peak,inner_selection_fit=True,full_trajectory_epochs=30,
            learned_fits=1,production_fit=False,candidate_selected=False,outer_scoring=False,
            fairness_closed=False,post_hoc=True)
        local.write(directory/'result.json',result)
    except BaseException as exc:
        local.write(directory/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),
                                                  outer_scoring=False))
        raise


def child(mode,bundle,run,shard,job,deadline,m):
    import psutil
    budget=m['budget']
    cap=budget['synthetic_seconds'] if mode=='qa' else budget['replay_seconds'] if mode=='replay' else budget['job_seconds']
    started=time.monotonic();limit=min(deadline,started+cap);peak=0;last_epoch=0
    name=job['id'] if job else 'synthetic_qa'
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
                peak=max(peak,rss);state=resources(run)
                local.write(run/'heartbeat.json',dict(state='RUNNING',shard=shard,mode=mode,job_id=name,
                    worker_pid=process.pid,peak_tree_rss_bytes=peak,resource=state,
                    elapsed_seconds=time.monotonic()-started))
                resource_gate(state,budget)
                local.require(rss<=budget['rss_bytes'] and time.monotonic()<limit and
                              not (run/'STOP').exists(),'REPAIR_FULL_INNER_STOP_TIME_RESOURCE')
                if mode=='job' and job:
                    history=run/'jobs'/name/'trajectory/history.json'
                    if history.is_file():
                        count=len(json.loads(history.read_text()))
                        if count!=last_epoch:
                            print('[REPAIR INNER EPOCH]',name,count,'/30',flush=True)
                            last_epoch=count
                time.sleep(2)
            local.require(process.returncode==0,'REPAIR_FULL_INNER_WORKER_FAILED:'+name+':'+mode)
        finally:
            inherited.terminate_owned(process,birth)
            process.wait(timeout=10)
    local.require(peak*1.3<=budget['rss_bytes'],'REPAIR_FULL_INNER_RSS_MARGIN')
    return dict(mode=mode,seconds=time.monotonic()-started,peak_tree_rss_bytes=peak)


def export(run,shard):
    target=ROOT.parent/(run_name(shard)+'_results.zip')
    files=sorted(p for p in run.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    local.write(run/'output_inventory.json',dict(files={p.relative_to(run).as_posix():
        dict(bytes=p.stat().st_size,sha256=local.sha(p)) for p in files},outer_scoring=False))
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files+[run/'output_inventory.json']:
            archive.write(path,run_name(shard)+'/'+path.relative_to(run).as_posix())
    print('DOWNLOAD REPAIR FULL INNER OUTPUT / FAILURE:',target,flush=True)


def run(bundle,shard):
    local.require(shard in SHARDS,'REPAIR_FULL_INNER_UNKNOWN_SHARD')
    directory=ROOT/'output'/run_name(shard)
    local.require(not directory.exists(),'REPAIR_FULL_INNER_FRESH_SESSION')
    directory.mkdir(parents=True);start=time.monotonic();m=None;error=None;ledger=[]
    try:
        m=manifest();gpu.configure(bundle);actual,mismatches=gpu.environment(m)
        local.write(directory/'delivery_manifest.snapshot.json',m)
        local.write(directory/'environment_check.json',dict(actual=actual,mismatches=mismatches))
        local.require(not mismatches,'REPAIR_FULL_INNER_ENVIRONMENT:'+json.dumps(mismatches))
        local.write(directory/'environment.lock.json',actual)
        local.write(directory/'input_mapping.json',inherited.input_checks(m,bundle))
        prefix='phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/'
        remapped=dict(config=m['source_bindings']['config'],inputs=[])
        for record in m['source_bindings']['inputs']:
            local.require(record['path'].startswith(prefix),'REPAIR_FULL_INNER_INPUT_PREFIX')
            name=record['path'][len(prefix):]
            local.require(name in m['input_files'],'REPAIR_FULL_INNER_INPUT_BINDING')
            remapped['inputs'].append(dict(record,path=str((Path(bundle)/name).resolve())))
        binding=ROOT/'protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json'
        local.write(binding,remapped);local.write(directory/'input_bindings.remapped.json',remapped)
        local.write(directory/'session_started.json',dict(shard=shard,assigned_job_ids=m['shards'][shard],
            delivery_sha256=local.sha(ROOT/MANIFEST),bindings_sha256=local.sha(binding),outer_scoring=False))
        local.require(inherited.can_start(resources(directory)),'REPAIR_FULL_INNER_PREFLIGHT')
        deadline=start+m['budget']['session_seconds']-m['budget']['reserve_seconds']
        child('qa',bundle,directory,shard,None,deadline,m)
        for job_id in m['shards'][shard]:
            job=next(j for j in m['jobs'] if j['id']==job_id)
            local.require(inherited.can_start(resources(directory)), 'REPAIR_FULL_INNER_START_RESOURCE')
            resource_gate(resources(directory),m['budget'])
            local.require(time.monotonic()+1.3*(m['budget']['job_seconds']+
                m['budget']['replay_seconds'])<deadline,'REPAIR_FULL_INNER_ADMISSION_TIME')
            entry=dict(job=job,state='STARTED',processes=[],accepted_fits=0,accepted_epochs=0)
            ledger.append(entry);local.write(directory/'compute_ledger.json',ledger)
            print('[REPAIR FULL INNER START]',job_id,flush=True)
            try:
                entry['processes'].append(child('job',bundle,directory,shard,job,deadline,m))
                entry['processes'].append(child('replay',bundle,directory,shard,job,deadline,m))
                entry.update(state='COMPLETE',accepted_fits=1,accepted_epochs=30)
            except BaseException as exc:
                entry.update(state='FAILED_OR_PARTIAL',error=str(exc))
                local.write(directory/'compute_ledger.json',ledger)
                raise
            local.write(directory/'compute_ledger.json',ledger)
            print('[REPAIR FULL INNER COMPLETE]',job_id,flush=True)
        # Start weights, full train/validation roles and support must match.
        records=[json.loads((directory/'jobs'/jid/'result.json').read_text()) for jid in m['shards'][shard]]
        roles=[json.loads((directory/'jobs'/jid/'roles.json').read_text()) for jid in m['shards'][shard]]
        support=[json.loads((directory/'jobs'/jid/'expected_support.json').read_text()) for jid in m['shards'][shard]]
        local.require(len({r['initial_parameter_sha256'] for r in records})==1 and
            all((r['train_cells'],r['validation_cells'],r['target_scaler'])==
                (roles[0]['train_cells'],roles[0]['validation_cells'],roles[0]['target_scaler']) for r in roles) and
            all(s==support[0] for s in support),'REPAIR_FULL_INNER_MATCHED_REPRESENTATIONS')
    except BaseException as exc:
        error=str(exc) or type(exc).__name__
        local.write(directory/'failure.json',dict(error=error,traceback=traceback.format_exc()))
    finally:
        assigned=m['shards'][shard] if m else []
        seen={row['job']['id'] for row in ledger}
        for jid in assigned:
            if jid not in seen:
                ledger.append(dict(job=next(j for j in m['jobs'] if j['id']==jid),state='NOT_STARTED',
                                   accepted_fits=0,accepted_epochs=0))
        local.write(directory/'compute_ledger.json',ledger)
        completed=sum(row['state']=='COMPLETE' for row in ledger)
        summary=dict(status='REPAIR_FULL_INNER_SHARD_COMPLETE_REVIEW_REQUIRED' if error is None and
            completed==3 else 'REPAIR_FULL_INNER_SHARD_PARTIAL_REVIEW_REQUIRED',
            shard=shard,completed_jobs=completed,planned_jobs=len(assigned),
            accepted_learned_fits=sum(row['accepted_fits'] for row in ledger),
            accepted_epoch_passes=sum(row['accepted_epochs'] for row in ledger),
            saved_epoch_checkpoints=len(list((directory/'jobs').glob('*/trajectory/checkpoint_epoch_*.pt'))),
            error=error,wall_seconds=time.monotonic()-start,candidate_selected=False,
            outer_scoring=False,fairness_closed=False)
        local.write(directory/'summary.json',summary)
        export(directory,shard)
        print(json.dumps(summary,indent=2),flush=True)
    return 0 if error is None and completed==3 else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('campaign','qa','job','replay'))
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--shard',choices=SHARDS,required=True)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--job-id')
    args=parser.parse_args()
    if args.mode=='campaign':
        sys.exit(run(args.bundle.resolve(),args.shard))
    local.require(args.run is not None and args.run.resolve()==ROOT/'output'/run_name(args.shard),
                  'REPAIR_FULL_INNER_WORKER_OUTPUT')
    worker(args.mode,args.bundle.resolve(),args.run.resolve(),args.shard,args.job_id)

"""Matched RUL task ablation using the sealed repair runtime and accepted joint controls."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1.runtime import capacity_repair_full_inner as prior
from protocol_and_provenance.neural_classical_fairness_v1.runtime import rul_task_ablation as ablation

local=prior.local
gpu=prior.gpu
engine=prior.engine
inherited=prior.inherited
MANIFEST='rul_representation_inner_delivery.json'
SHARDS=prior.SHARDS
ORIGINAL_TRAJECTORY=engine.run_trajectory
ORIGINAL_WORKER=prior.worker


def rul_policy(policy):
    local.require(set(policy['policies'])=={'S0','S1'},'RUL_REPR_POLICY_WRAPPER')
    for name in ('S0','S1'):
        local.require(all(t in policy['policies'][name] for t in ('rul90','rul80')),
                      'RUL_REPR_POLICY_ENDPOINTS')
        policy['policies'][name].pop('soh30',None)
    policy['soh30_policy_removed_untrained']=True
    return policy


def run_name(shard):
    local.require(shard in SHARDS,'RUL_REPR_SHARD')
    return 'batteryscope_nfr1_rul_representation_inner_v1_'+shard


def jobs():
    return [dict(id=f'ablation.rul.{architecture}.{representation}.CALB.{fold}.2026',
        kind='neural',source='CALB',inner_fold=fold,seed=2026,architecture=architecture,
        candidate='P1',representation=representation,epochs=30,trained_heads=['rul90','rul80'])
        for architecture in ('TCN','GRU') for fold in range(3) for representation in ('R0','R1')]


def assignments():
    return {s:[j['id'] for j in jobs() if (j['architecture'],j['inner_fold'])==pair]
            for s,pair in prior.ASSIGNMENT.items()}


def validate(m):
    local.require(m['schema']=='rul-representation-inner-v1' and m['jobs']==jobs() and
        m['shards']==assignments() and m['loss_contract']==ablation.LOSS_CONTRACT and
        m['representations']==['R0','R1'] and m['epochs']==30 and
        m['selection']['primary']=='S1_logical_prefix_patience5' and
        m['selection']['endpoints']==['rul90','rul80'] and
        m['selection']['outer_labels_used'] is False and
        m['selection']['automatic_cross_fold_selection'] is False and
        m['after_outer_observations_post_hoc'] is True and
        m['baseline_controls']['reused_joint_fits']==12 and
        len(m['baseline_controls']['records'])==12 and
        m['budget']==dict(session_seconds=28800,reserve_seconds=1800,job_seconds=5400,
            replay_seconds=300,synthetic_seconds=900,workers=1,rss_bytes=int(4.5*2**30),output_bytes=8*2**30) and
        all(m[k] is False for k in ('outer_scoring','candidate_selected','fairness_closed',
                                    'automatic_retry','automatic_resume')),'RUL_REPR_CONTRACT')
    local.require(len(m['jobs'])==12 and len(m['shards'])==6 and
        len(set(sum(m['shards'].values(),[])))==12 and all(len(v)==2 for v in m['shards'].values()),
        'RUL_REPR_ASSIGNMENTS')
    return m


def manifest():
    m=validate(json.loads((ROOT/MANIFEST).read_text()))
    for name,seal in m['files'].items():
        path=(ROOT/name).resolve()
        local.require(path.is_relative_to(ROOT) and path.is_file() and
            dict(bytes=path.stat().st_size,sha256=local.sha(path))==seal,'RUL_REPR_CODE_HASH:'+name)
    return m


def trajectory(ns,job,train_iterator,valid_iterator,scaler,support,guard,directory,*,device='cpu',epochs=3):
    import numpy as np
    import torch
    result=ORIGINAL_TRAJECTORY(ns,job,train_iterator,valid_iterator,scaler,support,guard,directory,
                               device=device,epochs=epochs)
    directory=Path(directory)
    if job['id'].startswith('ablation.rul.'):
        m=manifest()
        baseline=next(b for b in m['baseline_controls']['records'] if
            (b['architecture'],b['fold'],b['representation'])==
            (job['architecture'],job['inner_fold'],job['representation']))
        roles=json.loads((directory.parent/'roles.json').read_text())
        local.require(result['initial_parameter_sha256']==baseline['initial_parameter_sha256'] and
            result['effective_seed']==baseline['effective_seed'] and support==baseline['expected_support'] and
            all(roles[k]==baseline['roles'][k] for k in ('train_cells','validation_cells','target_scaler','cycle_scaler','feature_adapter')),
            'RUL_REPR_BASELINE_INITIAL_ROLES_SCALERS_SUPPORT')
        history=json.loads((directory/'history.json').read_text())
        local.require(all(h['validation_order_sha256']==baseline['validation_order_sha256'] for h in history),
                      'RUL_REPR_BASELINE_VALIDATION_ORDER')
        # Infer train endpoints at the two fixed diagnostic epochs, after all training is complete.
        # Initialization is for checkpoint loading only, not an additional learned fit.
        model,_,_=engine.initialize(ns,job,device)
        diagnostic=[]
        for epoch in (1,30):
            cp=result['checkpoints'][epoch-1]
            local.require(local.sha(directory/cp['file'])==cp['sha256'],'RUL_REPR_TRAIN_CP_HASH')
            state=torch.load(directory/cp['file'],map_location=device,weights_only=True)
            model.load_state_dict(state['state_dict'],strict=True)
            frame=train_iterator.rows if hasattr(train_iterator,'rows') else None
            local.require(frame is not None,'RUL_REPR_ITERATOR_ROWS')
            expected=dict(all_rows=len(frame))
            for t in local.TARGETS:
                spec=ns['TARGET_SPECS'][t]
                expected[t]=int((ns['normalize_bool'](frame[spec['mask_col']],spec['mask_col']) &
                    np.isfinite(frame[spec['y_col']].to_numpy(float))).sum())
            metrics,stats,order,_,seconds=engine.validate(ns,model,train_iterator,scaler,guard,device,expected)
            local.write(directory/f'train_cell_statistics_epoch_{epoch:02d}.json',stats)
            diagnostic.append(dict(epoch=epoch,checkpoint_sha256=cp['sha256'],metrics=metrics,
                                   sample_order_sha256=order,seconds=seconds))
        local.write(directory/'train_endpoint_diagnostics.json',dict(rows=diagnostic,
            trained_heads=['rul90','rul80'],soh30_head_untrained_do_not_compare=True,
            checkpoint_selected=False,optimizer_steps=0))
    result.update(task_loss_contract=ablation.LOSS_CONTRACT,trained_heads=['rul90','rul80'],
        soh30_head_untrained_do_not_compare=True,baseline_match_verified=job['id'].startswith('ablation.rul.'))
    local.write(directory/'result.json',result)
    return result


def worker(mode,bundle,run,shard,job_id):
    ORIGINAL_WORKER(mode,bundle,run,shard,job_id)
    if mode=='job':
        directory=run/'jobs'/job_id
        result=json.loads((directory/'result.json').read_text())
        result.update(status='RUL_REPR_INNER_COMPLETE_REPLAY_PENDING',task_loss_contract=ablation.LOSS_CONTRACT,
                      trained_heads=['rul90','rul80'],soh30_head_untrained_do_not_compare=True)
        local.write(directory/'result.json',result)
        policy=json.loads((directory/'inner_policy.json').read_text())
        local.write(directory/'inner_policy.json',rul_policy(policy))


def recover_job(run,job,m):
    """Restore a complete sealed trajectory; never train or resume optimizer state."""
    import hashlib
    from protocol_and_provenance.neural_classical_fairness_v1.inner_checkpoint_inputs import resolve_inner_input
    recovery=m['recovery_controls'][job['id']]
    discovery={}
    try:
        source=resolve_inner_input('/kaggle/input',recovery['inventory'],discovery)
    finally:
        local.write(run/'recovery_input_discovery.json',discovery)
    directory=run/'jobs'/job['id']
    local.require(not directory.exists(),'RUL_REPR_RECOVERY_FRESH_DIRECTORY')
    prefix=recovery['inventory']['root_prefix']+'/jobs/'+job['id']+'/'
    directory.mkdir(parents=True)
    with source.reader() as archive:
        for name in source.namelist():
            if not name.startswith(prefix):
                continue
            suffix=name[len(prefix):]
            target=(directory/suffix).resolve()
            local.require(target.is_relative_to(directory.resolve()) and target!=directory.resolve(),
                          'RUL_REPR_RECOVERY_PATH')
            raw=archive.read(name)
            seal=recovery['inventory']['members'][name]
            local.require(len(raw)==seal['bytes'] and hashlib.sha256(raw).hexdigest()==seal['sha256'],
                          'RUL_REPR_RECOVERY_MEMBER_HASH')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as handle:handle.write(raw)
    result=json.loads((directory/'result.json').read_text())
    local.require(result['epochs']==30 and result['baseline_match_verified'] is True and
        result['task_loss_contract']==ablation.LOSS_CONTRACT and
        result['history_sha256']==local.sha(directory/'trajectory/history.json') and
        result['history_sha256']==recovery['history_sha256'],'RUL_REPR_RECOVERY_RESULT')
    for cp in result['checkpoints']:
        local.require(local.sha(directory/'trajectory'/cp['file'])==cp['sha256'],
                      'RUL_REPR_RECOVERY_CHECKPOINT_HASH')
    roles=json.loads((directory/'roles.json').read_text())
    baseline=next(b for b in m['baseline_controls']['records'] if
        (b['architecture'],b['fold'],b['representation'])==
        (job['architecture'],job['inner_fold'],job['representation']))
    local.require(all(roles[k]==baseline['roles'][k] for k in baseline['roles']) and
        result['initial_parameter_sha256']==baseline['initial_parameter_sha256'] and
        result['effective_seed']==baseline['effective_seed'],'RUL_REPR_RECOVERY_BASELINE_MATCH')
    history=json.loads((directory/'trajectory/history.json').read_text())
    local.require(len(history)==30 and all(h['validation_order_sha256']==baseline['validation_order_sha256']
                                         for h in history),'RUL_REPR_RECOVERY_HISTORY')
    policy=engine.policy_map(history,result['checkpoints'],production=True)
    local.write(directory/'inner_policy.json',rul_policy(policy))
    result.update(status='RUL_REPR_RECOVERED_TRAJECTORY_REPLAY_PENDING',
                  recovered_without_training=True,recovery_source_archive=recovery['inventory']['archive'])
    local.write(directory/'result.json',result)
    local.write(directory/'recovery.json',dict(source=source.receipt(),
        original_delivery_sha256=recovery['original_delivery_sha256'],new_training_fits=0,
        optimizer_resumed=False,policy_schema_fixed=True,fresh_process_replay_required=True))


def install():
    prior.MANIFEST=MANIFEST
    prior.__file__=__file__  # inherited subprocess supervisor launches this explicit entry point
    prior.manifest=manifest
    prior.jobs=jobs
    prior.run_name=run_name
    engine.selection.training_loss=ablation.training_loss
    engine.run_trajectory=trajectory


def run(bundle,shard):
    directory=ROOT/'output'/run_name(shard)
    local.require(not directory.exists(),'RUL_REPR_FRESH_SESSION')
    directory.mkdir(parents=True);start=time.monotonic();m=None;error=None;ledger=[]
    try:
        m=manifest();gpu.configure(bundle);actual,mismatches=gpu.environment(m)
        local.write(directory/'delivery_manifest.snapshot.json',m)
        local.write(directory/'environment_check.json',dict(actual=actual,mismatches=mismatches))
        local.require(not mismatches,'RUL_REPR_ENVIRONMENT:'+json.dumps(mismatches))
        local.write(directory/'environment.lock.json',actual)
        local.write(directory/'input_mapping.json',inherited.input_checks(m,bundle))
        prefix='phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/'
        remapped=dict(config=m['source_bindings']['config'],inputs=[])
        for record in m['source_bindings']['inputs']:
            local.require(record['path'].startswith(prefix),'RUL_REPR_INPUT_PREFIX')
            name=record['path'][len(prefix):]
            local.require(name in m['input_files'],'RUL_REPR_INPUT_BINDING')
            remapped['inputs'].append(dict(record,path=str((Path(bundle)/name).resolve())))
        path=ROOT/'protocol_and_provenance/neural_classical_fairness_v1/source_bindings.json'
        local.write(path,remapped);local.write(directory/'input_bindings.remapped.json',remapped)
        local.write(directory/'session_started.json',dict(shard=shard,assigned_job_ids=m['shards'][shard],
            delivery_sha256=local.sha(ROOT/MANIFEST),bindings_sha256=local.sha(path),outer_scoring=False))
        local.require(inherited.can_start(prior.resources(directory)),'RUL_REPR_PREFLIGHT')
        deadline=start+m['budget']['session_seconds']-m['budget']['reserve_seconds']
        prior.child('qa',bundle,directory,shard,None,deadline,m)
        for jid in m['shards'][shard]:
            job=next(j for j in m['jobs'] if j['id']==jid)
            prior.resource_gate(prior.resources(directory),m['budget'])
            local.require(inherited.can_start(prior.resources(directory)) and
                time.monotonic()+1.3*(m['budget']['job_seconds']+m['budget']['replay_seconds'])<deadline,
                'RUL_REPR_ADMISSION_TIME')
            entry=dict(job=job,state='STARTED',processes=[],accepted_fits=0,accepted_epochs=0)
            ledger.append(entry);local.write(directory/'compute_ledger.json',ledger)
            print('[RUL REPR START]',jid,flush=True)
            try:
                if jid in m.get('recovery_controls',{}):
                    recover_job(directory,job,m)
                    entry.update(recovered_fits=1,new_fits=0)
                else:
                    entry['processes'].append(prior.child('job',bundle,directory,shard,job,deadline,m))
                    entry.update(recovered_fits=0,new_fits=1)
                entry['processes'].append(prior.child('replay',bundle,directory,shard,job,deadline,m))
                entry.update(state='COMPLETE',accepted_fits=1,accepted_epochs=30)
            except BaseException as exc:
                entry.update(state='FAILED_OR_PARTIAL',error=str(exc));raise
            finally:
                local.write(directory/'compute_ledger.json',ledger)
    except BaseException as exc:
        error=str(exc) or type(exc).__name__
        local.write(directory/'failure.json',dict(error=error,traceback=traceback.format_exc()))
    finally:
        assigned=m['shards'][shard] if m else []
        seen={r['job']['id'] for r in ledger}
        for jid in assigned:
            if jid not in seen:
                ledger.append(dict(job=next(j for j in m['jobs'] if j['id']==jid),state='NOT_STARTED',
                                   accepted_fits=0,accepted_epochs=0))
        completed=sum(r['state']=='COMPLETE' for r in ledger)
        local.write(directory/'compute_ledger.json',ledger)
        summary=dict(status='RUL_REPR_SHARD_COMPLETE_REVIEW_REQUIRED' if error is None and completed==2 else
            'RUL_REPR_SHARD_PARTIAL_REVIEW_REQUIRED',shard=shard,completed_jobs=completed,planned_jobs=len(assigned),
            accepted_learned_fits=completed,accepted_epoch_passes=completed*30,
            recovered_accepted_fits=sum(r.get('recovered_fits',0) for r in ledger if r['state']=='COMPLETE'),
            new_accepted_fits=sum(r.get('new_fits',0) for r in ledger if r['state']=='COMPLETE'),
            error=error,wall_seconds=time.monotonic()-start,candidate_selected=False,
            outer_scoring=False,fairness_closed=False)
        local.write(directory/'summary.json',summary)
        prior.export(directory,shard)
        print(json.dumps(summary,indent=2),flush=True)
    return 0 if error is None and completed==2 else 1


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('campaign','qa','job','replay'))
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--shard',choices=SHARDS,required=True)
    p.add_argument('--run',type=Path)
    p.add_argument('--job-id')
    a=p.parse_args();install()
    if a.mode=='campaign':
        sys.exit(run(a.bundle.resolve(),a.shard))
    local.require(a.run is not None and a.run.resolve()==ROOT/'output'/run_name(a.shard),'RUL_REPR_WORKER_OUTPUT')
    worker(a.mode,a.bundle.resolve(),a.run.resolve(),a.shard,a.job_id)

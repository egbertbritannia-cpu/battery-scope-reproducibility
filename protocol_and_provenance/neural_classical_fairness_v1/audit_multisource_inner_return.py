"""Audit one source context from the frozen multi-source inner refits.

The reviewer rebuilds input roles/preprocessors/sampling and neural metric
arithmetic. Classical metrics have no exported cell sufficient statistics;
their arithmetic/fixtures/replay receipts, not full predictions, are reviewed.
"""
from __future__ import annotations
import ast
import base64
import io
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1 import audit_remaining_cpu_return as h
from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_trajectory as engine
from protocol_and_provenance.neural_classical_fairness_v1.runtime import multisource_inner_contract as contract
from protocol_and_provenance.neural_classical_fairness_v1.runtime import selection_controls as selection

SOURCE = None
FOLDER = None
OUTPUT = None
read, require, sha = h.strict_json, h.require, h.digest


def safe_member(name, prefix):
    require(name.startswith(prefix) and '\\' not in name and ':' not in name and
            all(p not in ('', '.', '..') for p in name.split('/')), 'FULL_INNER_UNSAFE_MEMBER')


def expected_profile_reads(train_roles, validation_groups):
    # Cycle scaler reads ALL training profiles, even without an eligible window;
    # validation iteration reads only cells with active eligible anchors.
    return set(train_roles) | set(validation_groups)


def classify_protected(records, current):
    """Do not silently re-seal edited paper text as immutable science."""
    require(len(records)==len(current)==25,'FULL_INNER_PROTECTED_COUNT')
    # The manuscript and its compiled PDF are editable editorial outputs, not
    # training inputs. Keep both historical/current hashes in the review;
    # never re-seal them as unchanged scientific inputs. All other drift blocks.
    allowed={'paper/manuscript/main_vi.md','paper/latex/main.tex',
             'paper/latex/build/main.pdf'}
    drift=[]
    for r in records:
        prior={k:r[k] for k in ('bytes','sha256')}
        if current[r['path']].get('disposition')=='ARCHIVE_REMOVED_CONTENTS_VERIFIED':
            require(current[r['path']]['original_binding']==prior,'FULL_INNER_CLEANUP_ORIGINAL_BINDING')
            continue
        if current[r['path']]!=prior:
            require(r['path'] in allowed,'FULL_INNER_FROZEN_SCIENCE_DRIFT:'+r['path'])
            drift.append(dict(path=r['path'],original_protected_binding=prior,current_binding=current[r['path']],
                disposition='MANUSCRIPT_DRIFT_REVIEW_REQUIRED_NOT_OVERWRITTEN_NOT_RESEALED'))
    return drift


def protected_snapshot(records):
    """Explicitly distinguish a deleted archive from a still-existing sealed file.

    Cleanup evidence is accepted only for a formerly protected ZIP whose exact
    historical hash was logged before deletion and whose complete member set
    is still represented by verified local replacement files. Other missing
    inputs remain fatal. No historical manifest or binding is rewritten.
    """
    log=ROOT/'paper/audits/zip_cleanup_2026-10-15/deletions.jsonl'
    evidence=[read(line.encode()) for line in log.read_text(encoding='utf-8').splitlines()] if log.exists() else []
    result={}
    for r in records:
        path=ROOT/r['path']
        if path.exists():
            result[r['path']]=h.binding(path)
            continue
        matches=[e for e in evidence if e['path']==r['path']]
        require(path.suffix=='.zip' and len(matches)==1,'FULL_INNER_MISSING_PROTECTED:'+r['path'])
        e=matches[0];prior={k:r[k] for k in ('bytes','sha256')}
        require({k:e[k] for k in prior}==prior and e.get('members'),'FULL_INNER_CLEANUP_EVIDENCE')
        checked=[]
        for member in e['members']:
            replacement=(ROOT/member['replacement']).resolve()
            require(replacement.is_relative_to(ROOT.resolve()),'FULL_INNER_CLEANUP_PATH')
            binding=h.binding(replacement)
            require(binding=={k:member[k] for k in ('bytes','sha256')},'FULL_INNER_CLEANUP_CONTENT_CHANGED')
            checked.append(dict(member=member['member'],replacement=member['replacement'],**binding))
        result[r['path']]=dict(disposition='ARCHIVE_REMOVED_CONTENTS_VERIFIED',original_binding=prior,
            cleanup_log=log.relative_to(ROOT).as_posix(),verified_members=checked,
            original_archive_present=False)
    return result


def provenance():
    release=ROOT/'notebook/neural-classical-fairness-v1/multisource-inner-v1'
    receipt=read((release/'build_receipt.json').read_bytes())
    spec=read((FOLDER/'delivery.frozen.json').read_bytes())
    contract.validate(spec)
    frozen=(release/'multisource_protocol.frozen.json').read_bytes()
    require(sha(frozen)==spec['protocol_sha256']==receipt['protocol']['sha256'],
        'MULTISOURCE_AUDIT_PROTOCOL')
    manifest_data=(FOLDER/'delivery.frozen.json').read_bytes()
    for shard in contract.SHARDS:
        r=receipt['notebooks'][SOURCE+'.'+shard]
        source=release/r['path']
        require(h.binding(source)=={k:r[k] for k in ('bytes','sha256')},
            'MULTISOURCE_AUDIT_NOTEBOOK')
        nb=read(source.read_bytes()); values={}
        for cell in nb['cells']:
            if cell['cell_type']!='code':continue
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('CODE_BLOB','EXPECTED_PAYLOAD_SHA256'):
                    values[node.targets[0].id]=ast.literal_eval(node.value)
        raw=zlib.decompress(base64.b64decode(values['CODE_BLOB'],validate=True))
        require(sha(raw)==values['EXPECTED_PAYLOAD_SHA256']==r['embedded_payload_sha256'],
            'MULTISOURCE_AUDIT_PAYLOAD')
        embedded={n:base64.b64decode(v,validate=True) for n,v in read(raw).items()}
        require(embedded['multisource_inner_delivery.json']==manifest_data and
            embedded['multisource_protocol.frozen.json']==frozen,'MULTISOURCE_AUDIT_MANIFEST')
        for name,seal in spec['files'].items():
            require(dict(bytes=len(embedded[name]),sha256=sha(embedded[name]))==seal,
                'MULTISOURCE_AUDIT_EMBEDDED_CODE:'+name)
            if name=='multisource_protocol.frozen.json':
                original=release/name
            elif name=='kaggle_bootstrap.py':
                original=ROOT/'notebook/neural-classical-fairness-v1/diagnostic-v4/batteryscope_nfr1_diagnostic_v4/scripts/kaggle_bootstrap.py'
            else:original=ROOT/name
            require(h.binding(original)==seal,'MULTISOURCE_AUDIT_CURRENT_CODE:'+name)
    for name,seal in spec['input_files'].items():
        require(h.binding(local.BUNDLE/name)==seal,'MULTISOURCE_AUDIT_INPUT:'+name)
    return spec,manifest_data


def reconstruct(fold, output):
    import numpy as np
    job = next(j for j in contract.jobs(SOURCE) if j['kind']=='neural' and j['inner_fold']==fold)
    guard = SimpleNamespace(check=lambda:None)
    nd = output/f'neural_{fold}'
    ns,tr,va,cycle,scaler,*_ = local.verified_namespace(job,guard,nd,cycles=True)
    train = ns['MultitaskCycleWindowIterator'](tr,cycle,scaler)
    val = ns['MultitaskCycleWindowIterator'](va,cycle,scaler)
    counts = {}; support = dict(all_rows=len(val.rows))
    for t in local.TARGETS:
        ts = ns['TARGET_SPECS'][t]
        frame = val.rows.loc[ns['normalize_bool'](val.rows[ts['mask_col']],ts['mask_col']) & np.isfinite(val.rows[ts['y_col']].to_numpy(float))]
        counts[t] = {(str(s),str(c)):int(n) for (s,c),n in frame.groupby(['source','canonical_cell_id']).size().items()}
        support[t] = len(frame)
    exposures = []
    encoded_masks = {c:ns['encode_multitask_targets'](rows,scaler)[1] for c,rows in train.groups.items()}
    microbatches = len(train.groups) + (-len(train.groups)) % ns['TCN_GRAD_ACCUM_STEPS']
    for epoch in range(1,31):
        rng = np.random.default_rng(ns['SEED']+1009*epoch)
        cells = np.asarray(sorted(train.groups),dtype=object); rng.shuffle(cells)
        padding = (-len(cells)) % ns['TCN_GRAD_ACCUM_STEPS']
        if padding: cells=np.concatenate([cells,cells[:padding]])
        count = {t:0 for t in local.TARGETS}
        for c in cells:
            rows = train.groups[str(c)]
            idx = rng.choice(len(rows),size=ns['TCN_MICRO_BATCH_SIZE'],replace=len(rows)<ns['TCN_MICRO_BATCH_SIZE'])
            mask = encoded_masks[str(c)][idx]
            for i,t in enumerate(local.TARGETS): count[t]+=int(mask[:,i].sum())
        exposures.append(count)
    neural = dict(roles=read((nd/'roles.json').read_bytes()),access=read((nd/'access_log.json').read_bytes()),
        support=support,counts=counts,exposures=exposures,microbatches=microbatches,
        order=sha(''.join(str(s)+'\n' for s in val.rows.sample_id).encode()),
        batch=next(val.iter_prediction_batches())[0],profiles=expected_profile_reads(
            read((nd/'roles.json').read_bytes())['train_cells'],val.groups),
        profile_index=ns['cycle_store_index'],effective_seed=ns['deterministic_fold_seed']('LODO',SOURCE,fold))
    del train,val,tr,va
    cd = output/f'classical_{fold}'
    ns,tr,va,_,_,reader,train_ids,val_ids = local.verified_namespace(job,guard,cd,cycles=False)
    _, executor = local.classical_adapter()
    cols = list(dict.fromkeys(['sample_id','canonical_cell_id']+executor.feature_columns))
    ft = reader(local.BUNDLE/'phase2/tabular_feature_index.parquet',cols,train_ids)
    fv = reader(local.BUNDLE/'phase2/tabular_feature_index.parquet',cols,val_ids)
    classical = {}
    for t in local.TARGETS:
        ts = ns['TARGET_SPECS'][t]
        def join(rows,features,ids):
            f = rows.loc[ns['normalize_bool'](rows[ts['mask_col']],ts['mask_col']) & np.isfinite(rows[ts['y_col']].to_numpy(float))].copy()
            labels = reader(local.BUNDLE/'phase2/sample_target_index.parquet',['sample_id','canonical_cell_id',ts['y_col']],ids)
            f = f.drop(columns=ts['y_col']).merge(labels,on=['sample_id','canonical_cell_id'],how='left',validate='one_to_one')
            f = f.merge(features,on=['sample_id','canonical_cell_id'],how='left',validate='one_to_one',indicator=True)
            require(f._merge.eq('both').all() and np.isfinite(f[ts['y_col']]).all(), 'FULL_INNER_RECONSTRUCTION_JOIN')
            return f.drop(columns='_merge').reset_index(drop=True)
        a,b = join(tr,ft,train_ids),join(va,fv,val_ids)
        states = {}
        for family in contract.CLASSICAL_PRIORITY:
            state = executor._fit_preprocessor(a,family)
            state['fit_role']='inner_train_only_new_runtime_reconstruction'; states[family]=state
        y = a[ts['y_col']].to_numpy(float)
        y = np.log1p(y) if t.startswith('rul') else y.copy()
        centre, scale = float(y.mean()),float(y.std())
        scale = scale if math.isfinite(scale) and scale>1e-12 else 1.
        ys=(y-centre)/scale
        target_state = dict(transform=ts['transform'],centre=centre,scale=scale,scaled_min=float(ys.min()),scaled_max=float(ys.max()))
        classical[t] = dict(support=dict(training_rows=len(a),validation_rows=len(b),
            training_cell_hash=local.digest(sorted(set(a.canonical_cell_id.astype(str)))),
            training_sample_hash=local.digest(sorted(a.sample_id.astype(str))),
            validation_sample_hash=local.digest(sorted(b.sample_id.astype(str))),
            validation_cells=b.canonical_cell_id.nunique(),validation_sources=sorted(b.source.unique())),
            states=states,ridge_target=target_state,fixture=b.iloc[:256][executor.feature_columns].reset_index(drop=True))
    rebuilt = read((cd/'access_log.json').read_bytes())
    for i,t in enumerate(local.TARGETS):
        classical[t]['access']=rebuilt[:4]+[rebuilt[4],rebuilt[6+2*i],rebuilt[5],rebuilt[7+2*i]]
    classical['roles']=read((cd/'roles.json').read_bytes())
    print('[INDEPENDENT INNER INPUTS]',fold,flush=True)
    return neural,classical


def check_epoch(record, stats, context, epoch):
    require(record['epoch']==epoch and record['microbatches']==context['microbatches'] and
        record['optimizer_updates']==context['microbatches']//8 and
        record['observed_target_exposures']==context['exposures'][epoch-1], 'FULL_INNER_EPOCH_SAMPLING')
    require(record['validation_order_sha256']==context['order'] and all(math.isfinite(record[k]) and record[k]>0
        for k in ('training_seconds','validation_seconds','checkpoint_io_seconds','loss_mean_microbatch','gradient_norm_before_clip_mean')) and
        0<=record['gradient_clipping_update_fraction']<=1 and set(record['gradient_norm_before_clip_by_head'])=={'trunk',*local.TARGETS} and
        all(math.isfinite(v) and v>=0 for v in record['gradient_norm_before_clip_by_head'].values()), 'FULL_INNER_EPOCH_FINITE')
    require(set(stats)==set(record['metrics'])==set(local.TARGETS),'FULL_INNER_EPOCH_HEADS')
    for t in local.TARGETS:
        counts = {(r['source'],r['cell']):r['n'] for r in stats[t]}
        require(len(counts)==len(stats[t]) and counts==context['counts'][t], 'FULL_INNER_CELL_SUPPORT')
        require(engine.reaggregate_statistics(stats[t],t)==record['metrics'][t], 'FULL_INNER_STATISTICS_REAGGREGATION')
        h.check_metrics(record['metrics'][t],t,dict(validation_rows=context['support'][t],validation_cells=len(counts),
            validation_sources=sorted({s for s,c in counts})))


def verify_session(get, spec, shard, manifest_data):
    summary, ledger, session = get('summary.json'),get('compute_ledger.json'),get('session_started.json')
    assigned=spec['shards'][shard]; jobs=[next(j for j in spec['jobs'] if j['id']==n) for n in assigned]
    require(summary['status']=='FULL_INNER_SHARD_COMPLETE_REVIEW_REQUIRED' and summary['error'] is None and
        summary['shard']==shard and summary['completed_jobs']==summary['planned_jobs']==len(jobs), 'FULL_INNER_COUNTS')
    require([r['job'] for r in ledger]==jobs and all(r['state']=='COMPLETE' for r in ledger),'FULL_INNER_LEDGER')
    n=sum(j['kind']=='neural' for j in jobs); c=sum(j['kind']=='classical' for j in jobs)
    refs=sum(2 if j['target']=='soh30' else 3 for j in jobs if j['kind']=='references')
    require(all(summary[k]==v for k,v in dict(accepted_learned_fits=n+c,accepted_neural_epoch_passes=30*n,
        saved_neural_epoch_checkpoints=30*n,accepted_reference_executions=refs,synthetic_fits_separate=14,
        started_job_attempts=len(jobs)).items()),'FULL_INNER_SUMMARY_COUNTS')
    require(0<summary['wall_seconds']<=spec['budget']['wall_seconds'] and all(summary[k] is False for k in
        ('candidate_selected','outer_scoring','fairness_closed')), 'FULL_INNER_SUMMARY_SCOPE')
    for j,r in zip(jobs,ledger):
        require([p['mode'] for p in r['processes']]==(['job'] if j['kind']=='references' else ['job','replay']) and
            r['accepted_learned_fits']==int(j['kind']!='references') and r['accepted_epoch_passes']==(30 if j['kind']=='neural' else 0) and
            r['accepted_reference_executions']==((2 if j['target']=='soh30' else 3) if j['kind']=='references' else 0), 'FULL_INNER_PROCESS_COUNTS')
        require(all(0<p['seconds']<=spec['budget']['job_seconds' if p['mode']=='job' else 'replay_seconds'] and
            0<p['peak_tree_rss_bytes']*1.3<=4.5*2**30 for p in r['processes']), 'FULL_INNER_PROCESS_BUDGET')
    require(session['shard']==shard and session['assigned_job_ids']==assigned and session['outer_scoring'] is False and
        session['delivery_sha256']==sha(manifest_data),'FULL_INNER_SESSION_SEAL')
    for n in ('heartbeat.json','current_job.json'):
        require(get(n)['state']=='SUPERVISOR_TERMINAL','FULL_INNER_TERMINAL')
    env=get('environment_check.json'); actual=env['actual']
    require(env['mismatches']=={} and actual==get('environment.lock.json') and all(actual['numerical_device_profile'][k]==v
        for k,v in spec['expected_environment'].items()) and actual['gpu_count'] in (1,2) and actual['device']=='cuda:0' and
        actual['deterministic_algorithms'] is True and actual['amp'] is False and actual['tf32'] is False and
        actual['cudnn_benchmark'] is False and actual['classical_prediction_policies']==spec['prediction_policies'], 'FULL_INNER_ENVIRONMENT')
    mapped=get('input_mapping.json'); remapped=get('input_bindings.remapped.json')
    require(len(mapped)==len(spec['input_files']) and {r['bundle_relative'] for r in mapped}==set(spec['input_files']), 'FULL_INNER_INPUT_MAPPING')
    for r in mapped:
        seal=spec['input_files'][r['bundle_relative']]
        require({k:r[k] for k in seal}==seal and r['mounted_path'].startswith('/kaggle/input/') and
            r['mounted_path'].endswith('/'+r['bundle_relative']), 'FULL_INNER_INPUT_MAPPING_SEAL')
    require(remapped['config']==spec['source_bindings']['config'] and len(remapped['inputs'])==len(spec['source_bindings']['inputs']), 'FULL_INNER_REMAP')
    for r,original in zip(remapped['inputs'],spec['source_bindings']['inputs']):
        require({k:v for k,v in r.items() if k!='path'}=={k:v for k,v in original.items() if k!='path'} and
            r['path'] in {x['mounted_path'] for x in mapped}, 'FULL_INNER_REMAP_PATH')
    qa=get('synthetic_neural_qa/summary.json')
    require(qa['status']=='ACTUAL_ENGINE_TCN_GRU_PREFIX_RELOAD_SYNTHETIC_PASS' and qa['synthetic_fits']==12 and
        qa['synthetic_epoch_passes']==30 and qa['battery_fits']==0 and qa['device']=='cuda:0' and qa['checks']==[
            dict(architecture=a,candidate=c,prefix_epoch=2,exact=True) for a in ('TCN','GRU') for c in selection.PRIORITY], 'FULL_INNER_NEURAL_QA')
    require(get('synthetic_classical_qa.json')==dict(status='SYNTHETIC_PREDICT_RELOAD_PASS',families=['extra_trees','hist_gradient_boosting'],
        synthetic_fits=2,battery_fits=0,full_dataset_replay=False),'FULL_INNER_CLASSICAL_QA')
    return summary,ledger,session,actual


def selector_record(record):
    """Map serialized engine metrics to the frozen selector's logical schema.

    Engine uses rmse_or_rmsle for both endpoint types; the selector names the
    RUL metric rmsle. This is an exact alias, never a recalculation or change
    to source weights, checkpoint selection, candidates or observed scores.
    Raw validation records retain their original metric keys.
    """
    if record['endpoint']=='soh30':
        return record
    result=dict(record)
    result['by_source']={s:dict(values,rmsle=values['rmse_or_rmsle'])
                         for s,values in record['by_source'].items()}
    return result


def run():
    import numpy as np
    import pandas as pd
    require(not (OUTPUT/'review.json').exists(),'FULL_INNER_REFUSE_OVERWRITE_ACCEPTED_REVIEW')
    spec,manifest_data=provenance()
    audit_source=h.binding(Path(__file__))
    protected=read((h.LOCAL/'launch_manifest.json').read_bytes())['protected']
    protected_before=protected_snapshot(protected)
    manuscript_drift=classify_protected(protected,protected_before)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    attempt=Path(tempfile.mkdtemp(prefix='independent_',dir=OUTPUT))
    contexts={f:reconstruct(f,attempt) for f in range(3)}
    rows=[]; curves=[]; timings=[]; reports=[]; seen=set(); policies=[]; updates=0
    for shard in contract.SHARDS:
        archive=FOLDER.parent/'returns'/SOURCE/shard/f'batteryscope_nfr1_multisource_inner_v1_{SOURCE}_{shard}_results.zip'; before=h.binding(archive)
        prefix=f'batteryscope_nfr1_multisource_inner_v1_{SOURCE}_{shard}/'
        with zipfile.ZipFile(archive) as z:
            members=z.infolist(); names=[m.filename for m in members]
            require(len(names)==len(set(names)) and sum(m.file_size for m in members)<=spec['budget']['output_bytes']+2**20, 'FULL_INNER_ZIP_MEMBERS_SIZE')
            for m in members:
                safe_member(m.filename,prefix)
                require(not m.is_dir() and ((m.external_attr>>16)&0o170000)!=0o120000,'FULL_INNER_ZIP_TYPE')
            get=lambda n:read(z.read(prefix+n))
            inventory=get('output_inventory.json'); seals=inventory['files']
            require(set(names)=={prefix+n for n in seals}|{prefix+'output_inventory.json'} and
                inventory['outer_scoring'] is False and inventory['fairness_closed'] is False,'FULL_INNER_INVENTORY_COVERAGE')
            for name,seal in seals.items():
                data=z.read(prefix+name)  # ZIP CRC checked on every complete read.
                require(dict(bytes=len(data),sha256=sha(data))==seal,'FULL_INNER_OUTPUT_HASH:'+name)
                if name.endswith('.json'):read(data)
            require(not any(n.endswith('/failure.json') or n.endswith('/STOP') or n.endswith('.partial') for n in names),'FULL_INNER_FAILURE_ARTIFACT')
            require(get('delivery_manifest.snapshot.json')==spec,'FULL_INNER_DELIVERY_SNAPSHOT')
            summary,ledger,session,actual=verify_session(get,spec,shard,manifest_data)
            require(session['bindings_sha256']==sha(z.read(prefix+'input_bindings.remapped.json')),'FULL_INNER_BINDINGS_BYTES')
            require({n[len(prefix):].split('/')[1] for n in names if n.startswith(prefix+'jobs/')}==set(spec['shards'][shard]),'FULL_INNER_JOB_DIRECTORIES')
            for entry in ledger:
                j=entry['job']; jid=j['id']; require(jid not in seen,'FULL_INNER_DUPLICATE_JOB'); seen.add(jid)
                p='jobs/'+jid+'/'; result=get(p+'result.json'); neural, classical=contexts[j['inner_fold']]
                require(get(p+'started.json')==dict(job=j,delivery_sha256=session['delivery_sha256'],fit_role='FULL_INNER_NEW_RUNTIME_NOT_OUTER',pilot_artifact_imported=False),'FULL_INNER_JOB_START')
                require(result['fit_role']=='FULL_INNER_NEW_RUNTIME_NOT_OUTER' and result['is_bounded_pilot'] is False and
                    all(result[k] is False for k in ('candidate_selected','outer_scoring','calibration_scoring','fairness_closed')) and
                    0<result['wall_seconds']<=3600 and 0<result['peak_sampled_rss_bytes']*1.3<=4.5*2**30,'FULL_INNER_JOB_ROLE')
                if j['kind']=='neural':
                    t=p+'trajectory/'; trajectory=get(t+'result.json'); history=get(t+'history.json')
                    require(result['status']=='FULL_INNER_NEURAL_COMPLETE_REPLAY_PENDING' and result['epochs']==result['full_trajectory_epochs']==result['epoch_passes']==30 and
                        result['learned_fits']==1 and result['inner_selection_fit'] is True and result['full_validation'] is True and
                        result['interval_eligible']==(j['candidate']=='Q0'),'FULL_INNER_NEURAL_ROLE')
                    require(all(result[k]==v for k,v in trajectory.items() if k!='status') and
                        trajectory['status']=='F2_TRAJECTORY_COMPLETE_REPLAY_PENDING' and result['production_fit'] is False and
                        result['cross_fold_candidate_selected'] is False,'FULL_INNER_ENGINE_RECEIPT')
                    require(get(p+'roles.json')==neural['roles'] and get(p+'access_log.json')==neural['access'] and
                        get(p+'expected_support.json')==neural['support'],'FULL_INNER_NEURAL_ROLES_SCALERS_ACCESS')
                    profiles=get(p+'profile_hashes.json')
                    require(set(profiles)==neural['profiles'] and all(v==str(neural['profile_index'].loc[c]['sha256']).lower() for c,v in profiles.items()),'FULL_INNER_PROFILE_HASHES')
                    require(result['history_sha256']==seals[t+'history.json']['sha256'] and result['validation_order_sha256']==neural['order'] and
                        result['effective_seed']==neural['effective_seed'] and len(history)==len(result['checkpoints'])==30,'FULL_INNER_NEURAL_SEED_HISTORY')
                    start=get(t+'trajectory_started.json')
                    require(start['job']==j and start['planned_epochs']==30 and start['learned_fit_initializations']==1 and
                        start['initial_parameter_sha256']==result['initial_parameter_sha256'] and start['effective_seed']==neural['effective_seed'],'FULL_INNER_INITIALIZATION')
                    require(len({r['parameter_sha256'] for r in history}|{result['initial_parameter_sha256']})==31,'FULL_INNER_PARAMETER_UPDATES')
                    for epoch,(record,cp) in enumerate(zip(history,result['checkpoints']),1):
                        require(cp['epoch']==epoch and cp['file']==f'checkpoint_epoch_{epoch:02d}.pt' and
                            {k:cp[k] for k in ('sha256','bytes')}==seals[t+cp['file']],'FULL_INNER_CHECKPOINT')
                        check_epoch(record,get(t+f'cell_statistics_epoch_{epoch:02d}.json'),neural,epoch)
                        updates+=record['optimizer_updates']
                        for endpoint in local.TARGETS:
                            curves.append(dict(job_id=jid,epoch=epoch,target=endpoint,metric=record['metrics'][endpoint],
                                gradient_clipping_update_fraction=record['gradient_clipping_update_fraction']))
                    policy=contract.accept_trajectory(history,result['checkpoints'])
                    require(policy==get(p+'inner_policy.json') and engine.policy_map(history,result['checkpoints'])==get(t+'policies.descriptive.json'),'FULL_INNER_POLICY_RECOMPUTE')
                    policies.append(dict(job_id=jid,**policy))
                    require(get(t+'replay.json')==dict(status='F2_FRESH_PROCESS_BOUNDED_EXACT_PASS',scope='last_epoch_first_full_validation_batch_only',full_validation_replay=False,labels_read=False),'FULL_INNER_NEURAL_REPLAY_RECEIPT')
                    require(seals[t+'replay_x.npy']['sha256']==result['fixture']['x_sha256'] and
                        seals[t+'replay_expected.npz']['sha256']==result['fixture']['expected_sha256'],'FULL_INNER_NEURAL_FIXTURE_HASH')
                    x=np.load(io.BytesIO(z.read(prefix+t+'replay_x.npy')),allow_pickle=False)
                    require(x.dtype==np.float32 and np.array_equal(x,neural['batch']),'FULL_INNER_NEURAL_INPUT_FIXTURE')
                    with np.load(io.BytesIO(z.read(prefix+t+'replay_expected.npz')),allow_pickle=False) as pred:
                        require(set(pred.files)==set(local.TARGETS) and all(pred[t].shape==(len(x),3) and np.isfinite(pred[t]).all() and
                            (np.diff(pred[t],axis=1)>=0).all() for t in local.TARGETS),'FULL_INNER_PREDICTION_FIXTURE')
                    require(0<result['peak_cuda_allocated_bytes']<=result['peak_cuda_reserved_bytes'],'FULL_INNER_CUDA_MEMORY')
                    for endpoint in local.TARGETS:
                        s0=policy['policies']['S0'][endpoint]; s1=policy['policies']['S1'][endpoint]
                        rows.append(dict(job_id=jid,kind='neural',source=SOURCE,architecture=j['architecture'],model=j['architecture'],
                            by_source=history[s1['best_epoch']-1]['metrics'][endpoint]['by_source'],candidate=j['candidate'],
                            inner_fold=j['inner_fold'],endpoint=endpoint,validation_scope='full_eligible_inner',selection_policy='S1',
                            best_epoch=s1['best_epoch'],best_score=s1['best_score'],S0_epoch=s0['best_epoch'],S0_score=s0['score_at_shared_epoch'],
                            score_epoch30=history[-1]['metrics'][endpoint]['selection_score'],artifact_audit_accepted=True,
                            seed=2026,outer_scoring=False))
                else:
                    target=j['target']; context=classical[target]; support=context['support']
                    require(get(p+'roles.json')==classical['roles'] and get(p+'access_log.json')==context['access'],'FULL_INNER_CLASSICAL_ROLES_ACCESS')
                    if j['kind']=='classical':
                        seal=get(p+'model.integrity.json'); grid=spec['classical_grid'][j['candidate']]; family=grid['family']
                        require(result['status']=='FULL_INNER_CLASSICAL_COMPLETE_REPLAY_PENDING' and result['learned_fits']==1 and result['inner_selection_fit'] is True and
                            result['full_validation'] is True and all(result[k]==support[k] for k in ('training_rows','validation_rows','validation_sample_hash')),
                            'FULL_INNER_CLASSICAL_SUPPORT')
                        require(all(seal[k]==support[k] for k in ('training_cell_hash','training_sample_hash')) and
                            seal['feature_state']==context['states'][family] and seal['target_state']==(context['ridge_target'] if family=='ridge' else
                                dict(transform='identity' if target=='soh30' else 'log1p')),'FULL_INNER_CLASSICAL_PREPROCESSOR')
                        require(all(seal['estimator_params'][k]==v for k,v in grid['params'].items()) and seal['prediction_policy']==result['prediction_policy']==spec['prediction_policies'][family] and
                            seal['provenance']==result['provenance']=='NEW_RUNTIME_INNER_RECONSTRUCTION_NOT_HISTORICAL_REPLAY','FULL_INNER_CLASSICAL_GRID_PROVENANCE')
                        for n,k in (('pipeline.joblib','sha256'),('replay_features.parquet','feature_sha256'),('replay_expected.npy','expected_sha256')):
                            require(seals[p+n]['sha256']==seal[k],'FULL_INNER_CLASSICAL_ARTIFACT_SEAL')
                        require(get(p+'replay.json')==dict(status='FRESH_PROCESS_FORWARD_FIXTURE_EXACT_PASS',scope='first_256_full_validation_rows_only',full_validation_replay=False,labels_read=False),
                            'FULL_INNER_CLASSICAL_REPLAY_RECEIPT')
                        pd.testing.assert_frame_equal(pd.read_parquet(io.BytesIO(z.read(prefix+p+'replay_features.parquet'))),context['fixture'])
                        pred=np.load(io.BytesIO(z.read(prefix+p+'replay_expected.npy')),allow_pickle=False)
                        require(pred.shape==(len(context['fixture']),) and np.isfinite(pred).all() and (pred>=0).all() and
                            (target!='soh30' or (pred<=1.5).all()),'FULL_INNER_CLASSICAL_FIXTURE')
                        h.check_metrics(result['metrics'],target,support)
                        rows.append(dict(job_id=jid,kind='classical',source=SOURCE,family=family,model=family,by_source=result['metrics']['by_source'],candidate=j['candidate'],inner_fold=j['inner_fold'],
                            endpoint=target,score=result['metrics']['selection_score'],metric=result['metrics'],validation_scope='full_eligible_inner',
                            artifact_audit_accepted=True,seed=2026,outer_scoring=False))
                    else:
                        expected={'soh_persistence','causal_linear_trend'} if target=='soh30' else {'rul_train_median','rul_train_log_mean','causal_linear_trend'}
                        require(result['status']=='FULL_INNER_REFERENCES_COMPLETE' and result['learned_fits']==0 and result['reference_executions']==len(expected) and
                            set(result['metrics'])==expected and result['metrics']==get(p+'reference_metrics.json'),'FULL_INNER_REFERENCES')
                        for name,metric in result['metrics'].items():
                            h.check_metrics(metric,target,support,reference=True)
                            rows.append(dict(job_id=jid,kind='references',reference=name,inner_fold=j['inner_fold'],endpoint=target,metric=metric))
                timings.append(dict(job_id=jid,kind=j['kind'],fold=j['inner_fold'],wall_seconds=result['wall_seconds'],processes=entry['processes'],
                    peak_sampled_rss_bytes=result['peak_sampled_rss_bytes']))
            reports.append(dict(shard=shard,archive=before,members=len(names),inventory_hashes=len(seals),summary=summary,
                environment=actual['numerical_device_profile'],gpu_count=actual['gpu_count']))
        require(before==h.binding(archive),'FULL_INNER_ARCHIVE_CHANGED')
        print('[SHARD AUDIT PASS]',shard,len(ledger),flush=True)
    require(seen=={j['id'] for j in spec['jobs']} and len(seen)==108,'FULL_INNER_COMPLETE_UNION')
    require(protected_before==protected_snapshot(protected) and
        h.binding(Path(__file__))==audit_source,'FULL_INNER_FILES_CHANGED_DURING_AUDIT')
    # Deterministic inner-rule recomputation, not an outer release/authorization.
    selection_preview=[]
    for a in ('TCN','GRU'):
        for t in local.TARGETS:
            selected=contract.choose([selector_record(r) for r in rows if r['kind']=='neural' and r['architecture']==a and r['endpoint']==t],SOURCE,a,t,neural=True)
            selection_preview.append(dict(kind='neural',**selected))
    for family in contract.CLASSICAL_PRIORITY:
        for t in local.TARGETS:
            selected=contract.choose([selector_record(r) for r in rows if r['kind']=='classical' and r['family']==family and r['endpoint']==t],SOURCE,family,t,neural=False)
            selection_preview.append(dict(kind='classical',**selected))
    review=dict(status='ACCEPTED_MULTISOURCE_ONE_CONTEXT99_ARTIFACTS_OUTER_AND_FAIRNESS_OPEN',source=SOURCE,date='2026-10-15',shards=reports,
        accepted_jobs=108,neural_fits=18,classical_fits=81,learned_fits=99,neural_epoch_passes=540,
        neural_checkpoints=540,reference_jobs=9,reference_executions=24,synthetic_fits_separate=56,
        neural_metric_records=len(curves),validation_records=len(rows),optimizer_updates=updates,
        independent_reconstruction_directory=str(attempt.relative_to(ROOT)),inner_contexts_reconstructed=3,
        protected_files_matching_original=sum(protected_before[r['path']]=={k:r[k] for k in ('bytes','sha256')} for r in protected),
        protected_file_or_replacement_snapshots_unchanged_during_this_audit=len(protected),
        cleaned_protected_archives={p:v for p,v in protected_before.items() if v.get('disposition')=='ARCHIVE_REMOVED_CONTENTS_VERIFIED'},
        manuscript_protected_binding_drift=manuscript_drift,manuscript_readiness='PROTECTED_BINDING_DRIFT_REVIEW_REQUIRED' if manuscript_drift else 'NOT_ASSESSED',
        audit_source=audit_source,new_model_fits=0,reviewer_deserialized_models=False,
        reviewer_reran_predictions=False,neural_cell_statistics_reaggregated=True,
        classical_full_metrics_independently_recomputed=False,fresh_process_bounded_replay_receipts=99,
        executed_notebooks_provided=False,candidate_promoted=False,outer_launch_authorized=False,
        calibration_scoring=False,outer_scoring=False,fairness_closed=False,global_fairness_closed=False,
        limitations=[f'{SOURCE} held-source context only; inner validation sources are not the held-source test cells.',
            'Exact replay receipts concern only last neural epoch first batch or first256 classical rows; not full prediction replay.',
            'Classical metrics lack full predictions/cell statistics; source-balanced arithmetic checked, full metrics not recomputed.',
            'No executed notebook supplied; frozen source, output hashes and runtime receipts are not independent execution attestation.',
            'Inner selection preview is not outer performance, convergence proof, production release or explanation of historical CALB seed2027.',
            'Input preprocessing rebuilt without fits; representation, objectives, precision, exposure and cost are not matched.'])
    local.write(OUTPUT/'epoch_metrics.json',curves)
    local.write(OUTPUT/'validation_records.json',rows)
    local.write(OUTPUT/'selection_rule_preview.json',dict(status='RECOMPUTED_INNER_RULES_NOT_PRODUCTION_RELEASE',rows=selection_preview,outer_scoring=False,candidate_promoted=False))
    local.write(OUTPUT/'policies.json',policies); local.write(OUTPUT/'timing_memory.json',timings)
    local.write(OUTPUT/'review.json',review)
    print('ACCEPTED',SOURCE,'FULL INNER: 99 fits / 540 epochs / 108 jobs; fairness open',flush=True)
    return review


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=contract.SOURCES,required=True)
    args=parser.parse_args()
    SOURCE=args.source
    FOLDER=ROOT/'notebook/neural-classical-fairness-v1/multisource-inner-v1'/SOURCE
    OUTPUT=ROOT/'paper/audits/neural_classical_fairness_2026-10-08_v1/multisource_inner_return_2026-10-15_v1'/SOURCE
    run()

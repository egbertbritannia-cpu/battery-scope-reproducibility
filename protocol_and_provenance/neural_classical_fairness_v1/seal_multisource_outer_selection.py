"""Bind all audited inner records to original archives; freeze selected outer jobs.

Reads inner outputs and outer membership metadata only. No target/profile reads,
model deserialization, fitting, prediction or test scoring.
"""
from __future__ import annotations
import hashlib,json,sys,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
from protocol_and_provenance.neural_classical_fairness_v1.runtime import multisource_inner_contract as inner
from protocol_and_provenance.neural_classical_fairness_v1 import audit_multisource_inner_return as audit
BASE=ROOT/'paper/audits/neural_classical_fairness_2026-10-08_v1'
RELEASE=ROOT/'notebook/neural-classical-fairness-v1/multisource-inner-v1'
OUT=BASE/'multisource_outer_selection_v1'
SEEDS=(2026,2027,2028,2029,2030)
def binding(p):return dict(bytes=p.stat().st_size,sha256=local.sha(p))
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def effective(seed,source):return seed+int(hashlib.sha256(f'{seed}|LODO|{source}|outer_refit|multitask'.encode()).hexdigest()[:8],16)%1000000
def jobs(source,selected):
 result=[];views=[]
 for architecture in ('TCN','GRU'):
  for candidate in ('Q0','P0','P1'):
   heads=[r for r in selected if r['kind']=='neural' and r['model']==architecture and r['candidate']==candidate]
   if not heads:continue
   exports=[dict(endpoint=r['endpoint'],policy='S1',epoch=r['outer_epoch'],interval_eligible=False,output_type='POINT_ONLY') for r in heads]
   for seed in SEEDS:
    jid=f'outer.{source}.{architecture}.{candidate}.{seed}'
    result.append(dict(id=jid,kind='neural',source=source,architecture=architecture,candidate=candidate,seed=seed,
     effective_seed=effective(seed,source),fit_epochs=max(r['epoch'] for r in exports),checkpoint_exports=exports,outer_scoring=False))
    views.extend(dict(view='S1-selected',architecture=architecture,endpoint=r['endpoint'],candidate=candidate,seed=seed,job_id=jid,epoch=r['outer_epoch']) for r in heads)
 for family in inner.CLASSICAL_PRIORITY:
  for target in local.TARGETS:
   r=next(r for r in selected if r['kind']=='classical' and r['model']==family and r['endpoint']==target)
   for seed in ((2026,) if family=='ridge' else SEEDS):
    candidate=r['candidate'];params=dict(local.grid()[candidate]['params'])
    if family!='ridge':params['random_state']=seed
    result.append(dict(id=f'outer.{source}.{candidate}.{target}.{seed}',kind='classical',source=source,family=family,
      candidate=candidate,target=target,seed=seed,config_roles=['selected'],params=params,outer_scoring=False))
 for target in local.TARGETS:
  result.append(dict(id=f'outer.{source}.references.{target}',kind='references',source=source,target=target,
    learned_fits=0,executions=3,outer_scoring=False))
 counts=dict(neural_fits=sum(j['kind']=='neural' for j in result),classical_fits=sum(j['kind']=='classical' for j in result),
  neural_epoch_passes=sum(j.get('fit_epochs',0) for j in result),reference_jobs=3,reference_executions=9,total_jobs=len(result))
 counts['learned_fits']=counts['neural_fits']+counts['classical_fits']
 assert counts['classical_fits']==33 and counts['learned_fits']<=63 and len(views)==30
 return result,views,counts
def verify_records(source,folder,review,records):
 assert len(records)==159 and sum(r['kind']=='neural' for r in records)==54 and sum(r['kind']=='classical' for r in records)==81
 expected={r['job_id'] for r in records};seen=set();archives=[]
 for shard in review['shards']:
  p=RELEASE/'returns'/source/shard['shard']/f'batteryscope_nfr1_multisource_inner_v1_{source}_{shard["shard"]}_results.zip'
  assert binding(p)==shard['archive'],'INNER_ARCHIVE_CHANGED'
  prefix=f'batteryscope_nfr1_multisource_inner_v1_{source}_{shard["shard"]}/'
  with zipfile.ZipFile(p) as z:
   inv=json.loads(z.read(prefix+'output_inventory.json'))['files']
   def get(name):
    data=z.read(prefix+name);assert dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==inv[name]
    return json.loads(data)
   assert get('delivery_manifest.snapshot.json')==json.loads((RELEASE/source/'delivery.frozen.json').read_text())
   for entry in get('compute_ledger.json'):
    j=entry['job'];jid=j['id'];assert entry['state']=='COMPLETE';seen.add(jid)
    rows=[r for r in records if r['job_id']==jid];pfx='jobs/'+jid+'/'
    result=get(pfx+'result.json')
    if j['kind']=='neural':
     policy=get(pfx+'inner_policy.json');history=get(pfx+'trajectory/history.json')
     assert len(rows)==3
     for r in rows:
      assert r['source']==source and r['model']==r['architecture']==j['architecture'] and r['candidate']==j['candidate'] and r['inner_fold']==j['inner_fold']
      t=r['endpoint'];s1=policy['policies']['S1'][t];assert r['best_epoch']==s1['best_epoch'] and r['selection_policy']=='S1'
      assert r['by_source']==history[r['best_epoch']-1]['metrics'][t]['by_source']
    elif j['kind']=='classical':
     assert len(rows)==1;r=rows[0]
     assert r['source']==source and r['model']==r['family']==local.grid()[j['candidate']]['family'] and r['candidate']==j['candidate'] and r['endpoint']==j['target'] and r['inner_fold']==j['inner_fold']
     assert r['metric']==result['metrics'] and r['by_source']==result['metrics']['by_source']
    else:
     assert len(rows)==len(result['metrics'])
     assert {r['reference']:r['metric'] for r in rows}==result['metrics']
  archives.append(dict(path=p.relative_to(ROOT).as_posix(),**shard['archive']))
 assert seen==expected
 return archives
def main():
 assert not OUT.exists(),'NO_OVERWRITE_SELECTION'
 protocol=json.loads((RELEASE/'multisource_protocol.frozen.json').read_text());assert protocol['outer_plan']['seed_bases']==list(SEEDS)
 import pandas as pd
 membership=local.BUNDLE/'phase2/battery_phase2_fold_membership.parquet';frame=pd.read_parquet(membership)
 assert binding(membership)==protocol['inputs']['phase2/battery_phase2_fold_membership.parquet']
 completion=json.loads((BASE/'source_group_dispatch_2026-10-15_v2/inner_completion_20261016.json').read_text())
 seals={r['source']:r for r in completion['sources']};all_sources={}
 for source in inner.SOURCES:
  folder=BASE/'multisource_inner_return_2026-10-15_v1'/source
  review=json.loads((folder/'review.json').read_text());assert binding(folder/'review.json')['sha256']==seals[source]['sha256']
  assert review['status']=='ACCEPTED_MULTISOURCE_ONE_CONTEXT99_ARTIFACTS_OUTER_AND_FAIRNESS_OPEN' and review['learned_fits']==99 and review['outer_scoring'] is False
  records=json.loads((folder/'validation_records.json').read_text());archives=verify_records(source,folder,review,records)
  chosen=[]
  for model in ('TCN','GRU',*inner.CLASSICAL_PRIORITY):
   neural=model in ('TCN','GRU')
   for target in local.TARGETS:
    rs=[audit.selector_record(r) for r in records if r['kind']==('neural' if neural else 'classical') and r['model']==model and r['endpoint']==target]
    chosen.append(dict(kind='neural' if neural else 'classical',**inner.choose(rs,source,model,target,neural=neural)))
  assert chosen==json.loads((folder/'selection_rule_preview.json').read_text())['rows'],'SELECTION_PREVIEW_DRIFT'
  held=frame.loc[frame.evaluation_mode.eq('LODO') & frame.outer_entity.eq(source)]
  assert len(held)>0 and held.canonical_cell_id.is_unique
  roles={};familysets={}
  for role in ('development','calibration','test'):
   f=held.loc[held.split.eq(role)];cells=sorted(f.canonical_cell_id.astype(str));families=sorted(set(f.duplicate_family_id.astype(str)));assert cells
   roles[role]=dict(cells=len(cells),cell_ids=cells,cell_hash=local.digest(cells),family_hash=local.digest(families));familysets[role]=set(families)
  for left in roles:
   for right in roles:
    if left!=right:assert not set(roles[left]['cell_ids'])&set(roles[right]['cell_ids']) and not familysets[left]&familysets[right]
  assert not held.loc[held.split.eq('development'),'source'].eq(source).any() and held.loc[held.split.eq('test'),'source'].eq(source).all()
  queue,views,counts=jobs(source,chosen)
  selected=dict(schema='nfr1-multisource-outer-selection-v1',source=source,status='SELECTION_FROZEN_OUTER_DELIVERY_PENDING',
   frozen_at_utc=datetime.now(timezone.utc).isoformat(),chronology=protocol['chronology'],protocol=binding(RELEASE/'multisource_protocol.frozen.json'),
   accepted_inner_review=binding(folder/'review.json'),accepted_inner_archives=archives,
   accepted_inner_artifacts={n:binding(folder/n) for n in ('validation_records.json','selection_rule_preview.json','policies.json','timing_memory.json')},
   outer_seed_bases=list(SEEDS),selection=chosen,jobs=queue,neural_prediction_views=views,counts=counts,
   outer_role_metadata=dict(membership=binding(membership),roles=roles,target_or_profile_reads=False),
   input_files=protocol['inputs'],source_bindings=protocol['source_bindings'],expected_environment=protocol['expected_environment'],
   neural_recipes=protocol['neural_recipes'],classical_grid=protocol['classical_grid'],classical_prediction_policies=protocol['prediction_policies'],
   reference_mean_soh30='Arithmetic mean of development eligible observed SoH30 labels, clipped to [0,1.5]; RUL mean is inherited expm1(mean(log1p(y)))',
   disclosure=['Post hoc source comparison; no preregistration claim.','Five refit seeds conditional on one inner-selection seed.',
    'Representations, training objectives, sampling and compute differ by family.','No intervals or calibration-role reads.','Prediction artifacts must pass audit before separate scoring.'],
   outer_scoring=False,outer_labels_accessed=False,calibration_labels_accessed=False,fairness_closed=False)
  dump(OUT/source/'selection.frozen.json',selected);all_sources[source]=dict(selection=binding(OUT/source/'selection.frozen.json'),counts=counts)
  print(source,counts,flush=True)
 dump(OUT/'index.frozen.json',dict(schema='nfr1-multisource-outer-selection-index-v1',sources=all_sources,protocol=binding(RELEASE/'multisource_protocol.frozen.json'),builder=binding(Path(__file__)),
  outer_labels_accessed=False,outer_fits_started=False,counts={k:sum(v['counts'][k] for v in all_sources.values()) for k in next(iter(all_sources.values()))['counts']}))
if __name__=='__main__':main()

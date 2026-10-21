"""Quota-bounded outer sessions: import completed fits, pause between whole jobs."""
from __future__ import annotations
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1.runtime import multisource_outer_kaggle as old
from protocol_and_provenance.neural_classical_fairness_v1.runtime import outer_continuation as carry
from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
def run(bundle):
 cfg=json.loads((ROOT/'outer_lease.json').read_text());m,s=old.manifest()
 assert cfg['source']==m['source'] and cfg['selection_sha256']==m['selection_sha256'] and cfg['scientific_manifest_sha256']==local.sha(ROOT/old.MANIFEST)
 for name,seal in cfg['continuation_code'].items():assert carry.binding(ROOT/name)==seal
 assert 3600<=cfg['lease_seconds']<=28800 and cfg['reserve_seconds']==1800
 run=ROOT/'output'/old.run_name('ALL');assert not run.exists();run.mkdir(parents=True)
 started=time.monotonic();deadline=started+cfg['lease_seconds']-cfg['reserve_seconds'];ledger=[];error=None;paused=False
 try:
  old.pilot.configure(bundle);actual,mismatches=old.pilot.environment(m)
  local.write(run/'delivery_manifest.snapshot.json',m);local.write(run/'selection.snapshot.json',s);local.write(run/'lease.snapshot.json',cfg)
  local.write(run/'environment_check.json',dict(actual=actual,mismatches=mismatches));assert not mismatches
  local.write(run/'environment.lock.json',actual);local.write(run/'input_mapping.json',old.inherited.input_checks(m,bundle))
  local.write(run/'session_started.json',dict(shard='ALL',assigned_job_ids=m['shards']['ALL'],delivery_sha256=local.sha(ROOT/old.MANIFEST),selection_sha256=m['selection_sha256'],outer_scoring=False))
  if cfg.get('carry'):
   c=cfg['carry'];matches=list(Path('/kaggle/input').rglob(c['filename']));assert len(matches)==1,'EXACT_PRIVATE_HANDOFF_REQUIRED'
   proof=carry.restore(matches[0],run,m,c);assert proof['environment']['numerical_device_profile']==actual['numerical_device_profile'],'CARRY_NUMERICAL_PROFILE_DRIFT'
   ledger=list(proof['completed']);local.write(run/'carry_import.json',dict(archive=proof['archive'],completed_job_ids=c['completed_job_ids'],prior_environment=proof['environment']))
   local.write(run/'carry_provenance.json',proof['origins'])
  local.write(run/'compute_ledger.json',ledger)
  snapshot=old.resources(run);local.write(run/'preflight.json',snapshot);assert old.inherited.can_start(snapshot)
  local.write(run/'synthetic_qa_process.json',old.child('qa',bundle,run,'ALL',None,deadline,m))
  done={e['job']['id'] for e in ledger}
  for jid in m['shards']['ALL']:
   if jid in done:continue
   job=next(j for j in m['jobs'] if j['id']==jid);caps=cfg['job_caps'][jid]
   assert 1<=caps['job_seconds']<=m['budget']['job_seconds'] and 1<=caps['replay_seconds']<=m['budget']['replay_seconds']
   replay=0 if job['kind']=='references' else caps['replay_seconds']
   if time.monotonic()+1.3*(caps['job_seconds']+replay)>=deadline:paused=True;break
   snapshot=old.resources(run);old.monitor_gate(snapshot,m['budget']);assert old.inherited.can_start(snapshot)
   entry=dict(job=job,state='WORKER_STARTED',processes=[],accepted_learned_fits=0,accepted_epoch_passes=0,accepted_reference_executions=0,lease_id=cfg['lease_id'],execution_caps=caps)
   ledger.append(entry);local.write(run/'compute_ledger.json',ledger);print('[LEASE START]',jid,flush=True)
   limited=dict(m,budget=dict(m['budget'],job_seconds=caps['job_seconds'],replay_seconds=caps['replay_seconds']))
   try:
    entry['processes'].append(old.child('job',bundle,run,'ALL',job,deadline,limited))
    if job['kind']!='references':entry['processes'].append(old.child('replay',bundle,run,'ALL',job,deadline,limited))
    result=json.loads((run/'jobs'/jid/'result.json').read_text());expected={'neural':'MULTI_OUTER_NEURAL_FIT_COMPLETE_REPLAY_PENDING','classical':'MULTI_OUTER_CLASSICAL_FIT_COMPLETE_REPLAY_PENDING','references':'MULTI_OUTER_REFERENCES_COMPLETE'}[job['kind']]
    assert result['status']==expected
    entry.update(state='COMPLETE',accepted_learned_fits=int(job['kind']!='references'),accepted_epoch_passes=job.get('fit_epochs',0),accepted_reference_executions=result.get('reference_executions',0));done.add(jid)
   except BaseException as exc:entry.update(state='FAILED_OR_PARTIAL',error=str(exc));raise
   local.write(run/'compute_ledger.json',ledger);print('[LEASE COMPLETE]',jid,flush=True)
 except BaseException as exc:
  error=str(exc) or type(exc).__name__;local.write(run/'failure.json',dict(error=error,traceback=traceback.format_exc()))
 finally:
  seen={e['job']['id'] for e in ledger}
  for j in m['jobs']:
   if j['id'] not in seen:ledger.append(dict(job=j,state='PENDING_QUOTA' if error is None else 'NOT_ACCEPTED_AFTER_STOP',accepted_learned_fits=0,accepted_epoch_passes=0,accepted_reference_executions=0))
  local.write(run/'compute_ledger.json',ledger);completed=sum(e['state']=='COMPLETE' for e in ledger)
  status='MULTI_OUTER_SHARD_COMPLETE_REVIEW_REQUIRED' if error is None and completed==len(m['jobs']) else 'MULTI_OUTER_LEASE_PAUSED_READY_FOR_HANDOFF' if error is None and paused else 'MULTI_OUTER_SHARD_PARTIAL_REVIEW_REQUIRED'
  summary=dict(status=status,shard='ALL',lease_id=cfg['lease_id'],completed_jobs=completed,planned_jobs=len(m['jobs']),error=error,
   accepted_learned_fits=sum(e['accepted_learned_fits'] for e in ledger),accepted_neural_epoch_passes=sum(e['accepted_epoch_passes'] for e in ledger),accepted_reference_executions=sum(e['accepted_reference_executions'] for e in ledger),
   wall_seconds=time.monotonic()-started,outer_scoring=False,outer_labels_accessed=False,calibration_labels_accessed=False,point_only=True,fairness_closed=False)
  local.write(run/'summary.json',summary);local.write(run/'heartbeat.json',dict(state='SUPERVISOR_TERMINAL',**summary));old.export(run,'ALL');print(json.dumps(summary),flush=True)
 return 0 if error is None else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);args=p.parse_args();sys.exit(run(args.bundle.resolve()))

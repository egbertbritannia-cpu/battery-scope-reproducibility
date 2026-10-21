"""Portable completed-job receipts; never deserialize or resume a partial model."""
from __future__ import annotations
import hashlib,json,zipfile
from pathlib import Path
def binding(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def inspect(path,m):
 prefix='batteryscope_nfr1_outer_'+m['source']+'_v1/'
 with zipfile.ZipFile(path) as z:
  names=z.namelist();assert len(names)==len(set(names)) and len(names)<30000
  assert all(n.startswith(prefix) and '\\' not in n and ':' not in n and all(s not in ('','.','..') for s in n.split('/')) for n in names)
  inv=json.loads(z.read(prefix+'output_inventory.json'));assert set(names)=={prefix+n for n in inv['files']}|{prefix+'output_inventory.json'}
  assert sum(i.file_size for i in z.infolist())<=m['budget']['output_bytes']+64*2**20
  for name,seal in inv['files'].items():
   h=hashlib.sha256();i=z.getinfo(prefix+name);assert not i.is_dir() and (i.external_attr>>16)&0o170000!=0o120000
   with z.open(i) as f:
    for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
   assert dict(bytes=i.file_size,sha256=h.hexdigest())==seal,'CARRY_INVENTORY_HASH:'+name
  get=lambda n:json.loads(z.read(prefix+n))
  assert get('delivery_manifest.snapshot.json')==m
  summary=get('summary.json');assert summary['error'] is None and summary['status'] in ('MULTI_OUTER_SHARD_COMPLETE_REVIEW_REQUIRED','MULTI_OUTER_LEASE_PAUSED_READY_FOR_HANDOFF')
  ledger=get('compute_ledger.json');assert [e['job']['id'] for e in ledger]==m['shards']['ALL']
  complete=[]
  for e in ledger:
   j=e['job'];assert j==next(j for j in m['jobs'] if j['id']==e['job']['id'])
   if e['state']!='COMPLETE':assert e['state']=='PENDING_QUOTA';continue
   p='jobs/'+j['id']+'/'
   assert not any(n.startswith(p) and n.endswith('failure.json') for n in inv['files'])
   assert get(p+'started.json')['job']==j and get(p+'started.json')['selection_sha256']==m['selection_sha256']
   result=get(p+'result.json');assert all(result[k] is False for k in ('outer_scoring','outer_labels_accessed','calibration_labels_accessed','selection_from_test'))
   assert e['accepted_learned_fits']==int(j['kind']!='references') and e['accepted_epoch_passes']==j.get('fit_epochs',0)
   if j['kind']!='references':
    replay=get(p+'replay.json');assert replay['status']=='FRESH_PROCESS_FULL_PREDICTION_REPLAY_EXACT_PASS' and replay['job_id']==j['id'] and replay['full_prediction_replay'] is True and all(c['exact'] for c in replay['checks'])
   complete.append(e)
  assert [e['job']['id'] for e in complete]==m['shards']['ALL'][:len(complete)],'CARRY_MUST_BE_COMPLETE_PREFIX'
  assert summary['completed_jobs']==len(complete)
  env=get('environment.lock.json');assert all(env['numerical_device_profile'].get(k)==v for k,v in m['expected_environment'].items())
  qa=get('synthetic_neural_qa/summary.json');assert qa['status']=='ACTUAL_OUTER_LOOP_PREFIX_ALL_CHECKPOINT_SYNTHETIC_PASS'
  assert get('synthetic_classical_qa.json')['status']=='SYNTHETIC_PREDICT_RELOAD_PASS'
  cfg=get('lease.snapshot.json');serialized=(json.dumps(m,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
  assert cfg['scientific_manifest_sha256']==hashlib.sha256(serialized).hexdigest()
  assert cfg['source']==m['source'] and cfg['selection_sha256']==m['selection_sha256']
  origins=get('carry_provenance.json') if prefix+'carry_provenance.json' in names else []
  origin=dict(archive=binding(path),lease=cfg,summary=summary,environment=env,synthetic_neural=qa,synthetic_classical=get('synthetic_classical_qa.json'))
  return dict(archive=binding(path),completed=complete,summary=summary,environment=env,inventory=inv['files'],origins=origins+[origin])
def restore(path,run,m,expected):
 assert binding(path)==expected['archive'],'CARRY_ARCHIVE_BINDING'
 proof=inspect(path,m);assert [e['job']['id'] for e in proof['completed']]==expected['completed_job_ids']
 prefix='batteryscope_nfr1_outer_'+m['source']+'_v1/';run=Path(run).resolve()
 with zipfile.ZipFile(path) as z:
  for e in proof['completed']:
   p='jobs/'+e['job']['id']+'/'
   for name in proof['inventory']:
    if not name.startswith(p):continue
    dest=(run/name).resolve();assert dest.is_relative_to(run) and not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True)
    with z.open(prefix+name) as src,dest.open('xb') as dst:
     for b in iter(lambda:src.read(8*2**20),b''):dst.write(b)
    assert binding(dest)==proof['inventory'][name]
 return proof

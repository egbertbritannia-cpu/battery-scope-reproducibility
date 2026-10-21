"""Selected point-only outer refits for twelve sources; fixed five refit seeds."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from . import local_cpu_pilots as local
from . import selection_controls as selection
from . import multisource_inner_contract as inner
from .calb_outer_contract import check_read,METADATA
SHARDS=('ALL',)
MANIFEST='multisource_outer_delivery.json'
SEEDS=(2026,2027,2028,2029,2030)
BUDGET=dict(wall_seconds=28800,reserve_seconds=1800,job_seconds=5400,replay_seconds=1800,synthetic_seconds=900,workers=1,output_bytes=8*2**30)
GUARDS=dict(outer_scoring=False,calibration_labels_accessed=False,outer_labels_accessed=False,point_only=True,intervals=False,
 automatic_resume=False,automatic_retry=False,import_inner_models=False,fairness_closed=False)
def effective_seed(seed,source):
 local.require(type(seed) is int and seed in SEEDS and source in inner.SOURCES,'MULTI_OUTER_SEED_SOURCE')
 return seed+int(hashlib.sha256(f'{seed}|LODO|{source}|outer_refit|multitask'.encode()).hexdigest()[:8],16)%1000000
def read_selection(path):
 path=Path(path);m=path.parent/MANIFEST
 if not m.exists():m=path.parent/'multisource_outer_delivery.frozen.json'
 manifest=json.loads(m.read_text());data=path.read_bytes()
 local.require(hashlib.sha256(data).hexdigest()==manifest['selection_sha256'],'MULTI_OUTER_SELECTION_HASH')
 s=json.loads(data);validate_selection(s);return s
def validate_selection(s):
 local.require(s['schema']=='nfr1-multisource-outer-selection-v1' and s['source'] in inner.SOURCES and
  s['outer_seed_bases']==list(SEEDS) and s['neural_recipes']==selection.candidate_recipes() and s['classical_grid']==local.grid() and
  all(s[k] is False for k in ('outer_scoring','outer_labels_accessed','calibration_labels_accessed','fairness_closed')),'MULTI_OUTER_SELECTION_SCOPE')
 chosen=s['selection'];source=s['source'];models=('TCN','GRU',*inner.CLASSICAL_PRIORITY)
 local.require(len(chosen)==15 and {(r['model'],r['endpoint']) for r in chosen}=={(m,t) for m in models for t in local.TARGETS},'MULTI_OUTER_SELECTION_GRID')
 for r in chosen:
  priority=selection.PRIORITY if r['model'] in ('TCN','GRU') else inner.CLASSICAL_PRIORITY[r['model']]
  values=r['primary_equal_source_scores'];local.require(set(values)==set(priority) and r['source']==source,'MULTI_OUTER_CANDIDATES')
  expected=next(c for c in priority if values[c]<=min(values.values())+selection.TOLERANCE)
  local.require(expected==r['candidate'],'MULTI_OUTER_SELECTED_RECIPE')
  if r['kind']=='neural':local.require(len(r['inner_epochs'])==3 and r['outer_epoch']==sorted(r['inner_epochs'])[1] and
    all(type(e) is int and 1<=e<=30 for e in r['inner_epochs']),'MULTI_OUTER_MEDIAN_EPOCH')
 jobs=s['jobs'];lookup={j['id']:j for j in jobs};expected_ids=set();views=[]
 local.require(len(lookup)==len(jobs),'MULTI_OUTER_JOB_DUPLICATES')
 for a in ('TCN','GRU'):
  for c in selection.PRIORITY:
   heads=[r for r in chosen if r['kind']=='neural' and r['model']==a and r['candidate']==c]
   if not heads:continue
   exports=[dict(endpoint=r['endpoint'],policy='S1',epoch=r['outer_epoch'],interval_eligible=False,output_type='POINT_ONLY') for r in heads]
   for seed in SEEDS:
    jid=f'outer.{source}.{a}.{c}.{seed}';expected_ids.add(jid);j=lookup[jid]
    local.require(j['kind']=='neural' and j['architecture']==a and j['candidate']==c and j['seed']==seed and
      j['checkpoint_exports']==exports and j['fit_epochs']==max(r['epoch'] for r in exports) and j['effective_seed']==effective_seed(seed,source),'MULTI_OUTER_NEURAL_JOB')
    views.extend(dict(view='S1-selected',architecture=a,endpoint=r['endpoint'],candidate=c,seed=seed,job_id=jid,epoch=r['outer_epoch']) for r in heads)
 for family in inner.CLASSICAL_PRIORITY:
  for t in local.TARGETS:
   c=next(r['candidate'] for r in chosen if r['model']==family and r['endpoint']==t)
   for seed in ((2026,) if family=='ridge' else SEEDS):
    jid=f'outer.{source}.{c}.{t}.{seed}';expected_ids.add(jid);j=lookup[jid];params=dict(local.grid()[c]['params'])
    if family!='ridge':params['random_state']=seed
    local.require(j['kind']=='classical' and j['family']==family and j['candidate']==c and j['target']==t and j['seed']==seed and j['params']==params,'MULTI_OUTER_CLASSICAL_JOB')
 for t in local.TARGETS:
  jid=f'outer.{source}.references.{t}';expected_ids.add(jid);local.require(lookup[jid]['kind']=='references' and lookup[jid]['target']==t and lookup[jid]['executions']==3,'MULTI_OUTER_REFERENCE_JOB')
 local.require(set(lookup)==expected_ids and all(j['source']==source and j['outer_scoring'] is False for j in jobs),'MULTI_OUTER_QUEUE')
 counts=dict(neural_fits=sum(j['kind']=='neural' for j in jobs),classical_fits=33,neural_epoch_passes=sum(j.get('fit_epochs',0) for j in jobs),
  reference_jobs=3,reference_executions=9,total_jobs=len(jobs));counts['learned_fits']=counts['neural_fits']+33
 local.require(s['counts']==counts and counts['learned_fits']<=63 and s['neural_prediction_views']==views,'MULTI_OUTER_COUNTS_VIEWS')
 roles=s['outer_role_metadata']['roles']
 local.require(set(roles)=={'development','calibration','test'},'MULTI_OUTER_ROLE_SET')
 for name,role in roles.items():
  local.require(len(role['cell_ids'])==len(set(role['cell_ids']))==role['cells']>0 and local.digest(sorted(role['cell_ids']))==role['cell_hash'],'MULTI_OUTER_ROLE_HASH')
  for other in roles:
   if other!=name:local.require(not set(role['cell_ids'])&set(roles[other]['cell_ids']),'MULTI_OUTER_ROLE_OVERLAP')
def assignments(s):
 return {'ALL':[j['id'] for j in s['jobs']]}
def export_epochs(job):return sorted({r['epoch'] for r in job['checkpoint_exports']})
def validate(m,s):
 validate_selection(s)
 local.require(m['schema']=='nfr1-multisource-outer-point-v1' and m['source']==s['source'] and m['jobs']==s['jobs'] and
  m['shards']==assignments(s) and m['budget']==BUDGET and m['guards']==GUARDS,'MULTI_OUTER_MANIFEST')
 for key in ('expected_environment','input_files','source_bindings','neural_recipes','classical_prediction_policies','counts'):
  local.require(m[key]==s[key],'MULTI_OUTER_MANIFEST_'+key)

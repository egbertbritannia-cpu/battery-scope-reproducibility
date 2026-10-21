"""Seal and score audited selected outer predictions; no fits or reselection."""
from __future__ import annotations
import argparse,hashlib,io,itertools,json,os,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
BASE=ROOT/'paper/audits/neural_classical_fairness_2026-10-08_v1'
CAMPAIGN=BASE/'multisource_outer_leases_v2'
DELIVERY=ROOT/'notebook/neural-classical-fairness-v1/multisource-outer-v1'
OUT=BASE/'multisource_outer_scoring_v1'
BUNDLE=ROOT/'phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2'
PROTOCOL=ROOT/'notebook/neural-classical-fairness-v1/multisource-inner-v1/multisource_protocol.frozen.json'
TARGETS={'soh30':'soh_t_plus_30','rul90':'rul90_cycles','rul80':'rul80_cycles'}
META=['sample_id','canonical_cell_id','t_cycle','duplicate_family_id','source','protocol_id']
NAMES={'ridge':'Ridge','hist_gradient_boosting':'HGB','extra_trees':'Extra Trees','causal_linear_trend':'Trend','soh_persistence':'Persistence','soh_train_mean':'Mean','rul_train_median':'Median','rul_train_log_mean':'Mean'}
METRICS=['primary','mae','rmse','rmsle','bias','lower_clip','upper_clip']
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def need(ok,why):
 if not ok:raise RuntimeError('MULTISOURCE_SCORE:'+why)
def binding(p):
 p=Path(p)
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(bytes=p.stat().st_size,sha256=digest)
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.partial');temp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,p)
def digest_ids(values):return hashlib.sha256(json.dumps(sorted(map(str,values)),separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def metadata(selection):
 import pyarrow.dataset as ds
 cells=selection['outer_role_metadata']['roles']['test']['cell_ids']
 def table(n,cols):return ds.dataset(BUNDLE/'phase2'/n,format='parquet').to_table(columns=cols,filter=ds.field('canonical_cell_id').isin(cells)).to_pandas()
 w=table('window_index.parquet',META[:3]);m=table('sample_target_index.parquet',META)
 need(w.sample_id.is_unique and m.sample_id.is_unique,'METADATA_DUPLICATES')
 f=w.merge(m,on=META[:3],how='left',validate='one_to_one',indicator=True);need(len(f)>0 and f._merge.eq('both').all(),'METADATA_JOIN')
 return f.drop(columns='_merge').sort_values(['canonical_cell_id','t_cycle'],kind='mergesort').reset_index(drop=True)
def freeze():
 need(not OUT.exists(),'NO_OVERWRITE');protocol=read(PROTOCOL);state=read(CAMPAIGN/'campaign.json');progress=read(CAMPAIGN/'progress.json')
 need(progress['state']=='ALL_OUTER_ARTIFACTS_AUDITED_SCORING_PENDING' and len(state['sources'])==12 and all(r['accepted'] for r in state['sources'].values()),'ALL_SOURCES_MUST_PASS_ARTIFACT_AUDIT')
 sources={}
 for source,row in state['sources'].items():
  folder=DELIVERY/source;selection=read(folder/'selection.frozen.json');manifest=read(folder/'multisource_outer_delivery.frozen.json');reviewfile=CAMPAIGN/'reviews'/source/'review.json';review=read(reviewfile)
  need(binding(folder/'selection.frozen.json')['sha256']==manifest['selection_sha256'],'SELECTION_HASH')
  need(review['status']=='ACCEPTED_PORTABLE_OUTER_POINT_ARTIFACTS_SCORING_PENDING' and review['outer_labels_accessed'] is False,'ARTIFACT_AUDIT_SCOPE')
  lease=row['leases'][-1];archive=ROOT/lease['result_path'];need(lease['downloaded'] and binding(archive)==lease['archive']==review['artifact']['archive'],'ARCHIVE_HASH')
  views=[]
  with zipfile.ZipFile(archive) as z:
   prefix='batteryscope_nfr1_outer_'+source+'_v1/';inv=read_bytes(z.read(prefix+'output_inventory.json'))['files']
   def get(n):
    data=z.read(prefix+n);need(dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==inv[n],'JSON_MEMBER_HASH');return read_bytes(data)
   need(get('selection.snapshot.json')==selection and get('delivery_manifest.snapshot.json')==manifest,'SNAPSHOTS')
   results={j['id']:get('jobs/'+j['id']+'/result.json') for j in selection['jobs']}
   audited={(v['file'],v['sha256']) for v in review['artifact']['predictions']}
   for v in selection['neural_prediction_views']:
    result=results[v['job_id']];export=next(e for e in result['exports'] if e['checkpoint']['epoch']==v['epoch']);f='jobs/'+v['job_id']+'/'+export['prediction']['file']
    need((f,inv[f]['sha256']) in audited,'NEURAL_VIEW_AUDIT')
    views.append(dict(system=v['architecture'],endpoint=v['endpoint'],seed=v['seed'],kind='neural',candidate=v['candidate'],epoch=v['epoch'],job_id=v['job_id'],file=f,**inv[f]))
   for j in selection['jobs']:
    if j['kind']=='neural':continue
    for v in results[j['id']]['predictions']:
     system=NAMES[j['family']] if j['kind']=='classical' else NAMES[v['reference']]
     f='jobs/'+j['id']+'/'+v['file'];need((f,inv[f]['sha256']) in audited,'CLASSICAL_REFERENCE_VIEW_AUDIT')
     views.append(dict(system=system,endpoint=j['target'],seed=j.get('seed') if j['kind']=='classical' else None,kind=j['kind'],candidate=j.get('candidate',v.get('reference')),epoch=None,job_id=j['id'],file=f,**inv[f]))
  need(len(views)==72 and len({(v['system'],v['endpoint'],v['seed']) for v in views})==72,'VIEWS_COUNT')
  meta=metadata(selection)
  sources[source]=dict(archive=dict(path=archive.relative_to(ROOT).as_posix(),**lease['archive']),selection=binding(folder/'selection.frozen.json'),manifest=binding(folder/'multisource_outer_delivery.frozen.json'),artifact_audit=dict(path=reviewfile.relative_to(ROOT).as_posix(),**binding(reviewfile)),test_cells=selection['outer_role_metadata']['roles']['test']['cell_ids'],metadata_windows=len(meta),metadata_sample_hash=digest_ids(meta.sample_id),views=views)
  print('[SEALED SOURCE]',source,len(views),flush=True)
 value=dict(schema='nfr1-multisource-point-scoring-v1',sealed_at_utc=now(),source_code=binding(Path(__file__)),parent_protocol=dict(path=PROTOCOL.relative_to(ROOT).as_posix(),**binding(PROTOCOL)),input_files=protocol['inputs'],sources=sources,
  chronology='Post hoc design; historical outer evidence and earlier CALB diagnostic scoring already observed. This seal precedes scoring this twelve-source selected campaign, not preregistration.',
  support='Stored endpoint loss_mask AND finite truth; RUL truth nonnegative. Full identical test metadata/sample universe is required for every model before masking. NE remains null, never zero.',
  metrics=dict(primary={'soh30':'mean_cell_MAE','rul90':'mean_cell_RMSLE','rul80':'mean_cell_RMSLE'},mae='mean_cell(mean_window(abs(p-y)))',rmse='mean_cell(sqrt(mean_window((p-y)^2))); not sqrt of pooled MSE',rmsle='mean_cell(sqrt(mean_window((log1p(p)-log1p(y))^2))) for RUL only',bias='mean_cell(mean_window(p-y))',predictions='Use sealed bounded point outputs; no clipping changes.'),
  aggregation='Equal cells within source; equal evaluable sources within each seed; mean and sample SD across five refit seeds. Ridge and references executed once; seed SD null. No best-seed selection.',
  comparisons='All paired systems per endpoint, seed means per source; delta left-right, wins/ties/losses using parent absolute tolerance. Remove one evaluable source from each aggregate; no selection from these results.',
  tolerance=protocol['outer_plan']['ties_absolute_tolerance'],seed_bases=protocol['outer_plan']['seed_bases'],uncertainty='Descriptive sample SD and per-source deltas only; no new p-values/bootstrap/independent-source confidence claim.',
  scope='Selected point predictions only; no intervals, calibration, survival endpoint or new training.',prohibited=['new fitting','checkpoint/candidate/seed reselection','model-specific missing-label/prediction removal','calibration-label reads','global architectural fairness claim'],fairness_closed=False)
 OUT.mkdir();write(OUT/'protocol.frozen.json',value);write(OUT/'protocol_seal.json',dict(at_utc=now(),protocol=binding(OUT/'protocol.frozen.json'),new_campaign_outer_labels_read=False))
 print('[SEALED]',binding(OUT/'protocol.frozen.json')['sha256'],flush=True)
def read_bytes(v):return json.loads(v.decode('utf-8'))
def score():
 import numpy as np
 import pandas as pd
 import pyarrow.dataset as ds
 import pyarrow
 need((OUT/'protocol_seal.json').exists(),'NO_SEAL');lock=read(OUT/'protocol.frozen.json');seal=read(OUT/'protocol_seal.json')
 need(binding(OUT/'protocol.frozen.json')==seal['protocol'] and binding(Path(__file__))==lock['source_code'],'PROTOCOL_OR_CODE_DRIFT')
 need(not (OUT/'score_started.json').exists(),'NO_REENTRY')
 need(binding(PROTOCOL)=={k:lock['parent_protocol'][k] for k in ('bytes','sha256')},'PARENT_DRIFT')
 for n,b in lock['input_files'].items():need(binding(BUNDLE/n)==b,'INPUT_DRIFT:'+n)
 for source,s in lock['sources'].items():
  for n in ('selection','manifest'):
   filename='selection.frozen.json' if n=='selection' else 'multisource_outer_delivery.frozen.json';need(binding(DELIVERY/source/filename)==s[n],'SELECTION_MANIFEST_DRIFT')
  need(binding(ROOT/s['archive']['path'])=={k:s['archive'][k] for k in ('bytes','sha256')},'ARCHIVE_DRIFT')
  need(binding(ROOT/s['artifact_audit']['path'])=={k:s['artifact_audit'][k] for k in ('bytes','sha256')},'AUDIT_DRIFT')
 write(OUT/'score_started.json',dict(at_utc=now(),protocol=seal['protocol'],new_campaign_label_read_follows=True,prior_historical_and_CALB_labels_already_observed=True))
 all_cells=[];all_source=[];supports=[]
 censor_cols=[f'eol{level}_{kind}' for level in (90,80) for kind in ('event_observed','left_censored','right_censored')]
 for source,s in lock['sources'].items():
  selection=read(DELIVERY/source/'selection.frozen.json');meta=metadata(selection);need(len(meta)==s['metadata_windows'] and digest_ids(meta.sample_id)==s['metadata_sample_hash'],'UNIVERSE_DRIFT')
  columns=['sample_id','canonical_cell_id',*TARGETS.values(),*(t+'_loss_mask' for t in TARGETS),*censor_cols]
  truth=ds.dataset(BUNDLE/'phase2/sample_target_index.parquet',format='parquet').to_table(columns=columns,filter=ds.field('canonical_cell_id').isin(s['test_cells'])).to_pandas()
  need(truth.sample_id.is_unique,'TRUTH_DUPLICATE')
  labels=meta.merge(truth,on=['sample_id','canonical_cell_id'],how='left',validate='one_to_one',indicator=True);need(labels._merge.eq('both').all(),'TRUTH_JOIN');labels=labels.drop(columns='_merge')
  active={}
  for t,col in TARGETS.items():
   mask=labels[t+'_loss_mask'];need(mask.notna().all() and mask.isin([True,False]).all(),'MASK_DOMAIN');y=labels[col].to_numpy(float);keep=mask.to_numpy(bool)&np.isfinite(y);need(not t.startswith('rul') or (y[keep]>=0).all(),'NEGATIVE_RUL_TRUTH');active[t]=keep
   f=labels.loc[keep];cens={}
   if t.startswith('rul'):
    cols=[f'eol{t[-2:]}_{k}' for k in ('event_observed','left_censored','right_censored')]
    bycell=labels.groupby('canonical_cell_id',sort=True)
    need(all(bycell[c].nunique(dropna=False).eq(1).all() for c in cols),'CENSOR_NOT_CELL_CONSTANT')
    c=bycell[cols].first();need(c.notna().all().all() and c.isin([True,False]).all().all(),'CENSOR_DOMAIN');need(c.sum(axis=1).eq(1).all(),'CENSOR_PARTITION');cens={n:int(c[n].sum()) for n in cols}
   supports.append(dict(source=source,endpoint=t,total_windows=len(meta),total_cells=meta.canonical_cell_id.nunique(),evaluable_windows=len(f),evaluable_cells=f.canonical_cell_id.nunique(),evaluable_families=f.duplicate_family_id.nunique(),sample_hash=digest_ids(f.sample_id),status='EVALUABLE' if len(f) else 'NE',**cens))
  with zipfile.ZipFile(ROOT/s['archive']['path']) as z:
   prefix='batteryscope_nfr1_outer_'+source+'_v1/'
   for file in sorted({v['file'] for v in s['views']}):
    aliases=[v for v in s['views'] if v['file']==file];raw=z.read(prefix+file);need(all(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())=={k:v[k] for k in ('bytes','sha256')} for v in aliases),'PREDICTION_HASH')
    pred=pd.read_parquet(io.BytesIO(raw));need(set(pred.columns)==set(META+['endpoint','y_pred','y_pred_unbounded']),'PRED_SCHEMA')
    for v in aliases:
     t=v['endpoint'];f=pred.loc[pred.endpoint.eq(t)].reset_index(drop=True);need(len(f)==len(meta) and f.sample_id.is_unique,'PRED_UNIVERSE_SIZE')
     need(f[META].astype(str).equals(meta[META].astype(str)),'PRED_METADATA_ORDER')
     p=f.y_pred.to_numpy(float);rawpred=f.y_pred_unbounded.to_numpy(float);need(np.isfinite(p).all() and (p>=0).all() and (t!='soh30' or (p<=1.5).all()),'PRED_BOUND')
     reference=v['kind']=='references'
     if reference:need(np.isnan(rawpred).all(),'REF_RAW_UNAVAILABLE')
     else:need(np.isfinite(rawpred).all() and np.array_equal(p,np.clip(rawpred,0,1.5) if t=='soh30' else np.maximum(rawpred,0)),'SEALED_CLIPPING')
     keep=active[t];y=labels[TARGETS[t]].to_numpy(float)[keep];bounded=p[keep];err=bounded-y;g=labels.loc[keep,['canonical_cell_id','duplicate_family_id']].copy()
     g['abs_error']=np.abs(err);g['sq_error']=err**2;g['error']=err
     if t.startswith('rul'):g['log_sq_error']=(np.log1p(bounded)-np.log1p(y))**2
     if not reference:g['lower_clip']=(rawpred[keep]<0).astype(float);g['upper_clip']=(rawpred[keep]>1.5).astype(float) if t=='soh30' else np.zeros(len(g))
     seed=v['seed'] if v['kind']!='references' else -1
     detail=dict(source=source,system=v['system'],endpoint=t,seed=seed,kind=v['kind'],candidate=v['candidate'],epoch=v['epoch'],job_id=v['job_id'])
     if len(g):
      need(g.groupby('canonical_cell_id').duplicate_family_id.nunique().eq(1).all(),'CELL_FAMILY')
      for cell,c in g.groupby('canonical_cell_id',sort=True):
       mae=float(c.abs_error.mean());rmsle=float(np.sqrt(c.log_sq_error.mean())) if t.startswith('rul') else None
       all_cells.append(dict(**detail,cell=cell,family=c.duplicate_family_id.iloc[0],windows=len(c),primary=mae if t=='soh30' else rmsle,mae=mae,rmse=float(np.sqrt(c.sq_error.mean())),rmsle=rmsle,bias=float(c.error.mean()),lower_clip=None if reference else float(c.lower_clip.mean()),upper_clip=None if reference else float(c.upper_clip.mean())))
      cells=[r for r in all_cells if r['source']==source and r['system']==v['system'] and r['endpoint']==t and r['seed']==seed]
      metrics={k:float(np.mean([c[k] for c in cells])) if cells[0][k] is not None else None for k in METRICS}
     else:metrics={k:None for k in METRICS}
     all_source.append(dict(**detail,status='EVALUABLE' if len(g) else 'NE',cells=g.canonical_cell_id.nunique(),windows=len(g),**metrics))
  print('[SCORED SOURCE]',source,flush=True)
  write(OUT/'scoring_progress.json',dict(at_utc=now(),last_source=source,source_seed_rows=len(all_source),calibration_labels_accessed=False))
 need(len(all_source)==864,'SCORED_VIEW_COUNT')
 cells=pd.DataFrame(all_cells);sources=pd.DataFrame(all_source);support=pd.DataFrame(supports)
 need(set(support.loc[support.status.eq('NE'),['source','endpoint']].itertuples(index=False,name=None))=={('CALB','rul90')},'UNEXPECTED_NE_SUPPORT')
 cells.to_parquet(OUT/'cell_metrics.private.parquet',index=False);sources.to_csv(OUT/'source_seed_metrics.csv',index=False);support.to_csv(OUT/'support.csv',index=False)
 aggregate(sources,support,lock)
 outputs=['cell_metrics.private.parquet','source_seed_metrics.csv','support.csv','source_summary.csv','seed_macro_metrics.csv','aggregate_metrics.csv','paired_source_deltas.csv','pairwise_summary.csv','leave_one_source_out.csv','snl_exclusion.csv']
 write(OUT/'score_manifest.json',dict(status='ALL_12_SOURCE_POINT_SCORING_COMPLETE_NO_RESELECTION',at_utc=now(),protocol=seal['protocol'],source_seed_rows=len(sources),cell_metric_rows=len(cells),outputs={n:binding(OUT/n) for n in outputs},environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,pyarrow=pyarrow.__version__),outer_labels_accessed=True,calibration_labels_accessed=False,new_fits=0,selection_changed=False,seed_dropping=False,intervals=False,global_architectural_fairness_closed=False))
 print('[SCORING COMPLETE]',len(sources),'source-seed rows',flush=True)
def aggregate(sources,support,lock):
 import numpy as np
 import pandas as pd
 valid=sources.loc[sources.status.eq('EVALUABLE')];seedrows=[];summary=[]
 for (system,t,seed),g in valid.groupby(['system','endpoint','seed'],sort=True):
  need(g.source.is_unique,'DUPLICATE_SOURCE_SYSTEM_SEED');seedrows.append(dict(system=system,endpoint=t,seed=int(seed),sources=len(g),**{k:None if g[k].isna().all() else float(g[k].mean()) for k in METRICS}))
 seeds=pd.DataFrame(seedrows);seeds.to_csv(OUT/'seed_macro_metrics.csv',index=False)
 for (system,t),g in seeds.groupby(['system','endpoint'],sort=True):
  expected=lock['seed_bases'] if system in ('TCN','GRU','HGB','Extra Trees') else ([2026] if system=='Ridge' else [-1]);need(sorted(g.seed.tolist())==expected,'SEED_SUPPORT')
  summary.append(dict(system=system,endpoint=t,sources=int(g.sources.iloc[0]),seeds=len(g),**{k:None if g[k].isna().all() else float(g[k].mean()) for k in METRICS},primary_seed_sd=float(g.primary.std(ddof=1)) if len(g)>1 else None,primary_seed_min=float(g.primary.min()),primary_seed_max=float(g.primary.max())))
 pd.DataFrame(summary).to_csv(OUT/'aggregate_metrics.csv',index=False)
 per_source=[]
 for (system,t,source),g in valid.groupby(['system','endpoint','source'],sort=True):
  per_source.append(dict(system=system,endpoint=t,source=source,seeds=len(g),cells=int(g.cells.iloc[0]),windows=int(g.windows.iloc[0]),**{k:None if g[k].isna().all() else float(g[k].mean()) for k in METRICS},primary_seed_sd=float(g.primary.std(ddof=1)) if len(g)>1 else None))
 ps=pd.DataFrame(per_source);ps.to_csv(OUT/'source_summary.csv',index=False);pairs=[];pairsummary=[];sensitivity=[];excluded=[]
 for t in TARGETS:
  f=ps.loc[ps.endpoint.eq(t)];systems=sorted(f.system.unique())
  for left,right in itertools.combinations(systems,2):
   a=f.loc[f.system.eq(left)].set_index('source');b=f.loc[f.system.eq(right)].set_index('source');need(set(a.index)==set(b.index),'PAIRED_SOURCE_SUPPORT');a=a.sort_index();b=b.loc[a.index];delta=a.primary-b.primary;tol=lock['tolerance']
   for source,d in delta.items():pairs.append(dict(endpoint=t,left=left,right=right,source=source,left_score=float(a.loc[source,'primary']),right_score=float(b.loc[source,'primary']),delta=float(d),left_wins=bool(d < -tol),tie=bool(abs(d)<=tol)))
   pairsummary.append(dict(endpoint=t,left=left,right=right,delta=float(delta.mean()),sources=len(delta),left_wins=int((delta < -tol).sum()),ties=int((delta.abs()<=tol).sum()),left_losses=int((delta>tol).sum()),min_source_delta=float(delta.min()),max_source_delta=float(delta.max())))
   for source in delta.index:sensitivity.append(dict(endpoint=t,left=left,right=right,omitted_source=source,delta=float(delta.drop(source).mean()),original_delta=float(delta.mean())))
  for system in systems:
   g=f.loc[f.system.eq(system)];without=g.loc[g.source.ne('SNL')];excluded.append(dict(system=system,endpoint=t,sources=len(without),primary=float(without.primary.mean()),original_primary=float(g.primary.mean()),delta=float(without.primary.mean()-g.primary.mean())))
 pd.DataFrame(pairs).to_csv(OUT/'paired_source_deltas.csv',index=False);pd.DataFrame(pairsummary).to_csv(OUT/'pairwise_summary.csv',index=False);pd.DataFrame(sensitivity).to_csv(OUT/'leave_one_source_out.csv',index=False);pd.DataFrame(excluded).to_csv(OUT/'snl_exclusion.csv',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','score']);a=p.parse_args();{'freeze':freeze,'score':score}[a.stage]()

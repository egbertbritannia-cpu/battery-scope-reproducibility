"""Frozen-model cell-exclusion sensitivity; no target, prediction or model reads."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'paper/audits/neural_classical_fairness_2026-10-08_v1'
SCORE=BASE/'multisource_outer_scoring_v1'
OUT=BASE/'censoring_proxy_reproducibility_20261018_v1'
PROTO=ROOT/'notebook/neural-classical-fairness-v1/multisource-inner-v1/multisource_protocol.frozen.json'
SEL=BASE/'multisource_outer_selection_v1'
COHORT=ROOT/'phase1-output/battery_model_cohort.csv'
EVIDENCE=ROOT/'paper/audits/snl_scope_and_corrected_protocol_2026-10-04_v1/snl_evidence.json'
RECIPE=ROOT/'phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2/code/inner_recipe.json'
ENVZIP=BASE/'multisource_outer_leases_v2/sources/CALB/L001/batteryscope_nfr1_outer_CALB_v1_results.zip'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def require(ok,msg):
    if not ok:raise RuntimeError('PROXY_SENSITIVITY:'+msg)
def table(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def fmt(x,n=4):return f'{x:.{n}f}'.replace('.',',')

def freeze():
    require(not (OUT/'protocol.frozen.json').exists(),'FRESH_NAMESPACE')
    manifest=read(SCORE/'score_manifest.json')
    paths=[SCORE/'cell_metrics.private.parquet',SCORE/'support.csv',SCORE/'aggregate_metrics.csv',SCORE/'snl_exclusion.csv',
        SCORE/'source_seed_metrics.csv',SCORE/'score_manifest.json',COHORT,EVIDENCE,PROTO,RECIPE,Path(__file__),
        *sorted(SEL.glob('*/selection.frozen.json'))]
    require(len(list(SEL.glob('*/selection.frozen.json')))==12,'SELECTIONS')
    p=dict(schema='frozen-model-proxy-cell-exclusion-v1',sealed_at_utc=now(),
        chronology='Post-hoc after published-in-draft NFR1 and inspection of saved cell-metric schema/first rows; no preregistration or independent confirmation.',
        inputs={x.relative_to(ROOT).as_posix():dict(bytes=x.stat().st_size,sha256=sha(x)) for x in paths},
        protected_outputs=manifest['outputs'],proxy_rule='SNL and q_ref_tier C and q_ref_source protocol_soc_window_scaled_nominal; exactly 4 cells, q_ref_ah=1.92; confirm IDs against prior SNL evidence.',
        scenarios=['full','exclude_four_SNL_proxy_cells','exclude_all_SNL'],endpoints=['soh30','rul90','rul80'],
        systems='All existing systems and seeds; main table TCN/GRU/Extra Trees/Trend, fixed before this reaggregation.',
        estimand='Mean of saved cell primary metric within source, then equal evaluable sources, then equal existing seeds; no pooled-window weighting.',
        no_reselection=True,no_refit=True,no_label_changes=True,no_new_target_or_prediction_reads=True,
        limits=['Exclusion at evaluation only; proxy rows remain in training for other outer folds.',
            'All-SNL exclusion does not bound the effect of excluding only proxy cells.',
            'No survival analysis or correction for censoring bias; no new independent sources or calibrated bootstrap.'],
        environment_observation='Read environment_check.json only from accepted CALB L001 ZIP; representative observed GPU session, separate from local scoring environment.',
        environment_archive=dict(path=ENVZIP.relative_to(ROOT).as_posix(),sha256=read(SCORE/'protocol.frozen.json')['sources']['CALB']['archive']['sha256']))
    write(OUT/'protocol.frozen.json',p);write(OUT/'protocol_seal.json',dict(sha256=sha(OUT/'protocol.frozen.json'),at_utc=now()))
    print('PROXY_SENSITIVITY_PROTOCOL_FROZEN')

def configs():
    import pandas as pd
    p=read(PROTO);rows=[];jobs=[]
    for path in sorted(SEL.glob('*/selection.frozen.json')):
        s=read(path)
        require(s['expected_environment']==p['expected_environment'],'SELECTION_ENV_DRIFT')
        for r in s['selection']:
            rows.append(dict(source=s['source'],kind=r['kind'],model=r['model'],endpoint=r['endpoint'],candidate=r['candidate'],
                inner_selection_seed=2026,outer_epoch=r.get('outer_epoch'),inner_epochs=r.get('inner_epochs'),
                params=p['classical_grid'].get(r['candidate'],p['neural_recipes'].get(r['candidate']))))
        for j in s['jobs']:
            jobs.append({k:j[k] for k in ['id','kind','source','architecture','family','candidate','target','seed','effective_seed','fit_epochs','checkpoint_exports','params'] if k in j})
    require(len(rows)==180,'EXPECTED_180_SELECTED_MODEL_ENDPOINTS')
    write(OUT/'selected_configurations.json',dict(rows=rows,outer_jobs=jobs,selection_by_outer_results=False))
    flat=[dict(**{k:r[k] for k in ['source','kind','model','endpoint','candidate','inner_selection_seed','outer_epoch']},
        inner_epochs=json.dumps(r['inner_epochs']),params=json.dumps(r['params'],sort_keys=True)) for r in rows]
    pd.DataFrame(flat).to_csv(OUT/'selected_configurations.csv',index=False)
    require(sha(ENVZIP)==read(OUT/'protocol.frozen.json')['environment_archive']['sha256'],'ENV_ARCHIVE_HASH')
    with zipfile.ZipFile(ENVZIP) as z:
        name=next(n for n in z.namelist() if n.endswith('/environment_check.json'));payload=z.read(name);e=json.loads(payload)
    require(not e['mismatches'],'OBSERVED_ENV_MISMATCH')
    actual=e['actual'];require(actual['numerical_device_profile']==p['expected_environment'],'GPU_ENV_PROFILE')
    observed=dict(scope='Representative accepted CALB L001 session, not a newly executed fit or observation of every session.',
        archive_sha256=sha(ENVZIP),member=name,member_sha256=hashlib.sha256(payload).hexdigest(),
        numerical_device_profile=actual['numerical_device_profile'],packages=actual['identity']['packages'],
        runtime_flags={k:v for k,v in actual.items() if k not in ['identity','numerical_device_profile']})
    write(OUT/'training_environment_observed.json',observed)
    write(OUT/'training_contract.json',dict(expected_environment=p['expected_environment'],classical_grid=p['classical_grid'],
        neural_recipes=p['neural_recipes'],budgets=p['budgets'],inner_selection=p['inner_selection'],
        literal_constants=read(RECIPE)['literal_constants'],source_blocks=p['source_blocks'],representation=p['representation'],
        sampling=p['sampling'],prediction_policies=p['prediction_policies']))
    return p

def run():
    import numpy as np,pandas as pd
    require(not (OUT/'completion.json').exists(),'NO_OVERWRITE')
    p=read(OUT/'protocol.frozen.json');require(sha(OUT/'protocol.frozen.json')==read(OUT/'protocol_seal.json')['sha256'],'SEAL')
    for path,b in p['inputs'].items():require(sha(ROOT/path)==b['sha256'],'INPUT_DRIFT:'+path)
    cohort=pd.read_csv(COHORT);proxy=cohort[cohort.source.eq('SNL')&cohort.q_ref_tier.eq('C')&cohort.q_ref_source.eq('protocol_soc_window_scaled_nominal')]
    ids=set(proxy.canonical_cell_id);require(len(ids)==4 and proxy.q_ref_ah.eq(1.92).all(),'PROXY_RULE')
    require(ids=={r['canonical_cell_id'] for r in read(EVIDENCE)['proxy_cells']},'PRIOR_PROXY_IDENTITIES')
    write(OUT/'proxy_ids.private.json',dict(cells=sorted(ids),cohort_sha256=sha(COHORT),rule=p['proxy_rule']))
    support=pd.read_csv(SCORE/'support.csv');cens=[]
    for source in sorted(support.source.unique()):
        record=dict(source=source)
        for target,threshold in [('rul90','90'),('rul80','80')]:
            r=support[support.source.eq(source)&support.endpoint.eq(target)].iloc[0]
            total=int(r.total_cells);observed=int(r['eol'+threshold+'_event_observed']);left=int(r['eol'+threshold+'_left_censored']);right=int(r['eol'+threshold+'_right_censored'])
            require(observed+left+right==total and observed==int(r.evaluable_cells),'CENSORING_SUPPORT')
            record['total_cells']=total
            for name,value in [('observed',observed),('left',left),('right',right),('censored',left+right),('percent',100*(left+right)/total)]:record[target+'_'+name]=value
        cens.append(record)
    c=pd.DataFrame(cens);require(int(c.rul90_censored.sum())==40 and int(c.rul80_censored.sum())==197,'TOTAL_CENSORING')
    c.to_csv(OUT/'censoring_by_source.csv',index=False)
    f=pd.read_parquet(SCORE/'cell_metrics.private.parquet')
    require(not f.duplicated(['source','system','endpoint','seed','cell']).any(),'CELL_METRIC_DUPLICATES')
    require(set(f.loc[f.cell.isin(ids),'source'])=={'SNL'},'PROXY_SOURCE')
    metriccols=['primary','mae','rmse','bias','lower_clip','upper_clip']
    sources=[];seeds=[];aggregates=[]
    for scenario in p['scenarios']:
        keep=f if scenario=='full' else (f[~f.cell.isin(ids)] if scenario=='exclude_four_SNL_proxy_cells' else f[f.source.ne('SNL')])
        grouped=keep.groupby(['system','endpoint','seed','source'],sort=True)
        s=grouped[metriccols].mean().reset_index();counts=grouped.agg(cells=('cell','size'),windows=('windows','sum')).reset_index()
        s=s.merge(counts,on=['system','endpoint','seed','source'],validate='one_to_one');s['scenario']=scenario;sources.append(s)
        t=s.groupby(['system','endpoint','seed'],sort=True)[metriccols].mean().reset_index()
        t['sources']=s.groupby(['system','endpoint','seed']).source.nunique().values;t['scenario']=scenario;seeds.append(t)
        for (system,target),g in t.groupby(['system','endpoint'],sort=True):
            aggregates.append(dict(scenario=scenario,system=system,endpoint=target,sources=int(g.sources.iloc[0]),seeds=len(g),
                **{m:float(g[m].mean()) for m in metriccols},primary_seed_sd=float(g.primary.std(ddof=1)) if len(g)>1 else None))
    source=pd.concat(sources,ignore_index=True);seed=pd.concat(seeds,ignore_index=True);agg=pd.DataFrame(aggregates)
    source.to_csv(OUT/'source_seed_sensitivity.csv',index=False);seed.to_csv(OUT/'seed_sensitivity.csv',index=False);agg.to_csv(OUT/'aggregate_sensitivity.csv',index=False)
    old=pd.read_csv(SCORE/'aggregate_metrics.csv');snl=pd.read_csv(SCORE/'snl_exclusion.csv')
    for scenario,baseline in [('full',old),('exclude_all_SNL',snl)]:
        a=agg[agg.scenario.eq(scenario)].merge(baseline,on=['system','endpoint'],suffixes=('_new','_old'),validate='one_to_one')
        require(len(a)==len(baseline) and np.allclose(a.primary_new,a.primary_old,atol=1e-12,rtol=0),'EXISTING_SCORE_PARITY:'+scenario)
    # Independent explicit nested sums check every aggregate, preserving cell→source→seed weights.
    for r in aggregates:
        cell=f[f.system.eq(r['system'])&f.endpoint.eq(r['endpoint'])]
        if r['scenario']=='exclude_four_SNL_proxy_cells':cell=cell[~cell.cell.isin(ids)]
        if r['scenario']=='exclude_all_SNL':cell=cell[cell.source.ne('SNL')]
        vals=[]
        for _,g in cell.groupby('seed'):
            sourcevals=[math.fsum(v.primary)/len(v) for _,v in g.groupby('source')];vals.append(math.fsum(sourcevals)/len(sourcevals))
        require(abs(math.fsum(vals)/len(vals)-r['primary'])<1e-12,'INDEPENDENT_REAGGREGATION')
    contracts=configs()
    for name,b in p['protected_outputs'].items():require(sha(SCORE/name)==b['sha256'],'PROTECTED_SCORING_CHANGED')
    mainrows=[]
    for target in p['endpoints']:
        for system in ['TCN','GRU','Extra Trees','Trend']:
            values=[float(agg[agg.system.eq(system)&agg.endpoint.eq(target)&agg.scenario.eq(s)].primary.iloc[0]) for s in p['scenarios']]
            mainrows.append([system,dict(soh30='SoH30',rul90='RUL90',rul80='RUL80')[target],*[fmt(v) for v in values]])
    (OUT/'main_sensitivity_table.md').write_text(table(['Hệ thống','Nhãn','Đầy đủ','Bỏ 4 proxy','Bỏ SNL'],mainrows)+'\n',encoding='utf8')
    total=int(c.total_cells.sum());cr=[[r.source,int(r.total_cells),int(r.rul90_observed),fmt(r.rul90_percent,2),int(r.rul80_observed),fmt(r.rul80_percent,2)] for r in c.itertuples()]
    cr.append(['Tổng',total,int(c.rul90_observed.sum()),fmt(100*c.rul90_censored.sum()/total,2),int(c.rul80_observed.sum()),fmt(100*c.rul80_censored.sum()/total,2)])
    (OUT/'main_censoring_table.md').write_text(table(['Nguồn','Cell','Sự kiện RUL90','Kiểm duyệt 90 (%)','Sự kiện RUL80','Kiểm duyệt 80 (%)'],cr)+'\n',encoding='utf8')
    outputs={q.name:dict(bytes=q.stat().st_size,sha256=sha(q)) for q in OUT.iterdir() if q.is_file() and q.name not in ['completion.json','protocol.frozen.json','protocol_seal.json']}
    write(OUT/'completion.json',dict(status='FROZEN_MODEL_PROXY_SENSITIVITY_AND_CONFIG_EXTRACTION_COMPLETE',at_utc=now(),
        outputs=outputs,cells_excluded=4,cohort_cells=total,source_cell_metrics_rows=len(f),baseline_parity='full and all-SNL scores reproduced to 1e-12',
        aggregate_rows=len(agg),independent_nested_sum_check=True,protected_scoring_unchanged=True,
        selected_configurations=180,new_fits=0,new_prediction_or_label_reads=0,censoring_bias_corrected=False,independent_reproduction=False))
    print((OUT/'main_sensitivity_table.md').read_text());print((OUT/'main_censoring_table.md').read_text())
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','run']);args=a.parse_args();{'freeze':freeze,'run':run}[args.stage]()

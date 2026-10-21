"""HCP Eq.(6), family -> eligible windows, with frozen quantile predictors.

Post-hoc source-shift stress test. No exchangeability or simultaneous guarantee
is claimed for this benchmark. No predictor fitting, seed or method selection.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import sys
import zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1 import score_coverage_extension as parent
e=parent.e;BASE=parent.BASE;SCORE=BASE/'coverage_extension_scoring_v1'
OUT=BASE/'hcp_family_window_20261017_v1';ARM='hcp_family_window'
def read(p):return parent.read(p)
def bind(p):return parent.bind(p)
def verify(b):return parent.verify(b)
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def table(arc,member):return parent.table(arc,member)
def fraction_info(x):return dict(numerator=x.numerator,denominator=x.denominator)

def calibration_frame(p,source,model,target):
    if model=='TCN.2026':
        j=next(j for j in p['historical_reference']['jobs'] if j['source']==source)
        b=j['members'][target+'.calibration_observed.parquet']
        arc=ROOT/p['historical_reference']['archives'][j['shard']]['path']
        with zipfile.ZipFile(arc) as z:data=z.read(b['member'])
        assert len(data)==b['bytes']
        import hashlib
        assert hashlib.sha256(data).hexdigest()==b['sha256']
        f=pd.read_parquet(io.BytesIO(data))
    else:
        arch,seed=model.split('.')
        j=next(j for j in p['jobs'] if j['source']==source and j['architecture']==arch and j['seed']==int(seed))
        f=table(ROOT/p['inputs']['inference_return']['path'],p['inference_prefix']+'jobs/'+j['id']+'/'+target+'.calibration_predictions.parquet')
        mem=pd.read_parquet(ROOT/p['inputs']['membership']['path'])
        cells=set(mem.loc[mem.evaluation_mode.eq('LODO')&mem.outer_entity.eq(source)&mem.split.eq('calibration'),'canonical_cell_id'])
        col,mask=e.SPECS[target];y=parent.labels(cells,e.META+[col,mask])
        assert set(f.sample_id)==set(y.sample_id) and y[mask].notna().all()
        y=y.loc[y[mask].astype(bool)&np.isfinite(y[col])].copy();y['y_true']=y[col]
        f=f.merge(y,on=e.META,validate='one_to_one')
    e.validate_frame(f,target,truth=True)
    return f

def test_frame(p,source,model,target,truth):
    if model=='TCN.2026':
        j=next(j for j in p['historical_reference']['jobs'] if j['source']==source)
        arc=ROOT/p['historical_reference']['archives'][j['shard']]['path'];member=j['members'][target+'.outer_predictions.parquet']['member']
    else:
        arch,seed=model.split('.')
        design=read(ROOT/p['inputs']['design']['path'])
        j=next(j for j in design['selected_existing_models'] if j['source']==source and j['architecture']==arch and j['seed']==int(seed))
        arc=ROOT/j['archive'];member=next(r['member'] for r in j['quantile_prediction_schemas'] if r['endpoint']==target)
    f=table(arc,member);assert f.sample_id.is_unique and set(f.sample_id)==set(truth.sample_id)
    pd.testing.assert_frame_equal(f[e.META].sort_values('sample_id').reset_index(drop=True),truth[e.META].sort_values('sample_id').reset_index(drop=True),check_dtype=False,check_exact=True)
    col,mask=e.SPECS[target];assert truth[mask].notna().all()
    y=truth.loc[truth[mask].astype(bool)&np.isfinite(truth[col]),e.META+['t_cycle',col]].rename(columns={col:'y_true'})
    return f[e.META+['q10','q50','q90']].merge(y,on=e.META,validate='one_to_one')

def hcp_quantile(scores,families):
    """Long-double search, then exact rational CDF and predecessor checks.

    Work in unnormalised group masses: target = 4*(K+1)/5.
    Infinite test-group mass is one; each finite group mass is one.
    """
    scores=np.asarray(scores,float);families=np.asarray(families)
    assert np.isfinite(scores).all() and (scores>=0).all()
    ids,counts=np.unique(families,return_counts=True);K=len(ids);threshold=Fraction(4*(K+1),5)
    if threshold>K:return math.inf,dict(families=K,infinite=True,atom_mass=1/(K+1))
    sizes=dict(zip(ids,map(int,counts)))
    order=np.argsort(scores,kind='stable');ordered=scores[order]
    weights=np.asarray([np.longdouble(1)/sizes[f] for f in families[order]],dtype=np.longdouble)
    cumulative=np.cumsum(weights);idx=int(np.searchsorted(cumulative,np.longdouble(4)*(K+1)/5,side='left'));idx=min(idx,len(ordered)-1)
    def cdf(q):
        keys,n=np.unique(families[scores<=q],return_counts=True)
        return sum((Fraction(int(v),sizes[k]) for k,v in zip(keys,n)),Fraction(0))
    unique=np.unique(ordered);j=int(np.searchsorted(unique,ordered[idx]))
    while cdf(unique[j])<threshold:j+=1
    while j>0 and cdf(unique[j-1])>=threshold:j-=1
    q=float(unique[j]);at=cdf(q);before=cdf(unique[j-1]) if j else Fraction(0)
    assert before<threshold<=at
    return q,dict(families=K,infinite=False,atom_mass=1/(K+1),min_group_windows=int(counts.min()),max_group_windows=int(counts.max()),
        cdf_before=fraction_info(before/(K+1)),cdf_at=fraction_info(at/(K+1)),threshold=fraction_info(threshold),
        selected_score=q,finite_mass=K/(K+1))

def load():
    p=read(OUT/'protocol.frozen.json');verify(read(OUT/'protocol_seal.json')['protocol'])
    for b in p['inputs'].values():verify(b)
    return p,read(ROOT/p['inputs']['parent_protocol']['path'])

def freeze():
    assert not OUT.exists(),'FRESH_NAMESPACE_REQUIRED'
    p=read(SCORE/'protocol.frozen.json');verify(read(SCORE/'protocol_seal.json')['protocol'])
    accepted=read(SCORE/'accepted_completion.json');verify(accepted['independent_audit'])
    for b in p['inputs'].values():verify(b)
    membership=pd.read_parquet(ROOT/p['inputs']['membership']['path'])
    sources=read(ROOT/p['inputs']['design']['path'])['held_sources']
    for source in sources:
        m=membership.loc[membership.evaluation_mode.eq('LODO')&membership.outer_entity.eq(source)]
        for col in ['canonical_cell_id','duplicate_family_id']:
            sets=[set(m.loc[m.split.eq(role),col]) for role in ['development','calibration','test']]
            assert all(sets) and all(sets[i].isdisjoint(sets[j]) for i,j in [(0,1),(0,2),(1,2)])
    inputs=dict(parent_protocol=bind(SCORE/'protocol.frozen.json'),accepted=bind(SCORE/'accepted_completion.json'),
        baseline_metrics=bind(SCORE/'interval_metrics_by_source.csv'),baseline_qhat=bind(SCORE/'calibration_recipes.parquet'),
        code=bind(Path(__file__)),parent_helper=bind(Path(parent.__file__)),arithmetic=bind(Path(e.__file__)),
        literature=bind(BASE/'hcp_method_design_20261017_v1/literature_and_design.md'))
    OUT.mkdir()
    protocol=dict(schema='hcp-family-window-v1',at_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,
        sources=sources,models=p['models'],targets=list(e.SPECS),alpha=.2,recipes=['window','family_max',ARM],
        method_reference=dict(url='https://arxiv.org/html/2306.06342v3',section='2.2',equation=6),
        group='duplicate_family_id',within_group_unit='all endpoint-eligible windows, pooled across member cells',
        distribution='finite score atom 1/((K+1)*N_family); infinity atom 1/(K+1)',
        score='max(g(q10)-g(y),g(y)-g(q90),0); identity SoH/log1p RUL; same bounds as E07',
        distinct_from_family_max='Uses weighted empirical scores within each family, not maxima; targets a random window from a new group, not all group windows.',
        primary_coverage='equal-family mean window coverage within source, then equal evaluable sources',
        other_metrics='All previous window/cell/simultaneous events, widths, RUL ratios, zero/near-zero/last20 subgroups; unchanged reporting definitions',
        no_guarantee='Group and within-group exchangeability not established under source shift/time dependence; no simultaneous guarantee or shift correction claimed.',
        score_precision='Preserve float32 calibration truth for historical E07 reference; native target precision for new calibration and all test labels.',
        chronology='Post hoc after baseline and diagnostic test outcomes seen; method choice from feasibility/literature, no new HCP qhat/test results seen before lock.',
        infinite_rule='If K<4 retain +infinity qhat and unbounded RUL interval; SoH [0,1.5]; count and disclose; never cap or replace silently.',
        new_fits=0,selected_by_test=False,baseline_outputs_mutated=False)
    write('protocol.frozen.json',protocol);write('protocol_seal.json',dict(protocol=bind(OUT/'protocol.frozen.json')))
    print('HCP_PROTOCOL_SEALED_BEFORE_NEW_CALIBRATION',flush=True)

def calibrate():
    d,p=load();assert not (OUT/'calibration_started.json').exists()
    write('calibration_started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),test_labels_read=False))
    old=pd.read_parquet(ROOT/d['inputs']['baseline_qhat']['path']);rows=[];proofs=[]
    for source in d['sources']:
        for model in d['models']:
            for target in d['targets']:
                f=calibration_frame(p,source,model,target)
                baselines=e.calibrate_frame(f,target)
                for arm,value in baselines.items():
                    ref=old.loc[old.source.eq(source)&old.model.eq(model)&old.target.eq(target)&old.recipe.eq(arm)].iloc[0]
                    assert value['qhat']==ref.qhat and value['calibration_units']==ref.calibration_units and value['quantile_rank']==ref.quantile_rank
                q,proof=hcp_quantile(e.scalar_scores(f,target),f.duplicate_family_id)
                assert q<=baselines['family_max']['qhat'] or math.isinf(q)
                rows.append(dict(source=source,model=model,target=target,recipe=ARM,qhat=q,calibration_rows=len(f),calibration_families=f.duplicate_family_id.nunique(),infinite=math.isinf(q)))
                proofs.append(dict(source=source,model=model,target=target,**proof))
        print('[HCP CALIBRATION]',source,flush=True)
    assert len(rows)==108
    pd.DataFrame(rows).to_parquet(OUT/'hcp_calibration.parquet',index=False);write('quantile_cdf_proofs.json',proofs)
    write('calibration_acceptance.json',dict(status='HCP_108_QHATS_SEALED_BEFORE_TEST_SCORING',test_labels_read=False,new_fits=0,
        at_utc=datetime.now(timezone.utc).isoformat(),qhat=bind(OUT/'hcp_calibration.parquet'),proofs=bind(OUT/'quantile_cdf_proofs.json'),infinite_qhats=sum(r['infinite'] for r in rows)))

def interval(f,target,q):
    if math.isinf(q):return np.zeros(len(f)),np.full(len(f),1.5 if target=='soh30' else math.inf)
    return e.bounds(f,target,q)

def report_metrics(g,target,lo,hi,raw):
    metrics=e.interval_metrics(g,lo,hi);w=g.copy();w['width']=hi-lo;w['raw_width']=raw
    hit=(g.y_true.to_numpy(float)>=lo)&(g.y_true.to_numpy(float)<=hi)
    metrics['family_balanced_window_coverage']=float(pd.Series(hit,index=g.duplicate_family_id).groupby(level=0).mean().mean())
    metrics['cell_macro_mean_width']=float(w.groupby('canonical_cell_id').width.mean().mean())
    metrics['cell_macro_mean_raw_width']=float(w.groupby('canonical_cell_id').raw_width.mean().mean())
    if target.startswith('rul'):
        w['ratio_stabilized']=w.width/np.maximum(w.y_true,10)
        metrics['cell_macro_stabilized_relative_width']=float(w.groupby('canonical_cell_id').ratio_stabilized.mean().mean())
        pos=w.loc[w.y_true>0].copy();pos['ratio']=pos.width/pos.y_true;metrics['positive_ratio_windows']=len(pos)
        if len(pos):
            metrics['cell_macro_positive_ratio_median']=float(pos.groupby('canonical_cell_id').ratio.median().mean())
            metrics['cell_macro_positive_ratio_p90']=float(pos.groupby('canonical_cell_id').ratio.quantile(.9).mean())
    return metrics

def score():
    d,p=load();accept=read(OUT/'calibration_acceptance.json');verify(accept['qhat']);verify(accept['proofs'])
    assert not (OUT/'score_started.json').exists()
    write('score_started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),calibration=bind(OUT/'calibration_acceptance.json')))
    oldq=pd.read_parquet(ROOT/d['inputs']['baseline_qhat']['path']);newq=pd.read_parquet(OUT/'hcp_calibration.parquet')
    recipes=pd.concat([oldq,newq],ignore_index=True);old=pd.read_csv(ROOT/d['inputs']['baseline_metrics']['path'])
    membership=pd.read_parquet(ROOT/p['inputs']['membership']['path']);rows=[];supports=[];checks=0
    for source in d['sources']:
        cells=set(membership.loc[membership.evaluation_mode.eq('LODO')&membership.outer_entity.eq(source)&membership.split.eq('test'),'canonical_cell_id'])
        truth=parent.labels(cells,e.META+['t_cycle']+sum((list(v) for v in e.SPECS.values()),[]))
        for model in d['models']:
            for target in d['targets']:
                f=test_frame(p,source,model,target,truth)
                for arm in d['recipes']:
                    q=float(recipes.loc[recipes.source.eq(source)&recipes.model.eq(model)&recipes.target.eq(target)&recipes.recipe.eq(arm),'qhat'].iloc[0])
                    identity=dict(source=source,model=model,target=target,recipe=arm)
                    if f.empty:
                        assert (source,target)==('CALB','rul90')
                        rows.append(dict(**identity,subgroup='all',status='NE',windows=0,cells=0));continue
                    lo,hi=interval(f,target,q)
                    raw=(np.expm1(np.log1p(f.q90.to_numpy(float))+q)-np.expm1(np.log1p(f.q10.to_numpy(float))-q)) if target.startswith('rul') else f.q90.to_numpy(float)-f.q10.to_numpy(float)+2*q
                    groups={'all':np.ones(len(f),bool)}
                    if target.startswith('rul'):
                        y=f.y_true.to_numpy(float);tau=f.t_cycle.to_numpy(float)+y
                        groups.update(last20=(tau>0)&(f.t_cycle.to_numpy(float)/np.maximum(tau,1)>=.8),zero=y==0,near_0_10=(y>0)&(y<=10),near_10_50=(y>10)&(y<=50),over50=y>50)
                    for subgroup,keep in groups.items():
                        g=f.loc[keep];row=dict(**identity,subgroup=subgroup,status='PASS' if len(g) else 'EMPTY_SUBGROUP',windows=len(g),cells=g.canonical_cell_id.nunique())
                        if len(g):
                            m=report_metrics(g,target,lo[keep],hi[keep],raw[keep]);row.update(m)
                            if arm!=ARM:
                                ref=old.loc[old.source.eq(source)&old.model.eq(model)&old.target.eq(target)&old.recipe.eq(arm)&old.subgroup.eq(subgroup)].iloc[0]
                                for k in [k for k in m if k!='family_balanced_window_coverage']:
                                    assert np.isclose(m[k],ref[k],atol=1e-10,rtol=1e-10,equal_nan=True),(source,model,target,arm,subgroup,k)
                                    checks+=1
                        supports.append(dict(**identity,subgroup=subgroup,windows=len(g),cells=g.canonical_cell_id.nunique(),families=g.duplicate_family_id.nunique(),
                            positive_windows=int(g.y_true.gt(0).sum()),positive_cells=g.loc[g.y_true.gt(0),'canonical_cell_id'].nunique(),zero_windows=int(g.y_true.eq(0).sum())))
                        rows.append(row)
        print('[HCP SCORING]',source,flush=True)
    frame=pd.DataFrame(rows);assert len(frame)==1359 and int(frame.status.eq('NE').sum())==9
    frame.to_csv(OUT/'interval_metrics_by_source.csv',index=False);pd.DataFrame(supports).to_csv(OUT/'support_by_subgroup.csv',index=False)
    keys=['model','target','recipe','subgroup'];columns=[c for c in frame.select_dtypes(include='number') if c not in ['windows','cells','positive_ratio_windows']]
    grouped=frame.loc[frame.status.eq('PASS')].groupby(keys,sort=True)
    summary=grouped[columns].mean().reset_index().merge(grouped.agg(sources=('source','nunique'),windows=('windows','sum'),cells=('cells','sum')).reset_index(),on=keys,validate='one_to_one')
    summary.to_csv(OUT/'source_balanced_summary.csv',index=False);grouped[columns].count().reset_index().to_csv(OUT/'summary_metric_source_counts.csv',index=False)
    for b in p['inputs'].values():verify(b)
    write('completion.json',dict(status='HCP_SCORED_BASELINE_PARITY_PASS_INDEPENDENT_AUDIT_PENDING',new_fits=0,
        rows=len(frame),baseline_metric_comparisons=checks,baseline_outputs_unchanged=True,independent_audit_complete=False,
        outputs={n:bind(OUT/n) for n in ['interval_metrics_by_source.csv','support_by_subgroup.csv','source_balanced_summary.csv','summary_metric_source_counts.csv']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','calibrate','score']);a=parser.parse_args();globals()[a.stage]()

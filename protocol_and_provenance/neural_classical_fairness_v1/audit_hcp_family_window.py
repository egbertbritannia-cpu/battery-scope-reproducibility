"""Independent rational-CDF and interval arithmetic for the HCP release.

Reuses the bound data readers, not the HCP quantile or interval/metric routines.
"""
from pathlib import Path
import json
import math
from fractions import Fraction
from datetime import datetime,timezone
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1 import hcp_family_window as h

def arithmetic(f,target,q):
    a,b=f.q10.to_numpy(float),f.q90.to_numpy(float)
    if math.isinf(q):lo=np.zeros(len(f));hi=np.full(len(f),1.5 if target=='soh30' else math.inf);raw=np.full(len(f),math.inf)
    else:
        if target=='soh30':lo,hi=a-q,b+q
        else:lo,hi=np.expm1(np.log1p(a)-q),np.expm1(np.log1p(b)+q)
        raw=hi-lo;lo=np.clip(lo,0,None);hi=np.maximum(hi,lo)
        if target=='soh30':hi=np.minimum(hi,1.5)
    assert (hi>=lo).all()
    f=f.copy();f['cover']=(f.y_true.to_numpy(float)>=lo)&(f.y_true.to_numpy(float)<=hi);f['width']=hi-lo;f['raw']=raw
    return f

def m(g,target):
    cell=g.groupby('canonical_cell_id');family=g.groupby('duplicate_family_id')
    row=dict(marginal_coverage=float(g.cover.mean()),macro_cell_coverage=float(cell.cover.mean().mean()),
        cell_simultaneous_coverage=float(cell.cover.min().mean()),family_simultaneous_coverage=float(family.cover.min().mean()),
        mean_interval_width=float(g.width.mean()),median_interval_width=float(g.width.median()),
        family_balanced_window_coverage=float(family.cover.mean().mean()),cell_macro_mean_width=float(cell.width.mean().mean()),
        cell_macro_mean_raw_width=float(cell.raw.mean().mean()))
    if target!='soh30':
        g=g.copy();g['stabilized']=g.width/g.y_true.clip(lower=10)
        row['cell_macro_stabilized_relative_width']=float(g.groupby('canonical_cell_id').stabilized.mean().mean())
        pos=g.loc[g.y_true>0].copy();pos['ratio']=pos.width/pos.y_true;row['positive_ratio_windows']=len(pos)
        if len(pos):
            row['cell_macro_positive_ratio_median']=float(pos.groupby('canonical_cell_id').ratio.quantile(.5).mean())
            row['cell_macro_positive_ratio_p90']=float(pos.groupby('canonical_cell_id').ratio.quantile(.9).mean())
    return row

def main():
    assert not (h.OUT/'independent_audit.json').exists()
    d,p=h.load();done=h.read(h.OUT/'completion.json')
    for b in done['outputs'].values():h.verify(b)
    accept=h.read(h.OUT/'calibration_acceptance.json');h.verify(accept['qhat']);h.verify(accept['proofs'])
    assert accept['at_utc']<h.read(h.OUT/'score_started.json')['at_utc']
    oldq=pd.read_parquet(h.ROOT/d['inputs']['baseline_qhat']['path']);newq=pd.read_parquet(h.OUT/'hcp_calibration.parquet')
    recipes=pd.concat([oldq,newq],ignore_index=True)
    proofs={(r['source'],r['model'],r['target']):r for r in h.read(h.OUT/'quantile_cdf_proofs.json')}
    actual=pd.read_csv(h.OUT/'interval_metrics_by_source.csv');key=['source','model','target','recipe','subgroup']
    assert not actual.duplicated(key).any()
    membership=pd.read_parquet(h.ROOT/p['inputs']['membership']['path']);rows=[];support=[];checked=0
    for source in d['sources']:
        cells=set(membership.loc[membership.evaluation_mode.eq('LODO')&membership.outer_entity.eq(source)&membership.split.eq('test'),'canonical_cell_id'])
        truth=h.parent.labels(cells,h.e.META+['t_cycle']+sum((list(v) for v in h.e.SPECS.values()),[]))
        for model in d['models']:
            for target in d['targets']:
                cal=h.calibration_frame(p,source,model,target)
                transform=math.log1p if target.startswith('rul') else float
                scores=np.array([max(transform(float(a))-transform(float(y)),transform(float(y))-transform(float(b)),0.)
                    for a,b,y in zip(cal.q10,cal.q90,cal.y_true)])
                qr=newq.loc[newq.source.eq(source)&newq.model.eq(model)&newq.target.eq(target)].iloc[0];q=float(qr.qhat)
                grouped=pd.DataFrame(dict(family=cal.duplicate_family_id.to_numpy(),score=scores)).groupby('family')
                K=len(grouped);threshold=Fraction(4*(K+1),5)
                assert K==qr.calibration_families and len(cal)==qr.calibration_rows
                if math.isinf(q):assert K<threshold
                else:
                    assert np.any(scores==q)
                    at=sum((Fraction(int(g.score.le(q).sum()),len(g)) for _,g in grouped),Fraction(0))
                    before=sum((Fraction(int(g.score.lt(q).sum()),len(g)) for _,g in grouped),Fraction(0))
                    assert before<threshold<=at
                    proof=proofs[source,model,target]
                    assert Fraction(**dict(numerator=proof['cdf_at']['numerator'],denominator=proof['cdf_at']['denominator']))==at/(K+1)
                    assert Fraction(proof['cdf_before']['numerator'],proof['cdf_before']['denominator'])==before/(K+1)
                checked+=1
                f=h.test_frame(p,source,model,target,truth)
                for arm in d['recipes']:
                    identity=dict(source=source,model=model,target=target,recipe=arm)
                    if f.empty:
                        assert (source,target)==('CALB','rul90')
                        rows.append(dict(**identity,subgroup='all',status='NE',windows=0,cells=0));continue
                    q=float(recipes.loc[recipes.source.eq(source)&recipes.model.eq(model)&recipes.target.eq(target)&recipes.recipe.eq(arm),'qhat'].iloc[0])
                    w=arithmetic(f,target,q);groups=dict(all=np.ones(len(w),bool))
                    if target!='soh30':
                        tau=w.t_cycle+w.y_true
                        groups.update(last20=(tau>0)&(w.t_cycle>=.8*tau),zero=w.y_true.eq(0),near_0_10=w.y_true.gt(0)&w.y_true.le(10),near_10_50=w.y_true.gt(10)&w.y_true.le(50),over50=w.y_true.gt(50))
                    for subgroup,select in groups.items():
                        g=w.loc[select].copy();identity2=dict(**identity,subgroup=subgroup)
                        row=dict(**identity2,status='PASS' if len(g) else 'EMPTY_SUBGROUP',windows=len(g),cells=g.canonical_cell_id.nunique())
                        if len(g):row.update(m(g,target))
                        rows.append(row)
                        support.append(dict(**identity2,windows=len(g),cells=g.canonical_cell_id.nunique(),families=g.duplicate_family_id.nunique(),
                            positive_windows=int(g.y_true.gt(0).sum()),positive_cells=g.loc[g.y_true.gt(0),'canonical_cell_id'].nunique(),zero_windows=int(g.y_true.eq(0).sum())))
        print('[INDEPENDENT HCP AUDIT]',source,flush=True)
    expected=pd.DataFrame(rows)
    pd.testing.assert_frame_equal(actual.sort_values(key).reset_index(drop=True),expected[actual.columns].sort_values(key).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=1e-10)
    actualsupport=pd.read_csv(h.OUT/'support_by_subgroup.csv')
    pd.testing.assert_frame_equal(actualsupport.sort_values(key).reset_index(drop=True),pd.DataFrame(support)[actualsupport.columns].sort_values(key).reset_index(drop=True),check_dtype=False)
    keys=['model','target','recipe','subgroup'];cols=[c for c in expected.select_dtypes(include='number') if c not in ['windows','cells','positive_ratio_windows']]
    grouped=expected.loc[expected.status.eq('PASS')].groupby(keys,sort=True)
    summary=grouped[cols].mean().reset_index().merge(grouped.agg(sources=('source','nunique'),windows=('windows','sum'),cells=('cells','sum')).reset_index(),on=keys,validate='one_to_one')
    summaryactual=pd.read_csv(h.OUT/'source_balanced_summary.csv')
    pd.testing.assert_frame_equal(summaryactual,summary[summaryactual.columns],check_dtype=False,atol=1e-10,rtol=1e-10)
    counts=pd.read_csv(h.OUT/'summary_metric_source_counts.csv')
    pd.testing.assert_frame_equal(counts,grouped[cols].count().reset_index()[counts.columns],check_dtype=False)
    for b in p['inputs'].values():h.verify(b)
    assert checked==108 and len(expected)==1359
    h.write('independent_audit.json',dict(status='PASS_HCP_EXACT_CDF_AND_ALL_REPORTED_INTERVAL_ARITHMETIC',at_utc=datetime.now(timezone.utc).isoformat(),
        hcp_qhats_checked=checked,reported_rows=len(expected),evaluable_rows=int(expected.status.eq('PASS').sum()),NE_rows=int(expected.status.eq('NE').sum()),
        baseline_outputs_unchanged=True,new_fits=0,quantile_algorithm_reused=False,interval_metric_algorithm_reused=False,
        data_readers_reused=True,all_summary_and_support_rows_checked=True,audit_source=h.bind(Path(__file__)),outputs=done['outputs'],
        qhats=accept['qhat'],quantile_proofs=accept['proofs']))
    print('PASS_HCP_INDEPENDENT_ARITHMETIC',flush=True)

if __name__=='__main__':main()

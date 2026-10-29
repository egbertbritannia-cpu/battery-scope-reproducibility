"""Reconstruct the post hoc residual SoH extension from released aggregates.

Standard library by default. --bootstrap additionally requires NumPy.
No training, model selection, raw prediction scoring or manuscript access.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parent
METRICS=('mae','rmse','bias')
SEEDS=(2026,2027,2028,2029,2030)
SYSTEM='Trend-residual TCN'


def read(relative):
    with (ROOT/relative).open(encoding='utf8',newline='') as stream:
        return list(csv.DictReader(stream))


def aggregate(rows):
    """Exactly one aggregate per source/seed, with equal source weights."""
    rows=list(rows)
    sources=sorted({r['source'] for r in rows})
    keys={(r['source'],int(r['seed'])) for r in rows}
    if len(rows)!=60 or len(keys)!=60 or len(sources)!=12 or keys!={(s,k) for s in sources for k in SEEDS}:
        raise ValueError('Expected exactly 12 sources by five seeds')
    if any(r['system']!=SYSTEM or r['endpoint']!='soh30' for r in rows):
        raise ValueError('Unexpected system or endpoint')
    if any(not math.isfinite(float(r[m])) for r in rows for m in METRICS):
        raise ValueError('Nonfinite metric')
    if any(float(r['mae'])<0 or float(r['rmse'])+1e-12<float(r['mae'])
           or abs(float(r['bias']))>float(r['mae'])+1e-12 for r in rows):
        raise ValueError('Inconsistent error metrics')
    metrics={}
    for metric in METRICS:
        by_seed=[st.mean(float(r[metric]) for r in rows if int(r['seed'])==seed) for seed in SEEDS]
        metrics[metric]=dict(mean=st.mean(by_seed),sample_seed_sd=st.stdev(by_seed),seed_means=by_seed)
    source_rows=[]
    for source in sources:
        rr=[r for r in rows if r['source']==source]
        counts={(int(r['cells']),int(r['windows'])) for r in rr}
        if len(counts)!=1 or any(a<=0 or b<=0 for a,b in counts):
            raise ValueError('Changed or invalid evaluation support')
        cells,windows=next(iter(counts))
        source_rows.append(dict(system=SYSTEM,endpoint='soh30',source=source,seeds=5,cells=cells,windows=windows,
            primary=st.mean(float(r['mae']) for r in rr),
            **{m:st.mean(float(r[m]) for r in rr) for m in METRICS},
            primary_seed_sd=st.stdev(float(r['mae']) for r in rr)))
    return metrics,source_rows


def close(actual,expected):
    if not math.isclose(float(actual),float(expected),rel_tol=0,abs_tol=1e-12):
        raise ValueError(('Aggregate mismatch',actual,expected))


def reconstruct(*,bootstrap=False):
    rows=read('results/residual_outer/source_seed_metrics.csv')
    metrics,sources=aggregate(rows)
    summary=json.loads((ROOT/'results/residual_outer/summary.json').read_text())
    for metric in METRICS:
        for field in ('mean','sample_seed_sd'):
            close(metrics[metric][field],summary['metrics'][metric][field])
        for actual,expected in zip(metrics[metric]['seed_means'],summary['metrics'][metric]['seed_means']):
            close(actual,expected)
    support=read('results/primary/support.csv')
    for r in sources:
        match=[s for s in support if s['source']==r['source'] and s['endpoint']=='soh30']
        if len(match)!=1 or r['cells']!=int(match[0]['evaluable_cells']) or r['windows']!=int(match[0]['evaluable_windows']):
            raise ValueError('Primary and residual SoH support differ')
    released=read('results/residual_outer/source_summary.csv')
    if len(released)!=12:raise ValueError('Expected twelve source summaries')
    for r in sources:
        matches=[s for s in released if s['source']==r['source']]
        if len(matches)!=1:raise ValueError('Missing or duplicated source summary')
        for field in ('primary','mae','rmse','bias','primary_seed_sd'):
            close(r[field],matches[0][field])
    baseline=read('results/primary/source_summary.csv')
    agg=read('results/primary/aggregate_metrics.csv')
    own={r['source']:r['primary'] for r in sources}
    for expected in summary['comparisons']:
        model=expected['reference']
        reference={r['source']:float(r['primary']) for r in baseline if r['system']==model and r['endpoint']=='soh30'}
        if reference.keys()!=own.keys():raise ValueError('Paired source support differs')
        delta=[own[s]-reference[s] for s in sorted(own)]
        close(st.mean(delta),expected['delta'])
        if sum(d<0 for d in delta)!=expected['wins']:raise ValueError('Source win count differs')
        if bootstrap:
            import numpy as np
            values=np.array(delta)
            rng=np.random.default_rng(20260914)
            band=np.percentile(values[rng.integers(0,12,(10000,12))].mean(axis=1),[2.5,97.5])
            for a,e in zip(band,expected['source_bootstrap_95']):close(a,e)
    persistence=next(float(r['primary']) for r in agg if r['system']=='Persistence' and r['endpoint']=='soh30')
    skill=100*(1-metrics['mae']['mean']/persistence)
    close(skill,summary['soh30_skill_percent'])
    table=dict(system=SYSTEM,soh30=metrics['mae']['mean'],soh30_seed_sd=metrics['mae']['sample_seed_sd'],
        soh30_skill_percent=skill,rul90='',rul90_seed_sd='',rul90_skill_percent='',rul80='',rul80_seed_sd='')
    return dict(metrics=metrics,sources=sources,table3_row=table)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--bootstrap',action='store_true')
    args=parser.parse_args()
    result=reconstruct(bootstrap=args.bootstrap)
    print(json.dumps(dict(status='RESIDUAL_OUTER_AGGREGATES_PASS',sources=12,seeds=5,source_seed_rows=60,
        comparisons=4,bootstrap_recomputed=args.bootstrap,metrics=result['metrics'],
        soh30_skill_percent=result['table3_row']['soh30_skill_percent'],new_fits=0,new_prediction_scoring=False),indent=2))


if __name__=='__main__':main()

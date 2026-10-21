"""Reconstruct common-input summaries from supplied source aggregates only."""
import csv,json,math,re,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'results/common_input'
def rows(name):
    with (BASE/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def mean(values):return math.fsum(values)/len(values)
def fmt(v):return f'{v:.6f}'.replace('.',',')
selected=rows('selected_source_metrics.csv')
groups=defaultdict(list);folds=defaultdict(list)
for r in selected:
    groups[r['family'],r['arm'],r['seed']].append(float(r['mae']))
    folds[r['family'],r['arm'],r['seed'],r['fold']].append(float(r['mae']))
assert len(folds)==54 and len(groups)==18
aggregate=defaultdict(list)
for (family,arm,seed),values in groups.items():
    assert len(values)==11
    aggregate[family,arm].append(mean(values))
summary=json.loads((BASE/'summaries.json').read_text(encoding='utf-8'))
for (family,arm),values in aggregate.items():
    assert len(values)==3
    expected=summary[family][arm]
    assert abs(mean(values)-expected['mean_equal_source'])<1e-12
    assert abs(statistics.stdev(values)-expected['sd_across_three_seeds'])<1e-12
history=rows('epoch_source_metrics.csv');trajectories=defaultdict(lambda:defaultdict(list));scores={}
for r in history:
    key=r['family'],r['arm'],r['seed'],r['fold'];epoch=int(r['epoch'])
    trajectories[key][epoch].append(float(r['mae']))
    scores[key,epoch]=float(r['fold_selection_score'])
modes=defaultdict(list);missed=defaultdict(int)
for key,epochs in trajectories.items():
    assert set(epochs)==set(range(1,31))
    family,arm,seed,fold=key
    best=min(epochs,key=lambda e:scores[key,e])
    for mode,values in [('selected',folds[key]),('last',epochs[30]),('best',epochs[best])]:
        modes[family,arm,seed,mode].extend(values)
    if scores[key,best]<mean(folds[key])-1e-8:missed[family,arm]+=1
expected=json.loads((BASE/'checkpoint_sensitivity.json').read_text(encoding='utf-8'))
for item in expected:
    family,arm=item['family'],item['arm']
    for mode in ['selected','last','best']:
        values=[]
        for seed in ['2026','2027','2028']:
            source_values=modes[family,arm,seed,mode]
            assert len(source_values)==11
            values.append(mean(source_values))
        assert abs(mean(values)-item['means'][mode])<1e-12
    assert missed[family,arm]==item['later_better']
print(json.dumps(dict(status='COMMON_INPUT_SOURCE_AGGREGATES_PASS',fits=54,neural_histories=len(trajectories),aggregate_model_rows=3,checkpoint_rows=4,manuscript_read=False,new_fits=0,new_prediction_scoring=False)))

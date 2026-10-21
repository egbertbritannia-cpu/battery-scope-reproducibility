"""Post-hoc arithmetic diagnosis of accepted frozen intervals; no fitting/selection."""
from pathlib import Path
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'paper/audits/neural_classical_fairness_2026-10-08_v1'
SCORE = BASE / 'coverage_extension_scoring_v1'
OUT = BASE / 'coverage_seed_width_diagnostic_20261017_v1'
META = ['sample_id', 'canonical_cell_id', 'duplicate_family_id', 'source']
TARGETS = {'soh30': ('soh_t_plus_30', 'soh30_loss_mask'), 'rul90': ('rul90_cycles', 'rul90_loss_mask'), 'rul80': ('rul80_cycles', 'rul80_loss_mask')}
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def bind(p):
    with p.open('rb') as f: sha = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha)
def verify(b): assert bind(ROOT / b['path']) == b, b['path']
def write(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
def table(arc, member):
    with zipfile.ZipFile(arc) as z: return pd.read_parquet(io.BytesIO(z.read(member)))
def bounds(f, target, q):
    a, b = f.q10.to_numpy(float), f.q90.to_numpy(float)
    if target.startswith('rul'): lo, hi = np.expm1(np.log1p(a)-q), np.expm1(np.log1p(b)+q)
    else: lo, hi = a-q, b+q
    raw = hi-lo
    lo = np.maximum(lo, 0); hi = np.maximum(hi, lo)
    if target == 'soh30': hi = np.minimum(hi, 1.5)
    assert np.isfinite(lo).all() and np.isfinite(hi).all() and (hi >= lo).all()
    return lo, hi, raw
def metrics(f, lo, hi):
    w = f[META].copy(); w['hit'] = (f.y.to_numpy(float) >= lo) & (f.y.to_numpy(float) <= hi)
    w['width'] = hi-lo
    return dict(marginal_coverage=float(w.hit.mean()), macro_cell_coverage=float(w.groupby('canonical_cell_id').hit.mean().mean()),
        cell_simultaneous_coverage=float(w.groupby('canonical_cell_id').hit.all().mean()),
        family_simultaneous_coverage=float(w.groupby('duplicate_family_id').hit.all().mean()),
        mean_interval_width=float(w.width.mean()), median_interval_width=float(w.width.median()),
        cell_macro_mean_width=float(w.groupby('canonical_cell_id').width.mean().mean()))

def main():
    assert not OUT.exists(), 'FRESH_DIAGNOSTIC_REQUIRED'
    p = read(SCORE / 'protocol.frozen.json'); accepted = read(SCORE / 'accepted_completion.json')
    assert accepted['status'] == 'ACCEPTED_COVERAGE_EXTENSION_COMPLETE_INDEPENDENT_ARITHMETIC_PASS'
    verify(accepted['independent_audit']); verify(read(SCORE / 'protocol_seal.json')['protocol'])
    done = read(SCORE / 'completion.json')
    for b in done['outputs'].values(): verify(b)
    qbinding = read(SCORE / 'calibration_acceptance.json')['qhat']; verify(qbinding)
    OUT.mkdir(); (OUT / 'before').mkdir()
    old_manifest = read(ROOT / 'paper/latex/provenance/md_export_manifest.json')
    paths = ['paper/manuscript/main_vi.md', 'paper/latex/tools/sync_from_markdown.py', 'paper/latex/provenance/md_export_manifest.json', 'paper/latex/README.md']
    paths += ['paper/latex/' + n for n in old_manifest['generated_outputs']]
    snapshots = {}
    for name in sorted(set(paths)):
        src = ROOT / name; dst = OUT / 'before' / name; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(src.read_bytes()); snapshots[name] = bind(src)
    write('before_manifest.json', snapshots)
    design = read(ROOT / p['inputs']['design']['path'])
    lock = dict(schema='coverage-seed-width-diagnostic-v1', at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Post hoc after all coverage scores seen. Mechanical decomposition, not causal effect identification; no recipe/seed selection.',
        inputs=dict(protocol=bind(SCORE/'protocol.frozen.json'), accepted=bind(SCORE/'accepted_completion.json'), qhat=qbinding,
            metrics=bind(SCORE/'interval_metrics_by_source.csv'), design=p['inputs']['design'], code=bind(Path(__file__))),
        analysis=['All source/model/target raw vs calibrated widths and coverages, both recipes',
            'Exact width identity: base quantile width + upper expansion + lower expansion, before/after bounds',
            'XJTU RUL90: fixed-prediction 2x2 TCN2026/2027 prediction and qhat swaps for both recipes; diagnostic only',
            'XJTU RUL90 family-level miss counts and maximum nonconformity; no subsequent fit'],
        new_fits=0, label_access='Previously scored eligible test labels only; no selection or new calibration')
    write('protocol.frozen.json', lock); write('protocol_seal.json', dict(protocol=bind(OUT/'protocol.frozen.json')))
    targets = ds.dataset(ROOT / p['inputs']['targets']['path'], format='parquet')
    membership = pd.read_parquet(ROOT/p['inputs']['membership']['path'])
    recipes = pd.read_parquet(SCORE/'calibration_recipes.parquet'); actual = pd.read_csv(SCORE/'interval_metrics_by_source.csv')
    rows = []; families = []; swaps = []; xjtu = {}; parity_checks = 0
    for source in design['held_sources']:
        cells = sorted(set(membership.loc[membership.evaluation_mode.eq('LODO') & membership.outer_entity.eq(source) & membership.split.eq('test'), 'canonical_cell_id']))
        truth = targets.to_table(columns=META+sum((list(v) for v in TARGETS.values()), []), filter=ds.field('canonical_cell_id').isin(cells)).to_pandas()
        for model in p['models']:
            arch, seed = model.split('.')
            if model == 'TCN.2026':
                j = next(j for j in p['historical_reference']['jobs'] if j['source'] == source)
                arc = ROOT / p['historical_reference']['archives'][j['shard']]['path']
                members = {t: j['members'][t+'.outer_predictions.parquet']['member'] for t in TARGETS}
            else:
                j = next(j for j in design['selected_existing_models'] if j['source'] == source and j['architecture'] == arch and j['seed'] == int(seed))
                arc = ROOT/j['archive']; members = {r['endpoint']:r['member'] for r in j['quantile_prediction_schemas']}
            for target, (col, mask) in TARGETS.items():
                pred = table(arc, members[target]); y = truth.loc[truth[mask].astype(bool) & np.isfinite(truth[col]), ['sample_id', col]].rename(columns={col:'y'})
                f = pred[META+['q10','q50','q90']].merge(y, on='sample_id', validate='one_to_one')
                if f.empty:
                    assert (source,target)==('CALB','rul90'); continue
                if (source,target)==('XJTU','rul90'): xjtu[model]=f
                for arm in ['window','family_max']:
                    r = recipes.loc[recipes.model.eq(model)&recipes.source.eq(source)&recipes.target.eq(target)&recipes.recipe.eq(arm)].iloc[0]
                    q = float(r.qhat); lo,hi,raw = bounds(f,target,q); m = metrics(f,lo,hi)
                    ref = actual.loc[actual.model.eq(model)&actual.source.eq(source)&actual.target.eq(target)&actual.recipe.eq(arm)&actual.subgroup.eq('all')].iloc[0]
                    for k,v in m.items(): assert np.isclose(v, ref[k], atol=1e-10, rtol=1e-10), (source,model,target,arm,k); parity_checks+=1
                    blo,bhi,_ = bounds(f,target,0); bm = metrics(f,blo,bhi)
                    w = f[META].copy(); w['base'] = f.q90.to_numpy(float)-f.q10.to_numpy(float)
                    w['upper'] = hi-f.q90.to_numpy(float); w['lower'] = f.q10.to_numpy(float)-lo
                    w['width'] = hi-lo; w['raw_width'] = raw
                    assert np.allclose(w.base+w.upper+w.lower,w.width,atol=1e-10,rtol=1e-10)
                    cm = w.groupby('canonical_cell_id')[['base','upper','lower','width','raw_width']].mean().mean()
                    rows.append(dict(source=source,model=model,target=target,recipe=arm,qhat=q,calibration_units=int(r.calibration_units),quantile_rank=int(r.quantile_rank),
                        windows=len(f),cells=f.canonical_cell_id.nunique(),families=f.duplicate_family_id.nunique(),
                        base_cell_macro_width=float(cm['base']),upper_expansion=float(cm['upper']),lower_expansion=float(cm['lower']),
                        raw_cell_macro_width=float(cm['raw_width']),**m,**{'base_'+k:v for k,v in bm.items()}))
                    if (source,target)==('XJTU','rul90'):
                        z = np.log1p(f.y.to_numpy(float)); s = np.maximum.reduce([np.log1p(f.q10.to_numpy(float))-z,z-np.log1p(f.q90.to_numpy(float)),np.zeros(len(f))])
                        w['score']=s; w['miss']=~((f.y.to_numpy(float)>=lo)&(f.y.to_numpy(float)<=hi))
                        for family,g in w.groupby('duplicate_family_id'):
                            families.append(dict(model=model,recipe=arm,family=family,windows=len(g),missed_windows=int(g.miss.sum()),
                                covered=not bool(g.miss.any()),max_score=float(g.score.max()),qhat=q,score_margin=float(q-g.score.max())))
        print('[WIDTH DIAGNOSIS]',source,flush=True)
    for arm in ['window','family_max']:
        for predictor in ['TCN.2026','TCN.2027']:
            for calibrator in ['TCN.2026','TCN.2027']:
                f=xjtu[predictor]; q=float(recipes.loc[recipes.model.eq(calibrator)&recipes.source.eq('XJTU')&recipes.target.eq('rul90')&recipes.recipe.eq(arm),'qhat'].iloc[0])
                lo,hi,_=bounds(f,'rul90',q)
                swaps.append(dict(source='XJTU',target='rul90',recipe=arm,prediction_model=predictor,qhat_model=calibrator,qhat=q,
                    status='ORIGINAL' if predictor==calibrator else 'DIAGNOSTIC_SWAP_NOT_A_CANDIDATE',**metrics(f,lo,hi)))
    data=pd.DataFrame(rows); data.to_csv(OUT/'width_decomposition_by_source.csv',index=False)
    data.groupby(['model','target','recipe'],sort=True).mean(numeric_only=True).reset_index().to_csv(OUT/'width_decomposition_source_means.csv',index=False)
    pd.DataFrame(families).to_csv(OUT/'xjtu_rul90_family_failures.csv',index=False)
    pd.DataFrame(swaps).to_csv(OUT/'xjtu_rul90_fixed_prediction_qhat_swaps.csv',index=False)
    for b in p['inputs'].values(): verify(b)
    for b in lock['inputs'].values(): verify(b)
    write('completion.json',dict(status='FROZEN_INTERVAL_DIAGNOSIS_COMPLETE',new_fits=0,original_metric_checks=parity_checks,
        source_rows=len(data),qhat_swap_rows=len(swaps),original_inputs_and_scores_unchanged=True,
        outputs={n:bind(OUT/n) for n in ['width_decomposition_by_source.csv','width_decomposition_source_means.csv','xjtu_rul90_family_failures.csv','xjtu_rul90_fixed_prediction_qhat_swaps.csv']}))

if __name__=='__main__': main()

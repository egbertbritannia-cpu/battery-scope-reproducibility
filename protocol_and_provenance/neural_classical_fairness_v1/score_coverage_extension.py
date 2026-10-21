"""Separate post-hoc coverage release: audit/seal -> calibration -> test scoring.

Never fit or select predictors. Requires an accepted complete inference return.
Reuses frozen E07 math; adds cell-weighted widths and retrospective RUL strata.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'paper/audits/neural_classical_fairness_2026-10-08_v1'
RELEASE = ROOT / 'notebook/neural-classical-fairness-v1/coverage-extension-inference-v3'
OUT = BASE / 'coverage_extension_scoring_v1'
BUNDLE = ROOT / 'phase2-ouput/batteryscope_corrected_inner_v2_kaggle/batteryscope_corrected_inner_v2'
E07 = ROOT / 'paper/audits/e07_calibration_unit_ablation_2026-10-05_v1'
from protocol_and_provenance.data_integrity import e07_calibration_ablation as e


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def bind(p):
    p = Path(p)
    with p.open('rb') as f: s = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=s)
def verify(b): assert bind(ROOT / b['path']) == b, 'BOUND_FILE_CHANGED:' + b['path']
def write(n, v):
    with (OUT / n).open('x', encoding='utf-8') as f: json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)
def table(archive, member):
    import pandas as pd
    with zipfile.ZipFile(archive) as z: return pd.read_parquet(io.BytesIO(z.read(member)))
def load():
    p = read(OUT / 'protocol.frozen.json'); assert bind(OUT / 'protocol.frozen.json') == read(OUT / 'protocol_seal.json')['protocol']
    for b in p['inputs'].values(): verify(b)
    return p


def freeze(archive):
    import pandas as pd
    assert archive and not OUT.exists(), 'FRESH_SCORING_RELEASE_REQUIRED'
    m = read(RELEASE / 'delivery.frozen.json'); receipt = read(RELEASE / 'build_receipt.json')
    assert {k: bind(RELEASE / 'delivery.frozen.json')[k] for k in ('bytes', 'sha256')} == receipt['manifest']
    prefix = 'batteryscope_coverage_extension_infer_v2/'
    with zipfile.ZipFile(archive) as z:
        inv = json.loads(z.read(prefix + 'output_inventory.json'))['files']
        assert len(z.namelist()) == len(set(z.namelist()))
        assert set(z.namelist()) == {prefix + n for n in inv} | {prefix + 'output_inventory.json'}
        for n, expected in inv.items():
            data = z.read(prefix + n); assert dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) == expected
        def get(n): return json.loads(z.read(prefix + n))
        assert get('delivery_manifest.snapshot.json') == m
        assert get('summary.json')['status'] == 'ALL24_CHECKPOINTS_REPLAYED_AND_CALIBRATION_INFERRED_AUDIT_PENDING'
        assert get('all_model_test_parity.json')['status'] == 'ALL24_FULL_TEST_QUANTILES_EXACT_PASS'
        assert len(get('ledger.json')) == 24 and all(r['phase'] == 'CALIBRATION_INFERRED' for r in get('ledger.json'))
        assert not get('environment_check.json')['mismatches']
        for log in get('read_log.json'): assert set(log['columns']) <= set(['sample_id', 'canonical_cell_id', 't_cycle', 'duplicate_family_id', 'source'])
        membership = pd.read_parquet(BUNDLE / 'phase2/battery_phase2_fold_membership.parquet')
        import pyarrow.dataset as ds
        profiles = pd.read_parquet(BUNDLE / 'phase2/multisensor_cycle_store_index.parquet').set_index('canonical_cell_id')
        for cell, expected in get('profile_hashes.json').items():
            assert str(profiles.loc[cell, 'sha256']).lower() == expected
        parity_records = 0
        for j in m['jobs']:
            f = membership.loc[membership.evaluation_mode.eq('LODO') & membership.outer_entity.eq(j['source']) & membership.split.eq('calibration')]
            grid = ds.dataset(BUNDLE / 'phase2/sample_target_index.parquet', format='parquet').to_table(
                columns=e.META + ['t_cycle'], filter=ds.field('canonical_cell_id').isin(sorted(set(f.canonical_cell_id)))).to_pandas()
            assert grid.sample_id.is_unique
            replay = get('jobs/' + j['id'] + '/test_parity.json')
            assert replay['status'] == 'FULL_TEST_QUANTILES_EXACT_PASS' and replay['targets'] == 3 and replay['labels_read'] is False
            parity_records += 1
            for t in e.SPECS:
                frame = pd.read_parquet(io.BytesIO(z.read(prefix + 'jobs/' + j['id'] + '/' + t + '.calibration_predictions.parquet')))
                assert frame.sample_id.is_unique and frame.endpoint.eq(t).all()
                assert e.id_hash(frame.canonical_cell_id) == e.id_hash(f.canonical_cell_id)
                assert e.id_hash(frame.duplicate_family_id) == e.id_hash(f.duplicate_family_id)
                e.validate_frame(frame, t)
                pd.testing.assert_frame_equal(frame[e.META + ['t_cycle']].sort_values('sample_id').reset_index(drop=True),
                    grid.sort_values('sample_id').reset_index(drop=True), check_dtype=False, check_exact=True)
    old = read(E07 / 'protocol.frozen.json')
    inputs = dict(inference_return=bind(archive), inference_manifest=bind(RELEASE / 'delivery.frozen.json'),
        design=bind(ROOT / m['parent_design']['path']), e07_protocol=bind(E07 / 'protocol.frozen.json'),
        e07_qhat=bind(E07 / 'calibration_recipes.parquet'), e07_metrics=bind(E07 / 'interval_metrics_by_source.parquet'),
        targets=bind(BUNDLE / 'phase2/sample_target_index.parquet'), windows=bind(BUNDLE / 'phase2/window_index.parquet'),
        membership=bind(BUNDLE / 'phase2/battery_phase2_fold_membership.parquet'),
        arithmetic=bind(Path(e.__file__)), scoring_source=bind(Path(__file__)))
    for name in m['original_archives']: inputs['archive:' + name] = bind(ROOT / name)
    for name, b in old['archives'].items():
        inputs['base2026:' + name] = bind(ROOT / b['path']); assert inputs['base2026:' + name] == b
    score = BASE / 'multisource_outer_scoring_v1'
    for name, expected in m['primary_score_outputs'].items():
        b = bind(score / name); assert {k: b[k] for k in expected} == expected; inputs['protected:' + name] = b
    OUT.mkdir()
    p = dict(schema='posthoc-coverage-extension-scoring-v1', at_utc=datetime.now(timezone.utc).isoformat(),
        inputs=inputs, inference_prefix=prefix, historical_reference=old, jobs=m['jobs'],
        chronology='Post hoc. Prior outer scores/intervals already seen; protocol precedes new calibration-label reads.',
        models=['TCN.2026', 'GRU.2026', 'TCN.2027'], arms=list(e.ARMS), alpha=.2,
        metrics_design=read(ROOT / m['parent_design']['path'])['reporting'],
        new_fits=0, selection_changed=False, calibrated_outputs_not_yet_computed=True)
    write('protocol.frozen.json', p); write('protocol_seal.json', dict(protocol=bind(OUT / 'protocol.frozen.json')))
    write('inference_return_audit.json', dict(status='ALL24_FULL_QUANTILE_PARITY_AND_RETURN_HASH_AUDIT_PASS',
        inventoried_files=len(inv), new_fits=0, calibration_labels_read=False, test_labels_read=False))
    print('COVERAGE_PROTOCOL_SEALED_BEFORE_NEW_CALIBRATION_LABELS', flush=True)


def labels(cells, columns):
    import pyarrow.dataset as ds
    return ds.dataset(BUNDLE / 'phase2/sample_target_index.parquet', format='parquet').to_table(
        columns=columns, filter=ds.field('canonical_cell_id').isin(sorted(cells))).to_pandas()


def calibrate():
    import numpy as np
    import pandas as pd
    p = load(); assert not (OUT / 'calibration_started.json').exists()
    write('calibration_started.json', dict(at_utc=datetime.now(timezone.utc).isoformat(), test_labels_read=False))
    membership = pd.read_parquet(BUNDLE / 'phase2/battery_phase2_fold_membership.parquet')
    rows = []
    with zipfile.ZipFile(ROOT / p['inputs']['inference_return']['path']) as z:
        for j in p['jobs']:
            f = membership.loc[membership.evaluation_mode.eq('LODO') & membership.outer_entity.eq(j['source']) & membership.split.eq('calibration')]
            y = labels(set(f.canonical_cell_id), e.META + sum((list(v) for v in e.SPECS.values()), []))
            assert y.sample_id.is_unique
            for t, (col, mask) in e.SPECS.items():
                assert y[mask].notna().all() and y[mask].isin([True, False, 0, 1]).all()
                data = pd.read_parquet(io.BytesIO(z.read(p['inference_prefix'] + 'jobs/' + j['id'] + '/' + t + '.calibration_predictions.parquet')))
                assert set(data.sample_id) == set(y.sample_id)
                observed = y.loc[y[mask].astype(bool) & np.isfinite(y[col]), e.META + [col, mask]].copy()
                observed['y_true'] = observed[col]
                observed = data.merge(observed, on=e.META, validate='one_to_one', how='inner')
                recipes = e.calibrate_frame(observed, t)
                for arm, r in recipes.items():
                    rows.append(dict(model=j['architecture'] + '.' + str(j['seed']), source=j['source'], target=t,
                        recipe=arm, **r, calibration_rows=len(observed)))
            print('[COVERAGE CALIBRATED]', j['id'], flush=True)
    old = pd.read_parquet(ROOT / p['inputs']['e07_qhat']['path'])
    for r in old.to_dict('records'):
        rows.append(dict(model='TCN.2026', **{k: r[k] for k in ['source', 'target', 'recipe', 'qhat', 'calibration_units', 'quantile_rank', 'calibration_rows']}))
    assert len(rows) == 216
    pd.DataFrame(rows).to_parquet(OUT / 'calibration_recipes.parquet', index=False)
    write('calibration_acceptance.json', dict(status='ALL_QHATS_SEALED_BEFORE_NEW_TEST_LABEL_READ',
        recipes=216, new_predictor_fits=0, test_labels_read=False, qhat=bind(OUT / 'calibration_recipes.parquet')))


def score():
    import numpy as np
    import pandas as pd
    p = load(); acceptance = read(OUT / 'calibration_acceptance.json'); verify(acceptance['qhat'])
    assert not (OUT / 'score_started.json').exists()
    write('score_started.json', dict(at_utc=datetime.now(timezone.utc).isoformat(), qhat=acceptance['qhat'], retrospective=True))
    recipes = pd.read_parquet(OUT / 'calibration_recipes.parquet')
    membership = pd.read_parquet(BUNDLE / 'phase2/battery_phase2_fold_membership.parquet')
    old_metrics = pd.read_parquet(ROOT / p['inputs']['e07_metrics']['path'])
    rows = []; support = []
    for source in sorted(membership.source.unique()):
        cells = set(membership.loc[membership.evaluation_mode.eq('LODO') & membership.outer_entity.eq(source) & membership.split.eq('test'), 'canonical_cell_id'])
        if not cells: continue
        truth = labels(cells, e.META + ['t_cycle'] + sum((list(v) for v in e.SPECS.values()), []))
        for model in p['models']:
            if model == 'TCN.2026':
                oldjob = next(j for j in p['historical_reference']['jobs'] if j['source'] == source)
                archive = ROOT / p['historical_reference']['archives'][oldjob['shard']]['path']
                members = {t: oldjob['members'][t + '.outer_predictions.parquet']['member'] for t in e.SPECS}
            else:
                arch, seed = model.split('.'); job = next(j for j in p['jobs'] if j['source'] == source and j['architecture'] == arch)
                archive = ROOT / job['checkpoint_origin']['archive']
                inv = read(ROOT / p['inputs']['design']['path'])['selected_existing_models']
                original = next(r for r in inv if r['source'] == source and r['architecture'] == arch and r['seed'] == int(seed))
                members = {r['endpoint']: r['member'] for r in original['quantile_prediction_schemas']}
            for t, (col, mask) in e.SPECS.items():
                pred = table(archive, members[t]); assert pred.sample_id.is_unique and set(pred.sample_id) == set(truth.sample_id)
                active = truth.loc[truth[mask].astype(bool) & np.isfinite(truth[col]), ['sample_id', 't_cycle', col]].rename(columns={col: 'y_true'})
                f = pred[e.META + ['q10', 'q50', 'q90']].merge(active, on='sample_id', how='inner', validate='one_to_one')
                if model == 'TCN.2026': support.append(dict(source=source, target=t, windows=len(f), cells=f.canonical_cell_id.nunique(), total_cells=len(cells)))
                for arm in e.ARMS:
                    q = recipes.loc[(recipes.model == model) & (recipes.source == source) & (recipes.target == t) & (recipes.recipe == arm)]
                    assert len(q) == 1; qhat = float(q.iloc[0].qhat)
                    if not len(f):
                        assert (source, t) == ('CALB', 'rul90')
                        rows.append(dict(model=model, source=source, target=t, recipe=arm, subgroup='all', status='NE', windows=0, cells=0))
                        continue
                    lo, hi = e.bounds(f, t, qhat); f = f.copy(); f['width'] = hi - lo
                    if t.startswith('rul'):
                        f['raw_width'] = np.expm1(np.log1p(f.q90.to_numpy(float)) + qhat) - np.expm1(np.log1p(f.q10.to_numpy(float)) - qhat)
                    else: f['raw_width'] = f.q90.to_numpy(float) - f.q10.to_numpy(float) + 2 * qhat
                    groups = {'all': np.ones(len(f), bool)}
                    if t.startswith('rul'):
                        y = f.y_true.to_numpy(float); tau = f.t_cycle.to_numpy(float) + y
                        groups.update(last20=(tau > 0) & (f.t_cycle.to_numpy(float) / np.maximum(tau, 1) >= .8),
                            zero=y == 0, near_0_10=(y > 0) & (y <= 10), near_10_50=(y > 10) & (y <= 50), over50=y > 50)
                    for subgroup, keep in groups.items():
                        g = f.loc[keep]; identity = dict(model=model, source=source, target=t, recipe=arm, subgroup=subgroup,
                            status='PASS' if len(g) else 'EMPTY_SUBGROUP', windows=len(g), cells=g.canonical_cell_id.nunique())
                        if not len(g): rows.append(identity); continue
                        metrics = e.interval_metrics(g, lo[keep], hi[keep])
                        metrics['cell_macro_mean_width'] = float(g.groupby('canonical_cell_id').width.mean().mean())
                        metrics['cell_macro_mean_raw_width'] = float(g.groupby('canonical_cell_id').raw_width.mean().mean())
                        if t.startswith('rul'):
                            g = g.copy(); g['stabilized_relative_width'] = g.width / np.maximum(g.y_true, 10)
                            metrics['cell_macro_stabilized_relative_width'] = float(g.groupby('canonical_cell_id').stabilized_relative_width.mean().mean())
                            positive = g.loc[g.y_true > 0].copy(); positive['ratio'] = positive.width / positive.y_true
                            metrics['positive_ratio_windows'] = len(positive)
                            if len(positive):
                                metrics['cell_macro_positive_ratio_median'] = float(positive.groupby('canonical_cell_id').ratio.median().mean())
                                metrics['cell_macro_positive_ratio_p90'] = float(positive.groupby('canonical_cell_id').ratio.quantile(.9).mean())
                        if model == 'TCN.2026' and subgroup == 'all':
                            ref = old_metrics.loc[(old_metrics.source == source) & (old_metrics.target == t) & (old_metrics.recipe == arm)].iloc[0]
                            assert all(np.isclose(metrics[k], ref[k], atol=1e-10, rtol=1e-10) for k in e.METRICS), 'E07_REFERENCE_PARITY'
                        rows.append(dict(**identity, **metrics))
        print('[COVERAGE SCORED]', source, flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(OUT / 'interval_metrics_by_source.csv', index=False)
    pd.DataFrame(support).to_csv(OUT / 'support.csv', index=False)
    columns = [c for c in frame.select_dtypes(include='number').columns if c not in ('windows', 'cells', 'positive_ratio_windows')]
    keys = ['model', 'target', 'recipe', 'subgroup']
    grouped = frame.loc[frame.status.eq('PASS')].groupby(keys, sort=True)
    summary = grouped[columns].mean().reset_index()
    summary = summary.merge(grouped.agg(sources=('source', 'nunique'), windows=('windows', 'sum'), cells=('cells', 'sum')).reset_index(), on=keys, validate='one_to_one')
    summary.to_csv(OUT / 'source_balanced_summary.csv', index=False)
    grouped[columns].count().reset_index().to_csv(OUT / 'summary_metric_source_counts.csv', index=False)
    load()  # Recheck immutable protocols, primary outputs and source artifacts.
    write('completion.json', dict(status='COVERAGE_SCORED_REFERENCE_PARITY_PASS_INDEPENDENT_AUDIT_PENDING',
        new_fits=0, qhat_recipes=216, output_rows=len(frame), primary_outputs_unchanged=True,
        independent_arithmetic_audit_complete=False, manuscript_updated=False,
        outputs={n: bind(OUT / n) for n in ['interval_metrics_by_source.csv', 'support.csv', 'source_balanced_summary.csv', 'summary_metric_source_counts.csv']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze', 'calibrate', 'score'])
    parser.add_argument('--archive', type=Path)
    a = parser.parse_args()
    freeze(a.archive.resolve()) if a.stage == 'freeze' else calibrate() if a.stage == 'calibrate' else score()

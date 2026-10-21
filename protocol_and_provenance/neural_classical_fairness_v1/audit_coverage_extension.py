"""Recompute the sealed coverage release independently of E07 arithmetic helpers.

This is a scientific return audit, not a model fitting or selection program.
All qhats have already been sealed and all reporting outputs already computed.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'paper/audits/neural_classical_fairness_2026-10-08_v1'
OUT = BASE / 'coverage_extension_scoring_v1'
META = ['sample_id', 'canonical_cell_id', 'duplicate_family_id', 'source']
TARGETS = {'soh30': ('soh_t_plus_30', 'soh30_loss_mask'),
           'rul80': ('rul80_cycles', 'rul80_loss_mask'), 'rul90': ('rul90_cycles', 'rul90_loss_mask')}
KEYS = ['model', 'source', 'target', 'recipe', 'subgroup']


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def binding(p):
    p = Path(p)
    with p.open('rb') as f: sha = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha)
def frame(archive, name):
    with zipfile.ZipFile(archive) as z: return pd.read_parquet(io.BytesIO(z.read(name)))
def close(a, b): assert np.isclose(float(a), float(b), atol=1e-10, rtol=1e-10), (a, b)
def verify(b): assert binding(ROOT / b['path']) == b, b['path']


def main():
    assert not (OUT / 'independent_audit.json').exists()
    p = read(OUT / 'protocol.frozen.json')
    verify(read(OUT / 'protocol_seal.json')['protocol'])
    c = read(OUT / 'completion.json')
    assert c['status'] == 'COVERAGE_SCORED_REFERENCE_PARITY_PASS_INDEPENDENT_AUDIT_PENDING'
    for b in [*p['inputs'].values(), *c['outputs'].values()]: verify(b)
    acceptance = read(OUT / 'calibration_acceptance.json'); verify(acceptance['qhat'])
    assert acceptance['test_labels_read'] is False
    assert read(OUT / 'calibration_started.json')['at_utc'] < read(OUT / 'score_started.json')['at_utc']
    targets = ds.dataset(ROOT / p['inputs']['targets']['path'], format='parquet')
    membership = pd.read_parquet(ROOT / p['inputs']['membership']['path'])
    design = read(ROOT / p['inputs']['design']['path'])
    recipes = pd.read_parquet(OUT / 'calibration_recipes.parquet')
    actual = pd.read_csv(OUT / 'interval_metrics_by_source.csv')
    assert not actual.duplicated(KEYS).any() and not recipes.duplicated(KEYS[:4]).any()
    assert len(recipes) == 216
    checked_qhat = 0; expected_rows = []; expected_support = []; relative_support = []; reference_grids = []; precision_checks = []
    for source in design['held_sources']:
        split = membership.loc[membership.evaluation_mode.eq('LODO') & membership.outer_entity.eq(source)]
        cal_cells = set(split.loc[split.split.eq('calibration'), 'canonical_cell_id'])
        test_cells = set(split.loc[split.split.eq('test'), 'canonical_cell_id'])
        assert cal_cells.isdisjoint(test_cells)
        cal_families = set(split.loc[split.split.eq('calibration'), 'duplicate_family_id'])
        test_families = set(split.loc[split.split.eq('test'), 'duplicate_family_id'])
        assert cal_families.isdisjoint(test_families)
        def truth(cells):
            return targets.to_table(columns=META + ['t_cycle'] + sum((list(v) for v in TARGETS.values()), []),
                filter=ds.field('canonical_cell_id').isin(sorted(cells))).to_pandas()
        cy, ty = truth(cal_cells), truth(test_cells)
        assert cy.sample_id.is_unique and ty.sample_id.is_unique
        for model in p['models']:
            arch, seed = model.split('.')
            if model == 'TCN.2026':
                j = next(j for j in p['historical_reference']['jobs'] if j['source'] == source)
                oldarc = ROOT / p['historical_reference']['archives'][j['shard']]['path']
                cal_arc = test_arc = oldarc
                cm = {t: j['members'][t + '.calibration_predictions.parquet']['member'] for t in TARGETS}
                tm = {t: j['members'][t + '.outer_predictions.parquet']['member'] for t in TARGETS}
            else:
                j = next(j for j in p['jobs'] if j['source'] == source and j['architecture'] == arch and j['seed'] == int(seed))
                cal_arc = ROOT / p['inputs']['inference_return']['path']
                cm = {t: p['inference_prefix'] + 'jobs/' + j['id'] + '/' + t + '.calibration_predictions.parquet' for t in TARGETS}
                original = next(j for j in design['selected_existing_models'] if j['source'] == source and j['architecture'] == arch and j['seed'] == int(seed))
                test_arc = ROOT / original['archive']
                tm = {r['endpoint']: r['member'] for r in original['quantile_prediction_schemas']}
            for target, (label, mask) in TARGETS.items():
                cp, tp = frame(cal_arc, cm[target]), frame(test_arc, tm[target])
                for pred, y, cells, families in [(cp, cy, cal_cells, cal_families), (tp, ty, test_cells, test_families)]:
                    assert pred.sample_id.is_unique
                    if model == 'TCN.2026' and pred is cp:
                        # The historical executor inferred calibration on any-target-observed rows.
                        # Require this exact historical grid and exclude no scorable row.
                        any_observed = np.column_stack([y[m].astype(bool) & np.isfinite(y[c]) for c, m in TARGETS.values()]).any(axis=1)
                        assert set(pred.sample_id) == set(y.loc[any_observed, 'sample_id'])
                        reference_grids.append(dict(source=source, target=target, prediction_rows=len(pred),
                            metadata_rows=len(y), omitted_all_three_masks_false=int((~any_observed).sum()),
                            omitted_eligible_rows=0))
                    else:
                        assert set(pred.sample_id) == set(y.sample_id)
                    assert set(pred.canonical_cell_id) == cells and set(pred.duplicate_family_id) == families
                    pd.testing.assert_frame_equal(pred[META].sort_values('sample_id').reset_index(drop=True),
                        y.loc[y.sample_id.isin(pred.sample_id), META].sort_values('sample_id').reset_index(drop=True), check_dtype=False, check_exact=True)
                    assert y[mask].notna().all() and y[mask].isin([True, False, 0, 1]).all()
                    q = pred[['q10', 'q50', 'q90']].to_numpy(float)
                    assert np.isfinite(q).all() and (q >= 0).all() and (np.diff(q, axis=1) >= 0).all()
                eligible_c = cy.loc[cy[mask].astype(bool) & np.isfinite(cy[label]), ['sample_id', label]]
                assert eligible_c[label].ge(0).all()
                if model == 'TCN.2026':
                    # Historical _all_target_rows encoded calibration labels as float32.
                    # Verify the saved original labels exactly against this projection,
                    # rather than loosening the qhat arithmetic tolerance.
                    saved = frame(cal_arc, j['members'][target + '.calibration_observed.parquet']['member'])
                    expected_labels = eligible_c.sort_values('sample_id').reset_index(drop=True)
                    saved = saved.sort_values('sample_id').reset_index(drop=True)
                    assert expected_labels.sample_id.equals(saved.sample_id)
                    encoded = expected_labels[label].to_numpy(dtype='float32').astype(float)
                    assert np.array_equal(encoded, saved.y_true.to_numpy(float))
                    assert np.array_equal(encoded, saved[label].to_numpy(float))
                    precision_checks.append(dict(source=source, target=target, rows=len(saved),
                        original_labels_match_float32_projection=True,
                        max_abs_rounding=float(np.max(np.abs(encoded-expected_labels[label].to_numpy(float))))))
                    eligible_c = eligible_c.copy(); eligible_c[label] = eligible_c[label].astype('float32')
                cal = cp[META + ['q10', 'q90']].merge(eligible_c, on='sample_id', validate='one_to_one')
                q10, q90, y = (cal[k].to_numpy(float) for k in ['q10', 'q90', label])
                if target != 'soh30': q10, q90, y = np.log1p(q10), np.log1p(q90), np.log1p(y)
                scores = np.maximum.reduce([q10-y, y-q90, np.zeros(len(y))])
                family_scores = pd.Series(scores, index=cal.duplicate_family_id).groupby(level=0).max().to_numpy()
                eligible_t = ty.loc[ty[mask].astype(bool) & np.isfinite(ty[label]), ['sample_id', 't_cycle', label]]
                assert eligible_t[label].ge(0).all()
                f = tp[META + ['q10', 'q90']].merge(eligible_t, on='sample_id', validate='one_to_one').rename(columns={label: 'y'})
                if model == 'TCN.2026': expected_support.append(dict(source=source, target=target, windows=len(f), cells=f.canonical_cell_id.nunique(), total_cells=len(test_cells)))
                for arm, values in [('window', scores), ('family_max', family_scores)]:
                    n = len(values); rank = math.ceil((n+1)*.8)
                    assert n >= 4 and rank <= n
                    qh = float(np.partition(values, rank-1)[rank-1])
                    r = recipes.loc[(recipes.model == model) & (recipes.source == source) & (recipes.target == target) & (recipes.recipe == arm)].iloc[0]
                    close(qh, r.qhat); assert (n, rank, len(cal)) == (r.calibration_units, r.quantile_rank, r.calibration_rows)
                    checked_qhat += 1
                    identity = dict(model=model, source=source, target=target, recipe=arm)
                    if f.empty:
                        assert (source, target) == ('CALB', 'rul90')
                        expected_rows.append(dict(**identity, subgroup='all', status='NE', windows=0, cells=0)); continue
                    a, b = f.q10.to_numpy(float), f.q90.to_numpy(float)
                    if target != 'soh30':
                        low = np.expm1(np.log1p(a)-float(r.qhat)); high = np.expm1(np.log1p(b)+float(r.qhat))
                    else: low, high = a-float(r.qhat), b+float(r.qhat)
                    raw_width = high-low
                    low = np.clip(low, 0, None); high = np.maximum(high, low)
                    if target == 'soh30': high = np.minimum(high, 1.5)
                    assert np.isfinite(low).all() and np.isfinite(high).all() and (high >= low).all()
                    f = f.copy(); f['cover'] = (f.y >= low) & (f.y <= high)
                    f['width'] = high-low; f['raw_width'] = raw_width
                    groups = dict(all=np.ones(len(f), bool))
                    if target != 'soh30':
                        tau = f.t_cycle + f.y
                        groups.update(last20=(tau > 0) & (f.t_cycle >= .8*tau), zero=f.y.eq(0), near_0_10=f.y.gt(0) & f.y.le(10), near_10_50=f.y.gt(10) & f.y.le(50), over50=f.y.gt(50))
                        # Positive, zero and near-life bins are disjoint and exhaustive.
                        assert (groups['zero'].astype(int) + groups['near_0_10'].astype(int) + groups['near_10_50'].astype(int) + groups['over50'].astype(int) == 1).all()
                    for subgroup, select in groups.items():
                        g = f.loc[select].copy(); nc = g.canonical_cell_id.nunique()
                        row = dict(**identity, subgroup=subgroup, status='PASS' if len(g) else 'EMPTY_SUBGROUP', windows=len(g), cells=nc)
                        if target != 'soh30':
                            relative_support.append(dict(**identity, subgroup=subgroup, windows=len(g), cells=nc,
                                families=g.duplicate_family_id.nunique(), positive_windows=int(g.y.gt(0).sum()),
                                positive_cells=g.loc[g.y.gt(0), 'canonical_cell_id'].nunique(),
                                zero_windows=int(g.y.eq(0).sum()), zero_cells=g.loc[g.y.eq(0), 'canonical_cell_id'].nunique()))
                        if len(g):
                            bycell = g.groupby('canonical_cell_id'); byfamily = g.groupby('duplicate_family_id')
                            row.update(marginal_coverage=float(g.cover.mean()), macro_cell_coverage=float(bycell.cover.mean().mean()),
                                cell_simultaneous_coverage=float(bycell.cover.min().mean()), family_simultaneous_coverage=float(byfamily.cover.min().mean()),
                                mean_interval_width=float(g.width.mean()), median_interval_width=float(g.width.median()),
                                cell_macro_mean_width=float(bycell.width.mean().mean()), cell_macro_mean_raw_width=float(bycell.raw_width.mean().mean()))
                            if target != 'soh30':
                                g['stab'] = g.width / g.y.clip(lower=10)
                                row['cell_macro_stabilized_relative_width'] = float(g.groupby('canonical_cell_id').stab.mean().mean())
                                pos = g.loc[g.y > 0].copy(); pos['ratio'] = pos.width/pos.y
                                row['positive_ratio_windows'] = len(pos)
                                if len(pos):
                                    row['cell_macro_positive_ratio_median'] = float(pos.groupby('canonical_cell_id').ratio.quantile(.5).mean())
                                    row['cell_macro_positive_ratio_p90'] = float(pos.groupby('canonical_cell_id').ratio.quantile(.9).mean())
                        expected_rows.append(row)
        print('[INDEPENDENT COVERAGE AUDIT]', source, flush=True)
    expected = pd.DataFrame(expected_rows)
    assert set(actual.columns) == set(expected.columns)
    a = actual.sort_values(KEYS).reset_index(drop=True); b = expected[actual.columns].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=1e-10, rtol=1e-10)
    support = pd.read_csv(OUT / 'support.csv')
    pd.testing.assert_frame_equal(support.sort_values(['source', 'target']).reset_index(drop=True),
        pd.DataFrame(expected_support).sort_values(['source', 'target']).reset_index(drop=True), check_dtype=False)
    keys = ['model', 'target', 'recipe', 'subgroup']
    metrics = [c for c in expected.select_dtypes(include='number').columns if c not in ['windows', 'cells', 'positive_ratio_windows']]
    grouped = expected.loc[expected.status.eq('PASS')].groupby(keys, sort=True)
    means = grouped[metrics].mean().reset_index().merge(grouped.agg(sources=('source', 'nunique'), windows=('windows', 'sum'), cells=('cells', 'sum')).reset_index(), on=keys, validate='one_to_one')
    actual_summary = pd.read_csv(OUT / 'source_balanced_summary.csv')
    pd.testing.assert_frame_equal(actual_summary, means[actual_summary.columns], check_dtype=False, atol=1e-10, rtol=1e-10)
    counts = pd.read_csv(OUT / 'summary_metric_source_counts.csv')
    pd.testing.assert_frame_equal(counts, grouped[metrics].count().reset_index()[counts.columns], check_dtype=False)
    for b in p['inputs'].values(): verify(b)
    relative_path = OUT / 'relative_width_support.csv'
    assert not relative_path.exists()
    pd.DataFrame(relative_support).to_csv(relative_path, index=False)
    reference_path = OUT / 'historical_calibration_grid_audit.csv'
    assert not reference_path.exists()
    pd.DataFrame(reference_grids).to_csv(reference_path, index=False)
    precision_path = OUT / 'historical_calibration_precision_audit.csv'
    assert not precision_path.exists()
    pd.DataFrame(precision_checks).to_csv(precision_path, index=False)
    receipt = dict(status='PASS_INDEPENDENT_ARITHMETIC_ALL_QHATS_AND_REPORTED_METRICS', at_utc=datetime.now(timezone.utc).isoformat(),
        qhat_recipes=checked_qhat, scored_rows=len(expected), evaluable_rows=int(expected.status.eq('PASS').sum()),
        not_evaluable_rows=int(expected.status.eq('NE').sum()), empty_subgroup_rows=int(expected.status.eq('EMPTY_SUBGROUP').sum()),
        historical_reference_all_metrics_recomputed=True, metadata_and_role_disjointness_checked=True,
        source_balanced_summaries_and_metric_counts_checked=True, protected_primary_outputs_unchanged=True,
        new_fits=0, manuscript_updated=False, audit_source=binding(Path(__file__)),
        protocol=binding(OUT / 'protocol.frozen.json'), outputs=c['outputs'], calibration_qhats=acceptance['qhat'],
        relative_width_support=binding(relative_path), historical_calibration_grid_audit=binding(reference_path),
        historical_calibration_precision_audit=binding(precision_path),
        label_precision='TCN2026 reference: original float32 calibration labels verified against raw targets; new GRU2026/TCN2027: native target precision; all test labels: native target precision')
    with (OUT / 'independent_audit.json').open('x', encoding='utf-8') as f: json.dump(receipt, f, indent=2, allow_nan=False)
    print(receipt['status'], flush=True)


if __name__ == '__main__': main()

"""Isolated retrospective E07: frozen TCN, two calibration units, no training."""
import argparse
import hashlib
import io
import json
import math
import platform
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/audits/e07_calibration_unit_ablation_2026-10-05_v1'
PREV = ROOT / 'paper/audits/corrected_outer_scoring_2026-10-05_v1'
BASE = ROOT / 'paper/audits/corrected_outer_results_review_2026-10-04_v1/returned_outer_acceptance.json'
META = ['sample_id', 'canonical_cell_id', 'duplicate_family_id', 'source']
SPECS = {'soh30': ('soh_t_plus_30', 'soh30_loss_mask'),
         'rul80': ('rul80_cycles', 'rul80_loss_mask'), 'rul90': ('rul90_cycles', 'rul90_loss_mask')}
ARMS = ('family_max', 'window')
METRICS = ['marginal_coverage', 'macro_cell_coverage', 'cell_simultaneous_coverage',
           'family_simultaneous_coverage', 'mean_interval_width', 'median_interval_width']


def require(ok, name):
    if not ok:
        raise RuntimeError(name)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    path = Path(path)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=file_sha(path), bytes=path.stat().st_size)


def verify(r):
    p = ROOT / r['path']
    require(p.stat().st_size == r['bytes'] and file_sha(p) == r['sha256'], 'INPUT_BINDING_DRIFT:' + r['path'])


def write_new(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def id_hash(values):
    return digest(json.dumps(sorted(set(map(str, values))), sort_keys=True,
                             separators=(',', ':'), ensure_ascii=False).encode())


def validate_frame(f, target, truth=False):
    require(len(f) > 0 and f.sample_id.is_unique, 'EMPTY_OR_DUPLICATE_KEYS')
    require(not f[META].isna().any().any(), 'MISSING_META')
    require(all(f[c].astype(str).str.strip().ne('').all() for c in META), 'BLANK_META')
    require(f.groupby('canonical_cell_id').duplicate_family_id.nunique().eq(1).all(), 'CELL_FAMILY_DRIFT')
    vals = f[['q10', 'q50', 'q90']].to_numpy(float)
    require(np.isfinite(vals).all() and (vals >= 0).all(), 'INVALID_QUANTILES')
    require((vals[:, 0] <= vals[:, 1]).all() and (vals[:, 1] <= vals[:, 2]).all(), 'CROSSED_QUANTILES')
    if truth:
        y, mask = SPECS[target]
        require(f[mask].isin([True, 1]).all(), 'UNOBSERVED_CALIBRATION_ROW')
        require(np.isfinite(f.y_true).all() and f.y_true.ge(0).all(), 'INVALID_TRUTH')
        require(np.array_equal(f.y_true.to_numpy(), f[y].to_numpy()), 'TRUTH_COLUMN_DRIFT')


def scalar_scores(f, target):
    transform = math.log1p if target.startswith('rul') else float
    return np.fromiter((max(transform(lo) - transform(y), transform(y) - transform(hi), 0.0)
                        for lo, hi, y in zip(f.q10, f.q90, f.y_true)), dtype=float, count=len(f))


def finite_quantile(values):
    a = np.sort(np.asarray(values, float))
    n = len(a)
    require(n >= 4 and np.isfinite(a).all() and (a >= 0).all(), 'INVALID_SCORE_UNITS')
    rank = math.ceil((n + 1) * .8)
    require(rank <= n, 'FINITE_SAMPLE_RANK_EXCEEDS_UNITS')
    return float(a[rank - 1]), rank


def calibrate_frame(f, target):
    validate_frame(f, target, truth=True)
    scores = scalar_scores(f, target)
    families = pd.Series(scores).groupby(f.duplicate_family_id.to_numpy()).max().to_numpy()
    return {arm: dict(qhat=q, calibration_units=len(v), quantile_rank=k)
            for arm, v in [('family_max', families), ('window', scores)]
            for q, k in [finite_quantile(v)]}


def bounds(f, target, qhat):
    require(math.isfinite(qhat) and qhat >= 0, 'INVALID_QHAT')
    lo, hi = f.q10.to_numpy(float), f.q90.to_numpy(float)
    if target.startswith('rul'):
        lo, hi = np.log1p(lo), np.log1p(hi)
        require((hi + qhat <= math.log(np.finfo(float).max)).all(), 'INTERVAL_OVERFLOW')
        lo, hi = np.expm1(lo - qhat), np.expm1(hi + qhat)
    else:
        lo, hi = lo - qhat, hi + qhat
    lo = np.maximum(lo, 0)
    hi = np.maximum(hi, lo)
    if target == 'soh30':
        hi = np.minimum(hi, 1.5)
    require(np.isfinite(lo).all() and np.isfinite(hi).all() and (hi >= lo).all(), 'INVALID_BOUNDS')
    return lo, hi


def interval_metrics(f, lo, hi):
    cover = (lo <= f.y_true.to_numpy(float)) & (f.y_true.to_numpy(float) <= hi)
    w = f[META].copy()
    w['cover'] = cover
    return dict(marginal_coverage=float(cover.mean()),
                macro_cell_coverage=float(w.groupby('canonical_cell_id').cover.mean().mean()),
                cell_simultaneous_coverage=float(w.groupby('canonical_cell_id').cover.all().mean()),
                family_simultaneous_coverage=float(w.groupby('duplicate_family_id').cover.all().mean()),
                mean_interval_width=float(np.mean(hi - lo)), median_interval_width=float(np.median(hi - lo)))


def verify_protocol():
    require(not (OUT/'failure.json').exists(), 'FAILED_NAMESPACE_CANNOT_RESUME')
    lock = read_json(OUT / 'protocol_lock.json')
    verify(lock['protocol'])
    p = read_json(OUT / 'protocol.frozen.json')
    for r in [*p['inputs'].values(), *p['archives'].values()]:
        verify(r)
    require(p['arms'] == list(ARMS) and p['alpha'] == .2, 'PROTOCOL_CHANGED')
    return p


def member(p, job, name):
    r = job['members'][name]
    with zipfile.ZipFile(ROOT / p['archives'][job['shard']]['path']) as z:
        data = z.read(r['member'])
    require(len(data) == r['bytes'] and digest(data) == r['sha256'], 'MEMBER_DRIFT:' + r['member'])
    return data


def frame(p, job, name):
    return pd.read_parquet(io.BytesIO(member(p, job, name)))


def freeze():
    require(not OUT.exists(), 'REFUSE_OVERWRITE_E07')
    old = read_json(PREV / 'scoring_protocol.frozen.json')
    for r in old['inputs'].values():
        verify(r)
    verify(read_json(PREV / 'protocol_lock.json')['protocol'])
    base = read_json(BASE)
    require(base['accepted_outer_fixed_fits'] == 12 and base['total_campaign_accepted_fits'] == 48,
            'BASELINE_ACCEPTANCE_REQUIRED')
    inputs = {k: bind(v) for k, v in dict(baseline=BASE, previous_protocol=PREV/'scoring_protocol.frozen.json',
        previous_acceptance=PREV/'scoring_acceptance.json', previous_qa=PREV/'scoring_qa_acceptance.json',
        support=PREV/'endpoint_support.parquet', intervals=PREV/'interval_metrics_by_source.parquet',
        implementation=Path(__file__), tests=Path(__file__).with_name('test_e07_calibration_ablation.py'),
        independent_qa=Path(__file__).with_name('qa_e07_calibration_ablation.py'),
        design=ROOT/'paper/planning/49_e06_standard_neural_comparator_design.md',
        plan=ROOT/'paper/planning/50_e07_controlled_calibration_ablation_protocol.md').items()}
    inputs['targets'] = old['inputs']['targets']
    archives = base['input_archives']
    for r in archives.values():
        verify(r)
    jobs = []
    for b in base['jobs']:
        prefix = f"batteryscope_corrected_outer_v2_{b['shard']}/outer.{b['source']}/"
        with zipfile.ZipFile(ROOT / archives[b['shard']]['path']) as z:
            data = z.read(prefix + 'job_result.json')
            jr = json.loads(data)
            names = ['job_result.json', 'qhat.json', 'roles_and_scalers.json']
            names += [t + '.' + suffix + '.parquet' for t in SPECS
                      for suffix in ['calibration_predictions', 'calibration_observed', 'outer_predictions']]
            members = {}
            for name in names:
                d = data if name == 'job_result.json' else z.read(prefix + name)
                r = dict(member=prefix + name, sha256=digest(d), bytes=len(d))
                if name != 'job_result.json':
                    require(r['sha256'] == jr['artifacts'][name]['sha256'] and
                            r['bytes'] == jr['artifacts'][name]['bytes'], 'ARTIFACT_INVENTORY_DRIFT')
                members[name] = r
        jobs.append(dict(source=b['source'], shard=b['shard'], roles=b['roles'], members=members))
    require(len(jobs) == 12 and len({j['source'] for j in jobs}) == 12, 'CONTEXT_COUNT')
    p = dict(schema='e07-calibration-unit-v1', date='2026-10-05',
        author_request='thiết kế E06 và làm E07 có kiểm soát',
        scope='RETROSPECTIVE_POST_OUTER_DIAGNOSTIC_NOT_CONFIRMATORY', arms=list(ARMS), alpha=.2,
        predictor='corrected fixed TCN base2026 only', new_predictor_fits=0, inputs=inputs,
        archives=archives, jobs=jobs, target_rules=SPECS,
        selection='no outer winner selection, no official registry/interval mutation',
        score='max(q10_z-y_z,y_z-q90_z,0); identity SoH/log1p RUL',
        primary=['family_simultaneous_coverage', 'mean_interval_width'],
        confidence=dict(replicates=10000, seed=20260914, level=.95, unit='paired held-out source',
                        scope='conditional descriptive, dependent sources; no p-values'),
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                         pyarrow=pyarrow.__version__), outer_labels_accessed_in_this_stage=False)
    OUT.mkdir(parents=True)
    write_new('protocol.frozen.json', p)
    write_new('protocol_lock.json', dict(protocol=bind(OUT/'protocol.frozen.json'), outer_labels_accessed=False))
    print('E07_PROTOCOL_FROZEN; no calibration/outer values decoded in freeze')


def calibrate():
    p = verify_protocol()
    require(not (OUT/'calibration_started.json').exists(), 'CALIBRATION_NOT_FRESH')
    write_new('calibration_started.json', dict(protocol=bind(OUT/'protocol.frozen.json'), outer_labels_accessed=False))
    rows = []
    for j in p['jobs']:
        source = j['source']
        oldq = json.loads(member(p, j, 'qhat.json'))
        side = json.loads(member(p, j, 'roles_and_scalers.json'))
        require(side['roles'] == j['roles'], 'ROLE_SIDECAR_DRIFT')
        grid = None
        for t in SPECS:
            cal = frame(p, j, t+'.calibration_predictions.parquet')
            obs = frame(p, j, t+'.calibration_observed.parquet')
            out = frame(p, j, t+'.outer_predictions.parquet')
            validate_frame(cal, t)
            validate_frame(obs, t, truth=True)
            validate_frame(out, t)
            require(not set([*sum((list(v) for v in SPECS.values()), []), 'y_true']) & set(out.columns), 'OUTER_HAS_TRUTH')
            require(source not in set(cal.source) and set(out.source) == {source}, 'SOURCE_ROLE_LEAK')
            for f, role in [(cal, 'calibration'), (out, 'test')]:
                require(id_hash(f.canonical_cell_id) == j['roles'][role]['cell_hash'] and
                        id_hash(f.duplicate_family_id) == j['roles'][role]['family_hash'], 'ROLE_HASH_MISMATCH')
            require(set(cal.canonical_cell_id).isdisjoint(out.canonical_cell_id) and
                    set(cal.duplicate_family_id).isdisjoint(out.duplicate_family_id), 'CALIBRATION_TEST_LEAK')
            want = cal.loc[cal.sample_id.isin(obs.sample_id), META+['q10','q50','q90']].sort_values('sample_id').reset_index(drop=True)
            pd.testing.assert_frame_equal(want, obs[META+['q10','q50','q90']].sort_values('sample_id').reset_index(drop=True), check_dtype=False, check_exact=True)
            current = out[META].sort_values('sample_id').reset_index(drop=True)
            if grid is not None:
                pd.testing.assert_frame_equal(current, grid, check_dtype=False, check_exact=True)
            grid = current
            require(out.target.eq(t).all() and out.model_id.eq('quantile_tcn_locked').all() and
                    np.array_equal(out.y_pred, out.q50), 'PREDICTOR_ID_DRIFT')
            recipes = calibrate_frame(obs, t)
            q = recipes['family_max']['qhat']
            require(np.isclose(q, oldq[t]['qhat'], atol=1e-10, rtol=1e-10) and
                    recipes['family_max']['calibration_units'] == oldq[t]['families'] and
                    len(obs) == oldq[t]['observed_calibration_rows'], 'BASELINE_QHAT_PARITY_FAILED')
            lo, hi = bounds(out, t, q)
            require(np.allclose(out.lower, lo, atol=1e-10, rtol=1e-10) and
                    np.allclose(out.upper, hi, atol=1e-10, rtol=1e-10), 'BASELINE_BOUNDS_PARITY_FAILED')
            for arm in ARMS:
                rows.append(dict(source=source, target=t, recipe=arm, **recipes[arm],
                    calibration_rows=len(obs), calibration_cells=obs.canonical_cell_id.nunique(),
                    calibration_families=obs.duplicate_family_id.nunique(), baseline_parity='PASS',
                    calibration_observed_sha256=j['members'][t+'.calibration_observed.parquet']['sha256']))
        print('CALIBRATION/PARITY PASS', source, flush=True)
    require(len(rows) == 72, 'RECIPE_COUNT')
    pd.DataFrame(rows).to_parquet(OUT/'calibration_recipes.parquet', index=False)
    write_new('calibration_acceptance.json', dict(status='ALL36_BASELINE_PARITIES_PASS_BEFORE_E07_OUTER_READ',
        contexts=36, recipes=72, outer_labels_accessed=False, protocol=bind(OUT/'protocol.frozen.json'),
        qhat_table=bind(OUT/'calibration_recipes.parquet')))


def score():
    p = verify_protocol()
    c = read_json(OUT/'calibration_acceptance.json')
    require(c['status'] == 'ALL36_BASELINE_PARITIES_PASS_BEFORE_E07_OUTER_READ', 'CALIBRATION_GATE')
    verify(c['qhat_table'])
    verify(c['protocol'])
    require(not (OUT/'score_started.json').exists(), 'SCORE_NOT_FRESH')
    write_new('score_started.json', dict(calibration_acceptance=bind(OUT/'calibration_acceptance.json'),
                                       retrospective=True, no_predictor_training=True))
    recipes = pd.read_parquet(OUT/'calibration_recipes.parquet')
    support = pd.read_parquet(ROOT/p['inputs']['support']['path'])
    previous = pd.read_parquet(ROOT/p['inputs']['intervals']['path'])
    previous = previous.loc[previous.track.eq('CORRECTED_TCN_BASELINE')]
    rows = []
    for j in p['jobs']:
        source = j['source']
        heads = {t: frame(p, j, t+'.outer_predictions.parquet') for t in SPECS}
        grid = heads['soh30'][META].sort_values('sample_id').reset_index(drop=True)
        write_new('outer_read_intent.'+source+'.json', dict(source=source, protocol_sha256=file_sha(OUT/'protocol.frozen.json'),
            role='outer_test_labels', additional_retrospective_read=True))
        columns = META + sum((list(v) for v in SPECS.values()), [])
        labels = ds.dataset(str(ROOT/p['inputs']['targets']['path']), format='parquet').to_table(
            columns=columns, filter=ds.field('canonical_cell_id').isin(sorted(grid.canonical_cell_id.unique()))).to_pandas()
        require(labels.sample_id.is_unique and set(labels.source) == {source}, 'INVALID_LABEL_GRID')
        pd.testing.assert_frame_equal(labels[META].sort_values('sample_id').reset_index(drop=True), grid, check_dtype=False, check_exact=True)
        labels.to_parquet(OUT/('private_outer_labels.'+source+'.parquet'), index=False)
        write_new('outer_read_complete.'+source+'.json', dict(source=source, rows=len(labels), columns=columns))
        for t, (y, mask) in SPECS.items():
            require(labels[mask].isin([True, False, 0, 1]).all(), 'INVALID_OUTER_MASK')
            eligible = labels.loc[labels[mask].astype(bool) & np.isfinite(labels[y]), META+[y]].rename(columns={y:'y_true'})
            require(eligible.y_true.ge(0).all(), 'INVALID_OUTER_TRUTH')
            counts = dict(n_samples=len(eligible), n_cells=eligible.canonical_cell_id.nunique(), n_families=eligible.duplicate_family_id.nunique())
            ref = support.loc[support.source.eq(source) & support.target.eq(t)].iloc[0]
            require(tuple(counts.values()) == (ref.evaluable_samples, ref.evaluable_cells, ref.evaluable_families), 'SUPPORT_PARITY_FAILED')
            if eligible.empty:
                require((source, t) == ('CALB','rul90'), 'UNEXPECTED_NOT_EVALUABLE')
            scored = heads[t].merge(eligible[['sample_id','y_true']], on='sample_id', how='inner', validate='one_to_one')
            for arm in ARMS:
                r = recipes.loc[recipes.source.eq(source) & recipes.target.eq(t) & recipes.recipe.eq(arm)].iloc[0]
                identity = dict(source=source, target=t, recipe=arm, seed_base=2026, model_id='quantile_tcn_locked',
                                status='PASS' if len(scored) else 'NOT_EVALUABLE', **counts, qhat=float(r.qhat), nominal_coverage=.8)
                if len(scored):
                    lo, hi = bounds(scored, t, float(r.qhat))
                    metrics = interval_metrics(scored, lo, hi)
                    if arm == 'family_max':
                        old = previous.loc[previous.source.eq(source) & previous.target.eq(t)].iloc[0]
                        require(all(np.isclose(metrics[m], old[m], atol=1e-10, rtol=1e-10) for m in METRICS), 'BASELINE_METRIC_PARITY_FAILED')
                    rows.append(dict(**identity, **metrics))
                else:
                    rows.append(identity)
        print('SCORING/SUPPORT PASS', source, flush=True)
    f = pd.DataFrame(rows)
    require(len(f) == 72 and f.status.eq('PASS').sum() == 70, 'EXPECTED_SCORE_COUNT')
    f.to_parquet(OUT/'interval_metrics_by_source.parquet', index=False)
    summaries, pairs = [], []
    for target in SPECS:
        g = f.loc[f.target.eq(target) & f.status.eq('PASS')]
        a = g.loc[g.recipe.eq('family_max')].sort_values('source').reset_index(drop=True)
        b = g.loc[g.recipe.eq('window')].sort_values('source').reset_index(drop=True)
        require(a.source.equals(b.source), 'PAIRED_SOURCE_DRIFT')
        indices = np.random.default_rng(20260914).integers(0, len(a), size=(10000, len(a)))
        for metric in METRICS:
            for arm, table in [('family_max', a), ('window', b)]:
                v = table[metric].to_numpy(float)
                ci = np.quantile(v[indices].mean(axis=1), [.025,.975])
                summaries.append(dict(target=target, recipe=arm, metric=metric, n_sources=len(a),
                                      mean=float(v.mean()), ci_low=float(ci[0]), ci_high=float(ci[1])))
            delta = b[metric].to_numpy(float) - a[metric].to_numpy(float)
            ci = np.quantile(delta[indices].mean(axis=1), [.025,.975])
            pairs.append(dict(target=target, metric=metric, n_sources=len(a), delta_window_minus_family=float(delta.mean()),
                              ci_low=float(ci[0]), ci_high=float(ci[1])))
    pd.DataFrame(summaries).to_parquet(OUT/'source_balanced_summary.parquet', index=False)
    pd.DataFrame(pairs).to_parquet(OUT/'paired_recipe_effects.parquet', index=False)
    verify_protocol()  # Inputs, plans, code, previous artifacts must remain immutable.
    outputs = {n:bind(OUT/n) for n in ['calibration_recipes.parquet', 'calibration_acceptance.json',
        'interval_metrics_by_source.parquet', 'source_balanced_summary.parquet', 'paired_recipe_effects.parquet']}
    outputs.update({('private_outer_labels.'+j['source']+'.parquet'):bind(OUT/('private_outer_labels.'+j['source']+'.parquet')) for j in p['jobs']})
    write_new('execution_acceptance.json', dict(status='E07_SCORED_PENDING_INDEPENDENT_QA', outputs=outputs,
        calibration_contexts=36, recipe_contexts=72, evaluable_recipe_contexts=70,
        additional_outer_label_reads=12, new_predictor_fits=0, original_inputs_unchanged=True,
        recipe_selected_by_outer=False, official_intervals_changed=False, protocol=bind(OUT/'protocol.frozen.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'calibrate', 'score'])
    action = parser.parse_args().action
    try:
        globals()[action]()
    except Exception as exc:
        if OUT.exists() and not (OUT/'failure.json').exists():
            write_new('failure.json', dict(action=action, error=repr(exc), stop_no_automatic_resume=True))
        raise

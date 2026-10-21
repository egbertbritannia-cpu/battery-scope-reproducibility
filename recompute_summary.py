"""Recompute accepted source/seed aggregate means and standard deviations."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'results/primary'
METRICS = ('primary', 'mae', 'rmse', 'bias', 'lower_clip', 'upper_clip')


def rows(name):
    with (BASE / name).open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))


def close(a, b):
    if a is None or b is None:
        assert a is None and b is None, (a, b)
        return
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12), (a, b)


def mean(values):
    values = [v for v in values if v is not None]
    return math.fsum(values) / len(values) if values else None


def number(value): return float(value) if value else None


grouped = defaultdict(list)
for row in rows('source_seed_metrics.csv'):
    if row['status'] == 'EVALUABLE': grouped[row['system'], row['endpoint'], row['seed']].append(row)
seed_results = {}
for key, source_rows in grouped.items():
    seed_results[key] = {metric: mean([float(r[metric]) for r in source_rows if r[metric]]) for metric in METRICS}
    seed_results[key]['sources'] = len(source_rows)
stored_seeds = rows('seed_macro_metrics.csv')
assert len(stored_seeds) == len(seed_results)
for row in stored_seeds:
    result = seed_results[row['system'], row['endpoint'], row['seed']]
    assert result['sources'] == int(row['sources'])
    for metric in METRICS: close(result[metric], number(row[metric]))

by_system = defaultdict(list)
for (system, endpoint, seed), result in seed_results.items(): by_system[system, endpoint].append(result)
stored_aggregates = rows('aggregate_metrics.csv'); aggregates = {}
assert len(stored_aggregates) == len(by_system) == 24
sd_checks = 0
for row in stored_aggregates:
    key = row['system'], row['endpoint']; per_seed = by_system[key]
    result = {metric: mean([r[metric] for r in per_seed]) for metric in METRICS}
    assert int(row['seeds']) == len(per_seed)
    for metric in METRICS: close(result[metric], number(row[metric]))
    if len(per_seed) > 1:
        result['primary_seed_sd'] = statistics.stdev(r['primary'] for r in per_seed)
        close(result['primary_seed_sd'], float(row['primary_seed_sd'])); sd_checks += 1
    else:
        assert not row['primary_seed_sd']; result['primary_seed_sd'] = None
    aggregates[key] = result


print(json.dumps(dict(status='SOURCE_SEED_AGGREGATES_PASS', aggregate_contexts=len(aggregates),
    seed_contexts=len(seed_results), sample_sd_checks=sd_checks,
    manuscript_read=False, raw_prediction_scoring=False, new_fits=0)))

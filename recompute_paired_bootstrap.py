"""Recompute descriptive paired source bootstrap bands (requires NumPy)."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
with (ROOT / 'results/primary/source_summary.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))

results = []
for target in ('soh30', 'rul90'):
    scores = {
        system: {r['source']: float(r['primary']) for r in rows
                 if r['endpoint'] == target and r['system'] == system}
        for system in ('Extra Trees', 'Trend')
    }
    if scores['Extra Trees'].keys() != scores['Trend'].keys():
        raise ValueError(f'Mismatched source support for {target}')
    sources = sorted(scores['Extra Trees'])
    if len(sources) != {'soh30': 12, 'rul90': 11}[target]:
        raise ValueError(f'Unexpected source count for {target}')
    delta = np.array([scores['Extra Trees'][s] - scores['Trend'][s] for s in sources])
    rng = np.random.default_rng(20260914)
    resampled = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    lower, upper = np.quantile(resampled, [0.025, 0.975])
    results.append({'target': target, 'mean_delta': float(delta.mean()),
                    'lower_95': float(lower), 'upper_95': float(upper),
                    'extra_trees_wins': int((delta < 0).sum()), 'sources': len(sources)})
print(json.dumps(results, indent=2))

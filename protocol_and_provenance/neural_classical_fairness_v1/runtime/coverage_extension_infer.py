"""Frozen-checkpoint inference only; calibration metadata allowed, no labels/fits."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1.runtime import local_cpu_pilots as local
from protocol_and_provenance.neural_classical_fairness_v1.runtime import f2_kaggle_pilot as pilot
from protocol_and_provenance.neural_classical_fairness_v1.runtime import kaggle_classical_remaining as inherited
from protocol_and_provenance.data_integrity.e06_gru_model import make_model

MANIFEST = 'coverage_extension_infer_delivery.json'
RUN_NAME = 'batteryscope_coverage_extension_infer_v2'
META = ['sample_id', 'canonical_cell_id', 't_cycle', 'duplicate_family_id', 'source']


def sha(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def ids(values): return local.digest(sorted(set(map(str, values))))


def manifest():
    m = read(ROOT / MANIFEST)
    local.require(m['schema'] == 'coverage-extension-inference-v1' and len(m['jobs']) == 24 and
        m['new_fits'] == 0 and m['labels_read'] is False and m['autocast_float16'] is True and
        m['batch_size'] == 256, 'COVERAGE_INFERENCE_SCOPE')
    for n, expected in m['files'].items():
        p = (ROOT / n).resolve()
        local.require(p.is_relative_to(ROOT) and p.is_file() and
            dict(bytes=p.stat().st_size, sha256=local.sha(p)) == expected, 'COVERAGE_PAYLOAD_HASH:' + n)
    return m


class Inference:
    def __init__(self, bundle, m, run):
        import pandas as pd
        self.bundle = Path(bundle).resolve(); self.phase = self.bundle / 'phase2'
        self.m = m; self.run = run; self.stage = 'metadata'; self.log = []; self.profiles = {}
        # Membership/index/spec metadata are shared across all frozen contexts.
        self.membership = pd.read_parquet(self.phase / 'battery_phase2_fold_membership.parquet')
        self.index = pd.read_parquet(self.phase / 'multisensor_cycle_store_index.parquet').set_index('canonical_cell_id')
        spec = read(self.phase / 'feature_spec.json')
        self.channels = spec['value_channels']; self.masks = spec['value_mask_columns']
        module_spec = importlib.util.spec_from_file_location('coverage_frozen_tcn', ROOT / m['tcn_source'])
        self.base = importlib.util.module_from_spec(module_spec); sys.modules[module_spec.name] = self.base
        module_spec.loader.exec_module(self.base)
        self.executor = object.__new__(self.base.BatteryScopeFoldExecutor)
        self.executor.value_channels = self.channels; self.executor.value_mask_columns = self.masks

    def select_context(self, job):
        f = self.membership.loc[self.membership.evaluation_mode.eq('LODO') & self.membership.outer_entity.eq(job['source'])]
        self.roles = {}
        for role in ('development', 'calibration', 'test'):
            rows = f.loc[f.split.eq(role)]; cells = set(rows.canonical_cell_id.astype(str)); families = set(rows.duplicate_family_id.astype(str))
            bound = job['roles'][role]
            local.require(len(cells) == bound['cells'] and ids(cells).lower() == bound['cell_hash'].lower()
                and ids(families).lower() == bound['family_hash'].lower(), 'COVERAGE_ROLE_HASH:' + role)
            self.roles[role] = cells
        for a in self.roles:
            for b in self.roles:
                if a != b:
                    local.require(not self.roles[a] & self.roles[b], 'COVERAGE_ROLE_OVERLAP')
                    left = set(f.loc[f.split.eq(a), 'duplicate_family_id']); right = set(f.loc[f.split.eq(b), 'duplicate_family_id'])
                    local.require(not left & right, 'COVERAGE_FAMILY_OVERLAP')

    def rows(self, role):
        import pyarrow.dataset as ds
        local.require(role in ('test', 'calibration') and self.stage == role, 'COVERAGE_METADATA_STAGE')
        cells = sorted(self.roles[role])
        def table(name, columns):
            # This fixed allowlist excludes all targets and eligibility masks.
            local.require(set(columns) <= set(META), 'COVERAGE_TARGET_READ_FORBIDDEN')
            f = ds.dataset(self.phase / name, format='parquet').to_table(columns=columns,
                filter=ds.field('canonical_cell_id').isin(cells)).to_pandas()
            self.log.append(dict(table=name, columns=columns, role=role, cells=len(cells), rows=len(f)))
            return f
        w = table('window_index.parquet', META[:3]); a = table('sample_target_index.parquet', META)
        local.require(w.sample_id.is_unique and a.sample_id.is_unique, 'COVERAGE_METADATA_DUPLICATE')
        f = w.merge(a, on=META[:3], how='left', validate='one_to_one', indicator=True)
        local.require(len(f) == len(w) and f._merge.eq('both').all(), 'COVERAGE_METADATA_JOIN')
        return f.drop(columns='_merge').sort_values(['canonical_cell_id', 't_cycle'], kind='mergesort').reset_index(drop=True)

    def profile(self, cell):
        import numpy as np
        import pandas as pd
        local.require(self.stage in ('test', 'calibration') and cell in self.roles[self.stage], 'COVERAGE_PROFILE_ROLE')
        r = self.index.loc[cell]; p = (self.phase / str(r['shard_path'])).resolve()
        local.require(p.is_relative_to(self.phase.resolve()), 'COVERAGE_PROFILE_PATH')
        expected = str(r['sha256']).lower()
        if cell not in self.profiles:
            local.require(local.sha(p).lower() == expected, 'COVERAGE_PROFILE_HASH')
            self.profiles[cell] = expected
        f = pd.read_parquet(p, columns=list(dict.fromkeys(['canonical_cell_id', 'cycle_index', *self.channels, *self.masks])))
        local.require(f.canonical_cell_id.astype(str).eq(cell).all() and f.cycle_index.is_unique, 'COVERAGE_PROFILE_ID')
        local.require(np.array_equal(np.isfinite(f[self.channels].to_numpy(float)), f[self.masks].to_numpy(bool)), 'COVERAGE_VALUE_MASK')
        return f.sort_values('cycle_index', kind='mergesort').reset_index(drop=True)

    def infer(self, model, rows, state, guard):
        import numpy as np
        import pandas as pd
        import torch
        parts = {t: [] for t in local.TARGETS}; model.eval()
        with torch.inference_mode():
            for cell, group in rows.groupby('canonical_cell_id', sort=True):
                f = self.profile(str(cell)); group = group.reset_index(drop=True)
                for begin in range(0, len(group), 256):
                    guard.check(); batch = group.iloc[begin:begin + 256].reset_index(drop=True)
                    x = self.executor._make_windows(f, batch, state['cycle_scaler'])
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        outputs = model(torch.from_numpy(x).to('cuda:0'))
                    for t in local.TARGETS:
                        q = outputs[t].detach().float().cpu().numpy()
                        st = state['target_scaler']['targets'][t]
                        # Preserve historical float32 inverse arithmetic and AMP inference.
                        q = q * st['std'] + st['mean']
                        q = np.expm1(q) if t.startswith('rul') else q
                        q = np.clip(q, 0., 1.5 if t == 'soh30' else None)
                        local.require(np.isfinite(q).all() and np.all(q[:, 0] <= q[:, 1]) and np.all(q[:, 1] <= q[:, 2]), 'COVERAGE_QUANTILES')
                        part = batch[META].copy(); part['endpoint'] = t
                        for i, name in enumerate(('q10', 'q50', 'q90')): part[name] = q[:, i]
                        parts[t].append(part)
        return {t: pd.concat(v, ignore_index=True) for t, v in parts.items()}


def run(bundle):
    import numpy as np
    import torch
    run = ROOT / 'output' / RUN_NAME
    local.require(not run.exists(), 'COVERAGE_FRESH_NAMESPACE')
    run.mkdir(parents=True); error = None; ledger = []; started = time.monotonic()
    try:
        m = manifest(); pilot.configure(bundle)
        # Historical E06/TCN AMP inference retained cuDNN TF32=True while
        # matmul TF32=False. Do not inherit the later NFR1 float32 profile.
        torch.backends.cudnn.allow_tf32 = True
        actual, mismatches = pilot.environment(m)
        actual['tf32'] = dict(matmul=torch.backends.cuda.matmul.allow_tf32,
                              cudnn=torch.backends.cudnn.allow_tf32)
        local.write(run / 'environment_check.json', dict(actual=actual, mismatches=mismatches))
        local.require(not mismatches, 'COVERAGE_ENVIRONMENT')
        local.write(run / 'delivery_manifest.snapshot.json', m)
        local.write(run / 'input_mapping.json', inherited.input_checks(m, bundle))
        ctx = Inference(bundle, m, run)
        # Phase1: reproduce full saved test quantiles for every model, without labels.
        for job in m['jobs']:
            local.require(time.monotonic() - started < 10000, 'COVERAGE_SESSION_WALL')
            directory = run / 'jobs' / job['id']; directory.mkdir(parents=True)
            guard = pilot.Guard(run, directory, seconds=1800)
            ctx.select_context(job); ctx.stage = 'test'; rows = ctx.rows('test')
            cp = torch.load(ROOT / job['weights_file'], map_location='cpu', weights_only=True)
            model = ctx.executor._make_tcn() if job['architecture'] == 'TCN' else make_model()
            model.load_state_dict(cp, strict=True); model.to('cuda:0'); model.eval()
            outputs = ctx.infer(model, rows, job['state'], guard)
            for t, f in outputs.items():
                expected = job['test_parity'][t]
                values = f[['q10', 'q50', 'q90']].to_numpy(np.float32)
                local.require(len(f) == expected['rows'] and ids(f.sample_id) == expected['sample_hash']
                    and sha(values.tobytes()) == expected['quantile_float32_sha256'], 'COVERAGE_FULL_TEST_PARITY:' + job['id'] + ':' + t)
            local.write(directory / 'test_parity.json', dict(status='FULL_TEST_QUANTILES_EXACT_PASS', targets=3,
                rows_per_target=len(rows), labels_read=False))
            ledger.append(dict(job_id=job['id'], phase='TEST_PARITY_PASS'))
            local.write(run / 'ledger.json', ledger)
            del model, cp, outputs; torch.cuda.empty_cache()
            print('[COVERAGE TEST PARITY]', job['id'], len(rows), flush=True)
        local.write(run / 'all_model_test_parity.json', dict(status='ALL24_FULL_TEST_QUANTILES_EXACT_PASS',
            labels_read=False, calibration_inference_follows=True))
        # Phase2: calibration features only, after all models pass the historical bridge.
        for job in m['jobs']:
            local.require(time.monotonic() - started < 10000, 'COVERAGE_SESSION_WALL')
            directory = run / 'jobs' / job['id']; guard = pilot.Guard(run, directory, seconds=1800)
            ctx.select_context(job); ctx.stage = 'calibration'; rows = ctx.rows('calibration')
            cp = torch.load(ROOT / job['weights_file'], map_location='cpu', weights_only=True)
            model = ctx.executor._make_tcn() if job['architecture'] == 'TCN' else make_model()
            model.load_state_dict(cp, strict=True); model.to('cuda:0'); model.eval()
            outputs = ctx.infer(model, rows, job['state'], guard)
            for t, f in outputs.items(): f.to_parquet(directory / (t + '.calibration_predictions.parquet'), index=False)
            local.write(directory / 'calibration_inference.json', dict(status='CALIBRATION_FEATURE_INFERENCE_COMPLETE',
                windows_per_target=len(rows), labels_read=False, checkpoint_retrained=False))
            next(r for r in ledger if r['job_id'] == job['id'])['phase'] = 'CALIBRATION_INFERRED'
            local.write(run / 'ledger.json', ledger)
            del model, cp, outputs; torch.cuda.empty_cache()
            print('[COVERAGE CALIBRATION INFERENCE]', job['id'], len(rows), flush=True)
        local.write(run / 'read_log.json', ctx.log); local.write(run / 'profile_hashes.json', ctx.profiles)
        local.write(run / 'summary.json', dict(status='ALL24_CHECKPOINTS_REPLAYED_AND_CALIBRATION_INFERRED_AUDIT_PENDING',
            models=24, new_fits=0, labels_read=False, qhat_computed=False, wall_seconds=time.monotonic() - started))
    except BaseException as exc:
        error = exc
        if 'ctx' in locals():
            local.write(run / 'read_log.json', ctx.log); local.write(run / 'profile_hashes.json', ctx.profiles)
        local.write(run / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc(), no_retry=True, new_fits=0))
    finally:
        files = sorted(p for p in run.rglob('*') if p.is_file())
        local.write(run / 'output_inventory.json', dict(files={p.relative_to(run).as_posix():
            dict(bytes=p.stat().st_size, sha256=local.sha(p)) for p in files}))
        target = ROOT.parent / (RUN_NAME + '_results.zip')
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for p in files + [run / 'output_inventory.json']: z.write(p, RUN_NAME + '/' + p.relative_to(run).as_posix())
        print('DOWNLOAD RESULT OR FAILURE:', target, flush=True)
    if error: raise error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle', type=Path, required=True)
    run(parser.parse_args().bundle)

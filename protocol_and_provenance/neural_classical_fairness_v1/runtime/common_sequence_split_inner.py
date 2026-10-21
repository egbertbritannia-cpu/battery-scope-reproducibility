"""Operational CPU/GPU split, reusing the frozen common-input scientific functions.

Neural fits retain the original T4 profile. ET runs on CPU with exact scientific
Python/NumPy/pandas/PyArrow/sklearn versions, no CUDA calls or GPU reservation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from protocol_and_provenance.neural_classical_fairness_v1.runtime import common_sequence_scale_inner as base

RUN_NAME = 'batteryscope_common_sequence_split_v2'
original_environment = base.pilot.environment


def split_environment(manifest):
    actual, mismatch = original_environment(manifest)
    profile = manifest['execution_profile']
    if profile == 'cpu_et':
        mismatch.pop('gpu_count', None)
        if actual['gpu_count'] != 0:
            mismatch['gpu_count'] = dict(expected=0, actual=actual['gpu_count'])
        actual['device'] = 'cpu'
        actual['compute_role'] = 'CPU_EXTRA_TREES_ONLY_NO_NEURAL_TRAINING'
    else:
        assert profile == 'gpu_neural'
        actual['compute_role'] = 'GPU_TCN_GRU_ONLY_NO_TREE_TRAINING'
    return actual, mismatch


def configure_adapter():
    m = base.manifest()
    profile = m['execution_profile']
    active = [j for j in m['jobs'] if j['id'] in m['active_job_ids']]
    assert active and m['parent_science_sha256'] == m['science_sha256']
    if profile == 'cpu_et':
        assert all(j['architecture'] == 'ET' for j in active)
        assert set(m['expected_environment']) == {'python', 'numpy', 'pandas', 'pyarrow', 'scikit_learn', 'cuda_available', 'gpu_name'}
        assert m['expected_environment']['cuda_available'] is False and m['expected_environment']['gpu_name'] is None
        # Same RAM/disk/time/STOP guard as before, without querying CUDA memory.
        base.pilot.Guard = base.local.Guard
    else:
        assert profile == 'gpu_neural' and all(j['architecture'] in ('TCN', 'GRU') for j in active)
    base.pilot.environment = split_environment
    base.RUN_NAME = RUN_NAME
    # The campaign's child workers must enter this adapter too. All learning,
    # sampling, validation and checkpoint functions remain in the sealed base.
    base.__file__ = __file__


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['campaign', 'job', 'replay'])
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--run', type=Path)
    p.add_argument('--job-id')
    p.add_argument('--resume-zip', type=Path)
    p.add_argument('--resume-sha256')
    args = p.parse_args(); configure_adapter()
    if args.mode == 'campaign': base.campaign(args.bundle, args.resume_zip, args.resume_sha256)
    else: base.worker(args.mode, args.bundle, args.run, args.job_id)

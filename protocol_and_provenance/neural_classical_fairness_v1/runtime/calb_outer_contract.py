"""Point-only CALB outer delivery. Selection is immutable; scoring is separate."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from . import local_cpu_pilots as local
from . import selection_controls as selection

SELECTION_SHA = 'bc81cb935b52ec4b80b6b95acc20ead772959471bfdb1e72cb21aeb7b99bbfc0'
SHARDS = ('A', 'B', 'C', 'D')
MANIFEST = 'calb_outer_delivery.json'
BUDGET = dict(wall_seconds=28800, reserve_seconds=1800, job_seconds=5400,
    replay_seconds=1800, synthetic_seconds=900, workers=1, output_bytes=8*2**30)
GUARDS = dict(outer_scoring=False, calibration_labels_accessed=False, outer_labels_accessed=False,
    point_only=True, intervals=False, automatic_resume=False, automatic_retry=False,
    import_inner_models=False, historical_seed2027_root_cause='UNRESOLVED', fairness_closed=False)
METADATA = ('sample_id', 'canonical_cell_id', 't_cycle', 'duplicate_family_id', 'source', 'protocol_id')


def read_selection(path):
    data=Path(path).read_bytes()
    local.require(hashlib.sha256(data).hexdigest()==SELECTION_SHA, 'CALB_OUTER_SELECTION_HASH')
    value=json.loads(data);validate_selection(value)
    return value


def effective_seed(seed):
    local.require(type(seed) is int and seed in (2026,2027,2028), 'CALB_OUTER_SEED')
    return seed+int(hashlib.sha256(f'{seed}|LODO|CALB|outer_refit|multitask'.encode()).hexdigest()[:8],16)%1_000_000


def validate_selection(s):
    local.require(s['schema']=='nfr1-calb-outer-selection-v1' and s['source']=='CALB' and
        s['status']=='SELECTION_FROZEN_OUTER_RUNTIME_NOT_RELEASED' and s['outer_scoring'] is False and
        s['neural_recipes']==selection.candidate_recipes() and s['classical_grid']==local.grid(), 'CALB_OUTER_SELECTION_SCOPE')
    jobs=s['jobs']
    local.require(len(jobs)==len({j['id'] for j in jobs})==66 and
        s['counts']==dict(neural_fits=18,classical_fits=45,neural_epoch_passes=198,
            reference_jobs=3,reference_executions=8,total_jobs=66,learned_fits=63), 'CALB_OUTER_COUNTS')
    for j in jobs:
        local.require(j['source']=='CALB' and j['outer_scoring'] is False, 'CALB_OUTER_JOB_SCOPE')
        if j['kind']=='neural':
            local.require(j['effective_seed']==effective_seed(j['seed']) and
                j['fit_epochs']==max(r['epoch'] for r in j['checkpoint_exports']) and
                all(type(r['epoch']) is int and 1<=r['epoch']<=30 and r['endpoint'] in local.TARGETS and
                    r['interval_eligible'] is False for r in j['checkpoint_exports']), 'CALB_OUTER_NEURAL_RECIPE')
    roles=s['outer_role_metadata']['roles']
    local.require({r:roles[r]['cells'] for r in roles}==dict(development=627,calibration=127,test=27), 'CALB_OUTER_ROLES')
    for name,role in roles.items():
        local.require(len(role['cell_ids'])==len(set(role['cell_ids']))==role['cells'] and
            local.digest(sorted(role['cell_ids']))==role['cell_hash'], 'CALB_OUTER_ROLE_HASH')
        for other in roles:
            if other!=name:local.require(not set(role['cell_ids'])&set(roles[other]['cell_ids']), 'CALB_OUTER_ROLE_OVERLAP')


def assignments(s):
    """Deterministic engineering balance, NOT an empirical runtime guarantee."""
    queues={k:[] for k in SHARDS};cost={k:0 for k in SHARDS}
    def weight(j):
        if j['kind']=='neural':return 12+j['fit_epochs']*2+len(export_epochs(j))*3
        if j['kind']=='classical':return 16 if j['family']=='extra_trees' else 12 if j['family']=='hist_gradient_boosting' else 8
        return 2
    for j in sorted(s['jobs'],key=lambda j:(-weight(j),j['id'])):
        k=min(SHARDS,key=lambda k:(cost[k],SHARDS.index(k)))
        queues[k].append(j['id']);cost[k]+=weight(j)
    # Neural first; no cross-shard dependencies or best-seed filtering.
    lookup={j['id']:j for j in s['jobs']}
    for k in SHARDS:queues[k].sort(key=lambda n:(('neural','classical','references').index(lookup[n]['kind']),n))
    return queues


def export_epochs(job):
    return sorted({r['epoch'] for r in job['checkpoint_exports']})


def validate(m,s):
    validate_selection(s)
    local.require(m['schema']=='nfr1-calb-outer-point-v1' and m['selection_sha256']==SELECTION_SHA and
        m['jobs']==s['jobs'] and m['shards']==assignments(s) and m['budget']==BUDGET and m['guards']==GUARDS,
        'CALB_OUTER_DELIVERY_CONTRACT')
    local.require(m['expected_environment']==s['expected_environment'] and m['input_files']==s['input_files'] and
        m['source_bindings']==s['source_bindings'] and m['neural_recipes']==s['neural_recipes'] and
        m['classical_prediction_policies']==s['classical_prediction_policies'], 'CALB_OUTER_INPUT_ENV_RECIPE')


def check_read(table,columns,cells,roles,stage,feature_columns):
    """Deny by allowlist, before Arrow materializes any test target/mask."""
    columns=list(columns);cells=set(cells)
    local.require(columns and len(columns)==len(set(columns)) and cells, 'CALB_OUTER_EMPTY_DUPLICATE_READ')
    role='development' if cells<=set(roles['development']['cell_ids']) else 'test' if cells<=set(roles['test']['cell_ids']) else None
    local.require(role is not None, 'CALB_OUTER_CALIBRATION_MIXED_OR_UNKNOWN_READ')
    local.require((role=='development' and stage=='development_fit') or
        (role=='test' and stage=='prediction'), 'CALB_OUTER_STAGE_ROLE_READ')
    targets={x for t in local.TARGETS for x in (dict(soh30='soh_t_plus_30',rul90='rul90_cycles',rul80='rul80_cycles')[t],t+'_loss_mask')}
    allowed={'window_index.parquet':set(METADATA[:3]),
        'sample_target_index.parquet':set(METADATA)|(targets if role=='development' else set()),
        'tabular_feature_index.parquet':set(METADATA)|set(feature_columns)}
    local.require(table in allowed and set(columns)<=allowed[table], 'CALB_OUTER_FORBIDDEN_COLUMNS_OR_TABLE')
    local.require(not set(feature_columns)&targets and len(feature_columns)==len(set(feature_columns))==112,
        'CALB_OUTER_FEATURE_SCHEMA')
    return role

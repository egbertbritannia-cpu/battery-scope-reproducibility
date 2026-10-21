"""Locked multi-source inner refits and source-balanced candidate selection.

Consumes audited full-inner summaries only. Never reads targets or starts fits.
Historical CALB selectors and their results remain separate.
"""
from __future__ import annotations
import math
from statistics import median
from . import local_cpu_pilots as local
from . import selection_controls as selection
from . import full_inner_contract as original

SOURCES = ('CALB', 'CALCE', 'HNEI', 'MATR', 'MICH', 'SDU', 'SNL',
           'Stanford', 'Stanford_2', 'Tongji', 'UL_PUR', 'XJTU')
SOURCE_BLOCKS = (('CALB', 'MATR', 'SNL', 'Tongji'),
                 ('CALCE', 'MICH', 'Stanford', 'UL_PUR'),
                 ('HNEI', 'SDU', 'Stanford_2', 'XJTU'))
SHARDS = original.SHARDS
CLASSICAL_PRIORITY = original.CLASSICAL_PRIORITY
BUDGET = original.BUDGET.copy()
GUARDS = original.GUARDS.copy()
SELECTION = dict(policies=['S0', 'S1'], primary_policy='S1', patience=5,
    min_improvement=1e-8, candidate_tolerance=1e-8, folds=[0, 1, 2], seed=2026,
    neural_epoch_cap=30, checkpoint_score='cell_macro_then_source_macro_within_fold',
    neural_selector='equal_evaluable_validation_source_at_S1_checkpoint',
    classical_selector='equal_evaluable_validation_source',
    sensitivity='equal_fold_same_checkpoints_no_reselection',
    outer_epochs='median_selected_fold_epochs', validation_scope='full_eligible_inner',
    S2=False, automatic_selection_on_kaggle=False)


def jobs(source):
    local.require(source in SOURCES, 'MULTISOURCE_UNKNOWN_SOURCE')
    # Same grids, weights, epochs, seeds and job counts as the accepted refit.
    return [dict(j, source=source, id=j['id'].replace('.CALB.', '.'+source+'.'))
            for j in original.jobs()]


def assignments(source):
    queues = {s: [] for s in SHARDS}
    for kind in ('neural', 'classical', 'references'):
        for i, j in enumerate(j for j in jobs(source) if j['kind'] == kind):
            queues[SHARDS[i % len(SHARDS)]].append(j['id'])
    return queues


def validation_sources(source, fold, endpoint):
    local.require(source in SOURCES and fold in (0, 1, 2) and
                  endpoint in selection.TARGETS, 'MULTISOURCE_CONTEXT')
    # CALB has no observed RUL90 event. NE is excluded, never treated as zero.
    return tuple(s for s in SOURCE_BLOCKS[fold] if s != source and
                 not (s == 'CALB' and endpoint == 'rul90'))


def validate(m):
    source = m['source']
    local.require(m['schema'] == 'nfr1-multisource-full-inner-v1' and
                  m['jobs'] == jobs(source) and m['shards'] == assignments(source),
                  'MULTISOURCE_QUEUE')
    local.require(m['neural_recipes'] == selection.candidate_recipes() and
        m['classical_grid'] == local.grid() and m['classical_priority'] ==
        {k: list(v) for k, v in CLASSICAL_PRIORITY.items()} and
        m['prediction_policies'] == local.CLASSICAL_PREDICTION_POLICIES,
        'MULTISOURCE_RECIPES')
    local.require(m['budget'] == BUDGET and m['guards'] == GUARDS and
                  m['selection'] == SELECTION, 'MULTISOURCE_RULES')
    contexts = m['inner_context_metadata']
    local.require(len(contexts) == 3 and
        {(r['source'], r['inner_fold']) for r in contexts} ==
        {(source, f) for f in range(3)}, 'MULTISOURCE_ROLE_METADATA')
    for r in contexts:
        local.require(r['train_cells'] and r['validation_cells'] and
            not set(r['train_cells']) & set(r['validation_cells']) and
            source not in r['validation_sources'] and
            set(r['validation_sources']) == set(SOURCE_BLOCKS[r['inner_fold']])-{source},
            'MULTISOURCE_ROLE_SOURCE_SET')


def accept_trajectory(history, checkpoints):
    return original.accept_trajectory(history, checkpoints)


def choose(records, source, model, endpoint, *, neural):
    """Exact 3 candidates × 3 folds, per-source points from audited S1 epochs.

    Audit must certify roles, full support, hashes, metric reaggregation and
    numerical profile before marking a record accepted. This selector alone
    cannot certify those properties. No outer observations are allowed.
    """
    local.require(source in SOURCES and endpoint in selection.TARGETS,
                  'MULTISOURCE_SELECTOR_CONTEXT')
    priority = selection.PRIORITY if neural else CLASSICAL_PRIORITY[model]
    local.require(not neural or model in ('TCN', 'GRU'), 'MULTISOURCE_MODEL')
    rows = list(records)
    local.require(len(rows) == 9 and len({(r['candidate'], r['inner_fold']) for r in rows}) == 9 and
        {(r['candidate'], r['inner_fold']) for r in rows} ==
        {(c, f) for c in priority for f in range(3)}, 'MULTISOURCE_SELECTOR_GRID')
    metric = 'mae' if endpoint == 'soh30' else 'rmsle'
    means, fold_means = {}, {}
    for candidate in priority:
        points, folds = {}, []
        for r in sorted((r for r in rows if r['candidate'] == candidate), key=lambda r: r['inner_fold']):
            local.require(r['source'] == source and r['model'] == model and
                r['endpoint'] == endpoint and r['artifact_audit_accepted'] is True and
                r['outer_scoring'] is False and r['seed'] == 2026 and
                r['validation_scope'] == 'full_eligible_inner' and
                (not neural or r['selection_policy'] == 'S1'), 'MULTISOURCE_SELECTOR_PROVENANCE')
            wanted = validation_sources(source, r['inner_fold'], endpoint)
            local.require(set(r['by_source']) == set(wanted), 'MULTISOURCE_ENDPOINT_SOURCE_SUPPORT')
            values = {s: r['by_source'][s][metric] for s in wanted}
            local.require(not set(points) & set(values) and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and
                math.isfinite(v) and v >= 0 for v in values.values()), 'MULTISOURCE_POINTS')
            if neural:
                local.require(type(r['best_epoch']) is int and 1 <= r['best_epoch'] <= 30,
                              'MULTISOURCE_EPOCH')
            points.update(values)
            folds.append(math.fsum(values.values()) / len(values))
        means[candidate] = math.fsum(points.values()) / len(points)
        fold_means[candidate] = math.fsum(folds) / len(folds)
    minimum = min(means.values())
    selected = next(c for c in priority if means[c] <= minimum + selection.TOLERANCE)
    result = dict(source=source, model=model, endpoint=endpoint, candidate=selected,
        primary_equal_source_scores=means, equal_fold_sensitivity_scores=fold_means,
        seed_base=2026, outer_scoring=False, production_ready=False,
        fairness_closed=False)
    if neural:
        epochs = [r['best_epoch'] for r in rows if r['candidate'] == selected]
        result.update(outer_epoch=int(median(epochs)), inner_epochs=epochs)
    return result

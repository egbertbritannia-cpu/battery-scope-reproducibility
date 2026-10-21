"""Bounded CALB three-fold F2/F3 contract; not global fairness closure."""
from __future__ import annotations
import math
from . import local_cpu_pilots as local
from . import selection_controls as selection

SHARDS = ("A", "B", "C", "D")
CLASSICAL_PRIORITY = dict(ridge=("ridge_1","ridge_10","ridge_100"),
    hist_gradient_boosting=("hgb_31","hgb_15","hgb_63"),extra_trees=("et_10","et_5","et_20"))
BUDGET = dict(wall_seconds=28800,reserve_seconds=1800,job_seconds=3600,replay_seconds=300,
    synthetic_seconds=900,workers=1,output_bytes=4*2**30)
GUARDS = dict(outer_scoring=False,calibration_scoring=False,fairness_closed=False,
    automatic_resume=False,automatic_retry=False,import_pilot_checkpoints=False,global_claim=False)


def jobs():
    rows=[]
    for f in range(3):
        for a in ("TCN","GRU"):
            for c in selection.PRIORITY:
                rows.append(dict(id=f"inner.CALB.{f}.{a}.{c}",kind="neural",source="CALB",
                    inner_fold=f,seed=2026,architecture=a,candidate=c,epochs=30))
    for f in range(3):
        for c in local.grid():
            for t in selection.TARGETS:
                rows.append(dict(id=f"classical.CALB.{f}.{c}.{t}",kind="classical",source="CALB",
                    inner_fold=f,seed=2026,candidate=c,target=t))
    for f in range(3):
        for t in selection.TARGETS:
            rows.append(dict(id=f"references.CALB.{f}.{t}",kind="references",source="CALB",
                inner_fold=f,seed=2026,target=t))
    return rows


def assignments():
    queues={s:[] for s in SHARDS}
    for kind in ("neural","classical","references"):
        for i,j in enumerate(j for j in jobs() if j["kind"]==kind):
            queues[SHARDS[i%4]].append(j["id"])
    return queues


def validate(m):
    local.require(m["schema"]=="nfr1-calb-full-inner-v1" and m["jobs"]==jobs() and
        m["shards"]==assignments(),"FULL_INNER_QUEUE")
    local.require(m["neural_recipes"]==selection.candidate_recipes() and m["classical_grid"]==local.grid() and
        m["classical_priority"]=={k:list(v) for k,v in CLASSICAL_PRIORITY.items()} and
        m["prediction_policies"]==local.CLASSICAL_PREDICTION_POLICIES,"FULL_INNER_RECIPE")
    local.require(m["budget"]==BUDGET and m["guards"]==GUARDS,"FULL_INNER_SCOPE_BUDGET")
    local.require(m["selection"]==dict(policies=["S0","S1"],patience=5,min_improvement=1e-8,
        candidate_tolerance=1e-8,folds=[0,1,2],seed=2026,neural_epoch_cap=30,
        neural_selector="S1_three_fold_mean_endpoint_score",outer_epochs="median_selected_fold_epochs",
        classical_selector="three_fold_mean_endpoint_score",validation_scope="full_eligible_inner",
        S2=False,automatic_selection_on_kaggle=False),"FULL_INNER_SELECTION_RULES")


def accept_trajectory(history, checkpoints):
    local.require(len(history)==len(checkpoints)==30,"FULL_INNER_NEEDS_30_EPOCHS_NOT_PILOT")
    from . import f2_trajectory as engine
    return engine.policy_map(history,checkpoints,production=True)


def choose_classical(records, family, endpoint):
    """Only after artifact audit: exact three folds/candidates, no outer records."""
    local.require(family in CLASSICAL_PRIORITY and endpoint in selection.TARGETS,"FULL_INNER_FAMILY_TARGET")
    priority=CLASSICAL_PRIORITY[family];records=list(records)
    local.require(len(records)==9 and len({(r['candidate'],r['inner_fold']) for r in records})==9 and
        {(r['candidate'],r['inner_fold']) for r in records}=={(c,f) for c in priority for f in range(3)} and
        all(r['source']=='CALB' and r['endpoint']==endpoint and r['validation_scope']=='full_eligible_inner' and
            r['artifact_audit_accepted'] is True and r['seed']==2026 and r['outer_scoring'] is False and
            isinstance(r['score'],(int,float)) and not isinstance(r['score'],bool) and
            math.isfinite(r['score']) and r['score']>=0 for r in records),"FULL_INNER_CLASSICAL_RECORDS")
    means={c:math.fsum(r['score'] for r in records if r['candidate']==c)/3 for c in priority}
    minimum=min(means.values());chosen=next(c for c in priority if means[c]<=minimum+selection.TOLERANCE)
    return dict(source='CALB',family=family,endpoint=endpoint,candidate=chosen,mean_inner_scores=means,
        selection_seed_base=2026,outer_scoring=False,fairness_closed=False)

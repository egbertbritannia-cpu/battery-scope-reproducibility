"""F2 objective/selection primitives; not a production runner or scoring release.

Only train losses and already-computed full INNER validation scores belong here.
No data readers, outer predictions, checkpoint promotion, or automatic launch.
The evaluator, role seals, runtime pilots and production manifest remain gates.
"""
from __future__ import annotations

import math
from statistics import median

TARGETS = ("soh30", "rul90", "rul80")
WEIGHTS = (1.0, 1.0, 0.5)
PRIORITY = ("Q0", "P0", "P1")
TOLERANCE = 1e-8
MAX_EPOCHS = 30


def require(ok, message):
    if not ok:
        raise ValueError("NFR1_F2:" + message)


def candidate_recipes():
    """Fresh serializable values; callers cannot mutate the locked constants."""
    return {name: dict(objective="joint_pinball" if name == "Q0" else "point_q50",
                       learning_rate=0.0003 if name == "P1" else 0.001,
                       head_weights=dict(zip(TARGETS, WEIGHTS)), quantiles=[0.1, 0.5, 0.9],
                       interval_eligible=name == "Q0") for name in PRIORITY}


def training_loss(predictions, targets, mask, candidate):
    """Q0 exactly retains pinball; P0/P1 use standardized q50 L1/L2.

    Targets have ALREADY been transformed/scaled using inner-train-only state.
    Q10/Q90 gaps receive no direct point-loss gradient; their outputs must not
    be treated as trained quantiles. Normalize over heads observed in this batch,
    as in the bound historical recipe, not over all three heads unconditionally.
    """
    import torch
    require(candidate in PRIORITY, "UNKNOWN_CANDIDATE")
    require(targets.ndim == 2 and targets.shape[1] == 3 and mask.shape == targets.shape,
            "TARGET_OR_MASK_SHAPE")
    require(mask.dtype == torch.bool, "MASK_MUST_BE_BOOL")
    require(set(predictions) == set(TARGETS), "PREDICTION_HEADS")
    losses, weights = [], []
    for index, (head, weight) in enumerate(zip(TARGETS, WEIGHTS)):
        prediction = predictions[head]
        require(prediction.shape == (len(targets), 3), "PREDICTION_SHAPE:" + head)
        observed = mask[:, index]
        if not bool(observed.any()):
            continue
        truth = targets[observed, index].float()
        pred = prediction[observed].float()
        require(bool(torch.isfinite(truth).all()) and bool(torch.isfinite(pred).all()),
                "NONFINITE_OBSERVED_INPUT:" + head)
        if candidate == "Q0":
            q = torch.tensor([0.1, 0.5, 0.9], device=truth.device, dtype=torch.float32).view(1, 3)
            error = truth.view(-1, 1) - pred
            loss = torch.maximum(q * error, (q - 1.0) * error).mean()
        else:
            error = truth - pred[:, 1]
            loss = error.abs().mean() if head == "soh30" else error.square().mean()
        losses.append(weight * loss)
        weights.append(weight)
    require(bool(losses), "NO_OBSERVED_TARGET")
    return sum(losses) / sum(weights)


def select_logical_prefix(scores, patience=5, min_improvement=TOLERANCE):
    """Earliest qualifying best in the logical prefix, never after early stop.

    The trajectory is still trained for 30 epochs; this simulates selection only.
    An improvement must exceed min_improvement. Ties keep the earlier epoch.
    Missing/nonfinite endpoints block the context instead of receiving score 0.
    """
    scores = list(scores)
    require(0 < len(scores) <= MAX_EPOCHS, "TRAJECTORY_LENGTH")
    require(type(patience) is int and patience > 0 and
            math.isfinite(min_improvement) and min_improvement >= 0, "STOP_RULE")
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and v >= 0 for v in scores), "UNEVALUABLE_SCORE")
    best, best_epoch, stale, stop = math.inf, None, 0, None
    for epoch, score in enumerate(scores, 1):
        if score < best - min_improvement:
            best, best_epoch, stale = float(score), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            stop = epoch
            break
    return dict(best_epoch=best_epoch, best_score=best, logical_stop_epoch=stop,
                considered_epochs=stop or len(scores), trajectory_epochs=len(scores))


def checkpoint_policies(full_inner_scores):
    """S0 shared RUL90 epoch versus S1 endpoint-specific epochs.

    Input lists must come from a separate FULL eligible inner evaluator. This
    function cannot certify role provenance or completeness of supplied metrics.
    S2 is intentionally not implemented/promoted as an alternate winner.
    """
    require(set(full_inner_scores) == set(TARGETS), "ENDPOINTS_MISSING")
    require(len({len(v) for v in full_inner_scores.values()}) == 1, "EPOCH_ALIGNMENT")
    selected = {head: select_logical_prefix(full_inner_scores[head]) for head in TARGETS}
    shared = selected["rul90"]["best_epoch"]
    return dict(S0={head: dict(best_epoch=shared,
                              score_at_shared_epoch=float(full_inner_scores[head][shared - 1]),
                              rul90_prefix=selected["rul90"].copy()) for head in TARGETS},
                S1=selected)


def choose_inner_candidate(records, endpoint):
    """Three candidates x three folds for ONE source/architecture/endpoint.

    Each record needs candidate, inner_fold, source, architecture, endpoint,
    validation_scope='full_eligible_inner', selection_policy='S1', best_epoch,
    best_score. Provenance is subsequently checked/sealed by the future runner.
    Tie is measured against the global minimum, not a transitive tolerance chain.
    """
    require(endpoint in TARGETS, "UNKNOWN_ENDPOINT")
    records = list(records)
    require(len(records) == 9, "NINE_CANDIDATE_FOLD_RECORDS_REQUIRED")
    require(len({(r["source"], r["architecture"]) for r in records}) == 1, "MIXED_CONTEXT")
    require(all(r["architecture"] in ("TCN", "GRU") and r["endpoint"] == endpoint
                and r["validation_scope"] == "full_eligible_inner"
                and r["selection_policy"] == "S1" for r in records), "SELECTION_ROLE_OR_POLICY")
    keys = [(r["candidate"], r["inner_fold"]) for r in records]
    require(len(set(keys)) == 9 and set(keys) == {(c, f) for c in PRIORITY for f in range(3)},
            "MISSING_DUPLICATE_OR_UNKNOWN_CANDIDATE_FOLD")
    for record in records:
        require(type(record["best_epoch"]) is int and 1 <= record["best_epoch"] <= MAX_EPOCHS,
                "SELECTED_EPOCH_RANGE")
        score = record["best_score"]
        require(isinstance(score, (int, float)) and not isinstance(score, bool)
                and math.isfinite(score) and score >= 0, "UNEVALUABLE_FOLD")
    means = {candidate: math.fsum(r["best_score"] for r in records if r["candidate"] == candidate) / 3
             for candidate in PRIORITY}
    minimum = min(means.values())
    chosen = next(c for c in PRIORITY if means[c] <= minimum + TOLERANCE)
    selected = sorted((r for r in records if r["candidate"] == chosen), key=lambda r: r["inner_fold"])
    source, architecture = records[0]["source"], records[0]["architecture"]
    return dict(source=source, architecture=architecture, endpoint=endpoint, candidate=chosen,
                mean_inner_scores=means, inner_epochs=[r["best_epoch"] for r in selected],
                outer_epoch=int(median(r["best_epoch"] for r in selected)),
                selection_seed_base=2026, outer_scoring=False, production_ready=False)

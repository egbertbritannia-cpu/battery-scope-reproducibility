"""RUL-only loss; preserve the RUL contribution scale of the accepted joint P1 loss."""
from . import selection_controls as selection

LOSS_CONTRACT='rul-only-preserve-joint-denominator-v1'
TRAINED_HEADS=('rul90','rul80')


def training_loss(predictions,targets,mask,candidate):
    import torch
    selection.require(candidate=='P1','RUL_ABLATION_P1_ONLY')
    selection.require(targets.ndim==2 and targets.shape[1]==3 and mask.shape==targets.shape,
                      'RUL_ABLATION_TARGET_SHAPE')
    selection.require(mask.dtype==torch.bool and set(predictions)==set(selection.TARGETS),
                      'RUL_ABLATION_MASK_OR_HEADS')
    terms=[]
    denominator=0.
    for index,(head,weight) in enumerate(zip(selection.TARGETS,selection.WEIGHTS)):
        observed=mask[:,index]
        if not bool(observed.any()):
            continue
        denominator+=weight
        prediction=predictions[head]
        selection.require(prediction.shape==(len(targets),3),'RUL_ABLATION_PREDICTION_SHAPE')
        if head not in TRAINED_HEADS:
            continue
        truth=targets[observed,index].float()
        q50=prediction[observed,1].float()
        selection.require(bool(torch.isfinite(truth).all()) and bool(torch.isfinite(q50).all()),
                          'RUL_ABLATION_NONFINITE')
        terms.append(weight*(truth-q50).square().mean())
    if not terms:
        # Keep sampler and optimizer cadence identical even for a SoH-only batch.
        # Zero RUL gradients let AdamW/momentum retain the scheduled step; SoH stays disconnected.
        return sum(predictions[t][:,1].sum()*0. for t in TRAINED_HEADS)
    selection.require(denominator>0,'RUL_ABLATION_DENOMINATOR')
    return sum(terms)/denominator

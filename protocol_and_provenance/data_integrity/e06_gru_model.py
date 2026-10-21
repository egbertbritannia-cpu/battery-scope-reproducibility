"""Lazy Torch GRU adapter. No TCN alias, no inference/calibration side effects."""
MODEL_ID = 'gru_sequence_locked_v1'
ARCHITECTURE = dict(input_size=96, hidden_size=64, num_layers=2, dropout=.1,
                    batch_first=True, bidirectional=False, window_size=30)
TARGETS = ('soh30', 'rul90', 'rul80')


def make_model():
    import torch
    from torch import nn
    from torch.nn import functional as F

    class OrderedHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(64, 3)

        def forward(self, x):
            raw = self.projection(x)
            return torch.stack([raw[:, 0]-F.softplus(raw[:, 1]), raw[:, 0],
                                raw[:, 0]+F.softplus(raw[:, 2])], dim=1)

    class QuantileGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(**{k:v for k,v in ARCHITECTURE.items() if k!='window_size'})
            self.heads = nn.ModuleDict({t:OrderedHead() for t in TARGETS})

        def forward(self, x):
            if x.ndim!=3 or tuple(x.shape[1:])!=(30,96):
                raise RuntimeError('GRU_EXPECTS_BATCH_30_96')
            sequence, _ = self.gru(x)
            hidden = sequence[:, -1, :]
            return {t:self.heads[t](hidden) for t in TARGETS}

    return QuantileGRU()


def pinball_loss(predictions, targets, masks):
    import torch
    if targets.ndim!=2 or targets.shape[1]!=3 or masks.shape!=targets.shape:
        raise RuntimeError('GRU_TARGET_MASK_SHAPE')
    q = torch.tensor([.1,.5,.9], device=targets.device, dtype=torch.float32).view(1,-1)
    weights = {'soh30':1., 'rul90':1., 'rul80':.5}
    total=None;active_weight=0.
    for i,t in enumerate(TARGETS):
        active=masks[:,i].bool()
        if not bool(active.any()):continue
        truth=targets[active,i].float().view(-1,1)
        error=truth-predictions[t][active].float()
        value=torch.maximum(q*error,(q-1)*error).mean()*weights[t]
        total=value if total is None else total+value
        active_weight+=weights[t]
    if total is None:raise RuntimeError('GRU_NO_OBSERVED_TARGET')
    return total/active_weight

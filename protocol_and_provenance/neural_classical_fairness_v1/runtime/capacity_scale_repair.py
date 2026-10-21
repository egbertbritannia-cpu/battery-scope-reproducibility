"""Causal capacity-aware feature adapter. Never targets or test-fitted state.

New feature semantics require retraining. No change to frozen old predictors.
"""
from __future__ import annotations

NORMALIZED = (
    'discharge_ah','charge_ah','discharge_wh','charge_wh',
    'cumulative_discharge_ah','cumulative_charge_ah','cumulative_absolute_energy_wh',
    'current_mean_a','current_std_a','current_rms_a','current_median_abs_a','current_peak_abs_a',
    'dqdv_peak1_magnitude','dqdv_peak2_magnitude','dqdv_area')
RECIPES = ('R0','R1','R2')
FEATURE_CONTRACT='capacity-scale-repair-v1'


def require(ok, message):
    if not ok:raise RuntimeError('CAPACITY_SCALE_REPAIR:'+message)


def transform(frame, recipe):
    import numpy as np
    require(recipe in RECIPES,'UNKNOWN_RECIPE')
    if recipe=='R0':return frame.copy(deep=True)
    require(all(c in frame for c in NORMALIZED) and
        all(c+'__mask' in frame for c in NORMALIZED) and
        all(c in frame for c in ('discharge_ah_over_q_ref','discharge_ah_over_q_ref__mask')),'CHANNEL_SCHEMA')
    out=frame.copy(deep=True)
    require(np.isin(frame.discharge_ah__mask,[0,1]).all() and
        np.isin(frame.discharge_ah_over_q_ref__mask,[0,1]).all(),'INVALID_REFERENCE_MASK')
    ah=frame.discharge_ah.to_numpy(float);ratio=frame.discharge_ah_over_q_ref.to_numpy(float)
    observed=frame.discharge_ah__mask.to_numpy()==1
    observed &= frame.discharge_ah_over_q_ref__mask.to_numpy()==1
    observed &= np.isfinite(ah)&np.isfinite(ratio)&(ah>0)&(ratio>0)
    qref=np.full(len(frame),np.nan)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):qref[observed]=ah[observed]/ratio[observed]
    observed &= np.isfinite(qref)&(qref>0)
    for c in NORMALIZED:
        value=frame[c].to_numpy(float);mask=frame[c+'__mask'].to_numpy()
        require(np.isin(mask,[0,1]).all(),'INVALID_MASK:'+c)
        valid=(mask==1)&np.isfinite(value)&observed
        normalized=np.full(len(frame),np.nan,np.float32)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            normalized[valid]=(value[valid]/qref[valid]).astype(np.float32)
        valid &= np.isfinite(normalized)
        normalized[~valid]=np.nan
        out[c]=normalized;out[c+'__mask']=valid.astype(np.uint8)
    return out


def bound_inputs(x,recipe):
    import numpy as np
    require(recipe in RECIPES and x.ndim==3 and x.shape[1:]==(30,96),'WINDOW_SCHEMA')
    require(np.isfinite(x).all() and np.isin(x[:,:,48:],[0,1]).all(),'NONFINITE_OR_BAD_WINDOW_MASK')
    if recipe!='R2':return x
    out=x.copy();out[:,:,:48]=np.clip(out[:,:,:48],-5,5)
    require(np.array_equal(out[:,:,48:],x[:,:,48:]),'MASK_MUTATION')
    return out


def attach(ns,recipe):
    """Attach BEFORE fit_cycle_scaler; inherited role guards remain in loader."""
    require(recipe in RECIPES and not ns.get('_capacity_scale_adapter'),'RECIPE_OR_DOUBLE_ATTACH')
    require(set(NORMALIZED)<=set(ns['VALUE_CHANNELS']),'VALUE_CHANNELS')
    load=ns['load_cycle_shard'];make=ns['make_window_inputs']
    ns['load_cycle_shard']=lambda cell:transform(load(cell),recipe)
    ns['make_window_inputs']=lambda frame,rows,scaler:bound_inputs(make(frame,rows,scaler),recipe)
    ns['_capacity_scale_adapter']=dict(feature_contract=FEATURE_CONTRACT,recipe=recipe,
        channels=list(NORMALIZED),capacity_rule='same_cycle_discharge_ah_divide_discharge_ah_over_q_ref',
        missing_capacity_rule='missing_derived_value_and_mask0_no_future_fill',
        standardized_cap=5 if recipe=='R2' else None,retraining_required=True)
    return ns['_capacity_scale_adapter']

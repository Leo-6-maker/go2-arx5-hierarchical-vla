"""Host admission checks; JAX forward/gradient assay is a separate CPU script."""
import numpy as np
import pytest
from go2arx5_vla.sla_adapter.masked_plan_loss_v61 import validate_host_batch


def test_partial_invalid_targets_are_accepted_only_outside_supervision():
    actions=np.zeros((2,40,32),np.float32);mask=np.zeros((2,40),bool);mask[0,:10]=True;mask[1,::2]=True
    actions[~mask]=np.nan
    assert np.array_equal(validate_host_batch(actions,mask),mask)
    actions[0,0,31]=np.inf
    with pytest.raises(ValueError,match='FINITE_SUPERVISED'):validate_host_batch(actions,mask)


@pytest.mark.parametrize('case',['empty_row','numeric_mask','wrong_mask_axis','FM7','float64','empty_batch'])
def test_empty_or_ambiguous_batches_refuse_before_the_model(case):
    actions=np.zeros((2,40,32),np.float32);mask=np.ones((2,40),bool)
    if case=='empty_row':mask[1]=False
    elif case=='numeric_mask':mask=mask.astype(np.int8)
    elif case=='wrong_mask_axis':mask=mask[...,None]
    elif case=='FM7':actions=actions[...,:7]
    elif case=='float64':actions=actions.astype(np.float64)
    elif case=='empty_batch':actions=actions[:0];mask=mask[:0]
    with pytest.raises(ValueError):validate_host_batch(actions,mask)

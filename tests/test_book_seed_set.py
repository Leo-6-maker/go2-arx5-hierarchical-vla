from copy import deepcopy
import numpy as np
import pytest
from scripts.pi0294.multitask.seed_set_export import validate_seal, fit_statistics, TRAIN_OFFSETS, DEV_OFFSETS, TAG


def seal():
    return dict(schema='BOOK5_TRAIN_DEV10_FROZEN_SEED_SET_V1',
        TRAIN=[dict(id=f'BOOK_SOURCE_SEED{i:03}',xy_offset_m=v) for i,v in enumerate(TRAIN_OFFSETS)],
        DEV=[dict(id=f'BOOK_DEV{i:03}',xy_offset_m=v) for i,v in enumerate(DEV_OFFSETS)],
        DEV_for_training_or_statistics=False)


@pytest.mark.parametrize('fault',['overlap','wrong_id','DEV_fit','missing_TRAIN'])
def test_frozen_source_population_and_split_cannot_change(fault):
    s=seal();validate_seal(s)
    if fault=='overlap':s['DEV'][0]['xy_offset_m']=s['TRAIN'][0]['xy_offset_m']
    elif fault=='wrong_id':s['TRAIN'][0]['id']='OLD_E1_TRAIN'
    elif fault=='DEV_fit':s['DEV_for_training_or_statistics']=True
    else:s['TRAIN'].pop()
    with pytest.raises(ValueError):validate_seal(s)


def test_relative_statistics_anchor_at_actual_state_and_only_completed_commands():
    packets=[]
    for i in range(5):
        state=np.tile([i*.01,0,0,1,0,0,0,1,0,.5],(641,1))
        command=state[:640].copy();command[:,0]+=.01+i*.001
        packets.append(dict(actual_state10=state,committed_absolute10=command,valid_window=np.arange(640)<601))
    fitted=fit_statistics(packets)[TAG]
    assert abs(fitted['state']['eef_9d']['max'][0]-.04)<1e-15
    assert abs(fitted['action']['eef_9d']['max'][0]-.054)<1e-15
    assert abs(fitted['relative_action']['eef_9d']['min'][0]-.01)<1e-15
    assert abs(fitted['relative_action']['eef_9d']['max'][0]-.014)<1e-15
    bad=deepcopy(packets);bad[0]['valid_window'][600]=False
    with pytest.raises(ValueError):fit_statistics(bad)

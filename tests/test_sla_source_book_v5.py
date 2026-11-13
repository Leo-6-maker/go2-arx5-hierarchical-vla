import copy

import numpy as np
import pytest

from go2arx5_vla.sla_adapter.source_book_v5 import IDS, full_source_rows, nominal_commands


def test_source_desired_width_and_state_separation():
    source = np.zeros((640,10));source[:,9] = np.tile([.002/.044,.042/.044],320)
    original = source.copy();actual_measured_grip = np.linspace(-.001/.044,1,641)
    measured_before = actual_measured_grip.copy()
    nominal = nominal_commands(source)
    np.testing.assert_array_equal(source,original)
    np.testing.assert_array_equal(actual_measured_grip,measured_before)
    np.testing.assert_array_equal(nominal[:,9],np.tile([0.,1.],320))
    np.testing.assert_allclose(.002+.04*nominal[:,9],.044*source[:,9],atol=1e-15,rtol=0)
    np.testing.assert_array_equal(nominal[:,:9],source[:,:9])


@pytest.mark.parametrize('opening',[0.,1.,float('nan')])
def test_unsupported_desired_mechanical_width_is_rejected_without_clipping(opening):
    source=np.zeros((640,10));source[:,9]=.042/.044;source[10,9]=opening
    with pytest.raises(ValueError):nominal_commands(source)


def test_full_seeds_exclude_corrections_and_dev_cannot_enter():
    full=[dict(id=e,task='BOOK',complete_commands=640,actual_observations=641,split='TRAIN',windows=601)for e in IDS]
    suffix=dict(id='EXPERT_SUFFIX',task='BOOK',complete_commands=600,actual_observations=601,split='TRAIN',windows=561)
    assert full_source_rows(dict(episodes=full+[suffix]))==full
    leaked=copy.deepcopy(full);leaked[0]['split']='DEV'
    with pytest.raises(ValueError):full_source_rows(dict(episodes=leaked+[suffix]))
    with pytest.raises(ValueError):full_source_rows(dict(episodes=full+[full[0]]))

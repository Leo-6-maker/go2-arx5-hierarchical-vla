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


def test_book_probes_keep_release_withdraw_hold_and_shared_probe_writer(tmp_path):
    from types import SimpleNamespace
    from go2arx5_vla.sla_adapter.runtime_book_v5 import PROBE_PHASES,book_probe_indices,book_probes
    dataset=SimpleNamespace(rows=[(dict(expert_stage_audit_only=list(PROBE_PHASES)),i)for i in range(8)])
    indices=book_probe_indices(dataset);assert indices==list(range(8))
    calls=[]
    def generate(index,noise):
        calls.append((index,noise));return np.full((1,40,32),index,dtype=np.float32),np.full((40,10),noise,dtype=np.float64)
    records=book_probes(SimpleNamespace(output=tmp_path,dataset=dataset,generate=generate),'book')
    assert len(records)==16 and calls==[(i,n)for i in indices for n in range(2)]
    assert (tmp_path/'book/MANIFEST.private.json').is_file()


def test_source_visit_bank_remains_source_and_deterministic():
    from types import SimpleNamespace
    from go2arx5_vla.sla_adapter.runtime_book_v5 import SourceBookEngine
    rows=[dict(split='TRAIN',task='BOOK',episode=e,phase='hold',anchor=0)for e in IDS]
    d=SimpleNamespace(manifest_sha='a'*64,catalog_rows=lambda:rows)
    e=SourceBookEngine.__new__(SourceBookEngine);a=e.visits_for_dataset(d);b=e.visits_for_dataset(d)
    assert a.shape==(16000,)and np.array_equal(a,b)and set(a)==set(range(21))
    assert e.visit_bank_binding['planned_target_visits']==0 and e.visit_bank_binding['planned_source_visits']==16000

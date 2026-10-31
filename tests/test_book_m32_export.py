import json
import numpy as np
import pytest

from scripts.pi0294.multitask.recovery_export import expert_suffix_packet
from scripts.pi0294.multitask.book_m32_export import COUNTS, fit_statistics, early_suffix, admit, SCHEMA
from scripts.pi0294.multitask.book_early_recovery import conditions


def full_packet():
    x=np.tile([0.,0.,0.,1.,0.,0.,0.,1.,0.,.5],(641,1))
    c=x[:-1].copy();c[:,0]=np.arange(640)/10000
    return dict(actual_state10=x,committed_absolute10=c,valid_window=np.arange(640)<601,
        rgb_paths=np.tile(['body.png','wrist.png'],(641,1)),rgb_sha256=np.full((641,2),'0'*64),
        anchor_time_s=np.arange(601)/20,offset_s=np.arange(40)/20,
        instruction=np.asarray('Put book away'),label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'))


@pytest.mark.parametrize('takeover',[8,16,32])
def test_actual_takeover_excludes_student_commands_and_keeps_last_full_h40(takeover):
    p=full_packet();p['committed_absolute10'][:takeover,0]=999.
    suffix=expert_suffix_packet(p,takeover);count=601-takeover
    assert suffix['valid_window'].sum()==count
    assert suffix['anchor_time_s'][0]==takeover/20
    np.testing.assert_array_equal(suffix['source_observation_index'],np.arange(takeover,641))
    np.testing.assert_array_equal(suffix['committed_absolute10'][count-1:count+39],p['committed_absolute10'][600:640])
    assert not np.any(suffix['committed_absolute10']==999.)
    got=fit_statistics([suffix],(count,))['new_embodiment']['action']['eef_9d']
    expected=p['committed_absolute10'][np.arange(takeover,601)[:,None]+np.arange(40),0]
    assert got['max'][0]==expected.max()
    assert got['mean'][0]==pytest.approx(expected.mean(),abs=1e-12)


@pytest.mark.parametrize('takeover',[0,7,24,40,True,8.0])
def test_undeclared_or_coerced_takeover_rejected(takeover):
    with pytest.raises(ValueError):expert_suffix_packet(full_packet(),takeover)


def test_padded_last_window_or_partial_population_cannot_fit_statistics():
    p=expert_suffix_packet(full_packet(),8);p['valid_window'][-1]=True
    with pytest.raises(ValueError):fit_statistics([p],(593,))
    with pytest.raises(ValueError):fit_statistics([full_packet()])
    assert sum(COUNTS)==12533 and len(COUNTS)==21


@pytest.mark.parametrize('condition',[conditions()[5],conditions()[6],dict(id='BOOK_DEV000')])
def test_failed_unrun_and_dev_recoveries_rejected_before_file_access(tmp_path,condition):
    with pytest.raises(PermissionError):early_suffix(tmp_path/condition['id'],condition,{})


def test_changed_statistics_cannot_be_admitted_even_with_a_codec_receipt(tmp_path):
    (tmp_path/'DATASET_MANIFEST.json').write_text(json.dumps(dict(schema=SCHEMA,training_admitted=False,TRAIN_windows=12533)))
    (tmp_path/'OFFICIAL_QUALIFICATION.json').write_text(json.dumps(dict(candidate_manifest_sha256='old_manifest')))
    with pytest.raises(PermissionError):admit(tmp_path)
    assert not (tmp_path/'admitted').exists()

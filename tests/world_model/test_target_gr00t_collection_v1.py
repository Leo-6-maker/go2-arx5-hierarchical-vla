"""Synthetic contracts and geometry. These do not qualify target physics."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from go2arx5_vla.world_model import target_gr00t_collection_v1 as vla
from go2arx5_vla.world_model import stable_task_data_v1 as data
from tests.world_model.test_stable_task_data_v1 import fixture, put


def new_spec(tmp_path, monkeypatch, command=vla.COMMANDS[0]):
    spec,_ = fixture(tmp_path,monkeypatch,np.eye(4))
    spec.update(target_gr00t_collection=vla.contract(),
        target_gr00t_attempt_role='QUALIFICATION', task_command=command,
        stable_task_data=vla.config(command,'TRAIN'))
    panel = json.loads(Path(spec['stable_task_panel']).read_bytes())
    panel.update(schema=vla.PANEL_SCHEMA,collection=vla.contract())
    panel['assignments'][0].update(command=command,purpose='QUALIFICATION')
    p = put(Path(spec['stable_task_panel']),panel)
    spec['stable_task_panel_sha256'] = p['sha256']
    return spec


@pytest.mark.parametrize('command',vla.COMMANDS)
def test_new_profile_directions_are_explicit_and_old_hold_is_excluded(tmp_path,monkeypatch,command):
    spec = new_spec(tmp_path,monkeypatch,command)
    assert data.validate(spec) == vla.config(command,'TRAIN')
    for patch in (dict(target_gr00t_collection=None),dict(task_command='HOLD_OPEN'),
                  dict(target_gr00t_attempt_role='DEMONSTRATION')):
        with pytest.raises(ValueError): data.validate(dict(spec,**patch))


@pytest.mark.parametrize('command,sign',tuple(zip(vla.COMMANDS,(1,-1))))
def test_full40_bounded_real_cube_approach_and_retreat(command,sign):
    actual = np.eye(4); cube = actual.copy();cube[:3,3]=[.2,.04,.03]
    base = np.array([0,0,0,1,0,0,0.])
    c = vla.config(command,'TRAIN')
    program,proposal = vla.build(actual,actual,base,cube,c)
    np.testing.assert_allclose(program.desired_path[-1,:3], sign*.004*cube[:3,3]/np.linalg.norm(cube[:3,3]),atol=1e-15)
    np.testing.assert_array_equal(proposal['actual_cube'],cube)
    assert program.decoded_H40.shape == (40,10)
    assert program.geometry_scale == 1 and program.time_stretch_factor == 2
    assert proposal['minimum_full_path_clearance_m'] >= .14
    assert proposal['native_execution_authorized'] is False


def samples(sign=1):
    out = []
    for i in range(177):
        tcp = np.eye(4);tcp[0,3]=sign*.003*min(i/120,1)
        cube = np.eye(4);cube[0,3]=.2
        out.append(dict(native_step=vla.B+25*i,time_s=(vla.B+25*i)*.002,
            tcp=tcp,object_pose=cube,qpos=np.array([0,0,.3,1,0,0,0.]),
            finger_q=np.array([.044,.044]),safety_valid=True,native_abort=False,
            robot_object_contact=False))
    return out


@pytest.mark.parametrize('sign,command',((1,vla.COMMANDS[0]),(-1,vla.COMMANDS[1])))
def test_new_direction_score_retains_old_approach_negative_and_native_safety(sign,command):
    s = samples(sign)
    r = vla.score(s,command)
    assert r['task_success'] is True
    assert r['original_approach_score']['approach_step_success'] is (sign == 1)
    for mutate in ('base','contact','closed_gripper','native_abort'):
        failed = copy.deepcopy(s)
        if mutate == 'base':failed[-1]['qpos'][0] = .0021
        if mutate == 'contact':failed[-1]['robot_object_contact'] = True
        if mutate == 'closed_gripper':failed[-1]['finger_q'][0] = .02
        if mutate == 'native_abort':failed[-1]['native_abort'] = True
        assert vla.score(failed,command)['task_success'] is False


def test_capture_grid_covers_whole_task_without_claiming_all_future_previews_are_labels():
    c=vla.contract()
    assert vla.CAPTURE_ATOMS[0] == 0 and vla.CAPTURE_ATOMS[-1] == 176
    assert max(np.diff(vla.CAPTURE_ATOMS)) == 7
    assert len(vla.CAPTURE_ATOMS) == 27
    assert c['command_period_s'] == .05 and c['native_dt_s'] == .002
    assert c['initialization_is_demonstration'] is False
    assert c['HOLD_controls_are_demonstrations'] is False
    assert c['rejected_or_unexecuted_suffix_is_demonstration'] is False

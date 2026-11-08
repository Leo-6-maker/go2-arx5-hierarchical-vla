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


def test_start_velocity_admission_is_readonly_and_preserves_original_standing_contract(tmp_path):
    from types import SimpleNamespace
    state=dict(native_step=vla.B,time_s=vla.B*.002,qvel=np.zeros(32))
    state['qvel'][:6]=[.0002,0.,0.,.001,0.,0.]
    before=state['qvel'].copy();c=vla.start_readiness_contract()
    adapter=SimpleNamespace(spec=dict(target_gr00t_start_readiness=c),output=tmp_path)
    descriptor=vla.require_start_readiness(adapter,state)
    actual=json.loads(Path(descriptor['path']).read_bytes())
    assert actual['passes'] and actual['physics_integrations_added']==0
    np.testing.assert_array_equal(state['qvel'],before)
    assert c['WBC_or_startup_changed'] is False
    assert c['original_standing_limits_changed'] is False


@pytest.mark.parametrize('linear,angular',[(.00318408849,.00939069236),
    (.03785497119,.19026044155),(.03807625260,.18632133126)])
def test_actual_failed_start_velocities_are_rejected_before_teacher_build(tmp_path,linear,angular):
    from types import SimpleNamespace
    state=dict(native_step=vla.B,time_s=vla.B*.002,qvel=np.zeros(32))
    state['qvel'][0]=linear;state['qvel'][3]=angular
    adapter=SimpleNamespace(spec=dict(target_gr00t_start_readiness=vla.start_readiness_contract()),output=tmp_path)
    with pytest.raises(ValueError,match='START_READINESS_REJECTED'):
        vla.require_start_readiness(adapter,state)
    proof=json.loads((tmp_path/'TARGET_GR00T_START_READINESS.private.json').read_bytes())
    assert proof['passes']is False and proof['new_task_teacher_proposals_before_check']==0


def test_new_start_rule_cannot_be_disabled_by_omitting_flag_or_using_nonfinite_velocity(tmp_path):
    from types import SimpleNamespace
    state=dict(native_step=vla.B,time_s=vla.B*.002,qvel=np.zeros(32))
    with pytest.raises(ValueError,match='DECLARED_START_READINESS'):
        vla.require_start_readiness(SimpleNamespace(spec={},output=tmp_path),state)
    state['qvel'][3]=np.nan
    with pytest.raises(ValueError,match='ACTUAL_DECISION_QVEL'):vla.start_readiness(state)


def test_live_capture_index_can_progress_and_seal_without_overwriting_observations(tmp_path):
    path=tmp_path/'index.json';captures=[]
    immutable=tmp_path/'capture-0.json';immutable.write_bytes(b'ACTUAL_IMMUTABLE_OBSERVATION')
    for atom in vla.CAPTURE_ATOMS:
        captures.append(dict(atom=atom,native_step=vla.B+25*atom))
        vla.save_capture_index(path,captures)
        assert json.loads(path.read_text())['captures']==captures
    vla.save_capture_index(path,captures,complete=True,task_success=True)
    v=json.loads(path.read_text())
    assert v['complete']is True and v['expert_training_eligible']is False
    assert immutable.read_bytes()==b'ACTUAL_IMMUTABLE_OBSERVATION'

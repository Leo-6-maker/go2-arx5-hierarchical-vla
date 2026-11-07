"""Synthetic static metadata/input fixtures; not real target qualification."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from scripts.pi0292.relay_protocol import unseal
from go2arx5_vla.world_model import stable_task_scene_v1 as scene
from go2arx5_vla.world_model import stable_task_data_v1 as data
from go2arx5_vla.world_model import fusion_task_phase_initialization as init
from tests.world_model.test_fusion_task_phase_initialization import setup


def put(path,value):
    path.write_text(json.dumps(value));return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(tmp_path,monkeypatch,nominal):
    from go2arx5_vla.world_model import fusion_source_phase_c01 as c01
    monkeypatch.setattr(c01,'nominal_reference',lambda descriptor:np.asarray(nominal).copy())
    model=tmp_path/'SCENE.private.mjb';model.write_bytes(b'EXPLICIT_NONPHYSICAL_MODEL_FIXTURE')
    pin=hashlib.sha256(model.read_bytes()).hexdigest()
    proof=dict(schema='WM02_STABLE_TASK_LAYOUT_STATIC_V1',layout='TRAIN01',parent='STABLE_TASK_LAYOUT_TRAIN01',
        split='TRAIN',status='STATIC_ONLY_PASSED_NEW_NATIVE_ADMISSION_REQUIRED',source_sha=scene.BUILDER_SOURCE,
        original_journal_sha256=scene.JOURNAL,WBC_changes=False,physics_dt_s=.002,
        native_integrations=0,new_transitions=0,native_execution_authorized=False,
        compiled=dict(layout='TRAIN01',scene_sha256=pin,base_scene_sha256=scene.layouts.BASE_SCENE,
            changes=dict(changed_model_arrays=['key_qpos'],offset_xy_m=[-.03,-.02],
                         robot_startup_joints_changed=False,robot_dynamics_changed=False)),
        dense_saved_geometry=dict(saved_pose_FK_calls=38703,contact_free_at_saved_poses=True,integrations=0))
    ref=dict(static_result=put(tmp_path/'STATIC.json',proof),
             nominal_calibration_reference=dict(path='/synthetic/nominal',sha256='a'*64))
    row=dict(native_run='new-TRAIN01-HOLD-r1',parent=proof['parent'],split='TRAIN',
        model_sha256=pin,static_sha256=ref['static_result']['sha256'],command='HOLD_OPEN')
    panel=put(tmp_path/'PANEL.json',dict(schema='WM02_STABLE_TASK_DATA_PANEL_V1',assignments=[row]))
    spec=dict(phase=data.PHASE,kind=data.KIND,data_role='TRAIN',data_parent=proof['parent'],
        scene_id='STABLE_TASK_TRAIN01',model_path=str(model),live_skill_start_step=data.B,max_ticks=data.E,
        stable_task_scene_reference=ref,stable_task_panel=panel['path'],stable_task_panel_sha256=panel['sha256'],
        registered_root=str(tmp_path/row['native_run']),task_command='HOLD_OPEN',
        stable_task_data=data.config('HOLD_OPEN','TRAIN'),snapshot_binding_mode='PRE_REGISTERED_STABLE_TASK_DATA_V1',
        live_skill_geometry_scale=1.,live_skill_time_stretch_factor=2)
    return spec,pin


def test_real_scene_wrapper_retains_role_and_explicit_calibration_only(tmp_path,monkeypatch):
    spec,pin=fixture(tmp_path,monkeypatch,np.eye(4));data.validate(spec)
    document=scene.episode(spec);view=scene.prefix_episode(document,spec)
    assert document['actual_scene_sha256']==pin and document['data_role']=='TRAIN'
    assert view['scene_sha256']==scene.layouts.BASE_SCENE and document['actual_scene_sha256']!=view['scene_sha256']
    with pytest.raises(ValueError,match='NOT_RELABELLED_T06'):
        scene.prefix_episode(view,spec)
    with pytest.raises(ValueError,match='PARENT_AND_SPLIT'):
        data.validate(dict(spec,data_role='DEV'))
    with pytest.raises(ValueError,match='EXPLICIT_NEW_CPU'):
        data.validate(dict(spec,paused_query_pair='/old/closed/pair'))


def test_new_layout_actual_cube_is_not_replaced_by_nominal_source_cube(tmp_path,monkeypatch,setup):
    spec,pin=fixture(tmp_path,monkeypatch,setup['cube'])
    state=copy.deepcopy(setup['args']['measured_decision_state']);state['object_pose'][:2,3]+=[-.03,-.02]
    args=dict(setup['args'],measured_decision_state=state,scene_sha256=pin,
              stable_task_scene_reference=spec['stable_task_scene_reference'])
    inputs=init.load_inputs(**args);body=unseal(inputs['frozen']);init._inputs(inputs)
    np.testing.assert_array_equal(body['frozen_cube_pose_target'],state['object_pose'])
    np.testing.assert_array_equal(body['nominal_calibration_cube_pose_target'],setup['cube'])
    np.testing.assert_array_equal(body['full_goal_TCP_target'],unseal(setup['inputs']['frozen'])['full_goal_TCP_target'])
    assert body['scene_sha256']==pin and inputs['provenance']['object_and_goal_are_policy_inputs'] is False
    with pytest.raises(ValueError,match='UNCHANGED_T06_SCENE'):
        init.load_inputs(**{k:v for k,v in args.items()if k!='stable_task_scene_reference'})
    with pytest.raises(ValueError,match='ONE_EXPLICIT'):
        init.load_inputs(**dict(args,nominal_calibration_reference={}))


def test_data_reference_is_same_bounded_full40_math_with_new_role():
    from go2arx5_vla.world_model import stable_start_short_task_v1 as old
    actual=np.eye(4);cube=actual.copy();cube[0,3]=.2;base=np.array([0,0,0,1,0,0,0.])
    p,proposal=old.build_for_config(actual,actual,base,cube,data.config('APPROACH_RED_4MM_OPEN','TRAIN'))
    assert proposal['config']['data_role']=='TRAIN' and proposal['native_execution_authorized'] is False
    assert p.decoded_H40.shape==(40,10) and p.end_atom+8<=176
    np.testing.assert_allclose(p.desired_path[-1,:3],[.004,0,0],atol=1e-15)


def test_actual_data_phase_dispatch_keeps_original_spec_identity(monkeypatch):
    from go2arx5_vla.world_model.auto1_native import phase_limit
    got=[]
    monkeypatch.setattr(data,'episode_limit',lambda spec,previous:got.append((spec,previous))or data.E)
    spec=dict(phase=data.PHASE,kind=data.KIND)
    assert phase_limit(data.PHASE,{'previous':'real'},spec)==data.E
    assert got==[(spec,{'previous':'real'})]

import json
import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from go2arx5_vla.world_model import stable_start_short_task_v1 as s


def geometry(distance=.18):
    tcp=np.eye(4);tcp[:3,3]=(.3,0.,.6)
    cube=tcp.copy();cube[0,3]+=distance
    return tcp,cube,[0.,0.,.4,1.,0.,0.,0.]


@pytest.mark.parametrize('method,delta',[('HOLD',0.),('GEOMETRIC_TEACHER_4MM',.004)])
def test_full40_preserves_held_reference_and_actual_intent(method,delta):
    tcp,cube,base=geometry();reference=tcp.copy();reference[1,3]+=.06
    p,r=s.build(tcp,reference,base,cube,method)
    assert p.decoded_H40.shape==(40,10) and p.end_atom+8<=176
    np.testing.assert_allclose(p.desired_path[-1,:3]-tcp[:3,3],[delta,0,0],atol=1e-15)
    np.testing.assert_allclose(p.reference_path[0,:3],reference[:3,3],atol=1e-15)
    np.testing.assert_allclose(p.reference_path[-1,:3]-reference[:3,3],[delta,0,0],atol=1e-15)
    assert r['native_execution_authorized'] is False
    assert s.config(method)['data_role']=='DIAGNOSTIC_NOT_AUTO_TRAIN'
    from go2arx5_vla.world_model.fusion_graded_short_skill_v1 import score_contract
    assert s.score_contract()==score_contract()


@pytest.mark.parametrize('distance',[.139,.14,.1439])
def test_no_extrapolation_beyond_original_clearance(distance):
    tcp,cube,base=geometry(distance)
    with pytest.raises(ValueError,match='FULL_STEP_ORIGINAL'):
        s.build(tcp,tcp,base,cube,'GEOMETRIC_TEACHER_4MM')


def registered(tmp_path):
    rows=[dict(native_run='hold-1',method='HOLD',split='DIAGNOSTIC'),
          dict(native_run='hold-2',method='HOLD',split='DIAGNOSTIC'),
          dict(native_run='teacher-1',method='GEOMETRIC_TEACHER_4MM',split='DIAGNOSTIC')]
    p=tmp_path/'panel.json';p.write_text(json.dumps(dict(schema='WM02_STABLE_START_G1_PANEL_V1',assignments=rows)))
    return dict(phase=s.PHASE,kind=s.KIND,task_condition=s.CONDITION,
        snapshot_binding_mode='PRE_REGISTERED_STABLE_START_CPU_V1',live_skill_start_step=s.B,
        max_ticks=s.E,live_skill_geometry_scale=1.,live_skill_time_stretch_factor=2,
        stable_start_diagnostic=s.config('HOLD'),stable_start_panel=str(p),
        stable_start_panel_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),registered_root=str(tmp_path/'hold-1'))


def test_frozen_panel_cannot_relabel_method_or_create_gpu_pair(tmp_path):
    spec=registered(tmp_path);s.validate(spec)
    with pytest.raises(ValueError,match='PANEL_METHOD_RUN_JOIN'):
        s.validate(dict(spec,stable_start_diagnostic=s.config('GEOMETRIC_TEACHER_4MM')))
    with pytest.raises(ValueError,match='TYPED_CPU'):
        s.validate(dict(spec,paused_query_gpu_spec='/unregistered'))


def test_lifecycle_retains_phase_and_original_safety_contract(tmp_path,monkeypatch):
    spec=registered(tmp_path)
    spec.update(profile='wm02-auto1-target',domain='WM02_AUTO1_TARGET_COMMISSION_V1',
        site_profile='4080-simulation',physics=True,scope='WM02_PILOT1_SIM_ONLY',
        pilot1_binding='/fresh/window',WBC_changes=True)
    with pytest.raises(ValueError,match='FROZEN_WBC_2MS'):
        s.operational.recipe_config(spec,actual_phase=s.PHASE,actual_kind=s.KIND,
                                    source_members=s.SOURCE_MEMBERS)


def test_actual_installed_preview_cannot_be_changed_or_reanchor_base():
    tcp,cube,base=geometry();p,_=s.build(tcp,tcp,base,cube,'GEOMETRIC_TEACHER_4MM')
    a=SimpleNamespace(spec=dict(phase=s.PHASE,kind=s.KIND),observe=lambda:dict(step=s.B),
        live_skill_program=p,base_pose_hold=np.asarray(base),trajectory_initial_base=np.asarray(base))
    s.validate_adapter_install(a,p.preview(0))
    wrong=p.preview(0).copy();wrong[0,0]+=.001
    with pytest.raises(ValueError,match='EXACT_COMPLETE_SUFFIX'):
        s.validate_adapter_install(a,wrong)

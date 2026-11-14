import copy
import pytest
from go2arx5_vla.world_model import pd4_prospective_v71 as p


def test_statistics_are_view_specific_and_never_reuse_old_transport(monkeypatch):
    pin=p.STATS['PD4_FINAL_PLAN'];monkeypatch.setattr(p,'digest',lambda path:pin)
    s={p.KEY:dict(version=p.VERSION,view='PD4_FINAL_PLAN',old_RP8_passed=False,full_WBC_state_restored=False),
       'expert_plan_capture_v61':dict(version=p.VERSION,transport_statistics=dict(path='/frozen/stats',sha256=pin),codec_module='scripts.sla.pd4_plan_transport_v71'),
       'execution_site':p.SITE,'scope':p.SCOPE,'phase':'collect-target-task-learning-grasp-v1',
       'task_learning_role':'CORELAB_EXPERT_TRAIN4_SEMANTICS_NOT_BC_V3','task_learning_teacher_mode':'NORMAL_TEACHER_V7',
       'expert_eef_stream_v2':dict(version='EEF_STREAM_V2',prefix_atoms=10)}
    assert p.configuration(s)==s['expert_plan_capture_v61']
    for key,value in [('scope','WM02_PILOT1_SIM_ONLY'),('execution_site','CORE_PILOT1_NATIVE'),('target_gr00t_policy',{'checkpoint':'any'})]:
        altered=copy.deepcopy(s);altered[key]=value
        with pytest.raises(PermissionError):p.configuration(altered)
    altered=copy.deepcopy(s);altered[p.KEY]['view']='PD4_EXECUTED_REF'
    with pytest.raises(PermissionError):p.configuration(altered)


def test_new_native_route_cannot_be_entered_by_unknown_site_without_binding(tmp_path):
    import json
    from go2arx5_vla.world_model.auto1_native import admitted_target
    f=tmp_path/'spec.json';f.write_text(json.dumps(dict(campaign_root=str(tmp_path),execution_site=p.SITE)))
    with pytest.raises(KeyError):
        with admitted_target(f,'native-01',tmp_path/'native-01'):pytest.fail('must not admit')
    assert not (tmp_path/'native-01').exists()


def test_fresh_recipe_does_not_allow_changed_physics_or_unbounded_endpoint(monkeypatch):
    monkeypatch.setattr(p,'configuration',lambda s:{})
    from go2arx5_vla.world_model import stable_task_scene_v1 as scene
    monkeypatch.setattr(scene,'pinned',lambda pin:dict(schema=p.VERSION+'_BINDING',scope=p.SCOPE,source_sha='source',native_roots=['/fresh/run']))
    s={p.KEY:{'binding':{}},'registered_root':'/fresh/run','profile':'wm02-auto1-target',
       'domain':'WM02_AUTO1_TARGET_COMMISSION_V1','physics':True,'WBC_changes':False,'physics_dt_s':.002,
       'wall_limit_s':3600,'max_ticks':21703,'live_skill_start_step':4703,'source_sha':'source',
       'reference_binding_version':'MEASURED_EEF_AND_BASE_POSE_HOLD_V2'}
    from go2arx5_vla.world_model.task_learning_grasp_v1 import PHASE,KIND
    for key,value in [('WBC_changes',True),('physics_dt_s',.004),('max_ticks',21704),('wall_limit_s',100000),('pilot1_binding','/expired')]:
        altered=dict(s,**{key:value})
        with pytest.raises(PermissionError,match='EXACT_PD4_FRESH_NATIVE_RECIPE'):
            p.recipe_config(altered,None,actual_phase=PHASE,actual_kind=KIND,source_members=())

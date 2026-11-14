"""Fixed prospective coverage/native custody;no fixture claims physics success."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from go2arx5_vla.world_model import sla_hold50_v6 as production,sla_five_object_hold_v5 as five
from go2arx5_vla.world_model import sla_target_coverage_v5 as coverage,task_learning_grasp_v1 as task
from go2arx5_vla.world_model import corelab_expert_semantics_v3 as expert
from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER
from go2arx5_vla.world_model.storage import digest
from test_sla_five_object_hold_v5 import spec as old_spec


def spec(obj,pose):
    s=dict(sla_five_object_hold_v5=dict(version=production.VERSION,object_id=obj,pose_id=pose),
        task_learning_role=five.ROLE,task_learning_teacher_mode='NORMAL_TEACHER_V7',
        expert_eef_stream_v2=dict(version='EEF_STREAM_V2',prefix_atoms=10),
        expert_recorded_success_tail_v3=expert.RECORDED_TAIL)
    s.update(phase=task.PHASE,kind=task.KIND,task_learning_grasp=five.protocol(s),
        task_command=five.command(obj,production.VERSION),live_skill_start_step=task.CALIBRATION_B,
        max_ticks=five.ENDPOINT,physics_dt_s=.002,WBC_changes=False)
    return s


def panel(monkeypatch):
    d,pin=production.declaration()
    monkeypatch.setattr('go2arx5_vla.world_model.task_learning_collection_v1.qualified_base',lambda _:None)
    return dict(schema=production.PANEL,expert_version=production.VERSION,source_sha='a'*40,
        tasks=['PICK_HOLD'],D100_new=False,prospective_D100_component='PICK_HOLD50',old_episodes_relabelled=False,
        base_grasp_Teacher_sha256=FROZEN_TEACHER,base_Teacher_qualification={},preregistration=pin,
        assignments=[dict(e,layout=e['object_id']+'-'+e['pose_id'],parent=five.scene_id(e['object_id'],e['pose_id']),
            split='TRAIN',prefix_atoms=10,command=production.command(e['object_id']),model_sha256='b'*64,
            native_run='new50-'+e['object_id'].lower()+'-'+e['pose_id'].lower())for e in d['conditions']])


def test_all_fixed_fifty_strata_keep_original_clock_and_score():
    assert teacher_fingerprint()==FROZEN_TEACHER
    for obj in coverage.OBJECTS:
        for pose in coverage.POSES:
            s=spec(obj,pose);task.declared(s)
            p=five.protocol(s)
            assert p['pose_id']==pose and p['object_id']==obj and not p['D100_new']
            assert p['prospective_D100_component']=='PICK_HOLD50'
            assert expert.native_endpoint(s)==21703 and p['prefix_atoms']==10
            assert five.restart(s)==13 and p['original_score_unchanged']
    d,_=production.declaration();assert d['unchanged_task_score']==task.score_contract()


def test_selective_teacher_repair_retains_frozen_original_stages():
    initial=dict(tcp=np.eye(4),object_pose=np.eye(4));initial['object_pose'][:3,3]=[.28,-.014,.62]
    old=task.Teacher(initial)
    for obj in coverage.OBJECTS:
        new=five.make_teacher(spec(obj,'POSE10'),initial)
        delta=[-.005,0.,0.]if obj in production.REPAIRED_OBJECTS else[0.,0.,0.]
        np.testing.assert_allclose(new.grasp_goal()-old.grasp_goal(),delta,rtol=0,atol=1e-16)
        assert(new.stage,new.grip)==(old.stage,old.grip)
        if obj not in production.REPAIRED_OBJECTS:assert type(new)is task.Teacher
    assert teacher_fingerprint()==FROZEN_TEACHER


@pytest.mark.parametrize('mutation',['missing','duplicate','reordered','restart','old_phase','D100','relabelled'])
def test_new_panel_cannot_select_strata_or_relabel_old_episodes(monkeypatch,mutation):
    p=panel(monkeypatch);assert len(five.validate_panel(p))==50
    if mutation=='missing':p['assignments'].pop()
    elif mutation=='duplicate':p['assignments'][1]=copy.deepcopy(p['assignments'][0])
    elif mutation=='reordered':p['assignments'].reverse()
    elif mutation=='restart':p['assignments'][0]['restart']=12
    elif mutation=='old_phase':p['expert_version']=five.REPAIRED_VERSION
    elif mutation=='D100':p['D100_new']=True
    else:p['old_episodes_relabelled']=True
    with pytest.raises(ValueError):five.validate_panel(p)


def test_original_pilot_does_not_admit_other_poses_or_production_alias():
    s=old_spec('RED_CUBE40');s['sla_five_object_hold_v5']['pose_id']='POSE10'
    with pytest.raises(ValueError):five.configuration(s)
    s=spec('RED_CUBE40','POSE10');s['sla_five_object_hold_v5']['version']=five.VERSION
    with pytest.raises(ValueError):task.declared(s)


def test_native_scene_custody_joins_exact_pose10_in_all50_closed_static_cases(tmp_path,monkeypatch):
    p=panel(monkeypatch);row=p['assignments'][-1];obj,pose=row['object_id'],row['pose_id'];s=spec(obj,pose)
    monkeypatch.setattr('go2arx5_vla.world_model.native_body_camera_capture_v1.session_configuration',lambda _:True)
    def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return dict(path=str(p),sha256=digest(p))
    model=tmp_path/'SCENE.private.mjb';model.write_bytes(b'no native model fixture')
    builder='c'*64
    proof=save(tmp_path/'STATIC_RESULT.private.json',dict(passes=True,declaration=coverage.declaration(obj,pose),
        builder_sha256=builder,saved_startup_checks=5703,actual_integrations=0,native_execution_authorized=False,
        robot_contract=dict(status='ROBOT_AND_NATIVE_SAFETY_EXACT_WITH_EXPLICIT_OBJECT_VARIATION'),scene_sha256=digest(model)))
    producer=tmp_path/'static50';sp=save(producer/'EXECUTION_SPEC.json',{})
    save(producer/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,spec_sha256=sp['sha256']))
    save(producer/'WORKER_EXIT.private.json',dict(exit_code=0))
    rows=[dict(declaration=coverage.declaration(o,pp),result_sha256=proof['sha256']if(o,pp)==(obj,pose)else'd'*64)for o in coverage.OBJECTS for pp in coverage.POSES]
    result=save(producer/'RESULT.private.json',dict(schema=coverage.VERSION,passed=50,completed=50,rows=rows))
    row['model_sha256']=digest(model);panel_pin=save(tmp_path/'PANEL.private.json',p)
    s.update(source_sha='a'*40,source_files={'src/go2arx5_vla/world_model/sla_target_coverage_v5.py':builder},
        model_path=str(model),task_learning_scene_reference=proof,sla_five_object_static_result_v5=result,
        data_role='TRAIN',scene_id=five.scene_id(obj,pose),data_parent=five.scene_id(obj,pose),
        stable_task_panel=panel_pin['path'],stable_task_panel_sha256=panel_pin['sha256'],
        registered_root=str(tmp_path/row['native_run']),expert_cadence_restart=13)
    actual=five.native_scene_admission(s)
    assert actual['parent']==five.scene_id(obj,'POSE10')and actual['static_evidence_projection_only']
    s['sla_five_object_hold_v5']['pose_id']='POSE09';s['task_learning_grasp']=five.protocol(s)
    with pytest.raises(ValueError):five.native_scene_admission(s)

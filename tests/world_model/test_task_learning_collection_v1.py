import copy
import json
from pathlib import Path
import pytest

from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_scene_v1 as scene
from go2arx5_vla.world_model import task_learning_teacher_panel_v1 as teacher
from go2arx5_vla.world_model import task_learning_collection_v1 as collection
from go2arx5_vla.world_model import task_learning_recovery_v1 as recovery
from go2arx5_vla.world_model.storage import digest


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return dict(path=str(path),sha256=digest(path))


def qualified_base(tmp_path):
    # Synthetic CLOSED metadata tests validation only; never a site receipt.
    rows=[dict(native_run=f'q-{layout}-{restart}',layout=layout,restart=restart,
        parent=scene.scene_id(layout),split='TRAIN',command=task.COMMAND,
        purpose='TEACHER_QUALIFICATION_NOT_BC',model_sha256='a'*64)
        for restart in(1,2)for layout in scene.QUALIFICATION_OFFSETS]
    p=dict(schema=teacher.SCHEMA,protocol=task.protocol(task.QUALIFICATION_ROLE),
        Teacher_implementation_source_sections_sha256=teacher.FROZEN_TEACHER,assignments=rows)
    outcomes=[dict(r,attempted=True,reset_ready=True,task_success=True,technical_valid=True,
        hard_safety_stop=False)for r in rows]
    root=tmp_path/'base';result=dict(status='TEACHER_QUALIFICATION_COMPLETE',exit_code=0,
        summary=teacher.summarize(p,outcomes))
    return dict(panel=save(root/'PANEL.private.json',p),result=save(root/'RESULT.private.json',result),
        lease=save(root/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0)))


def plan(tmp_path,role):
    rows=[dict(native_run=f'{role.lower()}-{layout.lower()}-r{restart}',layout=layout,restart=restart,
        Teacher_mode=mode,parent=scene.scene_id(layout),split='TRAIN',command=task.COMMAND,
        purpose=collection.purpose(role),model_sha256='b'*64)
        for layout,restart,mode in collection.expected_assignments(role)]
    p=dict(schema=collection.SCHEMA,source_sha='a'*40,task_learning_role=role,protocol=task.protocol(role),
        Teacher_implementation_source_sections_sha256=teacher.FROZEN_TEACHER,
        recovery_implementation_sha256=digest(recovery.__file__),assignments=rows,
        base_Teacher_qualification=qualified_base(tmp_path))
    if role==task.COLLECTION_ROLE:
        rp=plan(tmp_path/'rproof',task.RECOVERY_QUALIFICATION_ROLE);root=tmp_path/'recovery'
        pd=save(root/'PANEL.private.json',rp)
        result=dict(status='TASK_LEARNING_NATIVE_GROUP_COMPLETE',exit_code=0,source_sha=rp['source_sha'],
            panel_sha256=pd['sha256'],summary=collection.summarize(rp,outcomes(rp)))
        p['recovery_qualification']=dict(panel=pd,result=save(root/'RESULT.private.json',result),
            lease=save(root/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0)))
    return p


def outcomes(p):
    return [dict(r,attempted=True,reset_ready=True,task_success=True,technical_valid=True,
        hard_safety_stop=False,genuine_recovery_success=r['Teacher_mode']==recovery.UNDER_CLOSE)
        for r in p['assignments']]


def test_new_layouts_are_disjoint_and_collection_is_fixed_25_times_4():
    old=set(scene.QUALIFICATION_OFFSETS.values())|{(0.,0.)}
    new=set(scene.TRAIN_OFFSETS.values());checks=set(scene.RECOVERY_OFFSETS.values())
    assert len(new)==25 and not(new&old or checks&old or checks&new)
    rows=collection.expected_assignments(task.COLLECTION_ROLE)
    assert len(rows)==100 and len(set(rows))==100
    assert sum(mode==recovery.NORMAL for _,_,mode in rows)==50
    assert sum(mode==recovery.UNDER_CLOSE for _,_,mode in rows)==50
    physical=task.protocol();physical.pop('stage')
    for role in(task.COLLECTION_ROLE,task.RECOVERY_QUALIFICATION_ROLE):
        c=task.protocol(role);assert c.pop('stage')==role;assert c==physical


def test_recovery_gate_requires_all_four_actual_recoveries_not_just_task_pass(tmp_path):
    p=plan(tmp_path,task.RECOVERY_QUALIFICATION_ROLE);v=outcomes(p)
    assert collection.summarize(p,v)['passes']
    v[0]['genuine_recovery_success']=False
    assert not collection.summarize(p,v)['passes']


def test_incomplete_base_cannot_admit_recovery_or_formal_collection(tmp_path):
    p=plan(tmp_path,task.RECOVERY_QUALIFICATION_ROLE);d=p['base_Teacher_qualification']['result']
    path=Path(d['path']);value=json.loads(path.read_text());value['summary']['attempted']=19
    p['base_Teacher_qualification']['result']=save(path,value)
    with pytest.raises(ValueError,match='ACTUAL_CLOSED_TWENTY'):collection.validate(p)


def test_collection_keeps_attempts_layouts_and_real_recovery_coverage_separate(tmp_path):
    p=plan(tmp_path,task.COLLECTION_ROLE);v=outcomes(p);s=collection.summarize(p,v)
    assert s['passes'] and s['attempted']==100 and s['genuine_recovery_successes']==50
    assert s['qualified_training_demonstrations']==0 and s['Processor_admitted']is False
    assert not collection.summarize(p,v[:50])['passes']
    for r in v[50:]:r['genuine_recovery_success']=False
    assert not collection.summarize(p,v)['passes']


def test_old_qualification_or_changed_mode_cannot_be_new_demonstration(tmp_path):
    p=plan(tmp_path,task.RECOVERY_QUALIFICATION_ROLE)
    for key,value in [('layout','QUAL01'),('purpose','TEACHER_QUALIFICATION_NOT_BC'),('Teacher_mode',recovery.NORMAL)]:
        bad=copy.deepcopy(p);bad['assignments'][0][key]=value
        with pytest.raises(ValueError):collection.validate(bad)


def test_recovery_classifier_requires_observed_miss_and_later_actual_grasp(tmp_path):
    p=plan(tmp_path,task.RECOVERY_QUALIFICATION_ROLE);row=p['assignments'][0]
    q=dict(schema=collection.native_schema(task.RECOVERY_QUALIFICATION_ROLE),
        excluded_intervention_decisions=1,expert_policy_decisions=14,
        recovery_evidence=dict(intervention_issued=True,actual_under_close_failed=True,
            native_grasp_after_correction=True))
    args=dict(native=dict(status='NATIVE_STAGE_PASSED',actual_target_integrations=task.E,
        shutdown_clean=True,owned_process_exits=[dict(name=n,exit_code=0)for n in('sim','solver','controller','planner')],
        reference_qualification=q),worker=dict(exit_code=0),lease=dict(status='CLOSED'),
        readiness=dict(passes=True,native_step=task.B),score=dict(task_success=True),
        spec=dict(task_learning_role=task.RECOVERY_QUALIFICATION_ROLE,phase=task.PHASE,source_sha='candidate'),
        expected_source='candidate',attempted=True)
    assert collection.classify(row,**args)['genuine_recovery_success']
    q['recovery_evidence']['actual_under_close_failed']=False
    v=collection.classify(row,**args)
    assert v['native_task_success'] and not v['task_success']and not v['genuine_recovery_success']


@pytest.mark.parametrize('role',[task.RECOVERY_QUALIFICATION_ROLE,task.COLLECTION_ROLE])
def test_unregistered_successor_group_cannot_start_compute(tmp_path,monkeypatch,role):
    import subprocess
    from scripts.wm02.task_learning_teacher_qualification_v1 import run
    (tmp_path/'control').mkdir();(tmp_path/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED')))
    f=tmp_path/'INPUT.private.json';f.write_text(json.dumps(dict(task_learning_role=role)))
    def forbidden(*args,**kwargs):raise AssertionError('COMPUTE_STARTED_BEFORE_ACTUAL_SCOPE')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    with pytest.raises(PermissionError,match='FRESH_ACTUAL_FINITE_TEACHER_PANEL'):run(f)

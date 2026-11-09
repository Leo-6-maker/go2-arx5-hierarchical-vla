"""Role/layout fixtures cannot grant native DEV or Teacher-reference success."""
import json
import pytest
from go2arx5_vla.world_model import task_learning_dev_v1 as dev
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_scene_v1 as scene
from go2arx5_vla.world_model.task_learning_dataset_v1 import audited_episode
from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER


def test_new_twenty_dev_layouts_are_disjoint_and_reference_four_are_fixed():
    other=set(scene.TRAIN_OFFSETS.values())|set(scene.QUALIFICATION_OFFSETS.values())|set(scene.RECOVERY_OFFSETS.values())|{(0.,0.)}
    assert len(dev.DEV_OFFSETS)==len(set(dev.DEV_OFFSETS.values()))==20
    assert not set(dev.DEV_OFFSETS.values())&other
    assert len(dev.REFERENCE_LAYOUTS)==4 and set(dev.REFERENCE_LAYOUTS)<=set(dev.DEV_OFFSETS)
    assert teacher_fingerprint()==FROZEN_TEACHER
    for name,(dx,dy)in dev.DEV_OFFSETS.items():
        assert max(abs(dx),abs(dy))<=.004
        assert scene.scene_id(name)=='TASK_LEARNING_UNDER_HAND_'+name
        assert scene.geometry(name)['role']=='NEW_DEV_TASK_LAYOUT_NOT_TRAIN'
        assert scene.geometry(name)['object_initial_xyz_m']==[.26+dx,-.014+dy,.62]


def test_dev_reference_role_keeps_clocks_safety_and_student_observations():
    a=task.protocol(task.COLLECTION_ROLE);b=task.protocol(task.DEV_REFERENCE_ROLE)
    assert {k:v for k,v in a.items()if k not in('stage','scene_family')}=={
        k:v for k,v in b.items()if k not in('stage','scene_family')}
    assert dev.purpose(task.DEV_REFERENCE_ROLE)=='DEV_REFERENCE_NOT_BC'
    with pytest.raises(ValueError):dev.purpose(task.COLLECTION_ROLE)
    with pytest.raises(ValueError,match='FOUR_FIXED_NEW_DEV_REFERENCES'):
        dev.validate(dict(assignments=[]))


def test_explicit_dev_audit_never_converts_old_training_or_qualification(tmp_path):
    with pytest.raises(ValueError,match='NEVER_ADMITTED_BY_A_ROLE_OVERRIDE'):
        audited_episode(tmp_path,{},'unused','unused',expected_role=task.QUALIFICATION_ROLE)
    (tmp_path/'EXECUTION_SPEC.json').write_text(json.dumps(dict(task_learning_role=task.COLLECTION_ROLE)))
    with pytest.raises(ValueError,match='NEVER_DIAGNOSTIC_RELABEL'):
        audited_episode(tmp_path,{},'unused','unused',expected_role=task.DEV_REFERENCE_ROLE)

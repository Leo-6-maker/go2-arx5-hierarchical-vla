import copy
import pytest
import numpy as np

from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_local_policy_v1 as local
from go2arx5_vla.world_model import task_learning_student_v1 as student
from go2arx5_vla.world_model import task_learning_live_pair_v1 as original_pair
from go2arx5_vla.world_model.target_gr00t_profile_v1 import PARENT


def panel():
    rows=[]
    for i,(layout,method,seed)in enumerate(local.ORDER):
        rows.append(dict(native_run=f'local-{i}',layout=layout,method=method,
            parent='TASK_LEARNING_UNDER_HAND_'+layout,split='TRAIN',purpose=local.PURPOSE,
            restart=1,command=task.COMMAND,model_sha256='a'*64,query_seeds=list(range(seed,seed+15))))
    return dict(schema=local.SCHEMA,task_learning_role=task.CORE_LOCAL_ROLE,
        protocol=task.protocol(task.CORE_LOCAL_ROLE),B1_global_step=2000,source_sha='b'*40,
        policies={'B0':{},'B1':{}},preregistration={'sha256':local.CONTRACT_SHA},assignments=rows)


def test_local_four_is_disjoint_from_original_DEV80():
    p=panel();assert len(local.validate_panel(p))==4
    assert len(local.selection(dict(schema=local.INPUT_SCHEMA,source_sha=p['source_sha'],assignment_slice=[0,4]),p))==4
    with pytest.raises(ValueError):student.validate(p)
    for key,value in [('split','DEV'),('layout','DEV01'),('method','B0')]:
        q=copy.deepcopy(p);q['assignments'][1][key]=value
        with pytest.raises(ValueError):local.validate_panel(q)


def test_local_policy_proof_cannot_satisfy_original_cross_hardware_contract():
    policy=dict(arm='B1',parent_manifest_sha256=PARENT,package={'path':'/unused','sha256':'c'*64},
                Core_local_qualification={'path':'/unused','sha256':'d'*64})
    with pytest.raises(PermissionError,match='EXPLICIT_ORIGINAL_B0_OR_TRAINED_INTERNAL_GR00T_B1'):
        original_pair.policy_contract(policy,{},verify_package=False)


def test_teacher_and_original_student_role_semantics_stay_separate():
    assert task.student_role(task.CORE_LOCAL_ROLE)
    assert task.student_role(task.STUDENT_DEV_ROLE)
    assert not task.student_role(task.COLLECTION_ROLE)
    assert task.protocol(task.CORE_LOCAL_ROLE)['expert_commands']=='NONE_STUDENT_NOT_EXPERT'
    assert task.protocol(task.STUDENT_DEV_ROLE)['scene_family']=='SEPARATE_UNDER_HAND_NEW_DEV_LAYOUTS_NOT_TRAIN'
    assert task.protocol(task.CORE_LOCAL_ROLE)['physics_dt_s']==.002


def test_generated_grip_classifies_original_contact_intent_without_action_edits():
    raw=np.zeros((40,10));raw[:,9]=np.linspace(1.,0.,40);before=raw.copy()
    assert local.execution_stage(raw,{'native_grasp':False})=='CLOSE'
    assert np.array_equal(raw,before)
    raw[:,9]=0.
    assert local.execution_stage(raw,{'native_grasp':True})=='HOLD'
    raw[:,9]=np.linspace(0.,1.,40)
    assert local.execution_stage(raw,{'native_grasp':True})=='PREGRASP'
    raw[:,9]=1.
    assert local.execution_stage(raw,{'native_grasp':False})=='PREGRASP'


@pytest.mark.parametrize('changed_member',['builder','model'])
def test_archived_scene_builder_custody_is_independent_of_new_runtime_and_rejects_tampering(tmp_path,monkeypatch,changed_member):
    from go2arx5_vla.world_model import task_learning_student_registration_v2 as registration
    from go2arx5_vla.world_model.task_learning_scene_v1 import MEMBER
    source=tmp_path/'archived';builder=source/MEMBER;builder.parent.mkdir(parents=True)
    builder.write_text('archived TRAIN builder; never executed')
    model=tmp_path/'scene.mjb';model.write_bytes(b'unchanged archived model')
    reference={'path':'sealed-static-proof','sha256':'a'*64}
    proof=dict(parent='TASK_LEARNING_UNDER_HAND_TRAIN01',builder_sha256=local.digest(builder),
        compiled={'scene_sha256':local.digest(model)})
    original=dict(data_role='TRAIN',task_learning_role=task.COLLECTION_ROLE,
        task_learning_scene_reference=reference,scene_id=proof['parent'],data_parent=proof['parent'],
        source_root=str(source),source_files={MEMBER:proof['builder_sha256']},model_path=str(model))
    monkeypatch.setattr(registration,'closed_base',lambda pin:original)
    monkeypatch.setattr(local,'pinned',lambda pin:proof)
    assert local.scene_builder_custody({'path':'successful-TRAIN-spec'},reference,model)==proof['builder_sha256']
    (builder if changed_member=='builder'else model).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='ORIGINAL_SCENE_BUILDER_CUSTODY'):
        local.scene_builder_custody({'path':'successful-TRAIN-spec'},reference,model)

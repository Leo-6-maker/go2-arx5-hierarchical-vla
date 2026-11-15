"""Typed N20 metadata fixtures; no model, physics or endpoint claim."""
import copy
import json
from pathlib import Path

import pytest

from go2arx5_vla.world_model import sla_dense_native_v5 as dense, sla_native_v1 as original
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from test_sla_dense_native_v5 import admission, save


def named_admission(tmp_path,monkeypatch):
    q,training,m=admission(tmp_path,monkeypatch)
    audit=save(Path(q['target_audit']['path']),dict(schema='SLA_FRESH_NAMED_N20_ALL_WINDOW_READBACK_V5',
        passes=True,read_windows=1220,episodes=20,dataset_id=dense.NAMED_DATASET,manifest_sha256='d'*64))
    b=m['bindings'];b.update(recipe=dense.NAMED_RECIPE,dataset_id=dense.NAMED_DATASET,
        target_audit=audit['sha256'],D300_new=False,old_2EP_reused=0,old_engineering16_reused=0,
        limited_fresh_named_N20_fitting=True)
    state=save(Path(q['complete_state']['path']),dict(schema='SLA_COMPLETE_PEFT_STATE_V1',
        step=3000,cursor=96000,arrays=[{}]*87,bindings=b))
    m.update(original_complete_sha256=state['sha256'],bindings=b)
    package=save(Path(q['package']['path'])/'MANIFEST.private.json',m)
    q['package']['sha256']=package['sha256']
    training.update(schema='SLA_FRESH_N20_FINITE_FM3K_ACTUAL_V5',operation='NAMED_PICK_HOLD_N20_FM_3K',
        noncandidate=False,complete3K=True,fresh_complete_episodes=20,old_2EP_reused=0,D100_new=False,
        D300_new=False,last_checkpoint={'sha256':state['sha256']})
    tr=save(Path(q['training_result']['path']),training)
    owner=save(Path(q['candidate_owner']['path']),dict(schema='SLA_CORE_FRESH_N20_FM3K_RGB_ACTUAL_V5',
        passes=True,complete3K=True,fresh_RGB16_bit_exact=True,step=3000,target_visits=96000,
        source_visits=0,fresh_complete_episodes=20))
    rs=json.loads(Path(q['ordinary_RGB_spec']['path']).read_bytes());rs['package']=q['package']
    rs=save(Path(q['ordinary_RGB_spec']['path']),rs)
    q.update(schema=dense.NAMED_ADMISSION,method=dense.NAMED_METHOD,target_audit=audit,
        complete_state=state,training_result=tr,candidate_owner=owner,ordinary_RGB_owner=owner,
        ordinary_RGB_spec=rs)
    monkeypatch.setattr(original,'closed',lambda pin:json.loads(Path(pin['path']).read_bytes()))
    return q,training,m


def test_actual_n20_family_uses87_arrays_and_its1220_window_audit(tmp_path,monkeypatch):
    q,_,_=named_admission(tmp_path,monkeypatch)
    assert dense.qualification(q,verify_package=False)==q
    # The old two-episode identity cannot consume this new package.
    q['schema']=dense.ADMISSION
    with pytest.raises(ValueError):dense.qualification(q,verify_package=False)


@pytest.mark.parametrize('mutation',['step500','old2ep','wrong_audit','old_owner','teacher','no_fresh_RGB'])
def test_pending_partial_or_relabelled_artifacts_never_admit_native(tmp_path,monkeypatch,mutation):
    q,tr,m=named_admission(tmp_path,monkeypatch)
    if mutation=='step500':
        tr.update(step=500,cursor=16000,complete3K=False)
        q['training_result']=save(Path(q['training_result']['path']),tr)
    elif mutation=='old2ep':
        m['bindings']['dataset_id']='T_PICK_HOLD_TRAIN01_P10_2EP_V4_1'
        q['package']['sha256']=save(Path(q['package']['path'])/'MANIFEST.private.json',m)['sha256']
    elif mutation=='wrong_audit':
        audit=json.loads(Path(q['target_audit']['path']).read_bytes());audit['read_windows']=122
        q['target_audit']=save(Path(q['target_audit']['path']),audit)
    elif mutation=='old_owner':
        o=json.loads(Path(q['candidate_owner']['path']).read_bytes());o['fresh_complete_episodes']=2
        pin=save(Path(q['candidate_owner']['path']),o);q['candidate_owner']=q['ordinary_RGB_owner']=pin
    elif mutation=='teacher':
        m['teacher_required']=True
        q['package']['sha256']=save(Path(q['package']['path'])/'MANIFEST.private.json',m)['sha256']
    else:
        o=json.loads(Path(q['candidate_owner']['path']).read_bytes());o['fresh_RGB16_bit_exact']=False
        pin=save(Path(q['candidate_owner']['path']),o);q['candidate_owner']=q['ordinary_RGB_owner']=pin
    with pytest.raises(ValueError):dense.qualification(q,verify_package=False)


def test_named_train_inputs_are_same_fixed_conditions_as_old_pilot_without_relabelling_it():
    d,pin=dense.declaration(named=True);old,_=dense.declaration()
    assert d['conditions']==old['conditions'] and d['dataset_id']==dense.NAMED_DATASET
    assert d['trained_layouts']==['TRAIN01','TRAIN04','TRAIN07','TRAIN10','TRAIN13','TRAIN16','TRAIN21','TRAIN25']
    path=Path(dense.__file__).resolve().parents[3]/dense.NAMED_DECLARATION
    p=dict(schema=dense.NAMED_SCHEMA,task_learning_role=task.SLA_DENSE_ROLE,protocol=dense.protocol(),
        B1_global_step=3000,population='TRAIN8',method=dense.NAMED_METHOD,source_sha='b'*40,
        policies={dense.NAMED_METHOD:dict(arm=dense.NAMED_METHOD)},preregistration=dict(path=str(path),sha256=pin),
        assignments=[dict(r,native_run=f'n20-{r["layout"]}-r{r["restart"]}',method=dense.NAMED_METHOD,
            split='TRAIN',parent='TASK_LEARNING_UNDER_HAND_'+r['layout'],command=task.COMMAND,
            purpose=dense.NAMED_EXPERIMENT,model_sha256='c'*64)for r in d['conditions']])
    assert len(dense.validate_panel(p))==8
    inp=dict(schema=dense.NAMED_INPUT_SCHEMA,source_sha=p['source_sha'],assignment_slice=[0,8])
    assert dense.selection(inp,p)==p['assignments']
    with pytest.raises(ValueError):dense.selection(dict(inp,assignment_slice=[0,1]),p)
    p['schema']=dense.SCHEMA
    with pytest.raises(ValueError):dense.validate_panel(p)


def test_archived_scene_requires_verified_producer_digest_and_still_binds_model_bytes(tmp_path):
    from go2arx5_vla.world_model import task_learning_scene_v1 as scene
    from go2arx5_vla.world_model.storage import digest
    model=tmp_path/'not-a-real-model';model.write_bytes(b'metadata fixture only')
    proof=dict(schema='TASK_LEARNING_NEAR_WORKSPACE_STATIC_V1',status='STATIC_ONLY_PASSED_NEW_NATIVE_ADMISSION_REQUIRED',
        builder_sha256='a'*64,layout='TRAIN01',geometry=scene.geometry('TRAIN01'),parent=scene.scene_id('TRAIN01'),
        split='TRAIN',saved_startup_pose_checks=5703,robot_contract=dict(status='COURSE_ROBOT_DYNAMICS_AND_NATIVE_SAFETY_PROPERTIES_EXACT'),
        compiled=dict(base_scene_sha256=scene.BASE_SCENE,fresh_XML_compilation=True,
            original_robot_contract_exact=True,scene_sha256=digest(model)),
        physics_integrations=0,new_transitions=0,native_execution_authorized=False,WBC_changes=False,physics_dt_s=.002)
    spec=dict(source_files={scene.MEMBER:'b'*64},scene_id=proof['parent'],data_parent=proof['parent'],
        data_role='TRAIN',task_learning_role=task.SLA_DENSE_ROLE,model_path=str(model))
    with pytest.raises(ValueError):scene.validate_static_provenance(spec,proof,expected_role=task.SLA_DENSE_ROLE)
    assert scene.validate_static_provenance(spec,proof,expected_role=task.SLA_DENSE_ROLE,
        verified_builder_sha256='a'*64)==proof
    model.write_bytes(b'changed model')
    with pytest.raises(ValueError):scene.validate_static_provenance(spec,proof,
        expected_role=task.SLA_DENSE_ROLE,verified_builder_sha256='a'*64)

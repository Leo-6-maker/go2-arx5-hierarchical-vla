import copy
import json
from pathlib import Path

import pytest

from go2arx5_vla.world_model import sla_dense_native_v5 as dense
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_live_pair_v1 as pair
from go2arx5_vla.world_model import stable_task_prefix_v1 as prefix
from go2arx5_vla.world_model import native_live_control_v3 as live
from go2arx5_vla.world_model import sla_native_v1
from go2arx5_vla.world_model.storage import digest


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path),sha256=digest(path))


def panel():
    d,pin=dense.declaration()
    p=Path(dense.__file__).resolve().parents[3]/dense.DECLARATION
    method=dense.METHODS[0]
    return dict(schema=dense.SCHEMA,task_learning_role=task.SLA_DENSE_ROLE,protocol=dense.protocol(),
        population='TRAIN8',method=method,B1_global_step=3000,source_sha='a'*40,
        policies={method:dict(arm=method)},preregistration=dict(path=str(p),sha256=pin),
        assignments=[dict(row,native_run=f'new-{row["layout"]}-r{row["restart"]}',method=method,
            parent='TASK_LEARNING_UNDER_HAND_'+row['layout'],split='TRAIN',model_sha256='b'*64,
            command=task.COMMAND,purpose=dense.EXPERIMENT)for row in d['conditions']])


def test_dense_p10_clock_is_separate_from_original_r1_clock():
    spec=dict(phase=task.PHASE,kind=task.KIND,task_learning_role=task.SLA_DENSE_ROLE,
        core_local_protocol=task.SLA_DENSE_ROLE,task_learning_grasp=dense.protocol(),
        task_command=task.COMMAND,live_skill_start_step=task.CALIBRATION_B,max_ticks=dense.ENDPOINT,
        physics_dt_s=.002,WBC_changes=False,registered_root='/not-executed')
    task.declared(spec)
    assert prefix.observation_steps(spec)==tuple(task.B+250*i for i in range(60))
    assert prefix.cycle(spec,59)['atoms']==10
    assert prefix.cycle(spec,60)==dict(index=60,kind='DETERMINISTIC_TERMINATION_HOLD',
        start_step=20703,end_step=21703,atoms=40)
    assert pair.query_count(spec)==60 and pair.query_count(dict(core_local_protocol=task.SLA_ROLE))==15
    assert pair.request_path(spec,59).name=='59.json'
    with pytest.raises(PermissionError):pair.request_path(spec,60)
    assert live.maximum_wall_seconds(spec,1200)==3600
    assert task.protocol(task.SLA_ROLE)['prefix_atoms']==40


def test_eight_conditions_are_fixed_before_outputs_and_cannot_be_selected():
    p=panel()
    assert len(dense.validate_panel(p))==8
    inp=dict(schema=dense.INPUT_SCHEMA,source_sha=p['source_sha'],assignment_slice=[0,8])
    assert dense.selection(inp,p)==p['assignments']
    with pytest.raises(ValueError):dense.selection(dict(inp,assignment_slice=[0,1]),p)
    for change in ('order','seed','old_step','DEV'):
        bad=copy.deepcopy(p)
        if change=='order':bad['assignments'].reverse()
        elif change=='seed':bad['assignments'][0]['query_seeds'][0]+=1
        elif change=='old_step':bad['B1_global_step']=500
        else:bad['assignments'][0]['split']='DEV'
        with pytest.raises(ValueError):dense.validate_panel(bad)


def admission(tmp_path,monkeypatch):
    owner=tmp_path/'candidate-owner';rgbroot=owner/'fresh_RGB';pkg=tmp_path/'package'
    parent=save(tmp_path/'base/MANIFEST.json',{'fixture':'immutable-parent'})
    params=tmp_path/'base/model/params';params.mkdir(parents=True)
    audit=save(tmp_path/'target-audit.json',dict(passes=True,read_windows=122,manifest_sha256='d'*64))
    b=dict(codec_version='SLA_GRIP_LOGIT_V2',recipe=dense.RECIPES[dense.METHODS[1]],
        dataset_id='T_PICK_HOLD_TRAIN01_P10_2EP_V4_1',dense_manifest_sha256='d'*64,
        derived_statistics_sha256='s'*64,target_audit=audit['sha256'],D100_new=False,PICK_PLACE_admitted=False)
    state=save(owner/'candidate3000/work/step3000/COMPLETE.json',dict(schema='SLA_COMPLETE_PEFT_STATE_V1',
        step=3000,cursor=96000,arrays=[{}]*105,bindings=b))
    training=dict(schema='SLA_CLEAN_BASE_DENSE_SPATIAL3K_ACTUAL_V5',step=3000,cursor=96000,
        exit_code=0,source_visits=0,physical_requests=0,last_checkpoint={'sha256':state['sha256']},
        actual=dict(optimizer_updates=3000,completed_training_visits=96008,
                    spatial_cache_training_visits=96008,teacher_model_loads=0))
    tr=save(owner/'candidate3000/RESULT.private.json',training)
    inventory=save(owner/'candidate3000/work/PARAMETERS.private.json',dict(
        parameters=[dict(shape=[5],numel=5,dtype='float32')]))
    manifest=dict(schema='SLA_GRIP_LOGIT_ORDINARY_RGB_INFERENCE_PACKAGE_V3',codec='SLA_GRIP_LOGIT_V2',
        label_epsilon=1e-4,step=3000,cursor=96000,qualification_only=False,
        original_complete_sha256=state['sha256'],bindings=b,parent_manifest=parent,
        statistics_sha256='s'*64,image_size=[224,224],image_masks=[True,True,False],
        action_horizon=40,label_interval_s=.05,sampler_steps=10,
        endpoint_basis_applied=False,state_gripper_transformed=False,teacher_required=False,
        training_cache_required=False,optimizer_required=False,
        decode_order='INVERSE_LATENT_NORMALIZATION_THEN_SIGMOID_ONCE_THEN_ORIGINAL_NATIVE_NOMINAL_MAPPING')
    m=save(pkg/'MANIFEST.private.json',manifest);package=dict(path=str(pkg),sha256=m['sha256'])
    rgb=save(rgbroot/'REPLAY_RESULT.private.json',dict(passes=True,all_bit_exact=True,
        ordinary_RGB_generations=16,records=[{}]*16,actual_parent_loads=1,optimizer_updates=0,
        physical_requests=0,teacher_loaded=False,optimizer_loaded=False,
        GPU_memory=dict(peak_bytes_in_use=20,bytes_limit=30)))
    runtime=dict(python='/fixture-python',upstream_root='/fixture-upstream',JAX_memory_fraction=.8)
    rs=save(rgbroot/'EXECUTION_SPEC.json',dict(package=package,GPU_uuid='fixture-GPU',
        parent_params=str(params),allocator_limit_bytes=30,neural_owner_lock='/fixture/GPU0.lock',**runtime))
    ownerpin=save(owner/'RESULT.private.json',dict(complete3K=True,passes=True,candidate_updates=3000,source_visits=0))
    closed_value=json.loads(Path(ownerpin['path']).read_bytes())
    monkeypatch.setattr(sla_native_v1,'closed',lambda pin:closed_value)
    q=dict(schema=dense.ADMISSION,method=dense.METHODS[1],old_DEV_TEST_admitted=False,
        candidate_owner=ownerpin,ordinary_RGB_owner=ownerpin,training_result=tr,
        ordinary_RGB_reconstruction=rgb,ordinary_RGB_spec=rs,complete_state=state,package=package,
        dense_manifest_sha256='d'*64,target_audit=audit,parent_manifest=parent,parent_params=str(params),
        runtime=runtime,GPU_uuid='fixture-GPU',allocator_limit_bytes=30,neural_owner_lock='/fixture/GPU0.lock',
        parameter_inventory=inventory,ordinary_parameter_bytes=20)
    return q,training,manifest


def test_actual_3k_metadata_allows_accounted_partial_visits_but_not_qualification(tmp_path,monkeypatch):
    q,training,_=admission(tmp_path,monkeypatch)
    assert dense.qualification(q,verify_package=False)==q
    training.update(step=4,cursor=128)
    q['training_result']=save(Path(q['training_result']['path']),training)
    with pytest.raises(ValueError,match='REAL_COMPLETE3K'):
        dense.qualification(q,verify_package=False)


def test_endpoint_basis_or_teacher_features_cannot_enter_raw_dense_deployment(tmp_path,monkeypatch):
    q,_,manifest=admission(tmp_path,monkeypatch)
    manifest['endpoint_basis_applied']=True
    m=save(Path(q['package']['path'])/'MANIFEST.private.json',manifest);q['package']['sha256']=m['sha256']
    with pytest.raises(ValueError,match='WITHOUT_ENDPOINT_BASIS'):
        dense.qualification(q,verify_package=False)


def test_student_dispatch_uses_raw_curve_and_next_query_after_ten_atoms(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from scripts.pi0292 import relay_protocol
    from go2arx5_vla.world_model import task_learning_student_v1 as student
    from go2arx5_vla.world_model import fusion_observation,task_learning_local_policy_v1
    raw=np.repeat(pose10(np.eye(4),.6)[None],40,axis=0)
    raw[:,0]=.0005*np.sin(np.linspace(0,np.pi,40))**2
    action=tmp_path/'ACTION.npz';np.savez(action,physical_H40=raw,normalized=np.zeros((1,40,32),np.float32))
    observation=save(tmp_path/'observation.json',dict(original_shadow_ABI=dict(
        source_policy_view=dict(T_source_from_target=np.eye(4).tolist()))))
    calls=[]
    def query(adapter,capture,index):
        calls.append(index);return action,digest(action),{'fixture':'no actual model call'}
    monkeypatch.setattr(pair,'query_from_native',query)
    monkeypatch.setattr(relay_protocol,'unseal',lambda value:value)
    monkeypatch.setattr(fusion_observation,'shadow_inference_payload',lambda value:dict(state_eef10=pose10(np.eye(4),.6)))
    monkeypatch.setattr(task_learning_local_policy_v1,'execution_stage',lambda decoded,sample:'STUDENT')
    policy=student.StudentPolicy.__new__(student.StudentPolicy)
    policy.adapter=SimpleNamespace(spec=dict(task_learning_role=task.SLA_DENSE_ROLE,task_learning_grasp=dense.protocol()),output=tmp_path)
    policy.index=0;policy.consumed_reference=None;policy.observation=observation
    base=np.array([0.,0.,.3,1.,0.,0.,0.])
    for index in (0,1):
        policy.observation=observation
        program,proposal=policy.decide(dict(native_step=task.B+250*index,tcp=np.eye(4)),np.eye(4),base)
        np.testing.assert_array_equal(program.decoded_H40,raw)
        assert proposal['original_H40_interior_preserved'] and not proposal['endpoint_basis_applied']
        assert policy.index==index+1
    assert calls==[0,1]


def test_dense_runtime_selects_the_reconstructed_card_and_its_owner(tmp_path,monkeypatch):
    from go2arx5_vla.world_model.sla_live_v1 import apply_inference_spec
    q,_,_=admission(tmp_path,monkeypatch)
    old=dict(GPU_uuid='old-GPU1',neural_owner_lock='/fixture/GPU1.lock')
    apply_inference_spec(old,q)
    assert old['GPU_uuid']=='fixture-GPU' and old['neural_owner_lock']=='/fixture/GPU0.lock'
    assert old['memory_execution_reference']==q['ordinary_RGB_reconstruction']
    q['neural_owner_lock']='/wrong-unqualified-owner'
    with pytest.raises(ValueError,match='DEVICE_ENVIRONMENT_AND_CAPACITY'):
        dense.qualification(q,verify_package=False)


def test_complete_p10_audit_requires_all_consumed_prefixes_not_full_raw_suffixes():
    from go2arx5_vla.world_model.task_learning_grasp_native_v1 import complete_consumption
    spec=dict(phase=task.PHASE,kind=task.KIND,task_learning_role=task.SLA_DENSE_ROLE,
        task_learning_grasp=dense.protocol(),task_command=task.COMMAND,
        live_skill_start_step=task.CALIBRATION_B,max_ticks=dense.ENDPOINT,physics_dt_s=.002,WBC_changes=False)
    windows=[dict(native_start_step=step,native_end_step=step+25,actual_ticks=25,
        solver_policies_with_exact_installed_target=True,gripper=dict(command_consumed=True))
        for step in range(2503,dense.ENDPOINT,25)]
    cycles=[]
    for index in range(-1,61):
        c=prefix.cycle(spec,index);offset=(c['start_step']-2503)//25
        cycles.append(dict(cycle=c,windows=windows[offset:offset+c['atoms']],complete_suffix_atoms=c['atoms'],
            solver_target_confirmed_atoms=c['atoms'],gripper_confirmed_atoms=c['atoms'],
            cancellation_status='SUPERSEDED_BY_ORIGINAL_NEW_REFERENCE_ACK'))
    result=dict(schema=dense.NATIVE_SCHEMA,diagnostic_control=None,teacher_decisions=[],
        GR00T_queries=0,SLA_queries=60,SLA_dense_H40_prefixes_consumed=True,
        SLA_dense_complete_raw_H40_consumed=False,student_commands_are_expert_labels=False,
        generated_cycles=cycles,student_decisions=[{}]*60,windows=windows,expert_training_eligible=False)
    assert complete_consumption(spec,result)
    result['SLA_dense_complete_raw_H40_consumed']=True
    assert not complete_consumption(spec,result)
    result['SLA_dense_complete_raw_H40_consumed']=False
    result['generated_cycles'][31]['windows']=result['generated_cycles'][31]['windows'][:-1]
    assert not complete_consumption(spec,result)


def test_pair_factory_sets_dense_endpoint_and_sixty_slots_without_expert_controls(tmp_path,monkeypatch):
    from go2arx5_vla.world_model import task_learning_student_batch_v1 as batch
    q,_,_=admission(tmp_path,monkeypatch)
    q['runtime'].update(initial_function_rule={'path':'/fixture/rule','sha256':'f'*64},
                        compilation_cache_dir='/fixture/cache')
    qpin=save(tmp_path/'qualification.json',q)
    p=panel();p['method']=q['method']
    for value in p['assignments']:value['method']=q['method']
    row=p['assignments'][0]
    policy=dict(arm=q['method'],package=q['package'],Native_body_qualification=qpin)
    p['policies']={q['method']:policy}
    scene=save(tmp_path/'STATIC.json',dict(layout=row['layout'],parent=row['parent'],
                                        compiled=dict(scene_sha256=row['model_sha256'])))
    inp=dict(source_root=str(tmp_path/'source'),source_sha=p['source_sha'],source_files={},
        scenes={row['layout']:scene},pilot1_binding='/fixture/P1',pilot1_binding_sha256='a'*64,
        fusion_binding='/fixture/Fusion',fusion_binding_sha256='b'*64,panel='/fixture/panel',
        panel_sha256='c'*64,scene_allowlist='/fixture/allowlist',scene_allowlist_sha256='d'*64,
        memory_execution='fixture',memory_execution_reference={})
    nbase=dict(native_environment={},scope='WM02_PILOT1_SIM_ONLY',kind=task.KIND,phase=task.PHASE,
        task_command=task.COMMAND,live_skill_start_step=task.CALIBRATION_B,max_ticks=task.E,
        physics_dt_s=.002,WBC_changes=False,task_learning_teacher_mode='NORMAL_TEACHER_V7',
        expert_eef_stream_v2={'fixture':'expert-only'},corelab_world_binding_v3={'fixture':'expert-owner'},
        native_startup_read_wait_s=600,native_live_control_v3={'fixture':'expert-only'})
    gbase=dict(checkpoint_files={},neural_owner_lock='/fixture/old-GPU1.lock',GPU_uuid='old-GPU1')
    n,g=batch.specs(inp,p,row,nbase,gbase,tmp_path/'native',tmp_path/'gpu',10**11)
    task.declared(n)
    assert n['max_ticks']==21703 and n['wall_limit_s']==3600
    assert g['max_queries']==60 and g['GPU_uuid']=='fixture-GPU'
    assert g['neural_owner_lock']=='/fixture/GPU0.lock'
    assert not any(key in n for key in ('task_learning_teacher_mode','expert_eef_stream_v2',
                                       'corelab_world_binding_v3','native_startup_read_wait_s','native_live_control_v3'))

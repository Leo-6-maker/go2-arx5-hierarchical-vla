"""Finite serving fixtures; these do not qualify CUDA, Core or native DEV."""
import json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from go2arx5_vla.world_model import task_learning_live_pair_v1 as pair
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import target_gr00t_profile_v1 as profile
from go2arx5_vla.world_model.storage import digest,write_json
from scripts.pi0292.relay_protocol import digest as document_hash
from scripts.pi0294.numerics import MODE


@pytest.fixture
def configuration(tmp_path,monkeypatch):
    native=tmp_path/'native';native.mkdir();(native/'native-01').mkdir()
    gpu=tmp_path/'gpu';gpu.mkdir();checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'mock-weights').write_bytes(b'MOCK NEVER LOAD')
    files={'mock-weights':digest(checkpoint/'mock-weights')}
    monkeypatch.setattr(pair,'CHECKPOINT_INVENTORY',document_hash(files))
    monkeypatch.setattr(pair,'memory_contract',lambda s:None)
    policy=dict(arm='B0',parent_manifest_sha256=profile.PARENT,package=None,Core_qualification=None)
    common=dict(source_sha='a'*40,task_learning_grasp=task.protocol(task.STUDENT_DEV_ROLE),
        target_gr00t_policy=policy,target_gr00t_pair_id='fixture-pair',
        target_gr00t_query_seeds=list(range(10,25)),deadline_epoch=time.time()+60)
    producer=dict(common,phase=task.PHASE,kind=task.KIND,registered_root=str(native),
        task_learning_role=task.STUDENT_DEV_ROLE,task_command=task.COMMAND,
        live_skill_start_step=task.CALIBRATION_B,max_ticks=task.E,physics_dt_s=.002,WBC_changes=False,
        target_gr00t_gpu_root=str(gpu))
    write_json(native/'EXECUTION_SPEC.json',producer)
    spec=dict(common,profile=pair.PROFILE,kind=pair.KIND,execution_site='CORE_FUSION_NEURAL',physics=False,
        device='cuda',max_queries=15,GR00T_updates=0,WM_updates=0,Actor_updates=0,
        numeric_mode=MODE,baseline_changed=False,checkpoint_manifest_sha256=profile.PARENT,
        checkpoint_files=files,checkpoint=str(checkpoint),registered_root=str(gpu),
        producer_native_spec=dict(path=str(native/'EXECUTION_SPEC.json'),sha256=digest(native/'EXECUTION_SPEC.json')))
    write_json(gpu/'EXECUTION_SPEC.json',spec)
    return spec,producer


def test_fifteen_current_slots_and_no_original_two_query_or_teacher_role(configuration):
    s,p=configuration;assert pair.validate_spec(s)==p
    for i in range(15):
        step=task.B+1000*i
        assert pair.request_body(s,dict(native_step=step,native_time_s=step*.002),i)['seed']==10+i
    for i in(-1,15,True):
        with pytest.raises(PermissionError):pair.request_path(p,i)
    with pytest.raises(PermissionError):pair.validate_spec(dict(s,max_queries=2))
    with pytest.raises(PermissionError):pair.request_body(s,dict(native_step=task.B,native_time_s=task.B*.002),1)
    policy=dict(s['target_gr00t_policy'],package={'path':'wrong','sha256':'f'*64})
    with pytest.raises(PermissionError,match='B0_ORIGINAL'):pair.policy_contract(policy,s['checkpoint_files'])


def install_model(configuration,tmp_path,monkeypatch,*,fail=None,early=None):
    s,p=configuration;out=Path(s['registered_root'])/'fit-01';out.mkdir()
    from go2arx5_vla.world_model import fusion_paused_query_bridge as bridge
    from scripts.pi0294 import numerics
    from scripts.wm02 import infer_fusion_paused_vla as original
    monkeypatch.setattr(bridge,'_registered',lambda *a,**k:{})
    monkeypatch.setattr(bridge,'_live',lambda *a,**k:None)
    monkeypatch.setattr(numerics,'prepare_environment',lambda *a:None)
    monkeypatch.setattr(numerics,'configure_torch',lambda *a:None)
    monkeypatch.setattr(pair,'native_terminal',lambda *a:None)
    calls=[];instance=SimpleNamespace(identity=dict(policy=s['target_gr00t_policy']))
    fake=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:8,max_memory_reserved=lambda:16))
    def load(*args):calls.append(('load',id(instance)));return instance,fake
    monkeypatch.setattr(pair,'_load',load)
    def wait(spec,producer,index):
        if index==early:return None,dict(worker_exit_code=1,result_sha256='e'*64)
        path=pair.request_path(producer,index);path.parent.mkdir(exist_ok=True)
        step=task.B+1000*index
        write_json(path,pair.request_body(s,dict(path='fixture',sha256='f'*64,
            native_step=step,native_time_s=step*.002),index))
        return path,None
    monkeypatch.setattr(pair,'wait_request',wait)
    def read(spec,producer,index):return json.loads(pair.request_path(producer,index).read_bytes()),dict(index=index)
    monkeypatch.setattr(pair,'read_request',read)
    def generate(model,payload,seed,torch):
        calls.append(('query',id(model),payload['index'],seed))
        if payload['index']==fail:raise RuntimeError('LOST_DRAW_NEVER_RETRY')
        return dict(identity=instance.identity,physics_integrations=0,extra_correction=False,
            physical_H40=np.full((40,10),seed/100.),normalized=np.full((1,40,132),1.25),
            noise=dict(shape=[1,40,132],dtype='torch.float32',sha256='a'*64)),{},dict(query_wall_s=.01)
    monkeypatch.setattr(original,'_generate_once',generate)
    return s,p,out,calls


def test_one_resident_model_all_fifteen_fresh_queries(configuration,tmp_path,monkeypatch):
    s,p,out,calls=install_model(configuration,tmp_path,monkeypatch)
    pair.infer(s,out)
    result=json.loads((out/'RESULT.private.json').read_bytes())
    assert result['actual_model_loads']==1 and result['actual_VLA_inferences']==15
    assert result['Processor_decode_calls']==15 and result['unused_query_slots']==0
    assert len({c[1]for c in calls})==1 and len(calls)==16
    with pytest.raises(FileExistsError):pair.infer(s,out)
    assert len(calls)==16


def test_native_early_stop_preserves_only_actual_queries(configuration,tmp_path,monkeypatch):
    s,_,out,calls=install_model(configuration,tmp_path,monkeypatch,early=2)
    pair.infer(s,out)
    result=json.loads((out/'RESULT.private.json').read_bytes())
    assert result['actual_VLA_inferences']==2 and result['unused_query_slots']==13
    assert result['actual_early_native_terminal']['worker_exit_code']==1
    assert len(calls)==3 and not(out/'queries/02').exists()


def test_lost_model_draw_retains_completed_results_and_is_never_replayed(configuration,tmp_path,monkeypatch):
    s,_,out,calls=install_model(configuration,tmp_path,monkeypatch,fail=1)
    with pytest.raises(RuntimeError,match='LOST_DRAW'):pair.infer(s,out)
    failed=json.loads((out/'FAILED.private.json').read_bytes())
    assert failed['generation_invocations']==2 and failed['completed_query_records']==1
    assert failed['query_completion_unknown_after_intent']is True
    assert(out/'queries/00/RESULT.json').exists()and not(out/'queries/01/RESULT.json').exists()
    with pytest.raises(FileExistsError):pair.infer(s,out)
    assert len(calls)==3


def test_sft_requires_actual_forty_query_core_reconstruction_with_same_package(tmp_path):
    from go2arx5_vla.world_model.target_gr00t_inference_v1 import CROSS_SITE_ATOL
    root=tmp_path/'Core';(root/'control').mkdir(parents=True);(root/'fit-01').mkdir()
    package=dict(path='/fixture/SFT',sha256='f'*64)
    spec=dict(registered_root=str(root),candidate_package=package,
        profile='wm02-task-learning-package-core-v1',kind='TASK_LEARNING_PACKAGE_CORE_RECONSTRUCTION_V1',
        target_inference_mode='CONTINUOUS_TASK_PACKAGE_RECONSTRUCTION_TRAIN40')
    sp=root/'EXECUTION_SPEC.json';write_json(sp,spec)
    write_json(root/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,spec_sha256=digest(sp)))
    write_json(root/'WORKER_EXIT.private.json',dict(exit_code=0,failure=None))
    result=dict(actual_model_loads=1,actual_VLA_queries=40,Processor_decode_calls=40,
        target_native_execution=False,model_identity=dict(candidate_package=package,global_step=2000),
        GR00T_updates=0,WM_updates=0,Actor_updates=0)
    rp=root/'fit-01/RESULT.private.json';write_json(rp,result)
    proof=dict(schema='TASK_LEARNING_CORE_PACKAGE_QUALIFICATION_V1',package=package,
        numerical=dict(status='PASS_ACTUAL_HOPPER_CORE_MATCHED_INPUT_NOISE_NUMERICAL_GATE',
            absolute_component_tolerances=CROSS_SITE_ATOL,actual_matched_queries=40,
            actual_maximum_absolute_error={k:0. for k in CROSS_SITE_ATOL}),
        Core_spec=dict(path=str(sp),sha256=digest(sp)),
        Core_result=dict(path=str(rp),sha256=digest(rp)))
    qp=tmp_path/'qualification.json';write_json(qp,proof)
    policy=dict(arm='B1',parent_manifest_sha256=profile.PARENT,package=package,
        Core_qualification=dict(path=str(qp),sha256=digest(qp)))
    pair.policy_contract(policy,{},verify_package=False)
    result['actual_VLA_queries']=4
    rp.write_text(json.dumps(result));proof['Core_result']['sha256']=digest(rp)
    qp.write_text(json.dumps(proof));policy['Core_qualification']['sha256']=digest(qp)
    with pytest.raises(PermissionError,match='FORTY_QUERY_RECONSTRUCTION'):
        pair.policy_contract(policy,{},verify_package=False)


def test_closed_native_rejection_ends_unused_slots_without_requiring_success(configuration,tmp_path):
    s,p=configuration;root=Path(p['registered_root']);(root/'control').mkdir()
    lease=root/'control/LEASE.json'
    write_json(lease,dict(status='ACTIVE'))
    assert pair.native_terminal(p)is None
    write_json(root/'WORKER_EXIT.private.json',dict(exit_code=1))
    write_json(root/'native-01/RESULT.json',dict(status='NATIVE_STAGE_FAILED'))
    lease.write_text(json.dumps(dict(status='CLOSED',run_uuid=root.name,worker_exit_code=1,
        spec_sha256=digest(root/'EXECUTION_SPEC.json'))))
    terminal=pair.native_terminal(p)
    assert terminal['worker_exit_code']==1
    path,actual=pair.wait_request(s,p,2)
    assert path is None and actual==terminal

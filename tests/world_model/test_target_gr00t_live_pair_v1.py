"""Synthetic pair/one-draw tests. No CUDA, native world or real GR00T load."""
import copy,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from go2arx5_vla.world_model import target_gr00t_live_pair_v1 as pair
from go2arx5_vla.world_model import target_gr00t_prefix_v1 as prefix
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
    policy=dict(arm='B0',parent_manifest_sha256=profile.PARENT,package=None,Core_qualification=None)
    producer=dict(phase=prefix.PHASE,kind=prefix.KIND,registered_root=str(native),
        source_sha='a'*40,target_gr00t_prefix_protocol=prefix.protocol(),
        live_skill_start_step=prefix.B,max_ticks=prefix.E,physics_dt_s=.002,WBC_changes=False,
        target_gr00t_gpu_root=str(gpu),target_gr00t_policy=policy,target_gr00t_pair_id='mock-pair',
        target_gr00t_query_seeds=[10,11],deadline_epoch=time.time()+60)
    write_json(native/'EXECUTION_SPEC.json',producer)
    s=dict(profile=pair.PROFILE,kind=pair.KIND,execution_site='CORE_FUSION_NEURAL',physics=False,
        device='cuda',max_queries=2,GR00T_updates=0,WM_updates=0,Actor_updates=0,
        numeric_mode=MODE,baseline_changed=False,checkpoint_manifest_sha256=profile.PARENT,
        checkpoint_files=files,checkpoint=str(checkpoint),target_gr00t_prefix_protocol=prefix.protocol(),
        producer_native_spec=dict(path=str(native/'EXECUTION_SPEC.json'),sha256=digest(native/'EXECUTION_SPEC.json')),
        registered_root=str(gpu),source_sha='a'*40,target_gr00t_policy=policy,
        target_gr00t_pair_id='mock-pair',target_gr00t_query_seeds=[10,11],deadline_epoch=time.time()+60)
    write_json(gpu/'EXECUTION_SPEC.json',s)
    return s,producer


def test_exact_pair_and_original_B0(configuration):
    s,p=configuration;assert pair.validate_spec(s)==p
    changed=copy.deepcopy(s);changed['target_gr00t_policy']['package']={'path':'not-baseline','sha256':'f'*64}
    with pytest.raises(PermissionError,match='B0_ORIGINAL'):pair.policy_contract(changed['target_gr00t_policy'],s['checkpoint_files'])


@pytest.mark.parametrize('key,value',[
    ('GR00T_updates',1),('WM_updates',200),('Actor_updates',1),('physics',True),
    ('max_queries',3),('baseline_changed',True),('target_gr00t_pair_id','another-pair'),
    ('target_gr00t_query_seeds',[11,10]),('source_sha','b'*40),('checkpoint_files',{})])
def test_no_extra_queries_changed_model_or_other_native_pair(configuration,key,value):
    s,_=configuration
    with pytest.raises(PermissionError):pair.validate_spec(dict(s,**{key:value}))


def test_second_query_cannot_reuse_first_observation_clock(configuration):
    s,_=configuration
    first=dict(path='synthetic',sha256='f'*64,native_step=34303,native_time_s=34303*.002)
    assert pair.request_body(s,first,0)['expected_native_step']==34303
    with pytest.raises(PermissionError,match='FRESH_OBSERVATION'):pair.request_body(s,first,1)
    second=dict(first,native_step=37303,native_time_s=37303*.002)
    assert pair.request_body(s,second,1)['seed']==11
    for i in (-1,2,True):
        with pytest.raises(PermissionError,match='FINITE_REQUEST'):pair.request_body(s,second,i)


@pytest.mark.parametrize('endpoint',[2000,4000,8000])
def test_B1_requires_matching_real_reconstruction_identity_and_original_tolerances(tmp_path,configuration,monkeypatch,endpoint):
    from go2arx5_vla.world_model.target_gr00t_inference_v1 import CROSS_SITE_ATOL
    from go2arx5_vla.world_model import target_gr00t_delivery_v1 as delivery
    s,_=configuration;root=tmp_path/'Core-reconstruction';(root/'control').mkdir(parents=True);(root/'fit-01').mkdir()
    package=dict(path='/synthetic/candidate',sha256='f'*64)
    cs=dict(registered_root=str(root),candidate_package=package,
        target_inference_mode='FORMAL_PACKAGE_RECONSTRUCTION_TRAIN4',
        target_dataset_sha256='d'*64,target_statistics_sha256='e'*64)
    write_json(root/'EXECUTION_SPEC.json',cs)
    write_json(root/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,spec_sha256=digest(root/'EXECUTION_SPEC.json')))
    write_json(root/'WORKER_EXIT.private.json',dict(exit_code=0,failure=None))
    result=dict(actual_model_loads=1,actual_VLA_queries=4,Processor_decode_calls=4,
        target_native_execution=False,model_identity=dict(candidate_package=package,global_step=endpoint),
        GR00T_updates=0,WM_updates=0,Actor_updates=0)
    rp=root/'fit-01/RESULT.private.json';write_json(rp,result)
    proof=dict(schema='WM02_TARGET_GR00T_CORE_PACKAGE_QUALIFICATION_V1',package=package,
        numerical=dict(status='PASS_ACTUAL_A40_CORE_MATCHED_INPUT_NOISE_NUMERICAL_GATE',
            absolute_component_tolerances=CROSS_SITE_ATOL,actual_matched_queries=4,
            actual_maximum_absolute_error={k:0. for k in CROSS_SITE_ATOL}),
        Core_spec=dict(path=str(root/'EXECUTION_SPEC.json'),sha256=digest(root/'EXECUTION_SPEC.json')),
        Core_result=dict(path=str(rp),sha256=digest(rp)))
    qp=tmp_path/'qualification.json';write_json(qp,proof)
    p=dict(arm='B1',parent_manifest_sha256=profile.PARENT,package=package,
        Core_qualification=dict(path=str(qp),sha256=digest(qp)))
    monkeypatch.setattr(delivery,'verify',lambda *args:dict(global_step=endpoint,recipe=dict(dataset_sha256='d'*64,statistics_sha256='e'*64)))
    pair.policy_contract(p,s['checkpoint_files'],expected_global_step=endpoint)
    if endpoint!=2000:
        with pytest.raises(PermissionError,match='FOUR_QUERY_RECONSTRUCTION'):
            pair.policy_contract(p,s['checkpoint_files'])
    result['model_identity']['global_step']=0;rp.write_text(json.dumps(result));proof['Core_result']['sha256']=digest(rp)
    qp.write_text(json.dumps(proof));p['Core_qualification']['sha256']=digest(qp)
    with pytest.raises(PermissionError,match='FOUR_QUERY_RECONSTRUCTION'):pair.policy_contract(p,s['checkpoint_files'],expected_global_step=endpoint)
    result['model_identity']['global_step']=endpoint;rp.write_text(json.dumps(result));proof['Core_result']['sha256']=digest(rp)
    proof['numerical']['actual_maximum_absolute_error']['XYZ_m']=.001
    qp.write_text(json.dumps(proof));p['Core_qualification']['sha256']=digest(qp)
    with pytest.raises(PermissionError,match='NUMERICAL_QUALIFICATION'):pair.policy_contract(p,s['checkpoint_files'],expected_global_step=endpoint)


def install_fake_model(configuration,tmp_path,monkeypatch,*,fail_second=False):
    s,p=configuration;output=Path(s['registered_root'])/'fit-01';output.mkdir()
    from go2arx5_vla.world_model import fusion_paused_query_bridge as bridge
    from scripts.pi0294 import numerics
    from scripts.wm02 import infer_fusion_paused_vla as original
    monkeypatch.setattr(bridge,'_registered',lambda *a,**k:{})
    monkeypatch.setattr(bridge,'_live',lambda *a,**k:None)
    monkeypatch.setattr(numerics,'prepare_environment',lambda *a:None)
    monkeypatch.setattr(numerics,'configure_torch',lambda *a:None)
    for i,step in enumerate((34303,37303)):
        path=pair.request_path(p,i);path.parent.mkdir(exist_ok=True)
        o=dict(path='mock-'+str(i),sha256=str(i)*64,native_step=step,native_time_s=step*.002)
        write_json(path,pair.request_body(s,o,i))
    instance=SimpleNamespace(identity=dict(policy=s['target_gr00t_policy']))
    fake=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:8,max_memory_reserved=lambda:16))
    calls=[]
    def load(spec,root):calls.append(('load',id(instance)));return instance,fake
    monkeypatch.setattr(pair,'_load',load)
    def read(spec,producer,index):
        r=json.loads(pair.request_path(producer,index).read_bytes())
        return r,dict(actual_mock_observation=index)
    monkeypatch.setattr(pair,'read_request',read)
    def generate(model,payload,seed,torch):
        calls.append(('query',id(model),payload['actual_mock_observation'],seed))
        if fail_second and seed==11:raise RuntimeError('MOCK_GENERATION_LOST_NO_RETRY')
        return dict(identity=instance.identity,physics_integrations=0,extra_correction=False,
            physical_H40=np.full((40,10),seed/100.),normalized=np.full((1,40,132),1.25),
            noise=dict(shape=[1,40,132],dtype='torch.float32',sha256=str(seed)*32)),{},dict(query_wall_s=.01)
    monkeypatch.setattr(original,'_generate_once',generate)
    return s,p,output,calls


def test_one_resident_model_two_distinct_inputs_no_redecode_or_clipping(configuration,tmp_path,monkeypatch):
    s,p,out,calls=install_fake_model(configuration,tmp_path,monkeypatch)
    pair.infer(s,out)
    assert calls==[('load',calls[0][1]),('query',calls[0][1],0,10),('query',calls[0][1],1,11)]
    final=json.loads((out/'RESULT.private.json').read_bytes())
    assert final['actual_model_loads']==1 and final['actual_VLA_inferences']==final['Processor_decode_calls']==2
    assert final['GR00T_updates']==final['native_integrations']==0
    for i in range(2):
        folder=out/'queries'/f'{i:02d}';r=json.loads((folder/'RESULT.json').read_bytes())
        request=json.loads(pair.request_path(p,i).read_bytes())
        assert pair.response(s,i,request,r,folder)==folder/'ACTION.npz'
        with np.load(folder/'ACTION.npz')as z:assert(z['normalized']==1.25).all()
    with pytest.raises(FileExistsError):pair.infer(s,out)
    assert len(calls)==3


def test_lost_second_generation_retains_first_and_cannot_redraw(configuration,tmp_path,monkeypatch):
    s,_,out,calls=install_fake_model(configuration,tmp_path,monkeypatch,fail_second=True)
    with pytest.raises(RuntimeError,match='LOST_NO_RETRY'):pair.infer(s,out)
    failure=json.loads((out/'FAILED.private.json').read_bytes())
    assert failure['completed_model_loads']==1 and failure['generation_invocations']==2
    assert failure['completed_query_records']==1 and failure['query_completion_unknown_after_intent']is True
    assert (out/'queries/00/ACTION.npz').is_file()and not(out/'queries/01/RESULT.json').exists()
    with pytest.raises(FileExistsError):pair.infer(s,out)
    assert len(calls)==3


def test_actual_wrapper_routes_and_closes_only_owned_neural_lease(tmp_path,monkeypatch):
    from scripts.wm02 import pilot1_neural_worker as worker
    import subprocess
    root=tmp_path/'job';root.mkdir();(root/'control').mkdir();lease=root/'control/LEASE.json';spec=root/'EXECUTION_SPEC.json'
    spec.write_text(json.dumps(dict(registered_root=str(root),lease=str(lease),kind=pair.KIND,
        source_sha='a'*40,wall_limit_s=60,deadline_epoch=time.time()+60)))
    lease.write_text(json.dumps(dict(status='ACTIVE',run_uuid=root.name,spec_sha256=digest(spec))))
    calls=[]
    def child(argv,**kw):calls.append(argv);return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',child)
    assert worker.main(['--spec',str(spec)])==0 and calls[0][3]=='scripts.wm02.infer_target_gr00t_live_v1'
    assert json.loads(lease.read_bytes())['status']=='CLOSED'

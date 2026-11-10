import copy
import numpy as np
import pytest
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_matched_native_v3_1 as m
from go2arx5_vla.world_model.task_learning_matched_policy_v3_1 import METHODS,configuration,compile_proposal


def panel():
    from go2arx5_vla.world_model.task_learning_scene_v1 import scene_id
    return dict(schema=m.SCHEMA,task_learning_role=task.MATCHED_ROLE,protocol=m.protocol(),B1_global_step=2000,
        source_sha='b'*40,preregistration={'sha256':m.CONTRACT_SHA},policies={v:{'arm':v}for v in METHODS},
        assignments=[dict(native_run=f'matched-{i}',layout=l,restart=r,method=k,parent=scene_id(l),split='TRAIN',purpose=m.PURPOSE,
            command=task.COMMAND,query_seeds=list(range(s,s+15)),model_sha256='a'*64)for i,(l,r,k,s)in enumerate(m.ORDER)])


def test_complete_counterbalanced_population_and_old_namespaces_stay_closed():
    from collections import Counter
    from go2arx5_vla.world_model.task_learning_student_batch_v1 import selection
    p=panel();assert len(m.validate_panel(p))==128
    assert Counter(row['method']for row in p['assignments'])=={method:16 for method in METHODS}
    assert p['assignments'][8]['method']==METHODS[1]
    for i in range(0,128,8):
        rows=p['assignments'][i:i+8]
        assert len({tuple(r['query_seeds'])for r in rows})==1 and {r['method']for r in rows}==set(METHODS)
    inp=dict(schema=m.INPUT_SCHEMA,source_sha=p['source_sha'],assignment_slice=[0,8],population_predecessor=None)
    assert selection(inp,p)==p['assignments'][:8]
    from go2arx5_vla.world_model.task_learning_student_v1 import validate as dev
    from go2arx5_vla.world_model.task_learning_endpoint_native_v3 import validate_panel as pilot
    for validator in(dev,pilot):
        with pytest.raises(ValueError):validator(p)
    for key,value in [('query_seeds',list(range(15))),('layout','TRAIN03'),('method','B1')]:
        bad=copy.deepcopy(p);bad['assignments'][0][key]=value
        with pytest.raises(ValueError):m.validate_panel(bad)


def test_all_method_denominators_partial_outcomes_and_no_duplicates():
    p=panel();row=dict(p['assignments'][0],attempted=True,task_success=False,technical_valid=True,hard_safety_stop=False,actual_GR00T_queries=1,actual_native_integrations=5703)
    result=m.summarize(p,[row]);assert result['requested']==128 and result['completed']==1 and not result['complete']
    assert all(v['requested']==16 for v in result['methods'].values())
    assert result['methods'][METHODS[0]]['valid_task_failures']==1
    with pytest.raises(ValueError):m.summarize(p,[row,row])


def test_raw_rejection_unchanged_and_all_smooth_methods_share_one_representation(tmp_path):
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.task_learning_student_v1 import StudentProposalRejected
    state=np.array([0,0,0,1,0,0,0,1,0,1.],float);raw=np.repeat(state[None],40,axis=0);raw[1:-1,0]=.001*(-1.)**np.arange(38);raw[-1,0]=.01
    file=tmp_path/'raw.npz';np.savez(file,physical_H40=raw,normalized=np.zeros((1,40,132)))
    context=dict(source_from_target=np.eye(4),query_state10=state,actual_TCP=np.eye(4),consumed_reference_TCP=np.eye(4),base_hold=[0,0,0,1,0,0,0]);outputs=[]
    for method in METHODS:
        _,basis=configuration(method)
        if not basis:
            with pytest.raises(StudentProposalRejected):compile_proposal(file,digest(file),method=method,consumed_nominal_grip=1.,**context)
        else:
            program,proof=compile_proposal(file,digest(file),method=method,consumed_nominal_grip=1.,**context)
            assert proof['composed_policy_method']==method and not proof['GR00T_raw_H40_consumed']and proof['command_clips']==0
            outputs.append(program.decoded_H40)
    assert all(np.array_equal(v,outputs[0])for v in outputs)
    with np.load(file)as z:assert np.array_equal(z['physical_H40'],raw)


def test_continuation_requires_actual_closed_prefix_not_a_start_index(monkeypatch):
    p=panel();inp=dict(schema=m.INPUT_SCHEMA,source_sha=p['source_sha'],assignment_slice=[3,8],population_predecessor={},panel='panel',panel_sha256='a'*64)
    monkeypatch.setattr(m,'pinned',lambda v:{})
    with pytest.raises((KeyError,ValueError)):m.selection(inp,p)
    outcomes=[dict(row,attempted=True,task_success=False,technical_valid=True,hard_safety_stop=False)for row in p['assignments'][:3]]
    prior=dict(schema='CORE_MATCHED128_POPULATION_ACTUAL_V3_1',panel={'path':'panel','sha256':'a'*64},source_sha=p['source_sha'],cold_continuation_safe=True,outcomes=outcomes,groups=[dict(lease='lease',result='result')],summary=m.summarize(p,outcomes))
    monkeypatch.setattr(m,'pinned',lambda v:dict(status='CLOSED',worker_exit_code=0)if v=='lease'else dict(exit_code=0,unresolved_pair=None)if v=='result'else prior)
    assert m.selection(inp,p)==p['assignments'][3:8]
    prior['summary']['methods'][METHODS[0]]['technical_failures']=1
    with pytest.raises(ValueError):m.selection(inp,p)


@pytest.mark.parametrize('method',METHODS)
def test_weight_and_processor_factory_routes_do_not_mix_controls(monkeypatch,tmp_path,method):
    import sys,types
    from types import SimpleNamespace
    import scripts.wm02.infer_fusion_paused_vla as original
    import go2arx5_vla.world_model.target_gr00t_core_v1 as core
    import go2arx5_vla.world_model.target_gr00t_delivery_v1 as delivery
    import go2arx5_vla.world_model.task_learning_endpoint_native_v3 as endpoint
    import scripts.pi0294.local_backbone as backbone
    q={'SFT2000_package':dict(path='/sealed/SFT2000',sha256='p')};calls=[]
    monkeypatch.setattr(m,'policy_contract',lambda *a,**k:q)
    def frozen(s,r,*,cpu_model_initializer=None):
        model=SimpleNamespace(weights='B0',placement='cpu')
        if cpu_model_initializer:cpu_model_initializer(model)
        model.placement='cuda';return SimpleNamespace(model=model,processor=SimpleNamespace(name='original_B0')),None
    def install(model,path,pin,files):
        assert model.placement=='cpu'and path=='/sealed/SFT2000';model.weights='SFT2000';calls.append('install_SFT')
    monkeypatch.setattr(original,'_load_frozen',frozen);monkeypatch.setattr(delivery,'install',install)
    monkeypatch.setattr(core,'trim_unused_logits',lambda model:None)
    monkeypatch.setattr(backbone,'local_model_name',lambda *a:'local')
    monkeypatch.setattr(endpoint,'load',lambda *a:(SimpleNamespace(model=SimpleNamespace(weights='C500',placement='cuda'),processor=SimpleNamespace(name='target_SFT2000')),None))
    fake=types.ModuleType('gr00t.model.gr00t_n1d7.processing_gr00t_n1d7')
    class Processor:
        @staticmethod
        def from_pretrained(path,**kwargs):
            assert str(path)=='/sealed/SFT2000/processor'
            return SimpleNamespace(name='target_SFT2000',eval=lambda:None)
    fake.Gr00tN1d7Processor=Processor
    monkeypatch.setitem(sys.modules,fake.__name__,fake)
    value={'arm':method};s=dict(target_gr00t_policy=value,checkpoint_files={},registered_root=str(tmp_path),neural_bindings={'cosmos_hub':'local'},numeric_mode='fixed')
    instance,_=m.load(s,tmp_path);family,_=configuration(method)
    assert instance.model.weights==('SFT2000'if family=='SFT2000'else'C500'if family=='C500'else'B0')
    assert instance.processor.name==('original_B0'if family=='B0'else'target_SFT2000')
    assert calls==(['install_SFT']if family=='SFT2000'else[])


def test_original_teacher_math_and_preregistered_protocol_remain_identical():
    from pathlib import Path
    import hashlib
    from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER
    assert teacher_fingerprint()==FROZEN_TEACHER
    assert task.student_role(task.MATCHED_ROLE)and task.protocol(task.MATCHED_ROLE)['physics_dt_s']==.002
    prereg=Path(__file__).resolve().parents[2]/'experiments/wm02/task_learning_core_matched_smooth_v3_1.json'
    assert hashlib.sha256(prereg.read_bytes()).hexdigest()==m.CONTRACT_SHA

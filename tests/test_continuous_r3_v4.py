import pytest
from go2arx5_vla.sla_adapter.continuous_r3_v4 import evaluate
from scripts.sla.prepare_r3_stream_v4 import choose


def rows():
    return [dict(index=i,ACK=i!=0,full_H40_pass=True,input_available_to_terminal_s=5. if i==0 else .2,
        capture_to_terminal_s=5.01 if i==0 else .21,release_lateness_s=.001,
        reservation_wait_capacity_s=0.,raw_output_transport_exact=True,guard_violation_count=0,
        TTL_rejected=i==0,dispatch_monotonic=100+i*.5)for i in range(32)]


def test_cold_expiry_is_preserved_and_not_averaged_into31_hot():
    r=evaluate(rows(),dict(archived=32,peak_pending=4))
    assert r['GO'] and r['warm_n']==31 and r['strict_ACK']==31 and r['TTL_rejections']==1
    assert r['cold']['input_available_to_terminal_s']==5.
    assert r['warm_latency_p50_p95_p99_max_s']==[.2]*4


@pytest.mark.parametrize('fault',['p95','TTL','H40','late_slot','backpressure','archive','guard','transport'])
def test_every_fixed_R3_failure_is_retained_and_blocks_qualification(fault):
    r=rows();archive=dict(archived=32,peak_pending=4)
    if fault=='p95':
        for q in r[1:]:q['input_available_to_terminal_s']=.351
    elif fault=='TTL':r[5]['input_available_to_terminal_s']=.4
    elif fault=='H40':r[5]['full_H40_pass']=False
    elif fault=='late_slot':r[5]['release_lateness_s']=.5
    elif fault=='backpressure':r[5]['reservation_wait_capacity_s']=.001
    elif fault=='archive':archive['archived']=31
    elif fault=='guard':r[5]['guard_violation_count']=1
    else:r[5]['raw_output_transport_exact']=False
    assert not evaluate(r,archive)['GO']


def test_fixed_stream_uses_contiguous16_around_first_close_and_checks_clocks():
    data=[dict(phase='CLOSE'if i==10 else'PREGRASP',native_step=5703+250*i,native_time_s=11.406+.5*i)for i in range(30)]
    assert choose(data)==list(range(4,20))
    data[12]['native_step']+=25
    with pytest.raises(ValueError,match='CLOCK'):choose(data)


def test_controller32_integration_with_synthetic_CPU_transport_and_real_archiver(tmp_path,monkeypatch):
    """Exercise lifecycle/archival end-to-end; no model, site or native calls."""
    import json,os,time
    import numpy as np
    from pathlib import Path
    from scripts.pi0292.durable import write
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import dirs
    from scripts.wm02 import continuous_gpu_control_v4 as m
    root=tmp_path/'run';dirs(root)
    pin=lambda p:dict(path=str(p),sha256=digest(p))
    state=np.array([.4,0.,.3,1.,0.,0.,0.,1.,0.,1.]);reference=np.eye(4);reference[:3,3]=state[:3]
    input_file=tmp_path/'input.npz'
    np.savez(input_file,state_eef10=state,body_rgb=np.zeros((16,16,3),np.uint8),wrist_rgb=np.zeros((16,16,3),np.uint8),
        T_policy_from_mount=np.eye(4),policy_from_world=np.eye(4),accepted_reference_TCP=reference)
    bank=[]
    for i in range(32):
        step=525+250*(i%16)
        bank.append(dict(index=i,family='F1'if i<16 else'F2',episode='cpu-fixture',anchor=i,phase_score_only='PREGRASP',noise=0,
            native_step=step,language='pick',input=pin(input_file),previous_reference=dict(width_m=.042,source_native_s=step*.002-.05,query_native_s=step*.002)))
    bp=tmp_path/'bank.json';write(bp,dict(records=bank))
    monkeypatch.setattr(m,'build',lambda spec,destination:pin(bp))
    def initialize(root):
        live=Path(root)/'live';live.mkdir();return live
    monkeypatch.setattr(m,'initialize',initialize)
    def register(spec,entries):
        for directory,value in entries:write(directory/'EXECUTION_SPEC.json',value)
    monkeypatch.setattr(m,'register',register)
    instances=[]
    class CPUTransport:
        def __init__(self,args,**kwargs):
            self.spec=m.read(args[-1]);self.root=Path(self.spec['registered_root']);self.pid=os.getpid();self.returncode=None;self.calls=0
            write(self.root/'READY.private.json',dict(synthetic_CPU_test=True));instances.append(self)
        def poll(self):return self.returncode
        def wait(self,timeout):
            self.returncode=0;write(self.root/'RESULT.private.json',dict(attempted_generations=self.calls,completed_generations=self.calls));return 0
    monkeypatch.setattr(m.subprocess,'Popen',CPUTransport)
    def receipt(path,**kwargs):
        path=Path(path)
        if path.name=='RESPONSE.json':
            process=instances[0];message=m.read(Path(process.spec['ipc_root'])/'REQUEST.json');q=message['request'];folder=path.parent
            value=dict(normalized=np.zeros((1,40,32)),physical_H40=np.repeat(state[None],40,axis=0))
            np.savez(folder/'SHIFT7_OUTPUT.npz',**value);np.savez(folder/'ACTION.npz',**value)
            write(path,dict(request_id=q['request_id'],bindings=q['bindings'],action=pin(folder/'ACTION.npz'),new_generations_this_request=1,branch_wall_s=[0.]))
            process.calls+=1
        return m.read(path)
    monkeypatch.setattr(m,'wait_receipt',receipt)
    grant=tmp_path/'grant.json';write(grant,dict(user_granted=True,stage='R3',max_requests=32,max_generations=32,native_requests=0))
    r1=tmp_path/'r1.json';write(r1,dict(status='R1_FULL_PASS',selected_architecture='SHIFT7_SINGLE'))
    binding=tmp_path/'site.json';write(binding,dict(paths={}))
    result=m.run(dict(resource_authorization=pin(grant),R1_result=pin(r1),physics=False,new_physical_requests_budget=0,
        new_optimizer_updates_budget=0,new_complete_generations_budget=32,unified_requests_budget=32,selected_combo='SHIFT7_SINGLE',
        registered_root=str(root),source_sha='synthetic-CPU-fixture',parent_params='not-loaded',source_root='not-executed',gpu_environment={},python='not-executed',
        site_binding=pin(binding),request_bindings=dict(synthetic='CPU_fixture'),deadline_epoch=time.time()+90))
    assert result['unified_requests']==32 and instances[0].calls==32
    assert result['archive']['archived']==32 and result['score']['strict_ACK']==32
    assert result['score']['raw_output_H40']==32 and result['score']['GO']
    assert result['new_native_requests']==result['new_updates']==0

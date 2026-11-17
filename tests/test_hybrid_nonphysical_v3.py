import copy
import numpy as np
import pytest
from go2arx5_vla.world_model.hybrid_nonphysical_v3 import Receiver, check_live, identity
from go2arx5_vla.world_model.storage import digest


def fixture(tmp_path):
    state=np.array([.4,0.,.3,1.,0.,0.,0.,1.,0.,1.])
    pin=tmp_path/'input.npz'
    with pin.open('wb') as f:
        np.savez(f,state_eef10=state,body_rgb=np.zeros((16,16,3),np.uint8),
            wrist_rgb=np.ones((16,16,3),np.uint8),T_policy_from_mount=np.eye(4),
            policy_from_world=np.eye(4),accepted_reference_TCP=np.array([[1,0,0,.4],[0,1,0,0],[0,0,1,.3],[0,0,0,1]]))
    r=Receiver(tmp_path/'receiver','one',525,dict(width_m=.042,source_native_s=1.,query_native_s=1.05),
        {'kind':'RECORDED_NATIVE_COMMAND','sha256':'a'*64},{'parent':'b'*64,'EEF':'c'*64,'grip':'d'*64})
    q=r.request(dict(path=str(pin),sha256=digest(pin)),'pick',0)
    action=tmp_path/'action.npz'
    with action.open('wb') as f:
        np.savez(f,physical_H40=np.repeat(state[None],40,axis=0),normalized=np.zeros((1,40,32)))
    response=dict(request_id=q['request_id'],bindings=q['bindings'],action=dict(path=str(action),sha256=digest(action)))
    return r,q,response


def test_real_compiler_atomic_ACK_and_partial_P10(tmp_path):
    r,q,a=fixture(tmp_path); before=copy.deepcopy(r.state['previous'])
    ack=r.install(q,a)
    assert ack==r.install(q,a)  # idempotent, no duplicate effect
    assert r.state['previous']==before
    assert ack['EEF_dimension']==7 and ack['full_H40_compiled']
    result=r.consume(7)
    assert result['consumed_atoms']==7 and not result['full_P10']
    assert result['previous']['source_native_s']==pytest.approx(1.35)
    with pytest.raises(RuntimeError): r.finish()
    with pytest.raises(RuntimeError): r.install(q,a)
    r.cancel()
    with pytest.raises(RuntimeError): r.consume(1)
    assert len(list(r.root.glob('install-*')))==1


def test_full_P10_requires_original_renewal(tmp_path):
    r,q,a=fixture(tmp_path);r.install(q,a)
    r.consume(7)
    with pytest.raises(ValueError,match='DEADLINE'):r.consume(1,renew_at_seven=False)
    assert r.state['emitted_atoms']==7
    assert r.consume(3)['full_P10']
    r.finish()
    q2=r.request(q['input'],'pick',0)
    assert q2['previous_reference']['width_m']==pytest.approx(.042)
    assert q2['previous_evidence']['kind']=='MOCK_EMISSION_ACK'
    assert q2['native_step']==q['native_step']+250
    assert (r.root/'renew-0001.json').exists()


@pytest.mark.parametrize('fault',['stop','cancel','restart','response_id','binding','input','native','request'])
def test_fault_has_no_install(tmp_path,fault):
    r,q,a=fixture(tmp_path)
    if fault=='stop': r.stop()
    elif fault=='cancel':r.cancel()
    elif fault=='restart':r.restart()
    elif fault=='response_id':a['request_id']='x'
    elif fault=='binding':a['bindings']={'other':True}
    elif fault=='input':open(q['input']['path'],'ab').write(b'changed')
    elif fault=='native':r.state['native_step']+=25;r.save()
    else:q['noise_id']=1
    with pytest.raises((ValueError,RuntimeError)):r.install(q,a)
    assert not list(r.root.glob('install-*'))
    assert r.state['emitted_atoms']==0


def test_whole_H40_suffix_is_not_skipped(tmp_path):
    r,q,a=fixture(tmp_path)
    with np.load(a['action']['path'])as z: raw=z['physical_H40'].copy()
    raw[30,0]+=.1
    np.savez(a['action']['path'],physical_H40=raw,normalized=np.zeros((1,40,32)))
    a['action']['sha256']=digest(a['action']['path'])
    with pytest.raises(ValueError,match='FULL_PROPOSAL_REJECTED'):r.install(q,a)
    assert not list(r.root.glob('install-*'))


def test_expired_wall_clock_rejected_despite_paused_native_time(tmp_path,monkeypatch):
    r,q,a=fixture(tmp_path)
    q['clock_mode']='WALL_CLOCK_STRICT';q['request_id']=identity({k:v for k,v in q.items()if k!='request_id'})
    r.state['pending']=q['request_id'];r.save();a['request_id']=q['request_id']
    monkeypatch.setattr('time.monotonic',lambda:q['created_monotonic']+.4)
    with pytest.raises(TimeoutError,match='TTL'):r.install(q,a)
    assert not list(r.root.glob('install-*'))


def test_read_receipt_retries_absence_without_repeating_any_effect(monkeypatch):
    from go2arx5_vla.world_model import hybrid_nonphysical_v3 as m
    calls=[]
    def delayed(path):
        calls.append(path)
        if len(calls)==1:raise FileNotFoundError(path)
        return {'request_id':'same','status':'GENERATED_NOT_INSTALLED'}
    monkeypatch.setattr(m,'read',delayed)
    assert m.wait_receipt('RESPONSE',alive=lambda:True,stopped=lambda:False,timeout_s=1)['request_id']=='same'
    assert len(calls)==2


def test_read_receipt_does_not_retry_invalid_bytes_or_terminal_writer(monkeypatch):
    from go2arx5_vla.world_model import hybrid_nonphysical_v3 as m
    def missing(path):raise FileNotFoundError(path)
    monkeypatch.setattr(m,'read',missing)
    with pytest.raises(RuntimeError,match='TERMINAL'):m.wait_receipt('R',alive=lambda:False,stopped=lambda:False,timeout_s=1)
    def invalid(path):raise ValueError('invalid')
    monkeypatch.setattr(m,'read',invalid)
    with pytest.raises(ValueError):m.wait_receipt('R',alive=lambda:True,stopped=lambda:False,timeout_s=1)

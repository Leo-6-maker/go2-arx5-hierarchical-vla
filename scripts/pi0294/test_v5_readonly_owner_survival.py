"""Site read outages keep only the adopted successor alive, never stale release."""
from types import SimpleNamespace
from scripts.pi0294.dec1 import cold_owner as c
from scripts.pi0294.dec1.owner_transport import ReadOnlyFailure
from scripts.pi0294.pv1 import successor_cd as s

def test_three_read_failures_then_recovery_does_not_replay_qsub_or_emit_stale_heartbeat(tmp_path):
    epoch=[1001.];events=[];count=[0];binding=dict(s.scope(1000.),grant_sha256=s.GRANT_SHA,before_sha256=dict(GLOBAL='a'*64,LOCAL='b'*64),authorized_storage_caps=s.NEW_STORAGE,old_clock_and_history_preserved=True,scope_only_not_GPU_or_physics_admission=True)
    o=SimpleNamespace(BASE=tmp_path,native=SimpleNamespace(T0=0),cold_rpc_bound=True,
        successor_scope=binding,preserve_readonly_until_deadline=True,HEARTBEAT='heartbeat')
    o.read=lambda p:({'request':{'submit_count':27},'directory':'pbs/alloc-27'}if p.name=='SOURCE_REQUEST.private.json'else {'job_id':'27.stdct-mgmt-02'}if p.name=='SOURCE_JOB_BOUND.private.json'else {'root':'registered'})
    o.record=lambda n,v:events.append((n,v));o.update=o.record
    o.previous=SimpleNamespace(rpc=lambda *args:events.append(('heartbeat',{})))
    o.auth=lambda secret:events.append(('auth',{}))
    def fresh(*args,**kwargs):
        count[0]+=1
        if count[0]<=3:raise ReadOnlyFailure('SITE_UNAVAILABLE_NOT_EMPTY_QUEUE')
        state='Q'if count[0]==4 else'F'
        return dict(rows={'vanda':{'jobs':{'27.stdct-mgmt-02':dict(Job_Name='pi0294_dec1_a40_train',project='CFP04-CF-007',job_state=state)}}},view={})
    o.fresh=fresh
    result=c.watch(o,None,scope_binding=binding,clock=lambda:epoch[0],monotonic=lambda:epoch[0],sleep=lambda n:epoch.__setitem__(0,epoch[0]+n))
    assert result['native']['job_state']=='F'
    assert sum(n=='heartbeat'for n,_ in events)==1
    assert sum(str(n).startswith('READ_ONLY_OBSERVATION_INTERRUPTION-')for n,_ in events)==3
    assert sum(str(n).startswith('READ_ONLY_SUSPENDED_NO_RELEASE-')for n,_ in events)==1
    assert count[0]==5

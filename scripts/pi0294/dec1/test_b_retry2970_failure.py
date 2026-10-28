"""CPU-only failed terminal contracts. No audit, settlement or recovery GO."""
from copy import deepcopy
from unittest.mock import patch
import pytest
from . import b_retry2970_failure as c
from .test_b_retry4096_completion import fixture as success_fixture


def fixture():
    e,v,digest=success_fixture();v['raw_native']['Exit_status']=1
    effects=v['effects'];effects.update(last_native_step=2970,optimizer_updates_committed=922)
    v['process_final'].update(status='FAILED_PRESERVED',global_step=2970,last_returned_training_step=2048,
        error=deepcopy(c.HEARTBEAT_ERROR),effects=deepcopy(effects))
    v['wrapper_final']['returncode']=1;v['native_proof']['Exit_status']=1
    for k in ('checkpoints','checkpoint_members'):del v[k]['4096']
    v['actual_complete_step_inventory']=[1024,2048]
    return e,v,digest


def test_exact_failed_token_owner_and_confirmed_last_heartbeat():
    s=dict(MainPID='0',ActiveState='failed',ExecMainStatus='1')
    stop=dict(epoch=1792755974.8454535,type='PermissionError',message='VANDA_PROJECT_TOKEN_EXPIRED',automatic_mutation_retry=False)
    rpc=dict(status='CONFIRMED',intent=dict(effect='HEARTBEAT'));c.owner(s,stop,rpc)
    for bad in (dict(s,MainPID='1'),dict(s,ActiveState='inactive',ExecMainStatus='0')):
        with pytest.raises(PermissionError):c.owner(bad,stop,rpc)
    with pytest.raises(PermissionError):c.owner(s,stop,dict(rpc,status='UNKNOWN'))


def test_two_complete_states_retained_but_922_paid_updates_do_not_become_a_checkpoint():
    e,v,d=fixture()
    with patch.object(c.prior.b,'PROBE_SHA',d):assert c.audit(v,e) is v
    with pytest.raises(PermissionError):c.prior.audit(v,e)
    assert c.CUMULATIVE_UPDATES==8744+922 and c.CUMULATIVE_UPDATES+2048==11714


@pytest.mark.parametrize('fault',['native_R','exit0','other_job','updates','unknown','query29','wrapper0','success',
    'last_returned','output_refund','seed','new_checkpoint','optimizer','sampler','model_sha','new_query'])
def test_relabeling_failure_or_incomplete_states_as_success_is_rejected(fault):
    e,v,d=fixture()
    if fault=='native_R':v['raw_native']['job_state']='R'
    elif fault=='exit0':v['raw_native']['Exit_status']=0
    elif fault=='other_job':e['job_id']='1480412.stdct-mgmt-02'
    elif fault=='updates':v['effects']['optimizer_updates_committed']=2048
    elif fault=='unknown':v['effects']['pending_updates']=[{}]
    elif fault=='query29':v['effects']['query_attempts']=29
    elif fault=='wrapper0':v['wrapper_final']['returncode']=0
    elif fault=='success':v['process_final']['status']='A40_SOURCE_4096_COMPLETE'
    elif fault=='last_returned':v['process_final']['last_returned_training_step']=2970
    elif fault=='output_refund':v['native_proof']['new_resident_allocated_upper_bytes']=0
    elif fault=='seed':v['checkpoints']['2048']['complete_sha256']='0'*64
    elif fault=='new_checkpoint':v['actual_complete_step_inventory'].append(3072)
    elif fault=='optimizer':v['checkpoint_members']['2048'].pop('optimizer.pt')
    elif fault=='sampler':v['checkpoints']['2048']['actual_consumed_cursor']=2970*32
    elif fault=='model_sha':v['checkpoint_members']['2048']['model-00002-of-00002.safetensors']='not-sha'
    elif fault=='new_query':v['new_queries']=1
    with patch.object(c.prior.b,'PROBE_SHA',d),pytest.raises(PermissionError):c.audit(v,e)

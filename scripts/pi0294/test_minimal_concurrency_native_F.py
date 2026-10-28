"""Full captured budget with future PBS/D25 fixtures; no native permissions."""
from copy import deepcopy
import json
import pytest
from scripts.pi0294.test_minimal_native_admission import actual,packet
from scripts.pi0294.test_minimal_concurrency import D_fixture
from scripts.pi0294.test_minimal_terminal import captured_CPU_terminal
from scripts.pi0294.generalization import minimal_execution as e,minimal_concurrency as n,minimal_terminal as terminal
from scripts.pi0294.dec1 import vanda_campaign as q


def inputs(packet,*,F=False):
    g=json.loads(packet['global_bytes']);record=D_fixture(g);proof=deepcopy(packet['queue_proof']);history=[]
    if F:
        job=json.loads(packet['native_job_bytes']);job.update(job_state='F',Exit_status=0)
        proof.update(qselect=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=0,stdout='',stderr=''),qstat=None)
        history=[dict(schema=q.TERMINAL_HISTORY,user='e1192548',observed_epoch=packet['now_epoch'],returncode=0,stderr='',
            argv=['/opt/pbs/bin/qstat','-x','-f','-F','json',packet['job_id']],stdout=json.dumps(dict(Jobs={packet['job_id']:job})))]
    return g,record,dict(queue_proof=proof,terminal_proofs=history,independent_job_bytes=None,now_epoch=packet['now_epoch'])


@pytest.mark.parametrize('F',[False,True])
def test_real_queue_shape_or_exact_native_F_does_not_erase_pending_financial_main(packet,F):
    g,record,args=inputs(packet,F=F);before=deepcopy(g);out=n.current_native_view(g,record,**args)
    assert out['known_native_F_waiting_for_independent_main_audit']is F
    assert g==before and e.rows(g)[0]['status']==e.STATES[1]
    assert out['GLOBAL_or_LOCAL_mutated']is out['new_model_or_update_or_physics_permission']is False


@pytest.mark.parametrize('change',['missing','nonce','stale','notF','both_current_and_F','other_Core',
    'wrong_main','pending','unknown','new_foreign_GPU'])
def test_missing_or_misbound_job_is_never_assumed_ended_or_filtered_out(packet,change):
    g,record,args=inputs(packet,F=True)
    if change=='missing':args['terminal_proofs']=[]
    elif change in('nonce','notF'):
        p=args['terminal_proofs'][0];v=json.loads(p['stdout'])
        if change=='nonce':v['Jobs'][packet['job_id']]['Variable_List']['PV1_SUBMISSION_NONCE']='other'
        else:v['Jobs'][packet['job_id']]['job_state']='R'
        p['stdout']=json.dumps(v)
    elif change=='stale':args['terminal_proofs'][0]['observed_epoch']-=121
    elif change=='both_current_and_F':args['queue_proof']=deepcopy(packet['queue_proof'])
    elif change=='other_Core':record['evaluation_candidate']='C50'
    elif change=='wrong_main':record[n.KEY]['main_request_id']='OTHER'
    elif change=='pending':g['pending_updates']=[{'PRESERVED':True}]
    elif change=='unknown':g['unknown_effects']=[{'PRESERVED':True}]
    else:
        proof=deepcopy(packet['queue_proof']);other='98765.stdct-mgmt-02';v=json.loads(packet['native_job_bytes'])
        v.update(Job_Name='other-training',project='CFP04-CF-009');proof['qselect']['stdout']=other+'\n'
        proof['qstat']['argv']=['/opt/pbs/bin/qstat','-f','-F','json',other]
        proof['qstat']['stdout']=json.dumps(dict(Jobs={other:v}));args['queue_proof']=proof
    before=deepcopy(g)
    with pytest.raises(PermissionError):n.current_native_view(g,record,**args)
    assert g==before


def test_new_D25_may_finish_after_main_is_already_audited_and_once_settled(packet):
    g,record,args=inputs(packet,F=True);r=e.rows(g)[0]
    proof=captured_CPU_terminal(g,success=True);g=terminal.settle(g,r['request_id'],**proof)
    args['terminal_proofs']=[];out=n.current_native_view(g,record,**args)
    assert not out['campaign_jobs']and not out['known_native_F_waiting_for_independent_main_audit']
    assert e.rows(g)[0]['scientific_endpoint_complete']is True

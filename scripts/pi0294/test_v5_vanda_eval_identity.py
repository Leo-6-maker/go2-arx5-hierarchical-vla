"""C/D PBS parser and closed-evidence joins; all native allocations are fixtures."""
import base64
from copy import deepcopy
import json
import time

import pytest
from scripts.pi0294.pv1 import mix_vanda_eval_identity as v, mix_vanda_cohost as c, mix_accounting as a
from scripts.pi0294.test_v5_vanda_cohost import captured


def request():
    return dict(T_CD=1000.,deadline_epoch=1000.+168*3600,
                request_id='72acd2ad-f3aa-42be-b93d-01993f9ee023',
                submission_nonce='2200fb80-2cab-47f6-b4ec-3fb6d21bc1b1')


def job():
    return dict(job_state='R',project='CFP04-CF-007',Job_Name=v.JOB_NAME,
        queue='batch_gpu',Rerunable=False,run_count=1,
        Job_Owner='e1192548@stdct-login-01',exec_host='GN-A40-022/0*36',
        Variable_List=dict(PV1_REQUEST_ID=request()['request_id'],PV1_SUBMISSION_NONCE=request()['submission_nonce']),
        Resource_List=dict(ncpus=36,ngpus=1,mem='240gb',walltime='08:00:00',select='1:ncpus=36:mem=240gb:ngpus=1:gpu_model=A40'),
        stime=time.strftime('%a %b %d %H:%M:%S %Y',time.localtime(1000.)))


def test_generic_native_identity_does_not_grant_model_or_physics():
    result=v.native_allocation(job(),request(),job_id='1234567.stdct-mgmt-02',hostname='GN-A40-022',now_epoch=1200.)
    assert result['numerical_model_permission'] is result['physics_authorized'] is False
    assert result['native_capacity']['ncpus']==36


@pytest.mark.parametrize('change',['queued','Core_host','login','CPU_only','other_nonce','other_project','too_few_cpus','long_wall','closeout'])
def test_wrong_or_expired_native_job_never_falls_back(change):
    j=job();r=request();host='GN-A40-022';now=1200.
    if change=='queued':j['job_state']='Q'
    elif change=='Core_host':host='corelab-4080s'
    elif change=='login':host='stdct-login-01'
    elif change=='CPU_only':j['Resource_List']['ngpus']=0
    elif change=='other_nonce':j['Variable_List']['PV1_SUBMISSION_NONCE']='wrong'
    elif change=='other_project':j['project']='OTHER'
    elif change=='too_few_cpus':j['Resource_List']['ncpus']=4
    elif change=='long_wall':j['Resource_List']['walltime']='16:00:00'
    else:now=1000.+28800-2700
    with pytest.raises(PermissionError):v.native_allocation(j,r,job_id='1234567.stdct-mgmt-02',hostname=host,now_epoch=now)


def rebound(captured):
    g=c.propose(**captured)['proposed_global'];raw=a.raw(g)
    cap=deepcopy(captured['captures']['C50']);cap['global_base64']=base64.b64encode(raw).decode()
    cap['handoff']=c.endpoint(raw,cap)
    req=dict(request(),evaluation_candidate='C50',run_uuid='v5-cd-c50-n294101-03',
             T_CD=g[c.FIELD]['T_CD'],deadline_epoch=g[c.FIELD]['deadline_epoch'],
             checkpoint_manifest_sha256=cap['handoff']['checkpoint_manifest_sha256'],
             training_handoff_sha256=a.sha(a.raw(cap['handoff'])))
    return g,req,cap


def test_new_request_requires_exact_fresh_global_but_same_native_job_may_advance_counters(captured):
    g,r,cap=rebound(captured);raw=a.raw(g)
    assert v.require_preparation_at_current_global(raw,r,cap)
    g['revision']+=1
    assert v.bind_closed_endpoint(a.raw(g),r,cap)['global_sha256']==a.sha(a.raw(g))
    with pytest.raises(PermissionError,match='EXACT_CURRENT_GLOBAL'):
        v.require_preparation_at_current_global(a.raw(g),r,cap)


def test_old_pre_D25_handoff_cannot_be_used_as_a_new_C50_request(captured):
    g,r,cap=rebound(captured);old=captured['captures']['C50'];r['training_handoff_sha256']=a.sha(a.raw(old['handoff']))
    with pytest.raises(PermissionError,match='EXACT_CURRENT_GLOBAL'):
        v.require_preparation_at_current_global(a.raw(g),r,old)


@pytest.mark.parametrize('change',['wrong_checkpoint','new_uuid','forged_exit','false_C_scope'])
def test_immutable_closed_endpoint_or_reserved_replacement_substitution_rejected(captured,change):
    g,r,cap=rebound(captured)
    if change=='wrong_checkpoint':r['checkpoint_manifest_sha256']='0'*64
    elif change=='new_uuid':r['request_id']='9c12ccce-f98d-4ac0-8a9b-9a7587d38403'
    elif change=='false_C_scope':r['T_CD']+=1
    else:
        key='OS_exit_bytes';row=json.loads(base64.b64decode(cap['terminal_capture_base64'][key]));row['returncode']=1
        cap['terminal_capture_base64'][key]=base64.b64encode(a.raw(row)).decode()
    with pytest.raises(PermissionError):v.bind_closed_endpoint(a.raw(g),r,cap)

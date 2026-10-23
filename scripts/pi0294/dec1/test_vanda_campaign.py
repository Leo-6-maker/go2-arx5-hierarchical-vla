"""Synthetic queue/ledger tests; public direction bytes are not native proof."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from . import vanda_campaign as c,dual_platform as dual
from .test_dual_platform import fixture as dual_fixture

def raw(v):return (json.dumps(v,indent=2)+'\n').encode()

def fixture(*,active=True):
    marker=dict(schema='WDR1_VANDA_SINGLE_CONTROLLER_HANDOFF_V1',authority_release=c.handoff.RELEASE,
        grant_sha256=c.handoff.GRANT_SHA,origin_global_sha256=c.handoff.GLOBAL_SHA,origin_DEC1_sha256=c.handoff.DEC1_SHA,
        origin_access='UNAVAILABLE_FENCED_BY_HUMAN_DIRECTION',origin_is_fresh_native_read=False,
        old_parent_mutations_permitted=False,old_parent_reconciliation_required_when_available=True,
        new_control_platform='VANDA_A40',one_mutation_owner=True,
        journal=dict(read_calls=542,failed_reads_preserved=1,heartbeat_calls=255,Vanda_release_calls=2,
                     inferred_Hopper_writes_after_snapshot=0,unresolved_effect_calls=0))
    failed=dict(job_id=c.handoff.failed.JOB,request_sha256=c.handoff.failed.REQUEST_SHA,
        source_sha=c.handoff.failed.SOURCE,native_platform='VANDA_A40',status='NATIVE_TERMINAL',
        native_exit_status=1,committed_optimizer_updates=922,query_attempts=17,
        conservative_H200_s=8481,conservative_CPU_s=11023)
    h=dict(campaign_run_uuid='fixture-campaign',unknown_effects=[],pending_updates=[],PBS_submissions=[
        dict(job_id='13.hopper-m-02',status='NATIVE_TERMINAL',native_exit_status=1),failed])
    h[c.handoff.FIELD]=marker;d=dict(campaign_run_uuid=h['campaign_run_uuid'],unknown_effects=[])
    d[c.handoff.FIELD]=deepcopy(marker)
    jobs={'15.stdct-mgmt-02':dict(Job_Name='personal-analysis',project='other-project',Job_Owner='e1192548@login',job_state='R',Resource_List=dict(ngpus=8))}
    if active:
        h['PBS_submissions'].append(dict(job_id='14.stdct-mgmt-02',native_platform='VANDA_A40',status='SUBMITTED_PENDING_NATIVE',project='CFP04-CF-007',job_name='pi0294_dec1_a40_train'))
        jobs['14.stdct-mgmt-02']=dict(Job_Name='pi0294_dec1_a40_train',project='CFP04-CF-007',Job_Owner='e1192548@login',job_state='R',Resource_List=dict(ngpus=1))
    ids=sorted(jobs)
    q=dict(schema=c.QUEUE,user='e1192548',observed_epoch=1000.,
        qselect=dict(argv=['/opt/pbs/bin/qselect','-u','e1192548'],returncode=0,stdout='\n'.join(ids)+'\n',stderr=''),
        qstat=dict(argv=['/opt/pbs/bin/qstat','-f','-F','json',*ids],returncode=0,stdout=json.dumps(dict(Jobs=jobs)),stderr=''))
    a=dict(schema=c.AUTHORITY,physical_platform='VANDA_A40',user='e1192548',observed_epoch=1000.,active_registered_control_lease=True,
        parent_STOP=False,control_STOP=False,single_mutation_owner=True,global_sha256=c.sha(raw(h)),DEC1_sha256=c.sha(raw(d)))
    return h,d,a,q

def observe(h,d,a,q):
    a=deepcopy(a);a.update(global_sha256=c.sha(raw(h)),DEC1_sha256=c.sha(raw(d)))
    grant=(Path(__file__).resolve().parents[3]/c.handoff.PLAN).read_bytes()
    return c.observe(grant_bytes=grant,global_ledger_bytes=raw(h),DEC1_ledger_bytes=raw(d),authority_readback=a,queue_readback=q,now_epoch=1001.)

def test_fresh_own_vanda_and_fenced_terminal_origin_are_explicitly_separate():
    args=fixture();before=deepcopy(args);h,d,a,q=args;v=observe(*args)
    assert args==before and v['status']==c.PASS and v['account_owned_job_count']==2
    assert len(v['campaign_jobs'])==len(v['outside_registered_campaign_jobs'])==1
    assert v['historical_Hopper_registered_terminal_jobs']==[dict(job_id='13.hopper-m-02',status='NATIVE_TERMINAL',native_exit_status=1)]
    assert v['Hopper_current_queue_observed'] is False and v['unavailable_Hopper_is_fresh_or_empty'] is False
    assert v['new_resource_or_science_grant'] is False
    c.validate(v,ledger=h,DEC1_ledger=d,now_epoch=1002.)

@pytest.mark.parametrize('fault',['missing_handoff','unfenced','fake_fresh_Hopper','unproven_journal','different_pair',
    'old_Hopper_active','Hopper_exit_unknown','unsettled922','refund922','refundCPU','unknown_effect',
    'stale_authority','STOP','not_registered','other_user','qselect162','fake_empty','dropped_job',
    'wrong_qstat_args','foreign_owner','changed_native_project','duplicate_active'])
def test_no_timeout_empty_queue_or_old_snapshot_can_replace_current_authority(fault):
    h,d,a,q=fixture()
    if fault=='missing_handoff':h.pop(c.handoff.FIELD)
    elif fault=='unfenced':h[c.handoff.FIELD]['old_parent_mutations_permitted']=True
    elif fault=='fake_fresh_Hopper':h[c.handoff.FIELD]['origin_is_fresh_native_read']=True
    elif fault=='unproven_journal':h[c.handoff.FIELD]['journal']['unresolved_effect_calls']=1
    elif fault=='different_pair':d[c.handoff.FIELD]['one_mutation_owner']=False
    elif fault=='old_Hopper_active':h['PBS_submissions'][0]['status']='SUBMITTED_PENDING_NATIVE'
    elif fault=='Hopper_exit_unknown':h['PBS_submissions'][0].pop('native_exit_status')
    elif fault=='unsettled922':h['PBS_submissions'][1]['status']='SUBMITTED_PENDING_NATIVE'
    elif fault=='refund922':h['PBS_submissions'][1]['committed_optimizer_updates']=0
    elif fault=='refundCPU':h['PBS_submissions'][1]['conservative_CPU_s']=8323
    elif fault=='unknown_effect':h['unknown_effects']=[{}]
    elif fault=='stale_authority':a['observed_epoch']=800.
    elif fault=='STOP':a['parent_STOP']=True
    elif fault=='not_registered':a['active_registered_control_lease']=False
    elif fault=='other_user':a['user']='other'
    elif fault=='qselect162':q['qselect']['returncode']=162
    elif fault=='fake_empty':q['qselect']['stdout']=''
    elif fault=='dropped_job':
        jobs=json.loads(q['qstat']['stdout']);jobs['Jobs'].pop('14.stdct-mgmt-02');q['qstat']['stdout']=json.dumps(jobs)
    elif fault=='wrong_qstat_args':q['qstat']['argv'].pop()
    else:
        jobs=json.loads(q['qstat']['stdout'])['Jobs']
        if fault=='foreign_owner':jobs['14.stdct-mgmt-02']['Job_Owner']='other@login'
        elif fault=='changed_native_project':jobs['14.stdct-mgmt-02']['project']='different'
        else:
            h['PBS_submissions'].append(dict(h['PBS_submissions'][-1],job_id='16.stdct-mgmt-02'))
            jobs['16.stdct-mgmt-02']=deepcopy(jobs['14.stdct-mgmt-02']);q['qselect']['stdout']+='16.stdct-mgmt-02\n';q['qstat']['argv'].append('16.stdct-mgmt-02')
        q['qstat']['stdout']=json.dumps(dict(Jobs=jobs))
    with pytest.raises(PermissionError):observe(h,d,a,q)

@pytest.mark.parametrize('fault',['missing_active','held','unregistered_pi0294'])
def test_unknown_or_missing_current_jobs_remain_unknown(fault):
    h,d,a,q=fixture();jobs=json.loads(q['qstat']['stdout'])['Jobs']
    if fault=='missing_active':
        jobs.pop('14.stdct-mgmt-02');q['qselect']['stdout']='15.stdct-mgmt-02\n';q['qstat']['argv']=['/opt/pbs/bin/qstat','-f','-F','json','15.stdct-mgmt-02']
    elif fault=='held':jobs['14.stdct-mgmt-02']['job_state']='H'
    else:jobs['15.stdct-mgmt-02']['Job_Name']='pi0294_unregistered'
    q['qstat']['stdout']=json.dumps(dict(Jobs=jobs));v=observe(h,d,a,q)
    assert v['status']=='STOP_UNKNOWN_VANDA_CAMPAIGN_STATE' and v['unknown_campaign_jobs']
    with pytest.raises(PermissionError):c.validate(v,ledger=h,DEC1_ledger=d,now_epoch=1002.)

def test_only_original_resource_budget_and_one_platform_are_reserved():
    h,d,a,q=fixture(active=False);v=observe(h,d,a,q);grant,_,_=dual_fixture()
    args=dict(grant_bytes=grant,view=v,used={},reserved={},extra=dict(H200_s=3600,PBS_submissions=1),platform='VANDA_A40',job_wall_s=3600,now_epoch=1002.,campaign_authority_ledgers=(h,d))
    with patch.object(dual,'GRANT_SHA',c.sha(grant)):
        p=dual.reserve_check(**args);assert p['concurrent_GPU_limit']==1
        for change in (dict(campaign_authority_ledgers=None),dict(platform='HOPPER_H200'),dict(used=dict(H200_s=90*3600)),dict(now_epoch=1201.)):
            with pytest.raises(PermissionError):dual.reserve_check(**dict(args,**change))

@pytest.mark.parametrize('fault',['old_ledger','old_DEC1','claimed_Hopper_view','status_only','removed_outside','future_timestamp'])
def test_reserve_release_and_heartbeat_revalidation_rejects_modified_view(fault):
    h,d,a,q=fixture();v=observe(h,d,a,q)
    if fault=='old_ledger':h['new_metadata']=1
    elif fault=='old_DEC1':d['new_metadata']=1
    elif fault=='claimed_Hopper_view':v['Hopper_current_queue_observed']=True
    elif fault=='status_only':v={'schema':c.SCHEMA,'status':c.PASS}
    elif fault=='removed_outside':v['outside_registered_campaign_jobs']=[]
    else:v['queue_observed_epoch']=1003.
    with pytest.raises(PermissionError):c.validate(v,ledger=h,DEC1_ledger=d,now_epoch=1002.)

def test_actual_collector_uses_only_bounded_own_vanda_commands_and_preserves_failure():
    calls=[]
    def runner(argv,**kw):
        calls.append((argv,kw));return SimpleNamespace(returncode=0,stdout='',stderr='')
    q=c.collect_queue(runner=runner,clock=lambda:1000.)
    assert len(calls)==1 and calls[0][0]==['/opt/pbs/bin/qselect','-u','e1192548'] and calls[0][1]['timeout']==10
    assert c.queue_rows(q,now_epoch=1001.)==[] and q['qstat'] is None
    def failed(argv,**kw):return SimpleNamespace(returncode=162,stdout='',stderr='real scheduler diagnostic')
    with pytest.raises(c.ReadOnlyFailure) as caught:c.collect_queue(runner=failed)
    assert caught.value.diagnostic['returncode']==162 and caught.value.diagnostic['stderr']=='real scheduler diagnostic'

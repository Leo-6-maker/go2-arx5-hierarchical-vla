"""Unit-exit evidence must not turn a timeout or partial execution into zero."""
import base64
import pytest
from scripts.pi0294.generalization import minimal_core_accounting as c
from scripts.pi0294.generalization import minimal_core_prebind_terminal as t,minimal_core_sharing as s


def fixture():
    files={}
    def add(name,value):
        b=c.raw(value);files[name]=dict(base64=base64.b64encode(b).decode(),sha256=c.sha(b));return c.sha(b)
    neural=s.RUN+'-neural.service';owner='arx5-e1-fixed512-eval-owner-03'
    handle=c.process_handle('22222222-2222-4222-8222-222222222222',2000,100)
    start=dict(source_sha=t.SOURCE,request_id=s.RID,request_sha256=t.REQUEST,
        run_manifest_sha256=None,native_model_permission=False,physics_authorized=False,
        boot_id='22222222-2222-4222-8222-222222222222',pid=2000,proc_start=100,
        native_handle=handle,started_epoch=1002.,observed_epoch=1003.)
    nh=add('hopper/RUN_MANIFEST.json',dict(code=dict(vla_commit=t.SOURCE),run_uuid=s.RUN))
    ph=add('core/RUN_MANIFEST.json',dict(code=dict(vla_commit=t.SOURCE),run_uuid=s.RUN))
    start['run_manifest_sha256']=nh;add('hopper/pbs/alloc-01/NATIVE_CORE_PROCESS_STARTED.json',start)
    add('hopper/pbs/alloc-01/NATIVE_PROCESS_FINAL.json',dict(job_id=handle,status='FAILED_PRESERVED',
        phase='WAIT_ACTUAL_GLOBAL_BIND_AND_PREMODEL_RELEASE',error='TimeoutError: PV1_CORE_NATIVE_RELEASE_NOT_OBSERVED_NO_RESTART',
        records=[],new_optimizer_updates=0,completed_epoch=1182.7))
    add('core/RESOURCE_LEDGER.json',dict(unknown_effects=[],reservations={},Core_CPU_s=300,
        main_starts=0,engine_starts=0,live_and_recovery_integrations=0,shadow_integrations=0))
    add('owner/V5_CD_NATIVE_OWNER_STARTED.private.json',dict(source_sha=t.SOURCE,request_id=s.RID,
        request_sha256=t.REQUEST,run_uuid=owner,pid=1000))
    add('owner/OWNER_PRESERVED_EXCEPTION.json',dict(error='EXACT_NATIVE_PROCESS_DURING_FINITE_ADOPTION'))
    unit=dict(MainPID='0',ActiveState='failed',ExecMainCode='1',ExecMainStatus='1',InvocationID='e'*32,
        ExecMainPID='2000',ExecMainStartTimestampMonotonic='2000000',
        ExecMainExitTimestampMonotonic='182730000',CPUUsageNSec='5304870000',
        ExecMainStartTimestamp='@1002',ExecMainExitTimestamp='@1182')
    core=dict(status='ACTUAL_SHARED03_PREMODEL_TIMEOUT_UNIT_EXIT_FULL_ZERO_RECORDS_READONLY',
        host='corelab-4080s',uid=2009,LOCAL_sha256='b'*64,observed_epoch=1195.,files=files,
        qualification_model_query_release_or_effect_records=[],boot_epoch=1000.,
        units={neural:unit,owner+'.service':dict(unit,ExecMainPID='1000')},
        terminal_storage=dict(both_native_units_terminal=True,followed_external_symlinks=False,
            deleted_files=0,allocated_bytes_upper=16*1024**2,charged_upper_bytes=64*1024**2))
    global_read=dict(GLOBAL_sha256='a'*64,all_three_paths_absent=True,unknown_effects=[],pending_updates=[],observed_epoch=1195.)
    failed=dict(sha256='a'*64,expected_before='a'*64,owned_live_actor_birth_window=[],
        files={k:dict(exists=False)for k in('INTENT.json','RECEIPT.json')})
    row=dict(request_id=s.RID,source_sha=t.SOURCE,request_sha256=t.REQUEST,native_handle=None,
        status=c.STATES[0],native_unit=neural,run_manifest_sha256=nh,physical_manifest_sha256=ph,
        query_limit=2097,CPU_upper_s=80000,created_epoch=999.)
    return {'core':core,'global':global_read,'failed_bind':failed},row


def check(p,r):return t.observed(p,r,global_sha256='a'*64,local_sha256='b'*64,now_epoch=1200.)


def test_native_exit_is_a_witness_only_and_padded_costs_are_not_zero():
    p,r=fixture();v=check(p,r)
    assert v==dict(GPU_s=241,neural_CPU_s=6,physical_CPU_s=300,
        Core_output_bytes=64*1024**2,new_runtime_transfer_bytes=0)
    assert r['native_handle']is None and r['status']==c.STATES[0]


@pytest.mark.parametrize('mutation',[
    lambda p:p['core'].update(qualification_model_query_release_or_effect_records=['START_NEURAL.json']),
    lambda p:p['global'].update(all_three_paths_absent=False),
    lambda p:p['global'].update(GLOBAL_sha256='f'*64),
    lambda p:p['failed_bind'].update(owned_live_actor_birth_window=[dict(pid=123)]),
    lambda p:p['failed_bind']['files']['INTENT.json'].update(exists=True),
    lambda p:p['core']['units'][s.RUN+'-neural.service'].update(MainPID='2000',ActiveState='active'),
    lambda p:p['core']['units'][s.RUN+'-neural.service'].update(ExecMainPID='2001'),
    lambda p:p['core']['units'][s.RUN+'-neural.service'].update(ExecMainExitTimestamp='@1282'),
    lambda p:p['core']['files']['core/RESOURCE_LEDGER.json'].update(sha256='0'*64),
    lambda p:p['core'].update(observed_epoch=1000.),
    lambda p:p['core']['terminal_storage'].update(followed_external_symlinks=True),
    lambda p:p['core']['terminal_storage'].update(allocated_bytes_upper=300*1024**2),
])
def test_partial_effect_unknown_write_or_live_actor_cannot_release_holds(mutation):
    p,r=fixture();mutation(p)
    with pytest.raises(PermissionError):check(p,r)

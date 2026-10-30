"""Raw512 premodel resource join; original device/qualification backend reused."""
import ctypes,errno,json,os,time
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import minimal_core_accounting as c
from scripts.pi0294.pv1 import mix_core_identity as identity
from scripts.pi0294.pv1.mix_core_accounting import digest as native_digest

need=c.need
SNAPSHOT='ARX5_MINIMAL_CORE_FIXED512_NATIVE_CAPACITY_V1'
PROOF='ARX5_MINIMAL_CORE_FIXED512_FULL_RESOURCE_RECOMPUTATION_V1'
ENDPOINT='plans/E1_FIXED512_ENDPOINT_REFERENCE.json'


def snapshot(root,request,*,global_bytes,local_bytes,PC_free_bytes,PC_observed_epoch):
    """Read only kernel space/quota metadata on the exact native filesystem."""
    root=Path(root);need(os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
        and 0<=time.time()-PC_observed_epoch<=120,'ACTUAL_SITE_AND_PC_OBSERVATION')
    from scripts.pi0294.pv1.node_local_custody import local_path
    local_path(root.resolve());matches=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before,after=line.split(' - ',1);mp=before.split()[4]
        if str(root)==mp or str(root).startswith(mp.rstrip('/')+'/'):matches.append((len(mp),line,after.split()))
    _,mount,fields=max(matches);need(fields[0]=='ext4','EXACT_CURRENT_NATIVE_FILESYSTEM')
    lib=ctypes.CDLL(None,use_errno=True);quota=[]
    for kind in range(3):
        value=ctypes.c_uint32();ctypes.set_errno(0)
        rc=lib.quotactl(ctypes.c_int((0x800004<<8)|kind),ctypes.c_char_p(fields[1].encode()),0,ctypes.byref(value))
        error=ctypes.get_errno();quota.append(dict(type=kind,operation='Q_GETFMT',returncode=rc,errno=error))
        need(rc==-1 and error==errno.ESRCH,'ACTUAL_NO_ENABLED_USER_GROUP_OR_PROJECT_QUOTA')
    space=os.statvfs(root);free=space.f_bavail*space.f_frsize
    checkpoint=read(root/'plans/EVALUATION_CHECKPOINT.private.json')
    need(sha(Path(checkpoint['checkpoint'])/'CHECKPOINT_COMPLETE.json')==c.CHECKPOINT,
         'ALREADY_DELIVERED_FIXED512_MODEL_PRESENT_NOT_ANOTHER_FUTURE_COPY')
    return dict(schema=SNAPSHOT,GLOBAL_sha256=c.sha(global_bytes),LOCAL_sha256=c.sha(local_bytes),
        observed_epoch=time.time(),Core_host=os.uname().nodename,Core_free_bytes=free,
        Core_free_inodes=space.f_favail,PC_free_bytes=PC_free_bytes,PC_observed_epoch=PC_observed_epoch,
        Core_quota_checked=True,Core_quota_available_bytes=free,
        kernel_quota_reads=quota,mountinfo=mount,existing_E1_checkpoint_manifest_sha256=c.CHECKPOINT,
        metadata_only=True,new_physics=0,new_models=0)


def recompute(*,manifest_bytes,request_bytes,global_bytes,local_bytes,snapshot_bytes,
              training_capture_bytes,population_bytes,plan_bytes,prior_bytes,native_facts_bytes,now_epoch):
    m,r,g,l,s,p=map(json.loads,(manifest_bytes,request_bytes,global_bytes,local_bytes,snapshot_bytes,native_facts_bytes))
    identity.request_identity(m,r,c.sha(manifest_bytes));c.paired(g,l)
    row=g[c.FIELD][r['request_id']]
    need(row['request_sha256']==c.sha(request_bytes) and row['status'] in c.STATES[1:3],
         'EXACT_ACTUALLY_BOUND_ALLOCATION')
    need(json.loads(training_capture_bytes)==c.endpoint_reference()
        and c.sha(training_capture_bytes)==r['training_handoff_sha256'], 'KNOWN_FIXED_QUALIFIED_E1_NOT_A_NEW_ENDPOINT')
    need(c.sha(population_bytes)==c.plan.POPULATION,'ORIGINAL_SEALED_POPULATION')
    plan=c.plan.candidate_plan(json.loads(population_bytes)['groups'],'E1',checkpoint_manifest_sha256=c.CHECKPOINT)
    need(json.loads(plan_bytes)==plan and c.sha(plan_bytes)==r['planned_panel_sha256'], 'EXACT_ORIGINAL24_PLAN')
    if r['remaining_conditions']<24:
        need(c.sha(prior_bytes)==m['execution']['sealed_inputs'].get('plans/PRIOR_EVALUATION_OUTCOMES.json'),
             'SEALED_ORIGINAL_PRIOR_FOR_REMAINING_PANEL')
    c.plan.labels(r,plan,json.loads(prior_bytes))
    need(s.get('schema')==SNAPSHOT and s.get('GLOBAL_sha256')==c.sha(global_bytes)
        and s.get('LOCAL_sha256')==c.sha(local_bytes)
        and 0<=now_epoch-s['observed_epoch']<120 and s.get('Core_host')=='corelab-4080s',
         'FRESH_ACTUAL_RESOURCE_READ_AND_FULL_LEDGER_BYTES')
    # This preserves every independent old projected future quantity. The new direct
    # directive removes project total caps for future scopes, not per-job
    # limits, historical charges, old native clocks or physical availability.
    from .minimal_scope import first512_forecast,components,INFERENCE_BYTES
    projection_g,projection_l=c.old_future_projection(g,l)
    forecast=first512_forecast(c.raw(projection_g),c.raw(projection_l))
    original_component=components()['E1_CORE_ORIGINAL24']
    original_package=projection_g['ARX5_minimal_first512_execution_package']
    need(original_package['remaining_resources']['Core_storage_bytes']>=original_component['Core_storage_bytes']
        and forecast['additional']['Core_storage_bytes']>=original_component['Core_storage_bytes']
        and not any(v.get('evaluation_candidate')=='E1' for v in projection_g.get(c.old.CORE_ROWS,{}).values())
        and s.get('existing_E1_checkpoint_manifest_sha256')==c.CHECKPOINT,
         'ONE_ALREADY_DELIVERED_E1_PACKAGE_STILL_IN_UNRUN_ORIGINAL_FUTURE')
    # Only remove the duplicate FUTURE model copy. Its actual inodes already
    # occupy the measured filesystem. Both old financial holds and the entire
    # old physical-output forecast remain; no ledger/package is changed.
    credit=INFERENCE_BYTES
    old_future=forecast['dimensions']['Core_storage_bytes']
    additional=c.HOLD['Core_storage_bytes']+c.CONTROL['Core_storage_bytes']
    required=max(0,old_future['projected']-old_future['used']-credit)+additional+2*1024**3
    need(type(s.get('Core_free_bytes')) is int and s['Core_free_bytes']>=required
        and type(s.get('Core_free_inodes')) is int and s['Core_free_inodes']>10000,
         'REAL_NODE_LOCAL_SPACE_INCLUDES_ALL_OLD_FUTURES_AND_NEW_OUTPUT')
    need(type(s.get('PC_free_bytes')) is int and s['PC_free_bytes']>=256*1024**2
        and s.get('Core_quota_checked') is True and s.get('Core_quota_available_bytes') is not None
        and s['Core_quota_available_bytes']>=required, 'ACTUAL_PC_SPACE_AND_SITE_QUOTA')
    start=p['started_epoch'];allocation={k:r[k]for k in ('source_sha','request_id','submission_nonce','stage',
        'evaluation_candidate','checkpoint_manifest_sha256','statistics_sha256','native_platform','allocation_number')}
    allocation.update(status=identity.STATUS,platform_evaluation_version=c.VERSION,run_uuid=r['run_uuid'],
        manifest_sha256=c.sha(manifest_bytes),request_sha256=c.sha(request_bytes),job_id=p['native_handle'],
        native_handle=p['native_handle'],native_process=p,native_process_sha256=native_digest(p),hostname=p['hostname'],
        allocation_start_epoch=start,allocation_end_epoch=min(start+r['walltime_s'],r['deadline_epoch']),
        model_loaded=False,new_optimizer_updates=0)
    identity.sealed_identity(m,r,allocation,run_uuid=r['run_uuid'],manifest_sha256=c.sha(manifest_bytes),
        request_sha256=c.sha(request_bytes),same_bound_process=row)
    need(0<=now_epoch-p['observed_epoch']<120 and now_epoch<allocation['allocation_end_epoch']-2700,
         'FRESH_SAME_NATIVE_PROCESS_AND_CLOSEOUT_RESERVE')
    return dict(schema=PROOF,request_id=r['request_id'],request_sha256=c.sha(request_bytes),
        GLOBAL_sha256=c.sha(global_bytes),LOCAL_sha256=c.sha(local_bytes),snapshot_sha256=c.sha(snapshot_bytes),
        native_process_sha256=native_digest(p),allocation=allocation,all_remaining_costs_fit=True,shortfalls={},
        full_original_future_forecast=forecast,Core_free_required_bytes=required,
        duplicate_future_E1_model_copy_credit_bytes=credit,old_financial_credit_bytes=0,
        project_internal_total_budget_limit=None,original_matrix_and_final_protected=True,
        model_permission=False,physics_authorized=False)


def budget_now(root,request_path,ledger):
    from scripts.pi0294.pv1 import mix_core_delegation as contract
    need(ledger.delegated and c.typed(ledger.request),'EXACT_FINITE_DELEGATED_RAW512_BACKEND')
    proof=ledger.live_proof();return proof,proof['allocation']['native_process']


def admit(root,request_path):
    from scripts.pi0294.pv1.mix_core_delegated_ledger import Ledger
    from scripts.pi0294.pv1.evaluation_core_runtime import put
    root=Path(root);request_path=Path(request_path);ledger=Ledger(root,request_path)
    proof=ledger.live_proof();allocation=proof['allocation']
    put(request_path.parent/'ALLOCATION_ADMISSION.json',allocation)
    return allocation

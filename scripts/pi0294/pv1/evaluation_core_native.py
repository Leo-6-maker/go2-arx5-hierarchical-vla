"""Core allocation seam for the original evaluator; no PBS or model alias.

Only the already registered, live, bound server process can obtain allocation
metadata. Model load additionally needs the original 600s/80-cycle proof and a
fresh complete resource recomputation after that proof. These functions do not
submit tasks, accept a queue, construct a model or authorize physical advance.
"""
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read,write
from . import evaluation_core_accounting as c,evaluation_core_identity as identity
from . import evaluation_core_admission as resources,evaluation_admission as original
from .evaluation_core_scope import validate,member
from .evaluation_core_ledger import Ledger
from .future_admission import _file


def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_ADMISSION_'+why)


def static_history(root,manifest,request):
    """Read every original attempt from the sealed original index."""
    root=Path(root);sealed=manifest['execution']['sealed_inputs']
    def source(name):
        raw=_file(root,name);pin=hashlib.sha256(raw).hexdigest()
        need(sealed.get(name)==request['sealed_inputs'].get(name)==pin,
             'EXACT_ORIGINAL_SEALED_HISTORY:'+name)
        return raw
    index=json.loads(source(original.INDEX));names=[original.INDEX,index['population']]
    need(index.get('schema')=='PI0294_V4_PRIMARY_HISTORY_INPUT_INDEX_V1'
         and index.get('campaign_run_uuid')==c.scope.CAMPAIGN
         and index.get('authority_release')==c.AUTHORITY
         and set(index.get('evidence',{}))==set(c.old.ENDPOINTS),'ORIGINAL_ALL_CANDIDATE_HISTORY_INDEX')
    population=source(index['population']);evidence={}
    for candidate,records in index['evidence'].items():
        need(type(records) is list,'ORIGINAL_COMPLETE_ATTEMPT_LIST')
        evidence[candidate]=[]
        for row in records:
            need(type(row) is dict and {'result','adjudication','audit','model_ready'}<=set(row)
                 and set(row)<={'result','adjudication','audit','model_ready','causal_chain','engineering_diagnosis'},
                 'ORIGINAL_ATTEMPT_EVIDENCE_KINDS')
            evidence[candidate].append({kind:source(name) for kind,name in row.items()});names.extend(row.values())
    need(len(names)==len(set(names)),'NO_ALIASED_ORIGINAL_HISTORY_INPUT')
    return population,evidence


def predecessor_history(root,request,global_ledger,evidence):
    """B's cold manifest stays fixed; actual closed A history is joined later."""
    if request.get('queue_position')!=1:return evidence
    root=Path(root);parent=root.parent.parent
    rows=sorted(global_ledger[c.FIELD].values(),key=lambda r:r['queue_position']);previous=rows[0]
    need(previous['status']==c.STATES[3] and previous['native_exit_status']==0
         and previous['evaluation_candidate']=='A_CONTROL' and evidence['A_CONTROL']==[],
         'ONLY_ACTUAL_CLOSED_FIRST_ARM_HISTORY_NOT_A_NEW_GATE')
    neural=parent/'neural-children'/previous['child_run_uuid'];physical=parent/'children'/previous['child_run_uuid']
    for peer in (neural,physical):
        binding=read(peer/'OUTPUT_BINDING.private.json');lease=read(binding['lease'])
        need(not peer.is_symlink() and peer.stat().st_uid==os.getuid()
             and Path(binding['run_root']).resolve()==peer.resolve()
             and lease['status']=='CLOSED' and lease['run_uuid']==previous['child_run_uuid'],
             'BOTH_ACTUAL_PREDECESSOR_ROLES_CLOSED')
    need(hashlib.sha256(_file(neural,'RUN_MANIFEST.json')).hexdigest()==previous['run_manifest_sha256'],
         'EXACT_CLOSED_PREDECESSOR_SOURCE')
    path='evidence/CORE_PREDECESSOR_HISTORY_BINDING.private.json'
    refresh=json.loads(_file(root,path))
    need(refresh.get('schema')=='PV1_CORE_STAGE_PREDECESSOR_HISTORY_V1'
         and refresh.get('request_id')==request['request_id']
         and refresh.get('previous_request_id')==previous['request_id']
         and refresh.get('previous_terminal_native_sha256')==previous['terminal_native_sha256']
         and refresh.get('queue_plan_sha256')==request['queue_plan_sha256']
         and type(refresh.get('attempts')) is list and len(refresh['attempts'])==24,
         'FULL_ACTUAL_A_PANEL_HISTORY_AFTER_NATIVE_TERMINAL')
    closure_raw=_file(neural,'evidence/NATIVE_DUAL_CLOSURE.json');closure=json.loads(closure_raw)
    need(refresh.get('previous_dual_closure_sha256')==hashlib.sha256(closure_raw).hexdigest()
         and closure.get('schema')=='PV1_CORE_GPU_AND_PHYSICS_DUAL_CLOSURE_V1'
         and closure.get('request_id')==previous['request_id']
         and closure.get('terminal_native_sha256')==previous['terminal_native_sha256']
         and closure.get('GPU_lease_status')==closure.get('physical_lease_status')=='CLOSED'
         and closure.get('active_native_actors')==0 and closure.get('remaining_holds')==0,
         'REAL_SETTLED_CLOSED_PREDECESSOR_NOT_AN_OWNER_ASSERTION')
    actual=[];identifiers=set()
    for record in refresh['attempts']:
        need(type(record) is dict and set(record)=={'attempt_id','members'}
             and Path(record['attempt_id']).name==record['attempt_id'] and record['attempt_id'] not in identifiers,
             'ONE_UNIQUE_ORIGINAL_CONDITION_ATTEMPT')
        identifiers.add(record['attempt_id']);item={}
        need(set(record['members'])=={'adjudication','result','audit','model_ready','causal_chain'},
             'FULL_INDEPENDENT_ACTUAL_ATTEMPT_AND_CAUSAL_EVIDENCE')
        for kind,row in record['members'].items():
            need(set(row)=={'role','relative','sha256'} and row['role'] in ('physical','neural'),
                 'OWN_PREDECESSOR_EVIDENCE_ROLE')
            raw=_file(physical if row['role']=='physical' else neural,row['relative'])
            need(hashlib.sha256(raw).hexdigest()==row['sha256'],'ACTUAL_PREDECESSOR_EVIDENCE_HASH')
            item[kind]=raw
        decision=json.loads(item['adjudication'])
        need(decision['attempt_id']==record['attempt_id'] and decision['model']=='A_CONTROL',
             'ACTUAL_FIRST_ARM_NOT_ANOTHER_MODEL_OR_RELABELED_RESULT')
        actual.append(item)
    return dict(evidence,A_CONTROL=actual)


def resource_inputs(root,request_path,ledger,process):
    root,request_path=Path(root),Path(request_path)
    m=read(root/'RUN_MANIFEST.json');r=read(request_path)
    population,evidence=static_history(root,m,r)
    global_ledger=ledger.state();evidence=predecessor_history(root,r,global_ledger,evidence)
    # The exact remote raw bytes, not a reserialized summary, bind the snapshot.
    raw_global=ledger.raw_state_bytes
    binding=read(member(root,'plans/CORE_RESIDENT_QUEUE_BINDINGS.private.json'))
    queue_root=Path(binding['queue_root'])
    need(queue_root.parent==root.parent.parent/'control'
         and queue_root.name=='v4a-core-resident-'+r.get('queue_id',r.get('control_id','MISSING')),'ORIGINAL_RESIDENT_CONTROL_CUSTODY')
    from scripts.pi0294.dec1.source_cpu_once_io import _file as ledger_file
    if 'startup_recovery_sha256' in r:
        from .evaluation_core_zero_start_recovery import current_boundary
        recovery=current_boundary(global_ledger,r['request_id'])
        need(recovery is not None and c.digest(recovery)==r['startup_recovery_sha256'],
             'SAME_FULL_DUAL_KNOWN_ZERO_BOUNDARY')
        original_control=root.parent.parent/'control'/('v4a-core-resident-'+global_ledger['Core_native_GPU_stage_queue']['plan']['queue_id'])
        need(binding.get('canonical_LOCAL_ledger')==str(original_control/'LOCAL_RESOURCE_LEDGER.private.json'),
             'ORIGINAL_CANONICAL_LOCAL_LEDGER_NOT_A_NEW_COPY')
        local_raw=ledger_file(original_control/'LOCAL_RESOURCE_LEDGER.private.json')
    else:local_raw=ledger_file(queue_root/'LOCAL_RESOURCE_LEDGER.private.json')
    return dict(manifest_bytes=_file(root,'RUN_MANIFEST.json'),request_bytes=_file(root,str(request_path.relative_to(root))),
        population_bytes=population,evidence_by_candidate=evidence,global_bytes=raw_global,local_bytes=local_raw,
        snapshot_bytes=_file(root,original.FRESH_SNAPSHOT),now_epoch=time.time(),
        native_facts_bytes=original.canonical(process))


def budget_now(root,request_path,ledger):
    r=read(request_path);process=identity.observe_current(root,dict(r,request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest()))
    result=resources.recompute(**resource_inputs(root,request_path,ledger,process),purpose='NATIVE_PREMODEL')
    need(result['all_remaining_costs_fit'] is True and result['shortfalls']=={},'FRESH_COMPLETE_BUDGET')
    return result,process


def admit(root,request_path):
    root,request_path=Path(root),Path(request_path);m,*_=validate(root)
    need(not os.environ.get('PBS_JOBID') and request_path.name=='REQUEST.json'
         and request_path.parent.parent.resolve()==(root/'pbs').resolve(),'EXACT_CORE_REGISTERED_REQUEST_PATH')
    r=read(request_path);resources.request_identity(m,r,hashlib.sha256(_file(root,'RUN_MANIFEST.json')).hexdigest())
    need(request_path.parent.name=='alloc-'+str(r['allocation_number']).zfill(2),
         'CURRENT_REQUEST_NAMESPACE_NOT_A_PBS_ORDINAL')
    out=request_path.parent/'ALLOCATION_ADMISSION.json'
    need(not out.exists(),'PRESERVE_FIRST_NATIVE_ADMISSION_NO_RESTART')
    process=identity.observe_current(root,dict(r,request_sha256=hashlib.sha256(_file(root,str(request_path.relative_to(root)))).hexdigest()))
    os.environ[identity.HANDLE_ENV]=process['native_handle']
    ledger=Ledger(root,request_path);proof,process_now=budget_now(root,request_path,ledger)
    need(process['native_handle']==process_now['native_handle'],'SAME_ACTUAL_CALLER_DURING_NATIVE_ADMISSION')
    value={k:r[k] for k in ('source_sha','request_id','submission_nonce','stage','evaluation_candidate',
        'checkpoint_manifest_sha256','statistics_sha256','native_platform','allocation_number')}
    value.update(status=identity.STATUS,platform_evaluation_version=c.VERSION,run_uuid=root.name,
        request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),manifest_sha256=r['run_manifest_sha256'],
        job_id=process['native_handle'],native_handle=process['native_handle'],hostname=process['hostname'],
        allocation_start_epoch=process['started_epoch'],allocation_end_epoch=min(process['started_epoch']+28800,c.old.STOP_PHYSICS),
        native_process=process,native_process_sha256=c.digest(process),model_loaded=False,new_optimizer_updates=0,
        resource_proof_sha256=c.digest(proof),native_model_permission=False,physics_authorized=False)
    if 'queue_dispatch_mode' in r:
        ack,plan=identity.current_queue(root,r,request_sha256=value['request_sha256'])
        value.update(native_queue_acceptance=ack,native_queue_plan=plan)
    identity.sealed_identity(m,r,value,run_uuid=root.name,manifest_sha256=r['run_manifest_sha256'],request_sha256=value['request_sha256'])
    write(out,value);out.chmod(0o400)
    proof_path=request_path.parent/'INITIAL_NATIVE_RESOURCE_PROOF.json';write(proof_path,proof);proof_path.chmod(0o400)
    return value

"""Exact GEN1 cap adoption and finite environment allocation, not physics GO.

GLOBAL is the original authority. The unavailable 4080 LOCAL is explicitly a
historical input; this module never invents or replaces a native LOCAL ledger.
"""
from copy import deepcopy
import hashlib
import json
import math

from . import amd_pilot_scope as pilot
from scripts.pi0294.pv1 import successor_cd as successor

FIELD='GEN1_AMD_layout_CORR1024_authority'
GRANT_SHA='73c7d2da1573c5ff440c8833da0f01910bca69ade2188645af866aef88c245e6'
CAPS=dict(PBS_submissions=36,Hopper_storage_bytes=1176*1024**3,
          Core_storage_bytes=477*1024**3,transfer_bytes=332*1024**3)
PROFILE='amd-gen1-teacher-environment-preparation'
ROLE='GEN1_AMD_ENVIRONMENT_ONLY_V1'
SIF_SHA='2cae97d18c6a41450bb102e3bd54819d0c30552605d222d73ffc387c6ec14c8c'


def raw(v):return (json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def need(ok,why):
    if not ok:raise PermissionError('GEN1_AMD_ADOPTION_'+why)


def validate(value):
    m=value.get(FIELD)
    need(type(m)is dict,'ACTUAL_ADOPTED_MARKER')
    s=successor.validated(value)
    need(m['schema']=='GEN1_AMD_EXISTING_GLOBAL_AUTHORITY_V1'
         and m['grant_sha256']==GRANT_SHA and m['caps']==CAPS
         and m['task']==pilot.TASK and m['T_CD']==s['T_CD']
         and m['deadline_epoch']==s['deadline_epoch']
         and m['stop_new_training_epoch']==s['stop_new_training_epoch']
         and m['LOCAL_sync']=='PENDING_NO_AMD_COPY'
         and m['new_physics_admitted']is False and m['new_training_admitted']is False
         and type(m['adopted_epoch'])in(int,float) and math.isfinite(m['adopted_epoch'])
         and s['T_CD']<=m['adopted_epoch']<s['stop_new_training_epoch'],
         'EXACT_FOUR_CAPS_CLOCK_AND_PENDING_LOCAL')
    for key in ('before_GLOBAL_sha256','historical_LOCAL_sha256','whole_future_proof_sha256'):
        need(type(m[key])is str and len(m[key])==64
             and all(c in '0123456789abcdef'for c in m[key]),'EXACT_BYTE_IDENTITIES')
    return m


def propose(global_bytes,historical_LOCAL_bytes,grant_bytes,*,now_epoch):
    need(type(grant_bytes)is bytes and sha(grant_bytes)==GRANT_SHA,'EXACT_HUMAN_GRANT')
    grant=json.loads(grant_bytes)
    need(grant['human_answer']=='批准仅这四项调整'
         and {k:v['after']for k,v in grant['caps'].items()}==CAPS
         and grant['new_door_or_bookshelf_physics_in_this_grant']is False,
         'FOUR_DIMENSIONS_ONLY')
    g=json.loads(global_bytes)
    need(FIELD not in g,'ONCE_NO_READOPTION')
    need(g['unknown_effects']==g['pending_updates']==[]
         and all(r['status']=='NATIVE_TERMINAL'for r in g['PBS_submissions'])
         and all(r['status']=='NATIVE_TERMINAL'for r in g.get('V5_CD_Core_allocations',{}).values()),
         'NO_ACTIVE_OR_UNKNOWN_COMPUTE')
    proof=pilot.full_forecast(global_bytes,historical_LOCAL_bytes)
    need({k:v['after']for k,v in proof['cap_changes_required'].items()}==CAPS,
         'EXACT_APPROVED_COMPLETE_FUTURE')
    s=successor.validated(g)
    need(s['T_CD']<=now_epoch<s['stop_new_training_epoch'],'REAL_EXISTING_WINDOW')
    for k,row in proof['dimensions'].items():
        need(row['projected']<=CAPS.get(k,row['cap']),'FULL_FUTURE_FITS:'+k)
    m=dict(schema='GEN1_AMD_EXISTING_GLOBAL_AUTHORITY_V1',task=pilot.TASK,
           grant_sha256=GRANT_SHA,caps=deepcopy(CAPS),adopted_epoch=now_epoch,
           T_CD=s['T_CD'],deadline_epoch=s['deadline_epoch'],
           stop_new_training_epoch=s['stop_new_training_epoch'],
           before_GLOBAL_sha256=sha(global_bytes),historical_LOCAL_sha256=sha(historical_LOCAL_bytes),
           LOCAL_sync='PENDING_NO_AMD_COPY',whole_future_proof_sha256=sha(raw(proof)),
           new_physics_admitted=False,new_training_admitted=False)
    out=deepcopy(g);out[FIELD]=m;out['revision']+=1;validate(out)
    need(all(out[k]==v for k,v in g.items()if k!='revision'),'OLD_COSTS_HISTORY_AND_CLOCK_IMMUTABLE')
    return dict(proposed_global=out,marker=m,full_future_proof=proof,
                actual_LOCAL_changed=False,native_admission=False)


def environment_allocation(value,*,run_uuid,manifest_sha256,now_epoch):
    """Consume only the declared control/IO slice, no Teacher/physics budget.

    The classification stays in GLOBAL. Per-run files are effect counters,
    not another source of CPU, storage or scientific authorization.
    """
    m=validate(value)
    need(m['T_CD']<=now_epoch<m['deadline_epoch'],'LIVE_ENVIRONMENT_WINDOW')
    need(type(run_uuid)is str and run_uuid=='gen1-amd-environment-01'
         and type(manifest_sha256)is str and len(manifest_sha256)==64,'ONE_FIXED_METADATA_ROLE')
    need('environment_preparation'not in value[FIELD],'ONCE_METADATA_ALLOCATION')
    out=deepcopy(value)
    row=dict(run_uuid=run_uuid,manifest_sha256=manifest_sha256,status='PREPAID_ENVIRONMENT_ONLY',
        site='corelab-amd-server',Core_CPU_upper_s=1800,LOCAL_CPU_upper_s=1800,
        retained_output_upper_bytes=2*1024**3,transfer_upper_bytes=2*1024**3,
        constructors=0,physics_ticks=0,models=0,PBS=0,
        included_in_existing_GEN1_full_forecast=True,actual_LOCAL_changed=False)
    out[FIELD]['environment_preparation']=row;out['revision']+=1
    need(all(out[k]==v for k,v in value.items()if k not in(FIELD,'revision')),
         'ENVIRONMENT_ONLY_CLASSIFICATION')
    return dict(proposed_global=out,row=row,new_GPU=0,new_PBS=0)


def static_allocation(value,*,run_uuid,manifest_sha256,environment_result,now_epoch):
    """Original GEN1 M2, exactly44 constructor upper and zero integrations."""
    m=validate(value)
    need(m['T_CD']<=now_epoch<m['deadline_epoch'],'LIVE_STATIC_WINDOW')
    need('static_previews'not in m and run_uuid=='gen1-amd-static24-01'
         and type(manifest_sha256)is str and len(manifest_sha256)==64,'ONE_FIXED_M2_CHILD')
    need(environment_result.get('status')=='PASS_ACTUAL_AMD_CONTAINER_METADATA_ONLY'
         and environment_result.get('SIF_sha256')==SIF_SHA
         and environment_result.get('returncode')==0
         and environment_result.get('constructors')==environment_result.get('physics_ticks')
             ==environment_result.get('models')==0,'REAL_ENVIRONMENT_METADATA_PREDECESSOR')
    out=deepcopy(value)
    row=dict(run_uuid=run_uuid,manifest_sha256=manifest_sha256,status='RESERVED_NOT_SPENT',
        site='corelab-amd-server',Core_CPU_upper_s=1200,engine_starts=44,
        physics_ticks=0,episode_starts=0,models=0,RGB_images=24,
        environment_result_sha256=sha(raw(environment_result)),
        included_in_existing_GEN1_full_forecast=True)
    out[FIELD]['static_previews']=row;out['revision']+=1
    return dict(proposed_global=out,row=row,new_GPU=0,new_PBS=0)


def capture_allocation(value,*,packet_bytes,workers,static_closure_sha256,now_epoch):
    """Bind all twelve predeclared first attempts inside the original envelope."""
    from .amd_capture import cards
    m=validate(value);prior=m.get('static_previews',{})
    need(prior.get('status')=='NATIVE_TERMINAL'and prior.get('native_OS_returncode')==0
         and prior.get('actual_images')==24 and prior.get('actual_constructor_attempts')==44
         and prior.get('actual_integrations')==0,'REAL_M2_TERMINAL_FIRST')
    need('teacher_pilot'not in m and type(workers)is dict and set(workers)=={'0','1'},
         'TWO_FIXED_WORKERS_NO_PILOT_REPLAY')
    need(m['T_CD']<=now_epoch<m['stop_new_training_epoch']-18.25*3600,
         'TIME_REMAINS_FOR_EXPORT_AND_ORIGINAL_TRAINING')
    need(type(static_closure_sha256)is str and len(static_closure_sha256)==64,'BOUND_STATIC_CLOSURE')
    for index in(0,1):
        row=workers[str(index)];expected=[c['id']for c in cards(packet_bytes,index)]
        need(row['attempt_ids']==expected and row['run_uuid']=='gen1-amd-teacher-'+str(index).zfill(2)
             and type(row['manifest_sha256'])is str and len(row['manifest_sha256'])==64
             and type(row['root'])is str and row['root'].endswith('/children/'+row['run_uuid']),
             'EXACT_GEN_TRAIN_WORKER_AND_NO_RESAMPLED_CELL')
    need(len({r['root']for r in workers.values()})==2,'DISJOINT_NATIVE_OUTPUTS')
    out=deepcopy(value)
    row=dict(status='RESERVED_FIRST_ATTEMPTS_ONLY',workers=deepcopy(workers),
        episode_starts=12,physics_ticks=384000,engine_starts=1008,Core_CPU_s=43200,
        per_cell_task_shadow_constructors=81,per_cell_readonly_audit_constructors=3,
        maximum_workers=2,static_closure_sha256=static_closure_sha256,
        static_native_counter_reused_with_same_capture_py_and_SIF=True,
        all_valid_failures_retained=True,extra_counter_probes=0,
        updates=0,queries=0,models=0)
    out[FIELD]['teacher_pilot']=row;out['revision']+=1
    return dict(proposed_global=out,row=row,new_GPU=0,new_PBS=0)

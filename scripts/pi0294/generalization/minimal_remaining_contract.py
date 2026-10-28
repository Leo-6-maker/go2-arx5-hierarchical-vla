"""Typed remaining-session semantics after the actual E0/E1 saved prefix.

Old request effects remain historical charges. The three new sessions expose
their own debits separately from15 already saved E2 zero-step observations.
This pure contract is not a registered request, release or native permission.
"""
from copy import deepcopy
import json
from . import minimal_execution as e,minimal_partial_completion_scope as authority

VERSION='ARX5_MINIMAL_REAL_SAVED_PREFIX_REMAINDER_V1'
PROFILE='vanda-arx5-minimal-raw-parent-first512-saved-prefix-completion'
PREFIX='plans/MINIMAL_ACTUAL_CLOSED_SAVED_PREFIX_RESULT.private.json'
CLOSURE='plans/MINIMAL_ACTUAL_SAVED_PREFIX_AUDIT_CLOSED.private.json'
RESULT='9c4ac800d8993809127a0808be98971f3a12b2c839fdfbff7ffcf3ab707300fb'
CLOSED='e7e1c24e941eb76a11946b32c6f0ea12d08b8e7d1a9ecfd260ee967715e226c7'
AUDIT_LEASE='d8164c5d6e110ecb0bc78f8e5735b3ec9380b0eb1994741081a7baa578a42ba8'
OLD_SOURCE='7c7e77d4b83a7b3a30f336a6f53f4788942c2373'
CP='dcda3434a368e5d08f8709bd6cbf0b3c261a0b3ddc3960ec7f40b22787c1917c'
OLD_PAID=dict(optimizer_updates=2,generation_queries=71,model_loads=5)
NEW_UPPER=dict(optimizer_updates=1026,generation_queries=76,model_loads=7)
KEYS=tuple(NEW_UPPER)


def saved_prefix(result_bytes,closure_bytes):
    e.need(type(result_bytes)is bytes and e.sha(result_bytes)==RESULT
        and type(closure_bytes)is bytes and e.sha(closure_bytes)==CLOSED,
        'REMAINDER_ACTUAL_FULL_NATIVE_AUDIT_AND_CLOSED_RECEIPT_BYTES')
    p,c=json.loads(result_bytes),json.loads(closure_bytes);cp=p['saved_E1_checkpoint']
    e.need(p['status']=='PASS_INDEPENDENT_ACTUAL_E0_E1_COMPLETE_SAVED_PREFIX_AND_E2_ZERO_OUTPUTS'
        and p['job_id']==authority.JOB and p['request_id']==authority.REQUEST
        and p['science_source_sha']==OLD_SOURCE
        and p['completed_prefix']==[s['id']for s in e.native_sessions()[:2]]
        and p['matched_zero_pairs']==30 and p['E2_saved_zero_queries']==15
        and p['E2_optimizer_calls']==0 and p['formal_E1_E2_steps']==[0,0]
        and p['new_model_loads']==p['new_queries']==p['new_updates']==p['new_physics']==0
        and p['CUDA_initialized']is False and p['completion_native_admission']is False,
        'REMAINDER_EXACT_COMPLETED_PREFIX_NOT_ALL_ARMS_OR_A_FORMAL_ENDPOINT')
    e.need(cp['checkpoint_manifest_sha256']==CP and cp['global_step']==2 and cp['cursor']==64
        and cp['scheduler_horizon']==1024 and cp['full_resume']is cp['qualification_only']is True
        and cp['raw_parent_files_identity']==e.training.RAW_PARENT_FILES_ID and cp['statistics_sha256']==e.R0
        and len(cp['frozen_parameter_identity'])==1032
        and p['restored_pair']['reference_sha256']==p['restored_pair']['reloaded_sha256']
        and p['actual_E1_restore']['optimizer_scheduler_model_RNG_sampler_checked']is True
        and p['actual_E1_restore']['qualification_never_becomes_formal']is True
        and c['lease_sha256']==AUDIT_LEASE and c['owner_OS_exit']==0
        and c['original_result_sha256']==RESULT,
        'REMAINDER_ORIGINAL_STEP2_FULL_STATE_RELOAD_AND_NATURAL_CLOSED_AUDITOR')
    return dict(schema='ARX5_MINIMAL_ACTUAL_PREFIX_IMMUTABLE_INPUT_V1',result_sha256=RESULT,
        closure_sha256=CLOSED,audit_lease_sha256=AUDIT_LEASE,old_request_id=authority.REQUEST,
        old_job_id=authority.JOB,old_source_sha=OLD_SOURCE,old_paid_effects=deepcopy(OLD_PAID),
        E1_full_qualification_manifest_sha256=CP,completed_sessions=deepcopy(p['completed_prefix']),
        E2_saved_zero_queries=15,extra_interrupted_E2_raw_load_retained_but_not_reused=1,
        formal_initialization='SAME_ORIGINAL_RAW_PARENT_FRESH_OPTIMIZER_SCHEDULER_RNG_SAMPLER',
        current_file_identity_and_native_release_still_required=True,native_admission=False)


def sessions():
    out=[]
    for i,spec in enumerate(e.native_sessions()[2:]):
        row=deepcopy(spec);carried=15 if i==0 else 0
        row.update(carried_zero_queries=carried,own_query_limit=spec['query_limit']-carried,
            own_model_load_limit=3 if i==0 else 2,own_optimizer_limit=spec['optimizer_limit'],
            never_initialize_formal_from_qualification=True)
        out.append(row)
    e.need(sum(s['own_query_limit']for s in out)==NEW_UPPER['generation_queries']
        and sum(s['own_model_load_limit']for s in out)==NEW_UPPER['model_loads']
        and sum(s['own_optimizer_limit']for s in out)==NEW_UPPER['optimizer_updates'],
        'REMAINDER_EXACT1026_76_7_NEW_NATIVE_BOUNDS')
    return out


def session(identifier):
    rows=[s for s in sessions()if s['id']==identifier]
    e.need(len(rows)==1,'REMAINDER_NO_E0_E1_REPLAY_OR_UNPLANNED_SESSION');return rows[0]


def logical_counters(identifier,own):
    s=session(identifier)
    e.need(set(own)==set(KEYS)and all(type(own[k])is int and own[k]>=0 for k in KEYS)
        and own['optimizer_updates']<=s['own_optimizer_limit']
        and own['generation_queries']<=s['own_query_limit']
        and own['model_loads']<=s['own_model_load_limit'],
        'REMAINDER_OWN_DEBITS_NOT_HISTORICAL_TOTALS')
    return dict(own,generation_queries=own['generation_queries']+s['carried_zero_queries'])


def tickets():
    return [dict(index=i,session_id=s['id'],phase=phase)for i,(s,phase)in enumerate(
        (s,p)for s in sessions()for p in('MODEL_RAW','FIRST_UPDATE','OFFICIAL_EXPORT'if s['formal_candidate']else'RESTORE_FULL'))]


def check_phase(identifier,phase,own):
    """Only counter prerequisites; actual R/queue/quota/STOP remain external."""
    s=session(identifier);logical=logical_counters(identifier,own)
    e.need(phase in('MODEL_RAW','FIRST_UPDATE','RESTORE_FULL','OFFICIAL_EXPORT'),
        'REMAINDER_NAMED_ORIGINAL_PHASE')
    if phase=='MODEL_RAW':expected=(0,0,0)
    elif phase=='FIRST_UPDATE':expected=(0,0,1)
    elif phase=='RESTORE_FULL':
        e.need(not s['formal_candidate'],'REMAINDER_RESTORE_ONLY_QUALIFICATION');expected=(2,1,1)
    else:
        e.need(s['formal_candidate'],'REMAINDER_OFFICIAL_EXPORT_ONLY_FIXED512');expected=(512,36,1)
    e.need(tuple(own[k]for k in KEYS)==expected,'REMAINDER_EXACT_OWN_PHASE_BOUNDARY')
    return dict(own_counters=deepcopy(own),logical_session_counters=logical,
        carried_zero_queries=s['carried_zero_queries'],model_or_optimizer_permission=False)


def completed_totals(own_rows):
    specs=sessions();e.need(set(own_rows)=={s['id']for s in specs},'REMAINDER_ONLY_THREE_REQUIRED_SESSIONS')
    totals={k:0 for k in KEYS}
    for s in specs:
        v=own_rows[s['id']];logical_counters(s['id'],v)
        expected=dict(optimizer_updates=s['own_optimizer_limit'],generation_queries=s['own_query_limit'],model_loads=s['own_model_load_limit'])
        e.need(v==expected,'REMAINDER_COMPLETE_ACTUAL_NEW_COUNTS_NOT_PARTIAL')
        for k in KEYS:totals[k]+=v[k]
    e.need(totals==NEW_UPPER,'REMAINDER_FULL1026_76_7_OWN_EFFECTS')
    return dict(own_native=totals,retained_old_native=deepcopy(OLD_PAID),
        actual_study_lifetime={k:totals[k]+OLD_PAID[k]for k in KEYS},
        old_E0_E1_or_E2_zero_queries_reexecuted=0,
        extra_old_interrupted_E2_model_load_retained=1,
        formal_endpoint_OR_native_success_proven=False)

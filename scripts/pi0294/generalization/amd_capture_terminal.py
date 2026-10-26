"""Pure exact R1 saved-effects closeout and separate original-data re-audit.

The original failed audit/unknown lists remain untouched. A successful worker
score is never substituted for independent raw or official R0 qualification.
"""
from copy import deepcopy
import math
from . import amd_adoption as a

SOURCE='13c2098e19f246e8c85f6bdb3ab94f5e0251398e'
ROLE='GEN1_AMD_SAVED_RAW_REAUDIT_V1'
PROFILE='amd-gen1-saved-raw-reaudit'


def failure_proposal(value,audit,*,now_epoch):
    mark=a.validate(value);old=mark['teacher_pilot'];pool=old['engineering_recovery']
    a.need(old['status']=='NATIVE_ENGINEERING_TERMINAL'and pool['status']=='RESERVED_ENGINEERING_CONTINUATION',
        'R1_AUDIT_FAILURE_ONCE')
    a.need(audit['status']=='PASS_R1_SAVED_NATIVE_EFFECTS_RAW_AUDIT_COMPATIBILITY_FAILURE'
        and audit['source']==SOURCE and audit['unknown_native_effects']==[]
        and audit['original_unknown_lists_unchanged']is True and audit['unattempted_conditions']==10
        and audit['admitted_demonstrations']==audit['new_constructors']==audit['new_integrations']==0,
        'R1_EXACT_SAVED_EFFECTS_PROOF_NOT_RAW_ADMISSION')
    a.need(mark['T_CD']<=audit['epoch']<=now_epoch and math.isfinite(now_epoch),
        'R1_FINANCIAL_ONLY_CURRENT_OR_LATE_CLOSE')
    a.need(set(audit['cases'])=={'0','1'},'R1_BOTH_COMPLETE_NATIVE_TRAJECTORIES')
    totals=dict(episode_starts=0,live_ticks=0,shadow_ticks=0,engine_starts=0);seed_errors=0;cases={}
    for key,case in audit['cases'].items():
        bound=pool['workers'][key]
        a.need(case['manifest_sha256']==bound['manifest_sha256']and case['source']==SOURCE
            and case['expected_ids']==bound['attempt_ids']and case['attempt_id']==bound['attempt_ids'][0]
            and case['raw_root']==bound['root']and case['actual_OS_returncode']==1
            and case['raw_status']=='VALID_COMPLETE'and case['paired_samples']==641
            and case['live_ticks']==case['shadow_ticks']==16000 and case['episode_starts']==1
            and case['engine_starts']==85+case['readonly_audit_constructor_attempts']
            and case['engine_starts']in(86,87)and case['native_safety_failures']==case['S1_rejections']==0
            and case['all_raw_member_hashes_verified']is True and case['old_unknown_list_preserved']is True
            and case['independent_raw_admitted']is False and type(case['original_score_success'])is bool,
            'R1_PHYSICAL_EFFECTS_COMPLETE_BUT_INDEPENDENT_AUDIT_FAILED')
        if case['original_raw_audit_error_type']=='KeyError':
            a.need(case['original_raw_audit_error']=="'seed'",'R1_ONLY_EXACT_LEGACY_SEED_METADATA_GAP');seed_errors+=1
        else:a.need(case['original_raw_audit_error_type']=='TimeoutError'
            and case['original_raw_audit_error']=='PV1_CAPTURE_ORIGINAL_BOUND_OR_STOP','R1_ONLY_SHARED_AUDIT_STOP')
        for name in ('raw_manifest_sha256','original_audit_sha256','worker_receipt_sha256','CARD_sha256'):
            a.need(type(case[name])is str and len(case[name])==64,'R1_BOUND_ORIGINAL_BYTES')
        cases[key]=deepcopy(case)
        for name in totals:totals[name]+=case[name]
    a.need(seed_errors>=1 and totals['episode_starts']==audit['actual_episode_starts']==2
        and totals['live_ticks']==audit['live_ticks']==32000 and totals['shadow_ticks']==audit['shadow_ticks']==32000
        and totals['engine_starts']==audit['engine_starts']==173,'R1_NATIVE_EFFECT_TOTALS')
    cost=audit['Core_CPU_s'];a.need(type(cost)in(int,float)and math.isfinite(cost)and 0<cost<=43200,
        'R1_OWNER_AND_CHILD_ACTUAL_CPU')
    out=deepcopy(value);row=out[a.FIELD]['teacher_pilot']['engineering_recovery']
    row.update(status='NATIVE_AUDIT_ENGINEERING_TERMINAL',actual=dict(**totals,Core_CPU_s=cost,
        raw_payload_bytes=audit['raw_payload_bytes'],unknown_native_effects=[],models=0,queries=0,updates=0,
        independently_admitted_demonstrations=0,original_worker_scores_successful=sum(x['original_score_success']for x in cases.values())),
        independent_saved_effects_audit_sha256=a.sha(a.raw(audit)),saved_full_cases=cases,
        unattempted_condition_ids=[x for key in ('0','1')for x in pool['workers'][key]['attempt_ids'][1:]],
        original_unknown_lists_and_failed_audits_preserved=True,original_R0_export_required=True,
        native_training_admitted=False)
    out['revision']+=1
    a.need(all(out[k]==v for k,v in value.items()if k not in(a.FIELD,'revision')),
        'R1_ONLY_EFFECT_CLASSIFICATION_OLD_ACCOUNTING_UNCHANGED')
    return dict(proposed_global=out,actual=deepcopy(row['actual']),new_PBS=0,new_GPU=0,native_admission=False)


def reaudit_allocation(value,*,run_uuid,manifest_sha256,closed_leases,now_epoch):
    mark=a.validate(value);row=mark['teacher_pilot']['engineering_recovery']
    a.need(row['status']=='NATIVE_AUDIT_ENGINEERING_TERMINAL'and 'saved_data_reaudit'not in row
        and len(row['saved_full_cases'])==2 and row['actual']['unknown_native_effects']==[],
        'R1_SAVED_DATA_ONCE_NO_RECAPTURE')
    a.need(mark['T_CD']<=now_epoch<mark['stop_new_training_epoch']-18.25*3600,
        'SAVED_AUDIT_WITH_REMAINING_TRAINING_TIME')
    a.need(run_uuid=='gen1-amd-saved-r1-reaudit-01'and type(manifest_sha256)is str
        and len(manifest_sha256)==64 and set(closed_leases)=={'owner','0','1'},'SAVED_AUDIT_OWN_ROLE')
    for key,lease in closed_leases.items():
        a.need(lease['status']=='CLOSED'and lease['terminal_kind']=='COMPLETE_NATIVE_TRAJECTORIES_WITH_RAW_AUDIT_ENGINEERING_FAILURE'
            and lease['independent_saved_effects_audit_sha256']==row['independent_saved_effects_audit_sha256']
            and lease['original_failed_audits_and_unknown_lists_preserved']is True,'SAVED_AUDIT_CLOSED_PREDECESSOR')
        if key!='owner':a.need(lease['manifest_sha256']==row['workers'][key]['manifest_sha256'],'SAVED_AUDIT_WORKER_IDENTITY')
    out=deepcopy(value);out[a.FIELD]['teacher_pilot']['engineering_recovery']['saved_data_reaudit']=dict(
        status='RESERVED_SAVED_DATA_ONLY',run_uuid=run_uuid,manifest_sha256=manifest_sha256,
        engine_starts=4,physics_ticks=0,episode_starts=0,Core_CPU_upper_s=600,LOCAL_CPU_upper_s=300,
        retained_output_upper_bytes_within_existing_320MiB=16*1024**2,
        original_raw_files_may_be_modified=False,Teacher_replay=False,models=0,queries=0,updates=0,
        input_cases_sha256=a.sha(a.raw(row['saved_full_cases'])))
    out['revision']+=1
    return dict(proposed_global=out,new_GPU=0,new_PBS=0)

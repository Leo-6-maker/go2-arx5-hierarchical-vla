"""Join verified parent receipts without inventing task results or stage success."""
from collections import Counter
from copy import deepcopy

from scripts.pi0294.scene_cards import digest
from .gates import TERMINAL,first_four,stage_gate
from .plan import capture_cards


def geometry_fingerprint(card):
    return digest({k:card[k] for k in ('source_xy','goal_xy','cube_yaw','table_top_m','visual')})


def joined_attempt(card,capture=None,export=None,*,actual_card=None):
    if capture is None:
        if export is not None:raise PermissionError('DATA_COV1_EXPORT_WITHOUT_CAPTURE')
        return dict(id=card['id'],scene_group=card['group_id'],task=card['task'],status='NOT_RUN',
            task_success=False,task_success_known=False,admitted=False,raw_admitted=False,codec_admitted=False,
            native_faults=0,native_safety_evaluated=False,unknown_effects=[],auditable_identity_and_clock=False)
    row=deepcopy(capture)
    if (row['id']!=card['id'] or row['scene_group']!=card['group_id'] or row['task']!=card['task']
            or row['status'] not in TERMINAL|{'RAW_CLOSED_PENDING_CODEC'}):
        raise PermissionError('DATA_COV1_EXACT_PLANNED_PARENT_RECEIPT')
    row.setdefault('task_success_known',row['status'] in ('VALID_TASK_FAIL','RAW_AUDIT_REJECT','RAW_CLOSED_PENDING_CODEC',
        'REFERENCE_REJECT','S1_PREDICTED_REJECT','CODEC_REJECT','ADMITTED'))
    row['native_safety_evaluated']=bool(row.get('auditable_identity_and_clock'))
    row['admitted']=row['codec_admitted']=False
    if row['status']=='ADMITTED':raise PermissionError('DATA_COV1_ADMISSION_NEEDS_INDEPENDENT_EXPORT')
    if actual_card is not None:
        if actual_card['id']!=card['id'] or actual_card['task']!=card['task']:
            raise PermissionError('DATA_COV1_ACTUAL_CARD_IDENTITY')
        row['actual_geometry_fingerprint']=geometry_fingerprint(actual_card)
        row['actual_geometry_revision']=actual_card.get('geometry_revision','V1')
    if export is not None:
        if (row['status']!='RAW_CLOSED_PENDING_CODEC' or not row['raw_admitted'] or not row['task_success']
                or actual_card is not None and actual_card!=card
                or not row.get('auditable_identity_and_clock') or row['native_faults'] or row['unknown_effects']
                or any(export[k]!=card[k] for k in ('id','task','split'))
                or export['scene_group']!=card['group_id'] or export['nominal_clearance_m']!=.004
                or export['source_card_sha256']!=digest(card)
                or export['source_raw_manifest_sha256']!=row['raw_manifest_sha256']
                or export['source_full_audit_sha256']!=row['raw_audit_sha256']
                or not export['full_task_success'] or not export['independent_raw_admitted']
                or export['windows']!=601 or export['codec']['H40_windows_actually_checked']!=601
                or export['codec']['statistics_refit'] is not False or export['codec']['encode_clip_components']!=0):
            raise PermissionError('DATA_COV1_RAW_CODEC_ACTUAL_CARD_AND_IMMUTABLE_LABEL_CHAIN')
        if export['admitted_for_training']:
            chain=export.get('official_chain',{})
            if (export['status']!='PASS_SAVED_TEACHER_EXPORT_AND_OFFICIAL_CHAIN'
                    or export['codec']['status']!='PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'
                    or chain.get('actual_anchors')!=[0,80,160,320,480,600]
                    or chain.get('deterministic_repeat') is not True
                    or chain.get('full_641_two_camera_raw_RGB256_verified') is not True
                    or chain.get('model_forward') is not False):
                raise PermissionError('DATA_COV1_FULL_OFFICIAL_READER_CHAIN_REQUIRED')
            row.update(status='ADMITTED',admitted=True,codec_admitted=True,
                actual_geometry_fingerprint=geometry_fingerprint(card),actual_geometry_revision=card.get('geometry_revision','V1'))
        elif (export['status']=='BLOCKED_FROZEN_CODEC_PHYSICAL_ERROR'
                and export['codec']['status']=='BLOCKED_FROZEN_CODEC_PHYSICAL_ERROR'):
            row['status']='CODEC_REJECT'
        else:raise PermissionError('DATA_COV1_UNKNOWN_EXPORT_FAILURE_NOT_A_SCIENTIFIC_NEGATIVE')
        row['export_receipt_identity']=digest(export)
    elif row['status']=='RAW_CLOSED_PENDING_CODEC':
        row['status']='ENGINEERING_INCOMPLETE'
        row['reason']='FULL_RAW_AVAILABLE_CODEC_NOT_YET_VERIFIED'
    if not row['task_success_known'] and row['task_success']:
        raise PermissionError('DATA_COV1_UNKNOWN_TASK_RESULT_CANNOT_BE_SUCCESS')
    return row


def stage_reports(groups,stage,captures,exports,*,actual_cards=None):
    cards=capture_cards(groups,stage);planned={c['id']:c for c in cards}
    if not set(captures)<=set(planned) or not set(exports)<=set(captures):
        raise PermissionError('DATA_COV1_ONLY_PLANNED_FIRST_ATTEMPTS')
    actual_cards=actual_cards or {}
    if not set(actual_cards)<=set(captures):raise PermissionError('DATA_COV1_ACTUAL_CARD_WITHOUT_ATTEMPT')
    rows=[joined_attempt(c,captures.get(c['id']),exports.get(c['id']),actual_card=actual_cards.get(c['id'])) for c in cards]
    admitted_raw=[r['source_raw_manifest_sha256'] for r in exports.values() if r.get('admitted_for_training')]
    if len(set(admitted_raw))!=len(admitted_raw):
        raise PermissionError('DATA_COV1_DISTINCT_PARENT_RAW_IDENTITY_NOT_RELABELED_DUPLICATES')
    gate=stage_gate(groups,stage,rows)
    pairs={task:{(r['scene_group'],r['actual_geometry_fingerprint']) for r in rows
        if r['task']==task and r['admitted']} for task in ('T1','T2')}
    real_pairs=len(pairs['T1']&pairs['T2'])
    gate['paired_groups']=real_pairs
    gate['checks']['paired_support']=real_pairs>=gate['paired_minimum']
    gate['count_gate_pass']=all(gate['checks'].values())
    strata={}
    for c,r in zip(cards,rows):
        key='/'.join((c['appearance_profile'],c['direction'],c['distance_bin']))
        item=strata.setdefault(key,{t:Counter() for t in ('T1','T2')})[c['task']]
        item.update(planned=1,attempted=int(c['id'] in captures),task_success=int(r['task_success']),
            task_result_known=int(r['task_success_known']),admitted=int(r['admitted']),
            native_faults=r['native_faults'],unknown_effect_parents=int(bool(r['unknown_effects'])))
    event_rows={}
    for identifier,e in exports.items():
        kinematics=e.get('committed_horizon_kinematics')
        compact=None if kinematics is None else dict(legal_windows=kinematics['legal_windows'],
            sample_interval_s=kinematics['sample_interval_s'],
            regions={region:{name:dict(maximum=values['maximum'],windows=len(values['per_window_max']))
                for name,values in metrics.items()} for region,metrics in kinematics['regions'].items()},
            full_per_window_arrays_in_bound_export_receipt=True)
        event_rows[identifier]=dict(
            codec_status=e['codec']['status'],first_codec_error=deepcopy(e['codec'].get('diagnostics',{}).get('first_error')),
            actual_physical_event_receipt=deepcopy(e.get('actual_physical_event_receipt')),
            committed_horizon_kinematics=compact,full_export_receipt_identity=digest(e),
            relative_coordinate_2_is_not_world_Z=True)
    first=first_four(groups,[r for r in rows if r['id'] in {c['id'] for c in cards[:4]}])
    return dict(
        all_attempts=dict(schema='PI0294_DATA_COV1_ALL_ATTEMPTS_V1',stage=stage,rows=rows,
            planned=len(cards),actual_first_attempts=len(captures),statuses=dict(Counter(r['status'] for r in rows)),
            physical_replacements=0,valid_task_failures=sum(r['status']=='VALID_TASK_FAIL' for r in rows),
            unknown_task_results=sum(not r['task_success_known'] and r['id'] in captures for r in rows)),
        coverage_and_events=dict(schema='PI0294_DATA_COV1_COVERAGE_AND_EVENTS_V1',stage=stage,
            strata=strata,paired_admitted_same_geometry_groups=real_pairs,parents=event_rows,
            metric='TEACHER_DATA_QUALIFICATION_NOT_POLICY_SUCCESS',PV1_bank_changed=False),
        stage_gate=gate,first_four_gate=first,selected='NONE',deployment='NONE')

"""Reconcile saved TRAIN attempts without changing any training bank or gate."""
from collections import Counter
import math


UNREVIEWED = 'NOT_EVALUATED_AFTER_FIXED_BANK_FREEZE'
BLOCKED = 'BLOCKED_FROZEN_CODEC_PHYSICAL_ERROR'
PASS = 'PASS_SAVED_TEACHER_EXPORT_AND_OFFICIAL_CHAIN'
GATES = dict(XYZ_m=1e-5, SO3_rad=1e-4, gripper=1e-5, relative_SE3_max_abs=1e-8)


def _unique(rows):
    result = {r['id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('D0_DUPLICATE_EPISODE_ID')
    return result


def reconcile(card, rejection_rows, new_rows):
    """Inputs are exact published/remote receipts; no inferred codec admission."""
    original = _unique(card['attempt_rows'])
    rejected = _unique(rejection_rows)
    fresh = _unique(new_rows)
    if len(original) != card['population']['actual_first_attempts']:
        raise ValueError('D0_INCOMPLETE_ORIGINAL_POPULATION')
    if any(r['split'] != 'TRAIN' for r in original.values()):
        raise PermissionError('D0_TRAIN_ONLY')
    expected_rejected = {k for k, r in original.items() if r['official_codec_row_status'] == BLOCKED}
    if set(rejected) != expected_rejected:
        raise ValueError('D0_REJECTION_JOIN_MUST_BE_EXACT')
    unreviewed = {k for k, r in original.items() if r['official_codec_row_status'] == UNREVIEWED}
    candidates = {k for k in unreviewed if original[k]['full_raw_audit_admitted']}
    if set(fresh) != candidates:
        raise ValueError('D0_REVIEW_ONLY_ALL_RAW_ADMITTED_UNREVIEWED_ROWS')
    for k, row in {**rejected, **fresh}.items():
        old = original[k]
        if (row['split'] != 'TRAIN' or row['task'] != old['task']
                or row['scene_group'] != old['scene_group']
                or row['source_raw_manifest_sha256'] != old['raw_manifest_sha256']
                or row['source_full_audit_sha256'] != old['full_raw_audit_sha256']
                or not old['full_raw_audit_admitted'] or row['codec']['H40_windows_actually_checked'] != 601
                or row['codec']['statistics_refit'] is not False
                or row['codec']['native_decode_unchanged'] is not True):
            raise ValueError('D0_ORIGINAL_RAW_AUDIT_AND_CODEC_IDENTITY')
        passed = row['status'] == PASS
        codec = row['codec']
        if (codec['frozen_numeric_gates'] != GATES or set(codec['maximum_error']) != set(GATES)
                or any(not math.isfinite(v) or v < 0 for v in codec['maximum_error'].values())
                or all(codec['maximum_error'][k] <= limit for k, limit in GATES.items()) is not passed):
            raise ValueError('D0_UNCHANGED_NUMERIC_GATES_AND_ACTUAL_ERRORS')
        if (row['status'] not in (PASS, BLOCKED) or row['admitted_for_training'] is not passed
                or (row['codec']['status'] == 'PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP') is not passed):
            raise ValueError('D0_NO_RELABELLED_CODEC_FAILURE')
    bank = [r for r in original.values() if r['fixed_bank_member']]
    baseline_groups = {r['scene_group'] for r in bank}
    alternates = [r for r in original.values() if r['official_codec_row_status']==PASS and not r['fixed_bank_member']]
    if any(not r['full_raw_audit_admitted'] or r['nominal_clearance_m']!=.012 for r in alternates):
        raise ValueError('D0_PRIOR_QUALIFIED_ALTERNATE_IDENTITY')
    admitted = [r for r in fresh.values() if r['status'] == PASS]
    original_reference = [r for r in admitted if r['nominal_clearance_m'] == .004]
    union = [*bank, *original_reference]
    tasks = {t: {r['scene_group'] for r in union if r['task'] == t} for t in ('T1', 'T2')}
    matrix = []
    for family in ('F0', 'F1', 'F2'):
        for direction in ('+Y', '-Y'):
            own = [r for r in original.values() if r['family'] == family and r['direction'] == direction]
            selected = [r for r in union if r['family'] == family and r['direction'] == direction]
            matrix.append(dict(family=family, direction=direction,
                original_independent_groups=len({r['scene_group'] for r in own}),
                frozen_T1=sum(r['fixed_bank_member'] and r['task'] == 'T1' for r in own),
                frozen_T2=sum(r['fixed_bank_member'] and r['task'] == 'T2' for r in own),
                future_original_reference_T1=sum(r['task'] == 'T1' for r in selected),
                future_original_reference_T2=sum(r['task'] == 'T2' for r in selected)))
    diagnostics = []
    for row in rejected.values():
        codec = row['codec']
        failed = [k for k, value in codec['maximum_error'].items() if value > codec['frozen_numeric_gates'][k]]
        diagnostics.append(dict(id=row['id'], task=row['task'], family=row['family'], direction=row['direction'],
            nominal_clearance_m=row['nominal_clearance_m'], failed_gates=failed,
            maximum_error=codec['maximum_error'], frozen_numeric_gates=codec['frozen_numeric_gates'],
            inverse_clip_by_EEF10_channel=codec['inverse_clip_by_EEF10_channel'],
            action_preclip_components=codec['action_preclip_components'],
            native_inverse_clip_components_float32=codec['native_inverse_clip_components_float32'],
            state_preclip_components=codec['state_preclip_components'],
            first_error_anchor_and_physical_phase='NOT_RETAINED_IN_ORIGINAL_CODEC_RECEIPT',
            original_status=BLOCKED, remains_excluded=True))
    return dict(schema='PI0294_FUTURE_DATA_D0_RECONCILIATION_V1',
        status='OFFLINE_QUALIFICATION_ONLY_NOT_PV1_BANK_UPDATE',
        resource_authority_release='9eff29a44a193dd4d3620dc0bfe5764a402eff42',
        original_campaign='20261018T182956Z-localpc-pv1',independent_CPU_child='future-data-d0-01',
        original_attempts=len(original), original_unreviewed=len(unreviewed),
        raw_audit_failure_excluded=sorted(unreviewed-candidates), completed_offline_reviews=len(fresh),
        new_official_chain_pass=len(admitted), new_codec_blocked=len(fresh)-len(admitted),
        new_pass_by_task=dict(Counter(r['task'] for r in admitted)),
        new_pass_original_reference=len(original_reference),
        new_pass_alternate_12mm=sum(r['nominal_clearance_m'] == .012 for r in admitted),
        frozen_bank_episodes=len(bank), frozen_bank_groups=len(baseline_groups),
        future_original_reference_union=dict(episodes=len(union), independent_groups=len(tasks['T1']|tasks['T2']),
            paired_groups=len(tasks['T1']&tasks['T2']), T1_groups=len(tasks['T1']), T2_groups=len(tasks['T2']),
            newly_covered_groups=sorted((tasks['T1']|tasks['T2'])-baseline_groups),
            candidate_only_not_a_sealed_or_authorized_training_bank=True),
        family_direction_matrix=matrix,
        prior_qualified_alternates=dict(episodes=len(alternates),
            additional_independent_groups=len({r['scene_group'] for r in alternates}-baseline_groups),
            rows=[{k:r[k] for k in ('id','scene_group','task','nominal_clearance_m',
                'raw_manifest_sha256','full_raw_audit_sha256','official_codec_row_sha256')} for r in alternates]),
        all_attempt_future_eligibility=[dict(id=k,scene_group=r['scene_group'],task=r['task'],
            family=r['family'],direction=r['direction'],nominal_clearance_m=r['nominal_clearance_m'],
            PV1_frozen_bank_member=r['fixed_bank_member'],original_codec_status=r['official_codec_row_status'],
            future_status='EXCLUDE_RAW_AUDIT_FAIL' if not r['full_raw_audit_admitted'] else fresh[k]['status'] if k in fresh else r['official_codec_row_status'],
            future_only_no_PV1_injection=True) for k,r in original.items()],
        old_rejection_summary=dict(episodes=len(diagnostics),
            failed_gate_combinations=dict(Counter('+'.join(r['failed_gates']) for r in diagnostics)),
            parents_with_inverse_clip_by_channel=[sum(r['inverse_clip_by_EEF10_channel'][i]>0 for r in diagnostics) for i in range(10)],
            channel_indices_are_relative_action_codec_coordinates_not_world_collision_axes=True,
            inverse_clip_does_not_identify_physical_task_failure_or_cause_of_student_failure=True),
        old_rejection_diagnostics=diagnostics,
        future_qualification_rows=[dict(id=r['id'], status=r['status'],
            scene_group=r['scene_group'], task=r['task'], nominal_clearance_m=r['nominal_clearance_m'],
            original_bank_member=False, original_raw_manifest_sha256=r['source_raw_manifest_sha256'],
            original_full_audit_sha256=r['source_full_audit_sha256'], codec=r['codec']) for r in fresh.values()],
        PV1_bank_changed=False, statistics_refit=False, new_physics=0, new_GPU_PBS=0,
        new_training_updates=0, D1_pilot='NOT_AUTHORIZED', selected='NONE', deployment='NONE')


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card',required=True)
    parser.add_argument('--rejections',required=True)
    parser.add_argument('--offline-result',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    def read(path):return json.loads(Path(path).read_text())
    card=read(args.card);prior=read(args.rejections);actual=read(args.offline_result)
    if (actual['status']!='CLOSED_D0_43_EXISTING_TRAIN_OFFICIAL_CHAIN_ACTUAL'
            or actual['lease']!='CLOSED' or not actual['PV1_bank_139_manifest_unchanged']
            or actual['ledger']['unknown_effects'] or actual['result']['CUDA_initialized']
            or actual['result']['neural_modules_imported']):
        raise PermissionError('D0_REQUIRES_ACTUAL_CLOSED_NATIVE_RECEIPT')
    if any(actual['ledger'][k]!=0 for k in ('engine_starts','episode_starts','ticks','live_ticks',
            'shadow_ticks','GR00T_queries','training_updates','CPU_reserved_s')):
        raise PermissionError('D0_REQUIRES_ZERO_SCIENTIFIC_EFFECTS_AND_SETTLED_RESERVATION')
    indexed=_unique(card['attempt_rows'])
    if any(indexed[r['row']['id']]['official_codec_row_sha256']!=r['row_sha256'] for r in prior['rejections']):
        raise ValueError('D0_ORIGINAL_REJECTION_RECEIPT_HASH_MISMATCH')
    rows=[r['row'] for r in prior['rejections']]
    report=reconcile(card,rows,actual['result']['rows'])
    report['provenance']=dict(original_card_sha256=hashlib.sha256(Path(args.card).read_bytes()).hexdigest(),
        prior_rejection_receipts_sha256=hashlib.sha256(Path(args.rejections).read_bytes()).hexdigest(),
        original_rejection_row_sha256={r['row']['id']:r['row_sha256'] for r in prior['rejections']},
        native_terminal_receipt_sha256=hashlib.sha256(Path(args.offline_result).read_bytes()).hexdigest(),
        native_run_manifest_sha256=actual['manifest_sha256'],native_result_sha256=actual['result_sha256'])
    report['actual_resources']=dict(Core_worker_CPU_s=actual['ledger']['CPU_s'],
        Core_control_charged_upper_s=360,local_control_charged_upper_s=600,
        Core_reserved_CPU_s=0,lease='CLOSED',output_files=actual['output_files'],
        output_logical_bytes=actual['output_logical_bytes'],
        exclusive_payload_allocated_bytes=actual['exclusive_payload_allocated_bytes'],
        reused_hardlinked_RGB_not_new_disk_payload=True,
        runtime_source='2ac48b1a666bb752c048ae7e03357e4dac7fe9f0',
        exact_runtime_CI_run=39364285377,exact_runtime_CI_job=118437383591)
    report['observed_epoch']=actual['observed_epoch']
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if any(s in text for s in ('/home/','/scratch/','/var/cache/')):
        raise PermissionError('D0_PUBLIC_REPORT_CONTAINS_PRIVATE_PATH')
    with Path(args.output).open('x') as f:f.write(text)
    print(json.dumps({k:report[k] for k in ('status','completed_offline_reviews','new_official_chain_pass','new_codec_blocked','future_original_reference_union')}))


if __name__=='__main__':main()

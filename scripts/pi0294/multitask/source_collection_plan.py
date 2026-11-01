"""Finite next collection batch; registration waits for the entire old chain.

No scheduler, resource ledger, worker launch or active-queue mutation lives
here. The existing owner uses these fixed scientific choices and gates.
"""
import hashlib
import json
from pathlib import Path
from .book_geometry_cards import conditions as geometry_conditions
from .book_late_recovery import conditions as recovery_conditions

VERSION='CORE_AFTER_PHASE4096_SOURCE_COLLECTION_V1'


def plan():
    return dict(schema=VERSION,current_chain_required='TRAIN_AND_BOTH_TRAIN32_DIAGNOSTICS_AND_ORIGINAL_DEV10_CLOSED',
        active_queue_hot_edit=False,GPU_serial=True,source_simulation_only=True,
        batches=[dict(id='A_CUSTOM_CABINET',pilot=[1],remainder=[2,3,4],
            card_set='CUSTOM_CABINET_PORT08_FIXTURE_XY_TRAIN_SEEDS_V1',
            continuation_requires='PILOT_FULL_NATIVE_SUCCESS_SAFETY_INTEGRITY_AND_OFFICIAL_EXPORT'),
            dict(id='B_BOOK_LATE_RECOVERY',pilot=recovery_conditions('pilot'),remainder=recovery_conditions('remainder'),
            continuation_requires='BOTH_PILOTS_ACTUAL_STUDENT_HELD_TAKEOVER_AND_FULL_SAFE_RECOVERY'),
            dict(id='C_BOOK_NEW_GEOMETRY',pilot=[geometry_conditions()[i]for i in(0,2)],
            remainder=[geometry_conditions()[i]for i in(1,3,4,5)],
            continuation_requires='BOTH_DIFFERENT_TYPE_PILOTS_FULL_NATIVE_SUCCESS_SAFETY_INTEGRITY'),
            dict(id='D_EXPORT_AND_NEW_JOINT_TRAIN_SNAPSHOT',old_T1_T2='REUSE_TEN_EACH_NO_NEW_CAPTURE',
            source_labels='ACTUAL_COMPLETED_EXPERT_COMMANDS_ONLY',
            shared_statistics='NEW_TRAIN_ONLY_VERSION_DO_NOT_OVERWRITE_ACTIVE_BOOK_STATS',
            official_Processor_codec_and_RGB_qualification_required=True,
            Rerun='REUSE_EXISTING_VIEWER_AND_SAVED_RGB_NO_NEW_RENDERING_PLATFORM')],
        first_attempts_max=dict(cabinet=4,book_recovery=8,book_geometry=6,total=18),
        nominal_integrations_if_all18_run=18*18500,nominal_RGB_pairs_if_all18_run=18*642,
        failed_and_qualification_work_also_charged=True,
        successful_full_Teacher_target=dict(cabinet_new=4,book_new=6),
        successful_recovery_suffix_target=dict(minimum=6,maximum=8),
        outcomes_are_targets_not_guarantees=True,automatic_same_condition_retry=False,
        no_new_model_training_in_this_queue=True,
        post_data_comparison='Same_parent_and_new_common_statistics_old_data_arm_vs_expanded_data_arm; matched_parameters_updates_sampler_eval',
        future_shared_pilot_requires='At_least_five_actual_full_custom_cabinet_demos_and_final_data_admission',
        Vanda='NO_PROBE_BEFORE_2026_11_02_1700_SGT_AND_ACTUAL_MAINTENANCE_END',
        WBC_safety_task_scoring_frozen=True,S1_T2_quarantine_unchanged=True)


def require_dependency_closed(root,*,expected_seal_sha256,unit_reader):
    """Read actual terminal files before the existing owner may register work."""
    root=Path(root);read=lambda p:json.loads(Path(p).read_bytes())
    if hashlib.sha256((root/'CAPSULE_SEAL.json').read_bytes()).hexdigest()!=expected_seal_sha256:
        raise PermissionError('EXACT_UNCHANGED_EXISTING_RESIDENT_CHAIN')
    config=read(root/'QUEUE.json');state=read(root/'STATE.json')
    if(state['status']!='COMPLETE'or state['TRAIN'].get('status')!='CLOSED_AUDITED'
            or state['TRAIN'].get('step')!=4096 or state['DEV'].get('status')!='CLOSED_AUDITED'
            or any(state['diagnostics'].get(arm)!='CLOSED_AUDITED_96_GENERATIONS'for arm in('uniform','phase'))):
        raise PermissionError('ENTIRE_OLD_CHAIN_MUST_FINISH_NOT_JUST_TRAIN4096')
    for path in (root/'steps').glob('*.json'):
        if read(path)['status']!='COMPLETE':raise PermissionError('UNRESOLVED_OLD_STEP_NO_NEW_REGISTRATION')
    roots=[root,Path(config['producer_root'])]
    for name in('BOOK_PHASE_TRAIN32_UNIFORM4096','BOOK_PHASE_TRAIN32_PHASE4096','BOOK_PHASE4096_DEV10'):
        staged=read(root/(name+'_STAGED.private.json'));closed=read(root/(name+'_COMPLETED_CLOSED.private.json'));final=read(root/(name+'_FINAL_INDEPENDENT.private.json'))
        producer=Path(staged['root']);lease=producer/'control/LEASE.json'
        if(final['lease_sha256']!=closed['lease_sha256']or hashlib.sha256(lease.read_bytes()).hexdigest()!=closed['lease_sha256']or read(lease)['status']!='CLOSED'):
            raise PermissionError('ACTUAL_DIAGNOSTIC_AND_DEV_LEASES_CLOSED')
        roots.append(producer)
    train_closed=read(root/'BOOK_PHASE_B2_S4096_COMPLETED_CLOSED.private.json');train_lease=roots[1]/'control/LEASE.json'
    if hashlib.sha256(train_lease.read_bytes()).hexdigest()!=train_closed['lease_sha256']or read(train_lease)['status']!='CLOSED':
        raise PermissionError('ACTUAL_TRAIN_PRODUCER_CLOSED')
    for actor in roots:
        unit=unit_reader(actor.name+'.service')
        if int(unit['MainPID'])!=0:raise PermissionError('OLD_ACTORS_STILL_LIVE_NO_NEW_GPU_OWNER')
    return dict(status='OLD_FULL_CHAIN_ACTUALLY_CLOSED_NEW_COLD_ADMISSION_MAY_BEGIN',
        previous_stop_files={str(r/'STOP.json'):hashlib.sha256((r/'STOP.json').read_bytes()).hexdigest()for r in roots if(r/'STOP.json').exists()},
        original_STOP_retained=True,new_native_reset_camera_safety_qualification_still_required=True)


def require_two_recovery_pilots(result):
    from .book_late_recovery import KIND
    expected={c['id']for c in recovery_conditions('pilot')}
    rows=result.get('conditions',[])
    if(result.get('scientific_kind')!=KIND or result.get('status')!='COMPLETE_LATE_TRAIN_RECOVERY_PANEL'
            or result.get('formal_DEV_conditions')!=0 or {r['id']for r in rows}!=expected
            or len(rows)!=2 or any(not r.get('qualified_expert_recovery')or not r.get('success')
                or r.get('hard_native_safety')or r.get('late_recovery',{}).get('takeover_index')is None for r in rows)):
        raise PermissionError('TWO_ACTUAL_SUCCESSFUL_HELD_RECOVERIES_BEFORE_EXPANSION')
    return True

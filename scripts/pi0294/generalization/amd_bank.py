"""Exact GEN1 bank contract; original PV1 and COV1 population gates stay put.

This is a data contract, not registration, process or optimizer admission.
The official exporter still has to produce each real parent receipt.
"""
from copy import deepcopy
import re
from . import amd_export as export,amd_pilot_scope as scope
from scripts.pi0294.scene_cards import digest
from scripts.pi0294.pv1.data_delivery import safe_json

SCHEMA='GEN1_LAYOUT_V0_ADMITTED_TRAIN_BANK_V1'
IDENTITY='GEN1_LAYOUT_V0_CORR1024'


def manifest(packet_bytes,capture_rows,export_rows):
    support=export.support_after_export(packet_bytes,capture_rows,export_rows)
    scope.need(support['status']=='DATA_SUPPORT_READY_FOR_FRESH_TRAINING_ADMISSION',
        'BANK_ALL_TASK_LAYOUT_STRATA_OR_NO_TRAINING')
    # Closed native roots/leases live in the private input binding, never in
    # the public bank. Keep all twelve outcomes, including valid negatives.
    outcomes=[{k:deepcopy(r[k])for k in ('attempt_id','unknown_effects','terminal',
        'native_safety_failures','full_task_success','independent_raw_audit',
        'raw_manifest_sha256','audit_sha256')}for r in capture_rows]
    episodes=[deepcopy(r)for r in export_rows if r['admitted_for_training']]
    value=dict(schema=SCHEMA,domain='FIXED_BASE_ARX5_TASK_SIM',training_identity=IDENTITY,
        source_packet_sha256=scope.PACKET_SHA,statistics_sha256=scope.a.STATS,
        statistics_refit=False,reset='LEGACY_MATCHED',train_visual_profile='V0',
        evaluation_rows_used_for_training=False,old_PV1_population_changed=False,
        old_COV1_population_changed=False,later_rows_hot_injected=False,
        capture_outcomes=outcomes,all_outcomes_sha256=digest(outcomes),
        support=support,episodes=episodes)
    validate(value,packet_bytes);return value


def validate(value,packet_bytes):
    scope.need(value.get('schema')==SCHEMA and value.get('training_identity')==IDENTITY
        and value.get('domain')=='FIXED_BASE_ARX5_TASK_SIM'
        and value.get('source_packet_sha256')==scope.PACKET_SHA
        and value.get('statistics_sha256')==scope.a.STATS and value.get('statistics_refit')is False
        and value.get('reset')=='LEGACY_MATCHED'and value.get('train_visual_profile')=='V0'
        and value.get('evaluation_rows_used_for_training')is False
        and value.get('old_PV1_population_changed')is False
        and value.get('old_COV1_population_changed')is False
        and value.get('later_rows_hot_injected')is False,'BANK_DISTINCT_GEN_IDENTITY_AND_ORIGINAL_R0')
    scope.need(value.get('all_outcomes_sha256')==digest(value['capture_outcomes']),
        'BANK_ALL_TWELVE_OUTCOMES_UNCHANGED')
    support=export.support_after_export(packet_bytes,value['capture_outcomes'],value['episodes'])
    scope.need(support==value['support']and support['status']=='DATA_SUPPORT_READY_FOR_FRESH_TRAINING_ADMISSION',
        'BANK_SUPPORT_RECOMPUTED_FROM_REAL_PARENT_RECEIPTS')
    expected={r['attempt_id']:r for r in scope.pilot_plan(packet_bytes)['attempts']}
    from .amd_capture import cards
    actual_cards={r['id']:r for i in (0,1)for r in cards(packet_bytes,i)}
    parents=set();raws=set()
    for row in value['episodes']:
        identity=expected[row['id']]
        scope.need(row['split']=='TRAIN'and row['study_split']=='G_TRAIN'
            and row['compatibility_TRAIN_field_is_not_old_PV1_population']is True
            and row['nominal_clearance_m']==.004 and row['visual_profile']=='V0'
            and row['file']=='numeric/'+row['id']+'.npz'
            and re.fullmatch(r'GEN1-PILOT-[0-9a-f]{16}-T[12]',row['id'])is not None
            and row['id']not in parents and row['source_raw_manifest_sha256']not in raws
            and row['geometry_id']==identity['geometry_id']
            and row['design_hash']==identity['design_hash']and row['lineage_id']==identity['lineage_id']
            and row['scene_group']==actual_cards[row['id']]['group_id'],
            'BANK_EXACT_UNIQUE_NATIVE_PARENTS_NOT_WINDOW_INFLATION')
        for key in ('sha256','source_raw_manifest_sha256','source_full_audit_sha256','source_card_sha256'):
            scope.need(type(row[key])is str and re.fullmatch('[0-9a-f]{64}',row[key])is not None,
                'BANK_REAL_NUMERIC_RAW_AUDIT_CARD_SHA')
        scope.need(type(row['bytes'])is int and row['bytes']>0
            and row['official_chain']['deterministic_repeat']is True
            and row['official_chain']['full_641_two_camera_raw_RGB256_verified']is True
            and row['official_chain']['model_forward']is False,'BANK_ORIGINAL_OFFICIAL_CHAIN_RECEIPT')
        parents.add(row['id']);raws.add(row['source_raw_manifest_sha256'])
    safe_json(value)
    return dict(admitted=sorted(parents),support_sha256=digest(support),
        native_training_admission=False)

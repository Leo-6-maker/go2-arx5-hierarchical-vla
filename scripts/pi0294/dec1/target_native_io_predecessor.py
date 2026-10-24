"""Exact closed zero-constructor predecessor for the existing ROS cold repair.

Pure byte verification; no clock, IO, resource writer, registry or permission.
Only the already-settled first import failure can fund this one cold wiring
continuation. A live/unknown/model/safety result cannot become a retry pool.
"""
import base64
from . import target_zero_counter_contract as common

REQUEST='e7cf534b-7f37-4a31-a2c6-89cb1aec7005'
RUN='dec1-target-native-zero-io-01'
SOURCE='8e2bcd7d7fc1e7140aa47d7ba16b2a0ef9b4efc1'
MANIFEST='cb782956d2db314e6814629bd92843f32db6d3aa520a1c90b6f1e8a65bb7d3aa'
ROLE='DEC1_FROZEN_TARGET_NATIVE_IO_QUALIFICATION_V1'
PACK='plans/ORIGINAL_ZERO_CONSTRUCTOR_IO_CLOSED_PREDECESSOR.private.json'
ANCHORS={
 'dual_bytes':'ca8a2e9faf01bd494416b1a0e5c51a79a2aab40df0468c5544859340a928d902',
 'lease_bytes':'626523ecb7d94c57758bb147a0e513f4c105af51e899836a1ff60a1b2817def5',
 'result_bytes':'84ca9ed64e6bc2578e075e6907d12f5e8d721975a4f931f4654711013d282ba6',
 'actor_proof_bytes':'6d303cf89a25e37afec45c400d0d952e8ec066a8c11db459a316602f27eb61a5',
 'independent_bytes':'4f721eb64adac7994243a74fcdbb769a3f482296bd1d4c38e3e3e6308d7f0830'}
need=common.need


def pack(fields):
    need(set(fields)==set(ANCHORS) and all(type(v) is bytes for v in fields.values()),'IO_PREDECESSOR_ALL_EXACT_BYTES')
    return common.encode(dict(schema='DEC1_ZERO_CONSTRUCTOR_IO_CLOSED_PACKET_V1',
        fields={k:base64.b64encode(v).decode() for k,v in fields.items()}))


def verify(packet,*,prior_row,prior_settlement):
    value=common.obj(packet)
    need(value['schema']=='DEC1_ZERO_CONSTRUCTOR_IO_CLOSED_PACKET_V1'
         and set(value['fields'])==set(ANCHORS),'IO_EXACT_CLOSED_PREDECESSOR_PACKET')
    fields={k:base64.b64decode(v,validate=True) for k,v in value['fields'].items()}
    need(sum(map(len,fields.values()))<256*1024 and all(common.sha(raw)==ANCHORS[k] for k,raw in fields.items()),
         'IO_ORIGINAL_CLOSED_RESULT_LEASE_DUAL_ACTOR_AND_INDEPENDENT_BYTES')
    dual,lease,result,actor,independent=(common.obj(fields[k]) for k in
        ('dual_bytes','lease_bytes','result_bytes','actor_proof_bytes','independent_bytes'))
    need(prior_row['request_id']==REQUEST and prior_row['child_run_uuid']==RUN
         and prior_row['source_sha']==SOURCE and prior_row['manifest_sha256']==MANIFEST
         and prior_row['scope']==ROLE and prior_row['status']=='SETTLED'
         and prior_row['unknown_effects']==[] and not any(common.vector(prior_row['remaining']).values())
         and prior_row['settlement_sha256']==common.sha(common.encode(prior_settlement))
         and dual['settlement']==prior_settlement,'IO_PAIRED_ORIGINAL_SETTLEMENT_AND_ALL_COSTS_UNCHANGED')
    need(dual['status']=='ACTUAL_CURRENT_NATIVE_IO_DUAL_ONCE_SETTLED_AND_LEASE_CLOSED'
         and dual['Core']['closed_lease_sha256']==ANCHORS['lease_bytes']
         and base64.b64decode(dual['Core']['closed_lease_base64'],validate=True)==fields['lease_bytes']
         and lease['status']=='CLOSED' and lease['scope']==ROLE and lease['run_uuid']==RUN
         and lease['source_sha']==SOURCE and lease['manifest_sha256']==MANIFEST
         and lease['settlement_sha256']==prior_row['settlement_sha256']
         and lease['all_registered_actors_terminal'] is True
         and lease['target_task_native_qualified'] is False,'IO_ORIGINAL_REAL_CLOSED_LEASE_NOT_JUST_SETTLEMENT')
    need(result['status']=='NATIVE_ZERO_IO_ENGINEERING_INCOMPLETE'
         and result['error']=="ModuleNotFoundError: No module named 'rclpy'"
         and result['completed_stages']==[] and result['constructor_labels']==[]
         and result['request_id']==REQUEST and result['manifest_sha256']==MANIFEST and result['source_sha']==SOURCE
         and result['target_task_native_qualified'] is False and result['native_prefix_safety_qualified'] is False
         and all(type(result[k]) is int and result[k]==0 for k in ('actual_constructor_attempts','physics_ticks',
            'policy_episodes','GR00T_queries','training_updates','GPU_submissions','core64_cells')),
         'IO_ONLY_EXACT_PRE_MODEL_IMPORT_FAILURE_NOT_SAFETY_OR_UNKNOWN_RETRY')
    need(actor['schema']=='DEC1_TARGET_NATIVE_IO_ACTUAL_TERMINAL_ACTOR_PROOF_V1'
         and actor['request_id']==REQUEST and actor['manifest_sha256']==MANIFEST
         and actor['all_registered_actors_terminal'] is True and actor['current_matching_actors']==0
         and actor['result_present'] is True and actor['result_sha256']==ANCHORS['result_bytes']
         and prior_settlement['actor_proof_sha256']==ANCHORS['actor_proof_bytes']
         and prior_settlement['result_sha256']==ANCHORS['result_bytes'],'IO_REAL_TERMINAL_ACTORS_AND_KNOWN_RESULT')
    local=dual['LOCAL_readback'];receipt=dual['LOCAL_receipt']
    need(local['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT' and local['receipt_binding_valid'] is True
         and local['actual_sha256']==receipt['after_sha256']==dual['LOCAL_after_sha256']
         and receipt['request_id']==prior_settlement['settlement_id']
         and receipt['status']=='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION'
         and independent['status']=='INDEPENDENT_CURRENT_NATIVE_IO_CLOSED_READBACK'
         and independent['lease_sha256']==ANCHORS['lease_bytes']
         and independent['Core_after_sha256']==dual['Core']['Core_after_sha256']
         and independent['LOCAL_after_sha256']==dual['LOCAL_after_sha256']
         and independent['all_8_holds_zero'] is independent['LOCAL_all_8_holds_zero'] is True
         and independent['registered_actors_active']==[] and independent['new_native_writes']==0
         and independent['old_SDK_UNKNOWN_unchanged'] is True,'IO_INDEPENDENT_CORE_LOCAL_CLOSE_READBACK')
    return dict(schema='DEC1_ONE_ZERO_CONSTRUCTOR_IO_COLD_REPAIR_PREDECESSOR_V1',
        request_id=REQUEST,packet_sha256=common.sha(packet),exact_original_bytes_sha256=dict(ANCHORS),
        old_costs_preserved=True,new_native_dispatch_admitted=False)

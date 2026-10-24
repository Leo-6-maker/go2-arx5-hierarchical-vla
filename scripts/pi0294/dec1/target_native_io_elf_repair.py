"""One approved LOCAL minute and exact CLOSED ELF-loader failure continuation.

Pure proposals only. Preserve original VAP1 grant bytes, paid rows and every
other dimension. No native safety, task or unknown-write retry permission.
"""
import base64
from copy import deepcopy
from . import target_zero_counter_contract as common

GRANT_SHA='ff4c05e898dd27fa3df31e0163baab6b10067a5815756352d06c50bc867507cc'
GRANT='plans/ONE_LOCAL_CONTROL_MINUTE_AUTHORIZATION.json'
MARKER='target_native_IO_LOCAL_minute_adoption'
REQUEST='b1cf63f5-040e-43fd-930d-64bc35ec49c2'
RUN='dec1-target-native-zero-io-ros-repair-01'
SOURCE='bc26e07b059ab2e6e00071971f0f0f739adcfcf6'
MANIFEST='7298520e5ef66a34a09b97cc353a2073cdf0f414b3fc348cee394043c3513ba9'
PACK='plans/ORIGINAL_ELF_LOADER_FAILURE_CLOSED_PREDECESSOR.private.json'
SCHEMA='DEC1_ONE_EXACT_CLOSED_ELF_LOADER_REPAIR_PREDECESSOR_V1'
ANCHORS=dict(dual_bytes='9b264c158efc3b880c2dac14b238c373d64417f87f41ec4526bfbefd6e9aede7',
 lease_bytes='11456f0022ff1386277b7b96f57ba0d46e8774b23e450e04f45c1825400b0023',
 result_bytes='2afedc5d02c75983fdf504ab3409057a3b12b09c5a4edfe9ac8754b750ded7a2',
 actor_proof_bytes='d22f5837d352068a3fd1b5da189430c8e0924fd19883564cd874044fec63409a',
 independent_bytes='49abc68f611bcbed8a1e9fc4f0f6466a91b023b74dcc8830f5dacf3200699103')
need=common.need

def marker():
    return dict(schema='DEC1_ONE_LOCAL_CONTROL_MINUTE_ADOPTED_V1',grant_sha256=GRANT_SHA,
        before_LOCAL_CPU_s=57600,after_LOCAL_CPU_s=57660,delta_LOCAL_CPU_s=60,
        exact_closed_predecessor_request=REQUEST,original_T0_epoch=1792348196,hard_close_epoch=1792952996,
        all_other_caps_clocks_and_costs_unchanged=True)

def adopted(p,c):
    need(p.get(MARKER)==c.get(MARKER)==marker(),'IO_EXACT_DUAL_ADOPTED_LOCAL_MINUTE')
    return 57660

def grant(raw):
    need(common.sha(raw)==GRANT_SHA,'IO_EXACT_DIRECT_HUMAN_LOCAL_MINUTE_GRANT')
    v=common.obj(raw)
    need(v['schema']=='WDR1_DIRECT_HUMAN_ONE_LOCAL_CONTROL_MINUTE_V1'
         and v['direct_human_answer']=='批准仅增加1分钟LOCAL CPU'
         and v['exact_cap_change']==dict(DEC1_LOCAL_CPU_s=dict(before=57600,after=57660,delta=60)),
         'IO_ONLY_ONE_LOCAL_MINUTE_NO_OTHER_CAP_CHANGE')
    return v

def pack(fields):
    need(set(fields)==set(ANCHORS),'IO_ELF_ALL_EXACT_CLOSED_BYTES')
    return common.encode(dict(schema='DEC1_CLOSED_ELF_LOADER_IO_PACKET_V1',
        fields={k:base64.b64encode(raw).decode() for k,raw in fields.items()}))

def verify(packet,*,prior_row,prior_settlement):
    v=common.obj(packet);need(v['schema']=='DEC1_CLOSED_ELF_LOADER_IO_PACKET_V1' and set(v['fields'])==set(ANCHORS),'IO_ELF_EXACT_PACKET')
    fields={k:base64.b64decode(raw,validate=True) for k,raw in v['fields'].items()}
    need(sum(map(len,fields.values()))<2*1024**2 and all(common.sha(raw)==ANCHORS[k] for k,raw in fields.items()),'IO_ELF_EXACT_REAL_CLOSURE_ANCHORS')
    d,l,r,a,i=(common.obj(fields[k]) for k in ('dual_bytes','lease_bytes','result_bytes','actor_proof_bytes','independent_bytes'))
    need(prior_row['request_id']==REQUEST and prior_row['child_run_uuid']==RUN and prior_row['source_sha']==SOURCE
         and prior_row['manifest_sha256']==MANIFEST and prior_row['status']=='SETTLED'
         and prior_row['unknown_effects']==[] and not any(prior_row['remaining'].values())
         and d['settlement']==prior_settlement and prior_row['settlement_sha256']==common.sha(common.encode(prior_settlement)),
         'IO_ELF_PAIRED_PAID_HISTORY_UNCHANGED')
    need(l['status']=='CLOSED' and l['run_uuid']==RUN and l['source_sha']==SOURCE and l['manifest_sha256']==MANIFEST
         and l['settlement_sha256']==prior_row['settlement_sha256'] and l['all_registered_actors_terminal'] is True
         and d['Core']['closed_lease_sha256']==ANCHORS['lease_bytes']
         and base64.b64decode(d['Core']['closed_lease_base64'],validate=True)==fields['lease_bytes'],
         'IO_ELF_REAL_CLOSED_LEASE')
    need(r['status']=='NATIVE_ZERO_IO_ENGINEERING_INCOMPLETE' and r['completed_stages']==['FROZEN_GEOMETRY']
         and r['error']=='PermissionError: TARGET_ZERO_COUNTER_IO_OWNED_NATIVE_CHILD_EXIT_solver'
         and r['request_id']==REQUEST and r['source_sha']==SOURCE and r['manifest_sha256']==MANIFEST
         and r['actual_constructor_attempts']==5
         and r['constructor_labels']==['ORIGINAL_FROZEN_COMPILED_MODEL','ORIGINAL_FROZEN_TASK_GEOMETRY',
             'ORIGINAL_PAUSED_SIM','ORIGINAL_SOLVER','ORIGINAL_CLOCK_CONTROLLER']
         and all(type(r[k]) is int and r[k]==0 for k in ('physics_ticks','policy_episodes','GR00T_queries','training_updates','GPU_submissions','core64_cells'))
         and r['native_prefix_safety_qualified'] is r['target_task_native_qualified'] is False,
         'IO_ONLY_EXACT_ZERO_PHYSICS_LOADER_FAILURE_NOT_SAFETY_REPLAY')
    need(a['all_registered_actors_terminal'] is True and a['current_matching_actors']==0 and a['result_present'] is True
         and a['result_sha256']==prior_settlement['result_sha256']==ANCHORS['result_bytes']
         and prior_settlement['actor_proof_sha256']==ANCHORS['actor_proof_bytes']
         and i['status']=='INDEPENDENT_ACTUAL_SERVER_IO_DUAL_CLOSURE_READBACK'
         and i['all_actors_terminal'] is i['all_current_IO_holds_zero'] is i['both_SDK_UNKNOWNs_unchanged'] is True
         and i['closed_lease_sha256']==ANCHORS['lease_bytes']
         and i['Core_after_sha256']==d['Core']['Core_after_sha256'] and i['LOCAL_after_sha256']==d['LOCAL_after_sha256'],
         'IO_ELF_INDEPENDENT_KNOWN_TERMINAL_READBACK')
    return dict(schema=SCHEMA,request_id=REQUEST,packet_sha256=common.sha(packet),exact_original_bytes_sha256=dict(ANCHORS),
        old_costs_preserved=True,new_native_dispatch_admitted=False)

def adopt(parent_bytes,Core_bytes,grant_bytes,predecessor_bytes):
    p,c=map(common.obj,(parent_bytes,Core_bytes));grant(grant_bytes)
    from . import target_native_io_budget as b
    b.identity(p,c);rows,_,held=b.rows(p,c)
    need(set(rows)=={b.predecessor.REQUEST,REQUEST} and not any(held.values()),'IO_MINUTE_ONLY_AFTER_BOTH_REAL_CLOSED_ROWS')
    verify(predecessor_bytes,prior_row=rows[REQUEST],prior_settlement=p[b.SETTLEMENTS][REQUEST])
    need(MARKER not in p and MARKER not in c and not any(p['CPU_reserved_s'].values()),'IO_MINUTE_ONCE_INACTIVE')
    for key in ('source_cpu_reserved','target_zero_counter_reserved','target_sdk_client_reserved','target_native_IO_reserved'):
        need(not any(c.get(key,{}).values()),'IO_MINUTE_NO_OTHER_PENDING_COMPONENT')
    proposed_p,proposed_c=deepcopy(p),deepcopy(c)
    for ledger in (proposed_p,proposed_c):ledger[MARKER]=marker()
    return dict(proposed_parent=proposed_p,proposed_Core=proposed_c,marker=marker(),all_actual_costs_unchanged=True,native_dispatch_admitted=False)

def extend_forecast(value,p,c):
    """Re-evaluate only the approved LOCAL cap; preserve all other STOPs."""
    cap=adopted(p,c);need(value['effective_DEC1_VAP1_caps']['local_CPU_s']==57600,'IO_ORIGINAL_LOCAL_CAP_NOT_DOUBLE_EXTENDED')
    value['effective_DEC1_VAP1_caps']['local_CPU_s']=cap
    difference=value['DEC1_shortfalls'].get('local_CPU_s',0)-60
    value['DEC1_shortfalls'].pop('local_CPU_s',None)
    if difference>0:value['DEC1_shortfalls']['local_CPU_s']=difference
    value['LOCAL_minute_authorization']=marker()
    return value

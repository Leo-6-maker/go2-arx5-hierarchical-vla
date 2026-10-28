"""Pure paired SDK rows and fixed component limits; no native imports or IO."""
from . import target_zero_counter_contract as common
ROLE='DEC1_FROZEN_TARGET_SDK_CLIENT_QUALIFICATION_V1'
CAMPAIGN='20261018T182956Z-localpc-pv1'
T0, STOP_NEW, STOP_PHYSICS, HARD_CLOSE = 1792348196, 1792866596, 1792924196, 1792952996
ROWS='target_sdk_client_reservations'
SETTLEMENTS='target_sdk_client_settlements'
ROW_SCHEMA='DEC1_TARGET_SDK_CPU_RESERVATION_V1'
WORK='DEC1_CORE_MATRIX_AND_COMMISSIONING'
COMPONENT='ORIGINAL_TARGET_SDK_COLD_BINDING_V1'
# Six finite engineering attempts inside the SAME original200000/3600 matrix
# CPU work; never an additive global cap or a new physics/episode allowance.
MAX_ATTEMPTS=6
CPU_CEILING=dict(CoreLab_CPU_s=10800,local_CPU_s=720)
REQUESTED=dict(CoreLab_CPU_s=3600,local_CPU_s=120,episode_starts=0,physics_ticks=0,
               constructors=2,CoreLab_output_bytes=512*1024**2,
               local_output_bytes=64*1024**2,transfer_bytes=64*1024**2)
def need(ok,why):
    if not ok:raise PermissionError('DEC1_SDK_BUDGET_'+why)


def rows(parent,core):
    """Only actual paired rows or an explicit proposed pair; never one side."""
    a=parent.get(ROWS,{});b=core.get(ROWS,{})
    need(common.same(a,b),'SDK_EXACT_PAIRED_ROWS')
    need(len(a)<=MAX_ATTEMPTS,'SDK_SIX_FINITE_SAME_WORK_ENGINEERING_ATTEMPTS')
    spent=dict.fromkeys(common.DIMS,0);held=dict.fromkeys(common.DIMS,0)
    for rid,row in a.items():
        wanted=common.vector(row['requested']);left=common.vector(row['remaining'])
        need(all(wanted[k]<=REQUESTED[k] for k in common.DIMS),'SDK_ORIGINAL_EIGHT_DIMENSION_COMPONENT_UPPER')
        need(row['schema']==ROW_SCHEMA and row['scope']==ROLE and row['request_id']==rid
             and row['funded_work_id']==WORK and row['component_id']==COMPONENT
             and row['unknown_effects']==[] and row['status'] in ('RESERVED','SETTLED')
             and wanted['episode_starts']==wanted['physics_ticks']==0,
             'SDK_TYPED_KNOWN_ZERO_PHYSICS_COMPONENT')
        if row['status']=='RESERVED':
            need(left==wanted,'SDK_FULL_HOLD_NOT_PARTIALLY_DROPPED')
            amount=left;destination=held
        else:
            need(not any(left.values()),'SDK_SETTLED_HOLDS_ZERO')
            s=parent[SETTLEMENTS][rid]
            need(common.same(s,core[SETTLEMENTS][rid]) and common.sha(common.encode(s))==row['settlement_sha256']
                 and s['schema']=='DEC1_TARGET_SDK_CPU_SETTLEMENT_V1' and s['request_id']==rid
                 and s['scope']==ROLE and s['manifest_sha256']==row['manifest_sha256']
                 and s['target_task_native_qualified'] is False,'SDK_EXACT_PAIRED_SETTLEMENT')
            amount=common.vector(s['financial_charged']);destination=spent
            need(all(amount[k]<=wanted[k] for k in common.DIMS),'SDK_NO_UNRESERVED_COST')
        for k in common.DIMS:destination[k]+=amount[k]
    need(sum(x['status']=='RESERVED' for x in a.values())<=1,'SDK_ONE_CURRENT_CPU_OWNER')
    for k,cap in CPU_CEILING.items():
        need(spent[k]+held[k]<=cap,'SDK_SINGLE_MATRIX_CPU_COMPONENT_CEILING_'+k)
    for ledger in (parent,core):
        need(common.same(ledger.get('target_sdk_client_reserved',dict.fromkeys(common.DIMS,0)),held),
             'SDK_FULL_EIGHT_DIMENSION_HOLD_AGGREGATE')
    need(core['CPU_s']>=spent['CoreLab_CPU_s'] and parent['CPU_s']['local_PC']>=spent['local_CPU_s']
         and core['constructors']>=spent['constructors']
         and parent['network_transfer_accounted_bytes']>=spent['transfer_bytes'],
         'SDK_CREDIT_MUST_ALREADY_BE_IN_ACTUAL_TOTALS')
    return a,spent,held


"""Exact unstarted D25 hold or a newly sealed independent D25 execution pair.

A cold Core row is not a live GPU or remote heartbeat prerequisite. It does
not gain permission to start with an old capsule that cannot read new rows.
"""
from . import minimal_execution as e,minimal_remaining_execution as execution
from scripts.pi0294.pv1 import mix_core_accounting as core,mix_core_d25_cold_source as cold

VERSION='ARX5_REMAINING_ONE_VANDA_PLUS_ORIGINAL_D25_CORE_COLD_V1'
LOCAL_MIRROR='ARX5_minimal_remaining_LOCAL_main_inventory_mirror'
COLD_REQUEST_SHA='999d1da66157ab01e6a189907ab5b0746529b175de0e25c75900852eb3b71d60'
COLD07_ROW_SHA='e510508076bfc5eed9ad70fc198796d5c072670becd0ff6b62fd9a95e4e9033a'


def cold_Core_rows(g):
    rows=[r for r in g.get(e.original.CORE_ROWS,{}).values()if r['status']!='NATIVE_TERMINAL']
    e.need(len(rows)<=1,'REMAINING_ONLY_ONE_ORIGINAL_CORE_SCOPE')
    if rows:
        row=rows[0]
        from scripts.pi0294.pv1 import mix_core_d25_admission_repair as repaired,mix_core_d25_local_successor as paid
        marker=repaired.validate(g)
        exact_repaired=marker is not None and row['request_id']==marker['replacement_request_id']
        identity=cold.effective_identity(g,paid.validated(g))if exact_repaired else None
        e.need((exact_repaired and e.sha(e.raw(row))==COLD07_ROW_SHA
            and row['source_sha']==identity['source_sha']and row['child_run_uuid']==identity['run_uuid']
            or row['request_id']==cold.REQUEST and row['request_sha256']==COLD_REQUEST_SHA and row['child_run_uuid']==cold.RUN)
            and row['evaluation_candidate']=='D25'
            and row['status']==core.STATES[0]and row['native_handle']is None
            and core.progress(row)==(0,0,0)
            and row.get('query_attempts',0)==row.get('model_load_attempts',0)==0
            and row.get('unknown_query_intents',[])==[],
            'REMAINING_ONLY_EXACT_UNSTARTED_D25_NOT_A_LIVE_UNBOUND_GPU')
    return rows


def validate_pair(g,pbs_rows,Core_rows):
    e.need(len(pbs_rows)==len(Core_rows)==1,'REMAINING_ONE_MAIN_ONE_CORE_MAXIMUM')
    m,main=execution.basic(g);row=Core_rows[0]
    e.need(pbs_rows[0]==main and main['status']in e.STATES[:2]
        and g['pending_updates']==g['unknown_effects']==[],'REMAINING_CURRENT_MAIN_AND_ALL_GLOBAL_FENCES')
    if row['status']==core.STATES[0]and row['native_handle']is None:
        e.need(cold_Core_rows(g)==[row],'REMAINING_EXACT_COLD_ROW_HOLD_ONLY')
        return dict(schema=VERSION,main_request_id=main['request_id'],Core_request_id=row['request_id'],
            Core_unstarted=True,Core_native_permission=False,remote_liveness_required=False)
    from . import minimal_concurrency as original
    expected=original.binding_identity(main)
    e.need(main['status']==e.STATES[1]and main['job_id']is not None
        and row['evaluation_candidate']=='D25'and row['platform_evaluation_version']==core.VERSION
        and row['native_platform']=='CORELAB_RTX4080'and row.get(original.KEY)==expected
        and row['created_epoch']>=m['registered_epoch']
        and row['source_sha']!='2a23eeb64e7f769fa5842eb72d371ac12ee99664',
        'REMAINING_LIVE_CORE_ONLY_A_NEW_EXPLICIT_PAIR_NOT_OLD_FROZEN_CAPSULE')
    return expected

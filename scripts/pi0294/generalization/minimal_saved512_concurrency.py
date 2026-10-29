"""One Vanda recovery plus the exact still-unstarted Core D25 hold only."""
from . import minimal_execution as e,minimal_saved512_execution as execution
from .minimal_remaining_concurrency import cold_Core_rows


def validate_pair(g,pbs_rows,Core_rows):
    e.need(len(pbs_rows)==len(Core_rows)==1 and execution.basic(g)[1]==pbs_rows[0]
        and pbs_rows[0]['status']in e.STATES[:2]and cold_Core_rows(g)==Core_rows
        and g['pending_updates']==g['unknown_effects']==[],
        'SAVED512_ONLY_EXACT_COLD_D25_NO_OLD_LIVE_SOURCE_PERMISSION')
    return dict(schema='ARX5_SAVED512_VANDA_WITH_ORIGINAL_UNSTARTED_D25_CORE_HOLD_V1',
        main_request_id=pbs_rows[0]['request_id'],Core_request_id=Core_rows[0]['request_id'],
        Core_unstarted=True,Core_native_permission=False,remote_liveness_required=False)

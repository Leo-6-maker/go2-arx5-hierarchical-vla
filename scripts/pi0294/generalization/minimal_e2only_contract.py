"""Only the unfinished E2 formal512 session; no E1 or qualification replay."""
from copy import deepcopy
from . import minimal_execution as e,minimal_e2only_scope as scope

VERSION='ARX5_MINIMAL_E2_ONLY_FRESH_RAW512_V1'
PROFILE='vanda-arx5-minimal-e2-only-raw-fixed512'
RUN='arx5-minimal-e2-only-raw-fixed512-01'
KEYS=('optimizer_updates','generation_queries','model_loads')


def run_name(attempt=1):
    e.need(type(attempt)is int and attempt>=1,'E2ONLY_EXPLICIT_POSITIVE_ATTEMPT')
    return 'arx5-minimal-e2-only-raw-fixed512-'+str(attempt).zfill(2)


def recovery_policy():
    """Execution-only saves; the scientific candidate remains fixed512."""
    return dict(periodic_steps=[256],terminal_step=512,max_full_states=2,
        per_full_state_upper_bytes=26000000000,inference_upper_bytes=16000000000,
        control_and_temporary_upper_bytes=2000000000,root_upper_bytes=70000000000,
        original_Trainer_full_resume=True,delete_old_states=False,development_selection=False)


def recovery_boundary(step,cursor,*,pending,unknown):
    e.need(type(step)is int and 0<step<=512 and cursor==step*32
        and pending==unknown==[],'E2ONLY_RECOVERY_COMMITTED_CURSOR_NO_UNKNOWN')
    return step==256


def sessions():
    rows=deepcopy(e.native_sessions()[4:])
    e.need(len(rows)==1 and rows[0]['id']=='E2_SHARED_ACTION_HEAD_FORMAL512'
        and rows[0]['optimizer_limit']==512 and rows[0]['query_limit']==37
        and rows[0]['fresh_raw_parent']is True,'E2ONLY_ORIGINAL_UNCHANGED_FORMAL_SESSION')
    return rows


def tickets():
    spec=sessions()[0]
    return [dict(index=i,session_id=spec['id'],arm=spec['arm'],phase=p)
        for i,p in enumerate(('MODEL_RAW','FIRST_UPDATE','OFFICIAL_EXPORT'))]


def check_phase(sid,phase,counters):
    expected=dict(MODEL_RAW=(0,0,0),FIRST_UPDATE=(0,0,1),OFFICIAL_EXPORT=(512,36,1))
    e.need(sid==sessions()[0]['id']and set(counters)==set(KEYS)and phase in expected
        and tuple(counters[k]for k in KEYS)==expected[phase],
        'E2ONLY_THREE_FRESH_PHASES_NO_SAVED122_OR_QUALIFICATION_INHERITANCE')


def binding():
    return dict(closed_predecessor_request_id=scope.REQUEST,closed_predecessor_row_sha256=scope.ROW,
        closed_predecessor_marker_sha256=scope.MARKER,closed_predecessor_lease_sha256=scope.LEASE,
        completion_scope_grant_sha256=scope.GRANT_SHA,E1_new_training_or_inference=0)

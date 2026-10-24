"""One human-confirmed independent job, never adopted or modified here."""
import hashlib
import json

RELEASE='730d0fde901be34f5ffeb9657a981b1677069823'
SHA='2223670a76ced2560f76df31eb68f374f7580584f4e4e34246935bef0a8ccb28'
PLAN='plans/V4_INDEPENDENT_ACCOUNT_JOB1481491_HUMAN_IDENTITY.json'


def checked_outside(rows,outside,confirmation_bytes=None):
    """Keep the complete queue. Only the exact confirmed other job is exempt."""
    identities={r['job_id'] for r in outside}
    active=[r for r in rows if r['job_id'] in identities and str(r['ngpus'])!='0' and r['state'] not in ('F','X')]
    if not active:return []
    if (type(confirmation_bytes) is not bytes or hashlib.sha256(confirmation_bytes).hexdigest()!=SHA):
        raise PermissionError('PV1_V4_OUTSIDE_GPU_IDENTITY_REQUIRES_EXACT_HUMAN_CONFIRMATION')
    confirmation=json.loads(confirmation_bytes)
    expected=dict(job_id='1481491.stdct-mgmt-02',name='l08b-1-e1192548-1',project='CFP04-CF-031')
    if (len(active)!=1 or any(active[0].get(k)!=v for k,v in expected.items())
            or str(active[0]['ngpus'])!='2' or active[0]['state'] not in ('Q','R')
            or confirmation['direct_human_answer']!='是，属于其他独立工作'):
        raise PermissionError('PV1_V4_UNCONFIRMED_OTHER_ACCOUNT_GPU_STILL_BLOCKS')
    return [dict(expected,human_confirmation_sha256=SHA,
        classification='USER_CONFIRMED_INDEPENDENT_NOT_ADOPTED_OR_CHARGED',may_use_modify_stop=False)]


def sealed_confirmation(root,manifest,request):
    from .future_admission import _file
    if PLAN not in manifest['execution']['sealed_inputs'] and PLAN not in request['sealed_inputs']:return None
    data=_file(root,PLAN)
    if (hashlib.sha256(data).hexdigest()!=SHA or manifest['execution']['sealed_inputs'].get(PLAN)!=SHA
            or request['sealed_inputs'].get(PLAN)!=SHA):
        raise PermissionError('PV1_V4_INDEPENDENT_JOB_IDENTITY_SEALED_IN_BOTH_INPUTS')
    return data

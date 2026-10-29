"""A fresh pre-qsub queue proof distinct from the immutable registration input."""
import json
from . import minimal_execution as e,minimal_e2only_execution as execution
from .minimal_native_launch import effective_queue

FILE='DISPATCH_EFFECTIVE_QUEUE_LIMITS.json'


def check(body,intent,request,*,now_epoch):
    execution.request(request)
    value=effective_queue(body,created_epoch=intent['observed_epoch'],walltime_s=request['walltime_s'])
    e.need(intent['request_id']==request['request_id']and intent['once']is True
        and intent['queue_proof_sha256']==e.sha(body)
        and request['created_epoch']<=value['observed_epoch']<=intent['observed_epoch']<=now_epoch,
        'REMAINING_ACTUAL_SUBMISSION_QUEUE_INTENT_NOT_RESTAMPED_OLD_INPUT')
    return value


"""A known new registered-self gap is not an unknown submitted job.

Only classification uses the narrowed view; canonical bytes/UNKNOWN are never
changed. A prior qsub journal/intent or any other gap still blocks dispatch.
"""
import base64,time
from pathlib import Path
from scripts.pi0292.durable import read
from . import minimal_execution as e,minimal_e2only_execution as execution
from scripts.pi0294.dec1 import vanda_campaign as queue
from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside


def validate(g,full,request,request_sha,*,now_epoch):
    execution.validate(g);_,row=execution.basic(g)
    e.need(row['request_sha256']==request_sha and row['status']==e.STATES[0]and row['job_id']is None
        and all(row.get(k)==v for k,v in request.items())and g['pending_updates']==g['unknown_effects']==[],
        'E2_ONLY_PREDISPATCH_ONLY_EXACT_ORIGINAL_REGISTERED_SELF_REQUEST')
    rows=queue.queue_rows(full,now_epoch=now_epoch);own,outside,unknown=queue.classify(g,rows)
    e.need(own==[]and unknown==[dict(job_id=None,reason='ACTIVE_LEDGER_WITHOUT_CURRENT_NATIVE_JOB_PROOF')],
        'E2_ONLY_PREDISPATCH_NO_OTHER_UNKNOWN_OR_ACTIVE_NATIVE_JOB')
    view=dict(g,PBS_submissions=[r for r in g['PBS_submissions']if r is not row]);a,b,c=queue.classify(view,rows)
    e.need(a==c==[]and b==outside,'E2_ONLY_PREDISPATCH_ONLY_EXACT_SELF_GAP_IN_CLASSIFICATION_VIEW')
    return rows,outside


def collect(owner):
    base=owner.BASE;saved=read(base/'SOURCE_REQUEST.private.json');path=Path(saved['directory'])/'REQUEST.json'
    e.need(e.sha(path.read_bytes())==saved['request_sha256'],'E2_ONLY_PREDISPATCH_ACTUAL_SAVED_REQUEST_BYTES')
    e.need(not(base/'SOURCE_QSUB_INTENT.private.json').exists()and not(base/'SOURCE_QSUB_REPLY.private.json').exists()
        and not(base/'SOURCE_JOB_BOUND.private.json').exists(),'E2_ONLY_PREDISPATCH_NO_PREVIOUS_QSUB_ATTEMPT')
    for p in(base/'setup-rpc').glob('*-INTENT.json'):
        v=read(p);e.need(v.get('intent',v).get('phase')!='QSUB','E2_ONLY_PREDISPATCH_NO_PRIOR_QSUB_JOURNAL')
    body=owner.global_path.read_bytes();g=read(owner.global_path);full=queue.collect_queue()
    rows,outside=validate(g,full,saved['request'],saved['request_sha256'],now_epoch=time.time())
    checked_outside(rows,outside,base64.b64decode(owner.CONFIG['outside_confirmation_base64'],validate=True))
    e.need(owner.global_path.read_bytes()==body,'E2_ONLY_PREDISPATCH_GLOBAL_CHANGED_DURING_QUEUE_READ')
    return dict(status='ACTUAL_E2_ONLY_EXACT_REGISTERED_SELF_UNSUBMITTED_NO_OTHER_UNKNOWN',
        GLOBAL_sha256=e.sha(body),complete_queue=full,observed_epoch=time.time(),canonical_mutation=False)


"""One Vanda recovery plus the exact still-unstarted Core D25 hold only."""
from . import minimal_execution as e,minimal_e2only_execution as execution
from .minimal_remaining_concurrency import cold_Core_rows


def validate_pair(g,pbs_rows,Core_rows):
    e.need(len(pbs_rows)==len(Core_rows)==1 and execution.basic(g)[1]==pbs_rows[0]
        and pbs_rows[0]['status']in e.STATES[:2]and cold_Core_rows(g)==Core_rows
        and g['pending_updates']==g['unknown_effects']==[],
        'E2_ONLY_ONLY_EXACT_COLD_D25_NO_OLD_LIVE_SOURCE_PERMISSION')
    return dict(schema='ARX5_E2_ONLY_VANDA_WITH_ORIGINAL_UNSTARTED_D25_CORE_HOLD_V1',
        main_request_id=pbs_rows[0]['request_id'],Core_request_id=Core_rows[0]['request_id'],
        Core_unstarted=True,Core_native_permission=False,remote_liveness_required=False)

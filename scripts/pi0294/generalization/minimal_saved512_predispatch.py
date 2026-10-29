"""A known new registered-self gap is not an unknown submitted job.

Only classification uses the narrowed view; canonical bytes/UNKNOWN are never
changed. A prior qsub journal/intent or any other gap still blocks dispatch.
"""
import base64,time
from pathlib import Path
from scripts.pi0292.durable import read
from . import minimal_execution as e,minimal_saved512_execution as execution
from scripts.pi0294.dec1 import vanda_campaign as queue
from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside


def validate(g,full,request,request_sha,*,now_epoch):
    execution.validate(g);_,row=execution.basic(g)
    e.need(row['request_sha256']==request_sha and row['status']==e.STATES[0]and row['job_id']is None
        and all(row.get(k)==v for k,v in request.items())and g['pending_updates']==g['unknown_effects']==[],
        'SAVED512_PREDISPATCH_ONLY_EXACT_ORIGINAL_REGISTERED_SELF_REQUEST')
    rows=queue.queue_rows(full,now_epoch=now_epoch);own,outside,unknown=queue.classify(g,rows)
    e.need(own==[]and unknown==[dict(job_id=None,reason='ACTIVE_LEDGER_WITHOUT_CURRENT_NATIVE_JOB_PROOF')],
        'SAVED512_PREDISPATCH_NO_OTHER_UNKNOWN_OR_ACTIVE_NATIVE_JOB')
    view=dict(g,PBS_submissions=[r for r in g['PBS_submissions']if r is not row]);a,b,c=queue.classify(view,rows)
    e.need(a==c==[]and b==outside,'SAVED512_PREDISPATCH_ONLY_EXACT_SELF_GAP_IN_CLASSIFICATION_VIEW')
    return rows,outside


def collect(owner):
    base=owner.BASE;saved=read(base/'SOURCE_REQUEST.private.json');path=Path(saved['directory'])/'REQUEST.json'
    e.need(e.sha(path.read_bytes())==saved['request_sha256'],'SAVED512_PREDISPATCH_ACTUAL_SAVED_REQUEST_BYTES')
    e.need(not(base/'SOURCE_QSUB_INTENT.private.json').exists()and not(base/'SOURCE_QSUB_REPLY.private.json').exists()
        and not(base/'SOURCE_JOB_BOUND.private.json').exists(),'SAVED512_PREDISPATCH_NO_PREVIOUS_QSUB_ATTEMPT')
    for p in(base/'setup-rpc').glob('*-INTENT.json'):
        v=read(p);e.need(v.get('intent',v).get('phase')!='QSUB','SAVED512_PREDISPATCH_NO_PRIOR_QSUB_JOURNAL')
    body=owner.global_path.read_bytes();g=read(owner.global_path);full=queue.collect_queue()
    rows,outside=validate(g,full,saved['request'],saved['request_sha256'],now_epoch=time.time())
    checked_outside(rows,outside,base64.b64decode(owner.CONFIG['outside_confirmation_base64'],validate=True))
    e.need(owner.global_path.read_bytes()==body,'SAVED512_PREDISPATCH_GLOBAL_CHANGED_DURING_QUEUE_READ')
    return dict(status='ACTUAL_SAVED512_EXACT_REGISTERED_SELF_UNSUBMITTED_NO_OTHER_UNKNOWN',
        GLOBAL_sha256=e.sha(body),complete_queue=full,observed_epoch=time.time(),canonical_mutation=False)

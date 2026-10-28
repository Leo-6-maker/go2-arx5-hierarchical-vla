"""A fresh pre-qsub queue proof distinct from the immutable registration input."""
import json
from . import minimal_execution as e,minimal_remaining_execution as execution
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

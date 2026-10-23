"""Pure exact completion proposal for the known pre-qsub two-ledger split.

The original intent already reserved DEC1; its global transaction failed before
commit. This module authorizes no IO, new request, qsub, model or owner restart.
Original unknown journal and once-intent must remain historical evidence.
"""
from copy import deepcopy
import hashlib
import json
from . import b_heartbeat_recovery as b
from .a40_allocation import request_scope
from scripts.pi0294.pv1.neural_ledger import check

REQUEST='185541e4-2348-4a3c-bee7-1a5703369780'
REQUEST_SHA='6c6d12d14895e4e22fa7a835c7edac8579ff341825911fd0d3b5fbbec76809ca'
GLOBAL_SHA='bb193e1c8183e608e68f0c8e1d8c395e861486688458de9f820b82a5be609516'
DEC1_SHA='65c5ec3699ef26d8fcb3481e5ddc0d63a82d72ff65e5527fbe31f4f631298dc2'
INTENT_SHA='7229c15e8d28ebcd80cce31b722cfcf19210e64742f655afb145119b4f194c29'


def need(ok,why):
    if not ok:raise PermissionError('B_RESERVATION_RECONCILIATION_'+why)


def load(raw,expected):
    need(type(raw) is bytes and hashlib.sha256(raw).hexdigest()==expected,'EXACT_ACTUAL_BYTES')
    def pairs(items):
        v={}
        for k,x in items:need(k not in v,'NO_DUPLICATE_KEYS');v[k]=x
        return v
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:need(False,'FINITE'))


def propose(global_bytes,DEC1_bytes,intent_bytes,request_bytes,*,now_epoch):
    h=load(global_bytes,GLOBAL_SHA);d=load(DEC1_bytes,DEC1_SHA)
    intent=load(intent_bytes,INTENT_SHA);request=load(request_bytes,REQUEST_SHA)
    request_scope(request,now=now_epoch);b.lineage(request);b.validate_adopted(h,d);check(h)
    need(request['request_id']==intent['request_id']==REQUEST and intent['request_sha256']==REQUEST_SHA,
         'SAME_ALREADY_CREATED_REQUEST_NOT_A_RETRY')
    need(h==intent['original_global'] and d==intent['planned_DEC1'],
         'ONLY_CONFIRMED_GLOBAL_ORIGINAL_DEC1_PLANNED_SPLIT')
    planned=deepcopy(intent['planned_global']);old_d=intent['original_DEC1']
    row=planned['PBS_submissions'][-1]
    need(planned['PBS_submissions'][:-1]==h['PBS_submissions'] and len(h['PBS_submissions'])==20
         and row['request_id']==REQUEST and row['request_sha256']==REQUEST_SHA
         and row['source_sha']=='2ed8aea9f782741dfc0298438f5762c592e87149'
         and row['status']=='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY' and row['job_id'] is None
         and row['start_native_step']==1024 and row['segment_stop_step']==2048
         and row['training_lineage_key']==b.LINEAGE,'EXACT_NINTH_B_SOURCE_ALLOCATION')
    need(h['optimizer_updates']==d['optimizer_updates']==7720 and planned['optimizer_updates']==7720
         and planned['used']==dict(h['used'],PBS_submissions=21)
         and planned['training_runs']==h['training_runs'],'ALL_SPENT_COSTS_AND_LINEAGES_PRESERVED')
    need(planned['reserved']==dict(h['reserved'],H200_s=22560,optimizer_updates=1024,neural_queries=29)
         and planned['CPU_reserved_s']==54000 and planned['storage_reserved_bytes']==70000000000,
         'ORIGINAL_FULL_GLOBAL_RESERVATION_ONLY')
    need(d['new_submissions'][:-1]==old_d['new_submissions'] and len(old_d['new_submissions'])==9
         and d['new_submissions'][-1]['request_id']==REQUEST
         and d['GPU_reserved_s']==22560 and d['CPU_reserved_s']==54000
         and d['optimizer_updates_reserved']==1024 and d['neural_queries_reserved']==29
         and d['GPU_site_output_reserved_bytes']==70000000000,'DEC1_ALREADY_HAS_EXACT_ORIGINAL_RESERVATION')
    # Match the mature global transaction's revision increment once; this is
    # completing its recorded plan, never recomputing another reservation.
    need(planned['revision']==h['revision'],'ORIGINAL_UNCOMMITTED_REVISION')
    planned['revision']+=1;check(planned);b.validate_adopted(planned,d)
    return dict(schema='B_RECOVERY_KNOWN_PREQSUB_RESERVATION_COMPLETION_PROPOSAL_V1',
        original_request_id=REQUEST,original_request_sha256=REQUEST_SHA,proposed_global=planned,
        expected_global_sha256=GLOBAL_SHA,expected_DEC1_sha256=DEC1_SHA,
        original_intent_sha256=INTENT_SHA,DEC1_rewrite=False,new_request=False,
        native_dispatch_admitted=False,automatic_retry=False,old_unknown_journal_preserved=True,
        new_GPU_PBS_models_queries_updates_physics=0)

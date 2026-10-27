"""Exact CPU-export join around the original complete-account queue reader.

The old campaign classifier's unresolved unbound intent and one-GPU rule stay
unchanged. Only the unique validated GEN1 CPU row is handled here, and its
unbound state remains explicit. This grants neither CPU nor model execution.
"""
from copy import deepcopy
from . import amd_export_accounting as accounting
from scripts.pi0294.dec1 import vanda_campaign as original

def view(g,rows,*,request_id):
    chosen=accounting
    from . import amd_export_repair_accounting as repair
    matches=[x for x in g['PBS_submissions']if x.get('request_id')==request_id]
    if len(matches)==1 and matches[0].get('platform_data_version')==repair.VERSION:chosen=repair
    items=chosen.rows(g)
    accounting.a.need(len(items)==1 and items[0]['request_id']==request_id
        and items[0]['status']!=accounting.STATES[-1],
        'QUEUE_ONE_ACTUAL_LIVE_OR_UNBOUND_CPU_REQUEST')
    item=items[0];job=item['job_id'];current=[];other=[]
    for row in rows:
        if job is not None and row['job_id']==job:current.append(row)
        else:other.append(row)
    accounting.a.need(len(current)<=1,'QUEUE_NO_DUPLICATE_CPU_HANDLE')
    for row in current:
        accounting.a.need(row['platform']==item['native_platform']
            and row['project']==item['project']and row['name']==item['job_name']
            and type(row['ngpus'])in(int,str)and str(row['ngpus'])=='0',
            'QUEUE_REAL_CPU_IDENTITY_AND_ZERO_GPU_NOT_OLD_RENDER_ROLE')
    # Only this exact validated row has been joined above. All other ledger
    # intents and all other selected handles still use the mature classifier.
    rest=deepcopy(g);rest['PBS_submissions']=[x for x in rest['PBS_submissions']
        if x.get('request_id')!=request_id]
    owned,outside,unknown=original.classify(rest,other)
    own=[{k:v for k,v in x.items()if k!='ngpus'}for x in current]
    if job is not None and not current:
        unknown.append(dict(job_id=job,reason='EXACT_BOUND_CPU_ABSENT_FROM_COMPLETE_NATIVE_QUEUE'))
    if any(x['state']not in('Q','R')for x in current):unknown.extend(own)
    accounting.a.need(len(owned)+len(own)<=1,'QUEUE_ONLY_ONE_CAMPAIGN_ALLOCATION')
    return dict(own=owned+own,outside=outside,unknown=unknown,
        own_CPU_unbound=job is None,request_id=request_id,
        native_job_id=job,native_CPU_release=False,model_permission=False,
        complete_selected_rows_retained=deepcopy(rows))

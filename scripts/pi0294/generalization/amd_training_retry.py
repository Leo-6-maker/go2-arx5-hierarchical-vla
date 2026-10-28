"""Conditional single cold retry in the original campaign and once ledger.

The exact additional direct human grant is pinned below. Its publication is
not native adoption: original once IO, registered own roots and full fresh
admission remain mandatory. Test fixture grants never become real evidence.
"""
import base64,json,math,re
from copy import deepcopy
from . import amd_training_accounting as a,amd_training_retry_context as context

FIELD='GEN1_exact1497713_retry_PBS37_authority'
SCHEMA='GEN1_EXACT1497713_ONE_PBS_ADOPTED_AUTHORITY_V1'
GRANT_SCHEMA='GEN1_EXACT1497713_ONE_PBS_DIRECT_HUMAN_AUTHORIZATION_V1'
GRANT_SHA='80489856b5e25066d13fb8d6e35f00b9d2ab427b9c727346ab62fbd033e8f3e4'
PROFILE=a.PROFILE+'-retry01'
REQUEST_KEYS={'retry_generation','retry_scope_grant_sha256','retry_predecessor_context_sha256'}

def validate_request(r):
    a.need(GRANT_SHA is not None and r.get('retry_generation')==1
        and type(r['retry_generation'])is int and r.get('retry_scope_grant_sha256')==GRANT_SHA
        and type(r.get('retry_predecessor_context_sha256'))is str
        and re.fullmatch('[0-9a-f]{64}',r['retry_predecessor_context_sha256'])is not None,
        'RETRY_EXACT_PINNED_NEW_COUNT_AUTHORITY_NOT_ORIGINAL_FOUR_CAP_GO')
    return r

def _grant(grant_bytes):
    a.need(type(GRANT_SHA)is str and re.fullmatch('[0-9a-f]{64}',GRANT_SHA)is not None
        and type(grant_bytes)is bytes and a.sha(grant_bytes)==GRANT_SHA,
        'RETRY_DIRECT_COUNT_GRANT_NOT_YET_PINNED_OR_WRONG_BYTES')
    v=json.loads(grant_bytes)
    a.need(v['schema']==GRANT_SCHEMA and v['status']=='DIRECT_HUMAN_AUTHORIZED'
        and type(v['human_answer'])is str and bool(v['human_answer'].strip())
        and v['caps']=={'PBS_submissions':dict(before=36,after=37)}
        and v['old_request_id']==context.failure.REQUEST and v['old_job_id']==context.failure.JOB
        and v['existing_four_cap_grant_sha256']==a.authority.GRANT_SHA
        and v['other_caps_clocks_science_and_final_protection_unchanged']is True,
        'RETRY_ONLY_ONE_PBS_NOT_A_NEW_RESOURCE_OR_SCIENCE_PACKAGE')
    return v

def validate(g):
    if FIELD not in g:return None
    m=g[FIELD];grant=_grant(base64.b64decode(m['grant_base64'],validate=True))
    prior=m['predecessor_context'];scope=a.authority.validate(g)
    a.need(m['schema']==SCHEMA and m['grant_sha256']==GRANT_SHA
        and m['caps']=={'PBS_submissions':37}and m['before_GLOBAL_sha256']==prior['closed_GLOBAL_sha256']
        and m['context_sha256']==a.sha(a.raw(prior))
        and m['historical_LOCAL_sha256']==scope['historical_LOCAL_sha256']
        and all(m[k]==prior[k]==scope[k]==grant[k]for k in
            ('T_CD','stop_new_training_epoch','deadline_epoch'))
        and type(m['adopted_epoch'])in(int,float)and math.isfinite(m['adopted_epoch'])
        and scope['T_CD']<=m['adopted_epoch']<scope['stop_new_training_epoch']
        and m['actual_LOCAL_written']is False and m['native_admission']is False,
        'RETRY_ONE_ADOPTED_COUNT_WITH_ORIGINAL_CLOCK_AND_HISTORICAL_LOCAL')
    context.verify_preserved_history(g,prior,check_package=False)
    return m

def effective_caps(g,caps):
    """One projection of the original cap; no second budget or reset."""
    out=dict(caps)
    if FIELD in g:
        m=validate(g);a.need(out['PBS_submissions']==36,'RETRY_ORIGINAL_PBS36_BEFORE37_ONLY')
        out['PBS_submissions']=m['caps']['PBS_submissions']
    return out

def propose(global_bytes,historical_LOCAL_bytes,grant_bytes,*,capture,closed_readback,now_epoch):
    grant=_grant(grant_bytes);g=json.loads(global_bytes);prior=context.closed_predecessor(capture,closed_readback)
    a.need(FIELD not in g and a.sha(global_bytes)==prior['closed_GLOBAL_sha256']
        and global_bytes==base64.b64decode(closed_readback['GLOBAL_base64'],validate=True),
        'RETRY_ONCE_FRESH_EXACT_CLOSED_GLOBAL_NOT_A_PREPARED_OVERWRITE')
    scope=a.authority.validate(g)
    a.need(a.sha(historical_LOCAL_bytes)==scope['historical_LOCAL_sha256']
        and all(grant[k]==scope[k]for k in ('T_CD','stop_new_training_epoch','deadline_epoch'))
        and scope['T_CD']<=now_epoch and now_epoch+22560<scope['stop_new_training_epoch']
        and now_epoch+600<prior['closed_export']['operation_deadline_epoch']
        and all(x['status']=='NATIVE_TERMINAL'for x in g['PBS_submissions'])
        and all(x['status']=='NATIVE_TERMINAL'for x in g.get(a.original.CORE_ROWS,{}).values()),
        'RETRY_EXACT_CLOSED_NATIVE_AND_REAL_REMAINING_ORIGINAL_WINDOW')
    before=a.scope.full_forecast(global_bytes,historical_LOCAL_bytes)
    a.need(before['cap_changes_required']=={'PBS_submissions':dict(before=36,after=37,after_GiB=None)},
        'RETRY_FRESH_COMPLETE_FUTURE_ONLY_THIS_ONE_COUNT_GAP')
    m=dict(schema=SCHEMA,grant_sha256=GRANT_SHA,grant_base64=base64.b64encode(grant_bytes).decode(),
        caps=dict(PBS_submissions=37),before_GLOBAL_sha256=a.sha(global_bytes),
        predecessor_context=prior,context_sha256=a.sha(a.raw(prior)),
        historical_LOCAL_sha256=a.sha(historical_LOCAL_bytes),actual_LOCAL_written=False,
        T_CD=scope['T_CD'],stop_new_training_epoch=scope['stop_new_training_epoch'],
        deadline_epoch=scope['deadline_epoch'],adopted_epoch=now_epoch,native_admission=False,
        full_before_future_proof_sha256=a.sha(a.raw(before)))
    out=deepcopy(g);out[FIELD]=m;out['revision']+=1;validate(out)
    a.need(all(out[k]==v for k,v in g.items()if k!='revision'),
        'RETRY_NO_COST_RESERVATION_HISTORY_OR_ORIGINAL_CAP_BYTES_CHANGED')
    a.original.validate_global(out);after=a.scope.full_forecast(a.raw(out),historical_LOCAL_bytes)
    a.need(after['cap_changes_required']=={},'RETRY_FULL_FUTURE_STILL_FITS_AFTER_ONE_COUNT_ONLY')
    return dict(proposed_global=out,full_remaining_proof=after,native_admission=False,
        actual_LOCAL_written=False,new_requests=0,new_PBS_submissions=0)

def marker_for(g,index):
    m=validate(g)
    a.need(m is not None and index==1 and type(m.get('training'))is dict,
        'RETRY_EXACT_SECOND_REQUEST_MARKER_NOT_ARBITRARY_AUTORETRIES')
    return m['training']

def require_register(g,r):
    m=validate(g);a.need(m is not None and 'training'not in m,'RETRY_ONE_NEW_COLD_REGISTRATION_ONLY')
    validate_request(r);prior=m['predecessor_context'];context.verify_preserved_history(g,prior)
    items=[x for x in g['PBS_submissions']
        if x.get('platform_training_version')==a.VERSION]
    a.need(len(items)==1 and items[0]==prior['original_row']
        and r['retry_predecessor_context_sha256']==m['context_sha256']
        and r['request_id']not in prior['all_prior_request_ids']
        and r['submission_nonce']not in prior['all_prior_submission_nonces']
        and r['source_sha']!=prior['predecessor_source_sha']
        and r['closed_export_sha256']==a.sha(a.raw(prior['closed_export']))
        and r['bank_manifest_sha256']==prior['bank_manifest_sha256']
        and r['recipe_sha256']==prior['recipe_sha256'],
        'RETRY_NEW_SEALED_IDENTITY_AND_SAME_ORIGINAL_SAVED_INPUTS')
    return m

def check_bound(g,r):
    m=validate(g);validate_request(r)
    a.need(m is not None and r['retry_predecessor_context_sha256']==m['context_sha256'],
        'RETRY_BOUND_ROW_MUST_JOIN_ITS_REAL_CLOSED_PREDECESSOR')
    return m

def verify_registered_root(root,request):
    """Runtime join of sealed context and actual unchanged old CLOSED lease."""
    from pathlib import Path
    from scripts.pi0292.durable import read
    from scripts.pi0294.native_dataset import sha
    root=Path(root);g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json');m=check_bound(g,request)
    path=root/'plans/GEN1_CLOSED_TRAIN_PREDECESSOR_CONTEXT.private.json'
    a.need(not path.is_symlink()and path.stat().st_mode&0o777==0o400
        and sha(path)==m['context_sha256']==request['retry_predecessor_context_sha256']
        and read(path)==m['predecessor_context'],
        'RETRY_ACTUAL_NEW_SEALED_CONTEXT_NOT_AN_IN_MEMORY_PREPARATION')
    prior=m['predecessor_context'];old=Path(prior['predecessor_root']);binding=read(old/'OUTPUT_BINDING.private.json')
    a.need(not old.is_symlink()and old.resolve().parent==root.resolve().parent
        and sha(old/'RUN_MANIFEST.json')==prior['original_row']['run_manifest_sha256']
        and sha(binding['lease'])==prior['closed_lease_sha256']and read(binding['lease'])['status']=='CLOSED',
        'RETRY_PREDECESSOR_REAL_CLOSED_HOST_FILES_NOT_ONLY_CPU_FIXTURE')
    context.verify_cold_science(read(root/'RUN_MANIFEST.json'),request,read(root/'SOURCE_SEAL.json'),prior)
    return prior

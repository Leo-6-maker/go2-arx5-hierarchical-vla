"""Read-only provenance for the exact closed1497713 cold successor.

No grant, allocation, request, PBS slot or training permission is created here.
The original single-request accounting still rejects a second registration.
This joins actual once receipts to the full original failed-byte proposal so
the next immutable owner can preserve the failure without treating it as a
successful checkpoint, a reusable request, or a refundable cost.
"""
import base64,json,math
from copy import deepcopy
from pathlib import Path
from . import amd_training_accounting as a,amd_training_bootstrap_failure as failure

STATUS='PASS_EXACT1497713_CLOSED_PRERELEASE_PREDECESSOR_NO_RETRY_PERMISSION'
READBACK='INDEPENDENT_ACTUAL_GEN1_1497713_SETTLED_OWN_LEASE_CLOSED_ORIGINALS_UNCHANGED'
SETTLE_KEY='GEN1-1497713-PRERELEASE-SETTLE-01'
CLOSE_KEY='GEN1-1497713-PRERELEASE-CLOSE-01'

def _decode(value):
    a.need(type(value)is str,'RETRY_CAPTURE_BASE64_STRING')
    return base64.b64decode(value,validate=True)

def _once(proofs,key,*,target,before_sha256,after_sha256):
    values={};hashes={}
    for suffix in ('INTENT.json','RECEIPT.json'):
        name=key+'.'+suffix;row=proofs[name];b=_decode(row['base64'])
        a.need(a.sha(b)==row['sha256'],'RETRY_EXACT_ONCE_BYTES:'+name)
        values[suffix]=json.loads(b);hashes[suffix]=a.sha(b)
    i,r=values['INTENT.json'],values['RECEIPT.json']
    a.need(i['schema']=='DEC1_SOURCE_CPU_ONCE_METADATA_INTENT_V1'
        and r['schema']=='DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1'
        and i['request_id']==r['request_id']==key and i['target']==r['target']==target
        and i['before_sha256']==r['before_sha256']==before_sha256
        and i['proposed_sha256']==r['after_sha256']==after_sha256
        and r['intent_sha256']==hashes['INTENT.json']
        and r['status']=='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION'
        and i['native_dispatch_admitted']is r['native_dispatch_admitted']is False,
        'RETRY_REAL_ONCE_PAIR_NOT_AN_INTENT_OR_UNKNOWN_REPLY')
    a.need(i['temporary']==target+'.source-once-'+key,'RETRY_ONCE_CANONICAL_TEMPORARY')
    return hashes

def closed_predecessor(capture,readback):
    """Return sealed cold input only after exact settlement and own closure."""
    result=failure.audit(capture)
    gb=_decode(readback['GLOBAL_base64']);lb=_decode(readback['lease_base64'])
    a.need(readback['status']==READBACK and a.sha(gb)==readback['GLOBAL_sha256']
        and a.sha(lb)==readback['lease_sha256']
        and readback['original_missing_effects_and_final_unchanged']is True
        and readback['new_scientific_stage_completed']is False
        and readback['models']==readback['updates']==readback['queries']==0,
        'RETRY_ACTUAL_INDEPENDENT_CLOSED_READBACK_NOT_PREPARED')
    before=json.loads(_decode(capture['GLOBAL_base64']));g=json.loads(gb)
    expected=failure.settle(before,capture)['proposed_global']
    a.need(gb==a.raw(expected),'RETRY_EXACT_ALL_DIMENSION_SETTLEMENT_NOT_COUNTERS_ONLY')
    rows=a.rows(g);a.need(len(rows)==1,'RETRY_ONE_EXACT_ORIGINAL_FAILURE')
    row=rows[0];mark=a.authority.validate(g)[a.FIELD]
    manifest=json.loads(_decode(capture['files']['RUN_MANIFEST.json']['base64']))
    seal=json.loads(_decode(capture['files']['SOURCE_SEAL.json']['base64']))
    binding=json.loads(_decode(capture['files']['OUTPUT_BINDING.private.json']['base64']))
    lease=json.loads(lb);old_lease=json.loads(_decode(capture['lease_base64']))
    a.need(row['request_id']==failure.REQUEST and row['job_id']==failure.JOB
        and row['status']==a.STATES[-1]and row['native_exit_status']==1
        and row['scientific_endpoint_complete']is False
        and row['known_prerelease_failure_version']==failure.STATUS
        and row['neural_output_charge_bytes']==0
        and row['committed_optimizer_updates']==row['query_attempts']==row['model_load_attempts']==0
        and row['terminal_native_sha256']==a.sha(a.raw(result['native']))
        and row['terminal_effect_summary_sha256']==a.sha(a.raw(result['effects']))
        and row['full_checkpoint_audit_sha256']==a.sha(a.raw(result['audit'])),
        'RETRY_EXACT_SETTLED_FAILURE_NOT_A_COMPLETE_MODEL_OR_OTHER_PHASE')
    added={'closed_epoch','terminal_native_sha256','terminal_audit_sha256',
        'terminal_effect_summary_sha256','payload_workers_terminal','known_prerelease_failure',
        'original_missing_effect_journal_and_final_preserved'}
    a.need(old_lease['status']=='ACTIVE'and lease['status']=='CLOSED'
        and set(lease)==set(old_lease)|added
        and all(lease[k]==v for k,v in old_lease.items()if k!='status')
        and lease['manifest_sha256']==row['run_manifest_sha256']==binding['run_manifest_sha256']
        and lease['run_uuid']==manifest['run_uuid']==row['run_uuid']
        and lease['terminal_native_sha256']==row['terminal_native_sha256']
        and lease['terminal_audit_sha256']==row['full_checkpoint_audit_sha256']
        and lease['terminal_effect_summary_sha256']==row['terminal_effect_summary_sha256']
        and all(lease[k]is True for k in ('payload_workers_terminal','known_prerelease_failure',
            'original_missing_effect_journal_and_final_preserved')),
        'RETRY_CLOSED_OWN_GPU_CONTROL_LEASE_NOT_ACTIVE_OR_REOPENED')
    a.need(type(lease['closed_epoch'])in(int,float)and math.isfinite(lease['closed_epoch'])
        and capture['epoch']<=lease['closed_epoch']<=readback['epoch']
        and readback['epoch']<lease['deadline_epoch'],
        'RETRY_REAL_CLOSURE_AND_READBACK_CLOCK_NOT_RESET')
    proofs=readback['proofs']
    a.need(set(proofs)=={k+'.'+s for k in (SETTLE_KEY,CLOSE_KEY)
        for s in ('INTENT.json','RECEIPT.json')},'RETRY_BOTH_REAL_ONCE_PAIRS')
    ledger=str(Path(manifest['output_root']).parent.parent/'HOPPER_RESOURCE_LEDGER.json')
    settled=_once(proofs,SETTLE_KEY,target=ledger,
        before_sha256=capture['GLOBAL_sha256'],after_sha256=a.sha(gb))
    closed=_once(proofs,CLOSE_KEY,target=binding['lease'],
        before_sha256=capture['lease_sha256'],after_sha256=a.sha(lb))
    scope=a.authority.validate(g)
    return dict(status=STATUS,predecessor_job_id=failure.JOB,
        predecessor_request_id=failure.REQUEST,predecessor_nonce=row['submission_nonce'],
        predecessor_source_sha=failure.SOURCE,predecessor_root=manifest['output_root'],
        closed_GLOBAL_sha256=a.sha(gb),closed_lease_sha256=a.sha(lb),
        original_row=deepcopy(row),original_marker_sha256=a.sha(a.raw(mark)),
        all_original_rows_sha256=[a.sha(a.raw(r))for r in g['PBS_submissions']],
        all_prior_request_ids=[r['request_id']for r in g['PBS_submissions']],
        # Older closed compatibility rows predate submission_nonce. Their
        # entire bytes and request IDs remain preserved; absence is not a
        # nonce to fabricate or an excuse to drop that historical row.
        all_prior_submission_nonces=[r['submission_nonce']for r in g['PBS_submissions']
            if 'submission_nonce'in r],
        original_C_D_package_sha256=a.sha(a.raw(g[a.original.PACKAGE])),
        original_C_D_package_identity={k:deepcopy(g[a.original.PACKAGE][k])for k in
            ('schema','scope_grant_sha256','order','created_epoch','original_resources','endpoints')},
        previous_paid_totals=dict(GPU_s=g['used']['H200_s'],PBS=g['used']['PBS_submissions'],
            updates=g['optimizer_updates'],queries=g['generation_queries'],CPU_s=g['CPU_s'],
            current_output_bytes=g['new_storage_current_bytes'],peak_output_bytes=g['new_storage_peak_bytes']),
        exact_once_pairs=dict(settlement=settled,closure=closed),
        closed_export=deepcopy(mark['closed_export']),bank_manifest_sha256=row['bank_manifest_sha256'],
        recipe_sha256=row['recipe_sha256'],parent_checkpoint_manifest_sha256=a.original.PARENT,
        frozen15=deepcopy(seal['frozen15']),frozen_overlay_origins=deepcopy(seal['frozen_overlay_origins']),
        statistics_sha256=a.original.STATS,T_CD=scope['T_CD'],
        stop_new_training_epoch=scope['stop_new_training_epoch'],deadline_epoch=scope['deadline_epoch'],
        optimizer_start_step=0,inherit_failed_or_C_D_weights=False,
        original_control_paid_once=deepcopy(mark['control_paid_once']),
        registration_admitted=False,native_GPU_admission=False,native_training_permission=False,
        current_PBS_cap=scope['caps']['PBS_submissions'],required_PBS_cap=37,
        additional_explicit_human_count_authority_required=True,
        repeat_Teacher_or_official_export=False,bank_pack_before_model_still_required=True)

def verify_preserved_history(g,context,*,check_package=True):
    """Fresh downstream readers retain exact prior rows and all paid totals.

    This deliberately supplies no second-row schema or mutation permission;
    those must be joined to a separately granted/adopted cold successor.
    """
    a.need(context['status']==STATUS and context['registration_admitted']is False
        and context['native_GPU_admission']is context['native_training_permission']is False,
        'RETRY_PREDECESSOR_CONTEXT_NEVER_COUNTS_AS_PERMISSION')
    rows=g['PBS_submissions'];old=context['all_original_rows_sha256']
    a.need(len(rows)>=len(old)and all(a.sha(a.raw(rows[i]))==h for i,h in enumerate(old)),
        'RETRY_EVERY_OLD_SUBMISSION_AND_FAILURE_IMMUTABLE')
    own=[r for r in rows if r.get('request_id')==failure.REQUEST]
    a.need(type(check_package)is bool and len(own)==1 and own[0]==context['original_row']
        and a.sha(a.raw(a.authority.validate(g)[a.FIELD]))==context['original_marker_sha256']
        and all(g[a.original.PACKAGE][k]==v for k,v in context['original_C_D_package_identity'].items()),
        'RETRY_EXACT_PREDECESSOR_AND_ORIGINAL_C_D_PACKAGE_IDENTITY')
    if check_package:
        a.need(a.sha(a.raw(g[a.original.PACKAGE]))==context['original_C_D_package_sha256'],
            'RETRY_BEFORE_REGISTRATION_ALL_ORIGINAL_C_D_ESCROW_UNCHANGED')
    else:
        # A permanent count marker must not freeze legitimate later original
        # Eval accounting. Existing package/order/resource validators retain
        # those gates; this flag grants no stage or model permission.
        a.original.package_view(g)
    paid=context['previous_paid_totals']
    current=dict(GPU_s=g['used']['H200_s'],PBS=g['used']['PBS_submissions'],
        updates=g['optimizer_updates'],queries=g['generation_queries'],CPU_s=g['CPU_s'],
        current_output_bytes=g['new_storage_current_bytes'],peak_output_bytes=g['new_storage_peak_bytes'])
    a.need(all(current[k]>=v for k,v in paid.items()),'RETRY_NO_PREVIOUS_PAID_COST_OR_PEAK_REFUND')
    scope=a.authority.validate(g)
    a.need(all(scope[k]==context[k]for k in ('T_CD','stop_new_training_epoch','deadline_epoch'))
        and g['pending_updates']==g['unknown_effects']==[],
        'RETRY_SAME_CLOCK_AND_NO_UNKNOWN_EFFECTS')
    return dict(status=STATUS,all_original_paid_history_preserved=True,
        registration_admitted=False,native_GPU_admission=False,native_training_permission=False)

def verify_cold_science(manifest,request,seal,context):
    """Check cold packet parity only; no new UUID, cap or role is registered.

    Current runtime remains untouched and still rejects a second row. The
    eventual owner must separately validate a real adopted count grant and
    its registered profile/lease, fresh quota/full queue and model releases.
    """
    a.need(context['status']==STATUS and context['registration_admitted']is False,
        'RETRY_EXACT_PREDECESSOR_CONTEXT')
    # This function checks science parity while count approval may still be
    # pending. Actual request/root/owner admission validates every retry
    # authority field separately; never turn CPU parity into such approval.
    from . import amd_training_retry as retry
    a.validate_request({k:v for k,v in request.items()if k not in retry.REQUEST_KEYS})
    a.need(request['request_id']not in context['all_prior_request_ids']
        and request['submission_nonce']not in context['all_prior_submission_nonces']
        and request['job_id']is None and request['status']==a.STATES[0],
        'RETRY_NEW_COLD_REQUEST_NONCE_NOT_AN_OLD_WRITE_REPLAY')
    a.need(manifest['run_uuid']==request['run_uuid']==Path(manifest['output_root']).name
        and manifest['output_root']!=context['predecessor_root']
        and manifest['scope']==manifest['platform_training_version']==a.VERSION
        and manifest['code']['vla_commit']==request['source_sha']==seal['source_sha']
        and seal['source_sha']!=context['predecessor_source_sha']
        and manifest['code']['source_seal_sha256']==a.sha(a.raw(seal))
        and request['run_manifest_sha256']==a.sha(a.raw(manifest))
        and request['retry_predecessor_context_sha256']==a.sha(a.raw(context))
        and manifest['execution']['agent_liveness_required']is False,
        'RETRY_NEW_IMMUTABLE_ROOT_AND_SOURCE_NO_HOT_EDIT_OR_AGENT_HEARTBEAT')
    a.need(seal['frozen15']==context['frozen15']
        and seal['frozen_overlay_origins']==context['frozen_overlay_origins']
        and all(seal['files'][n]==h for n,h in context['frozen15'].items()),
        'RETRY_ORIGINAL_FIFTEEN_NUMERICAL_FILES_AND_TRUE_OVERLAY_ORIGIN')
    a.need(request['closed_export_sha256']==a.sha(a.raw(context['closed_export']))
        and request['bank_manifest_sha256']==context['bank_manifest_sha256']
        and request['recipe_sha256']==context['recipe_sha256']
        and request['CPU_bank_deadline_epoch']==context['closed_export']['operation_deadline_epoch']
        and request['T_CD']==context['T_CD']and request['deadline_epoch']==context['deadline_epoch'],
        'RETRY_SAME_SAVED_EXPORT_RECIPE_BANK_AND_UNCHANGED_CLOCK')
    science=manifest['science']
    a.need(science['arm']==request['arm']==a.bank.IDENTITY
        and science['parent_checkpoint_manifest_sha256']==context['parent_checkpoint_manifest_sha256']
        and science['statistics_sha256']==context['statistics_sha256']
        and science['optimizer_updates']==1024 and science['resume_only_saves']==[512]
        and science['sole_behavioral_endpoint']==1024 and science['C_D_weights_not_inherited']is True,
        'RETRY_INDEPENDENT_COMMON_CORR_FRESH_OPTIMIZER_NOT_FAILED_OR_C_D_ENDPOINT')
    limits=manifest['limits']
    a.need(all(limits[k]==v for k,v in dict(CPU_control_upper_s=1800,GPU_s=22560,
        optimizer_updates=1024,queries=29,training_CPU_upper_s=54000,
        output_upper_bytes=70000000000,new_physics=0).items()),
        'RETRY_ORIGINAL_BOUNDED_ENVELOPE_NOT_AN_UNAPPROVED_EXPANSION')
    return dict(status=STATUS,science_and_saved_input_parity=True,
        original_bank_and_model_gates_still_required=True,registration_admitted=False,
        native_GPU_admission=False,native_training_permission=False)

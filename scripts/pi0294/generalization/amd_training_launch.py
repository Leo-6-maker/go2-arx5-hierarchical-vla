"""Cold identity/config factory around the original PBS submit program."""
from copy import deepcopy
from . import amd_training_accounting as a,amd_learning

def template(original_template,*,source_sha,run_uuid,request_id,submission_nonce,scope_marker,closed,original_recipe,retry_marker=None):
    v=deepcopy(original_template)
    v.update(schema='GEN1_ACTUAL_NATIVE_TRAINING_REQUEST_V1',platform_training_version=a.VERSION,
        scope_grant_sha256=a.authority.GRANT_SHA,arm=a.bank.IDENTITY,stage='GEN1_INDEPENDENT_TRAIN1024',
        source_sha=source_sha,run_uuid=run_uuid,request_id=request_id,submission_nonce=submission_nonce,
        T_CD=scope_marker['T_CD'],deadline_epoch=scope_marker['deadline_epoch'],job_id=None,status=a.STATES[0],
        bank_manifest_sha256=closed['bank_manifest_sha256'],new_bank_manifest_sha256=closed['bank_manifest_sha256'],
        CPU_bank_deadline_epoch=closed['operation_deadline_epoch'],
        recipe_sha256=a.sha(a.raw(amd_learning.recipe(original_recipe))),closed_export_sha256=a.sha(a.raw(closed)),
        training_block=dict(arm=a.bank.IDENTITY,replica=False,stop_after_step=1024,resume_checkpoint_relative=None),
        expected_native_start_step=0,reserved_optimizer_updates=1024,reserved_neural_queries=29)
    # Manifest and actual cold audit hashes are joined after these immutable
    # files exist. They must never be fabricated/self-referential placeholders.
    for k in ('run_manifest_sha256','cold_input_audit_sha256','request_sha256','submit_count','created_epoch'):v.pop(k,None)
    if retry_marker is not None:
        from . import amd_training_retry as retry
        a.need(retry.GRANT_SHA is not None and retry_marker['grant_sha256']==retry.GRANT_SHA,
            'TRAIN_COLD_TEMPLATE_REQUIRES_REAL_PINNED_RETRY_AUTHORITY')
        v.update(retry_generation=1,retry_scope_grant_sha256=retry.GRANT_SHA,
            retry_predecessor_context_sha256=retry_marker['context_sha256'])
    else:
        from .amd_training_retry import REQUEST_KEYS
        a.need(not any(k in v for k in REQUEST_KEYS),'TRAIN_ORIGINAL_TEMPLATE_HAS_NO_HIDDEN_RETRY')
    return v

def programs(original):
    v=deepcopy(original);s=v['SUBMIT']
    old='scripts.pi0294.dec1.a40_training_runtime'
    a.need(s.count(old)==1,'TRAIN_ORIGINAL_SEALED_SUBMIT_PROGRAM_SINGLE_ENTRY')
    s=s.replace(old,'scripts.pi0294.generalization.amd_training_runtime')
    a.need('walltime=06:15:00'in s and '#PBS -m n'in s and 'PV1_SUBMISSION_NONCE'in s
        and 'select=1:ngpus=1:ncpus=4:mem=32gb'in s,'TRAIN_ORIGINAL_BOUNDED_NATIVE_SCRIPT')
    # Preserve the original40s outer RPC; qsub itself must fit within it.
    a.need(s.count('timeout=45')==1,'TRAIN_ORIGINAL_QSUB_TIMEOUT_LITERAL')
    v['SUBMIT']=s.replace('timeout=45','timeout=30');compile(v['SUBMIT'],'<original-GEN1-submit-entry>','exec')
    return v

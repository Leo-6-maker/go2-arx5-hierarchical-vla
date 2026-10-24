"""Same global lock/ledger, with request-bound evaluation escrow consumption.

The existing model and query journals remain the effect owners. This adapter
does not load a model, submit a job or supply native/resource qualification.
"""
import os
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import evaluation_accounting as accounting
from . import evaluation_platform as platform
from .neural_ledger import Ledger as GlobalLedger


class EvaluationLedger(GlobalLedger):
    def __init__(self,root,request_path):
        root,request_path=Path(root),Path(request_path)
        m=read(root/'RUN_MANIFEST.json');r=read(request_path)
        if (not platform.cold_profile(m) or r.get('platform_evaluation_version')!=platform.VERSION
                or m.get('owner')!='EXISTING_LOCAL_PC' or m.get('owner_user')!='e1192548'
                or m['execution'].get('uid')!=os.getuid()
                or (root/'RUN_MANIFEST.json').is_symlink() or (root/'RUN_MANIFEST.json').stat().st_mode&0o222
                or request_path.is_symlink() or request_path.name!='REQUEST.json'
                or request_path.stat().st_mode&0o222 or request_path.stat().st_uid!=os.getuid()
                or request_path.parent.parent.resolve()!=(root/'pbs').resolve()
                or type(r.get('submit_count')) is not int
                or request_path.parent.name!='alloc-'+str(r.get('submit_count')).zfill(2)
                or root.is_symlink() or root.stat().st_uid!=os.getuid()
                or m.get('run_uuid')!=root.name or r.get('run_uuid')!=root.name
                or m['code']['vla_commit']!=r.get('source_sha')
                or sha(root/'RUN_MANIFEST.json')!=r.get('run_manifest_sha256')):
            raise PermissionError('PV1_EVALUATION_LEDGER_EXACT_OWN_SEALED_REQUEST')
        super().__init__(root)
        value=read(self.path);row=accounting.item(value,r['request_id']);job_id=os.environ.get('PBS_JOBID')
        if (not job_id or row['job_id']!=job_id or row['status'] not in accounting.STATES[1:3]
                or row['child_run_uuid']!=root.name or row['request_sha256']!=sha(request_path)
                or row['run_manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
                or row['source_sha']!=r['source_sha'] or row['submission_nonce']!=r['submission_nonce']
                or value['PBS_submissions'].index(row)+1!=r['submit_count'] or row['stage']!=r.get('stage')
                or row['evaluation_candidate']!=r.get('evaluation_candidate')
                or row['checkpoint_manifest_sha256']!=r.get('checkpoint_manifest_sha256')
                or row['project']!=r.get('project') or row['statistics_sha256']!=r.get('statistics_sha256')
                or os.environ.get('USER')!='e1192548'):
            raise PermissionError('PV1_EVALUATION_LEDGER_SAME_REGISTERED_BOUND_NATIVE_JOB')
        self.request_id=r['request_id'];self.job_id=job_id;self.sequence=0
        proc_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        self.actor=str(os.getpid())+':'+proc_start

    def _debit(self,kind,count):
        if os.environ.get('PBS_JOBID')!=self.job_id or self.actor.split(':')[0]!=str(os.getpid()):
            raise PermissionError('PV1_EVALUATION_DEBIT_NO_JOB_REBIND_OR_FORKED_WRITER')
        effect=self.actor+':'+str(self.sequence)+':'+kind
        with self.transaction() as value:
            updated=accounting.charge(value,request_id=self.request_id,job_id=self.job_id,
                kind=kind,count=count,effect_id=effect)
            value.clear();value.update(updated)
        self.sequence+=1

    def charge(self,key,count):
        if key not in ('generation_queries','model_loads'):
            raise ValueError('PV1_EVALUATION_ONLY_NATIVE_QUERY_OR_MODEL_CHARGE')
        self.guard()
        return self._debit(key,count)

    def cpu(self,cost):
        # Known closeout CPU is still recorded after STOP or an unrelated
        # unknown effect, exactly as in the original global ledger.
        return self._debit('CPU_s',cost)


def for_request(root,request_path,*,final=False):
    if type(final) is not bool:raise PermissionError('PV1_EXPLICIT_BOOLEAN_FINAL_FLAG')
    m=read(Path(root)/'RUN_MANIFEST.json');r=read(request_path)
    if platform.cold_profile(m) or 'platform_evaluation_version' in r:
        if final:raise PermissionError('PV1_VANDA_DEV_IS_NOT_FINAL_LOCKBOX_PERMISSION')
        return EvaluationLedger(root,request_path)
    return GlobalLedger(root,final=final)


def reserve_once(root,request_path,*,now_epoch=None):
    """Existing global owner lock/once CAS; no native dispatch or retry.

    This is callable only from the registered cold login scope. All proposals
    are recomputed from its actual sealed history and fresh owner files. The
    native project/quota/queue/lease release is still independently required.
    """
    import time
    from . import evaluation_admission as admission
    from .neural_scope import validate
    from .future_admission import _file
    from scripts.pi0294.dec1 import source_cpu_once_io as once
    if now_epoch is not None:
        raise PermissionError('PV1_EVALUATION_RESERVE_NATIVE_CLOCK_ONLY')
    if os.environ.get('PBS_JOBID'):
        raise PermissionError('PV1_EVALUATION_RESERVE_ORIGINAL_LOGIN_OWNER_ONLY')
    root,request_path=Path(root),Path(request_path)
    manifest,*_=validate(root)
    if not platform.cold_profile(manifest):
        raise PermissionError('PV1_EVALUATION_RESERVE_TYPED_COLD_SCOPE_ONLY')
    inputs=admission.file_inputs(root,request_path,now_epoch=time.time())
    proposal=admission.reserve_proposal(inputs['global_bytes'],recompute_inputs=inputs)
    from .evaluation_account_jobs import sealed_confirmation
    request,_=admission._read(inputs['request_bytes'])
    identity=sealed_confirmation(root,manifest,request)
    site=admission.prequeue_site(_file(root,admission.FRESH_SITE),request_bytes=inputs['request_bytes'],
        global_bytes=inputs['global_bytes'],resource_proof=proposal['recomputed_proof'],now_epoch=inputs['now_epoch'],
        independent_job_bytes=identity)
    request,_=admission._read(inputs['request_bytes']);path=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json'
    after=proposal['proposed_global'];after['revision']+=1
    # Same canonical path used by GlobalLedger.transaction(). An existing
    # once intent remains a hard no-retry latch even if old bytes are observed.
    suffix='V4-DEV-'+request['request_id']
    receipt=once.commit(path,expected_before_sha256=admission.hashlib.sha256(inputs['global_bytes']).hexdigest(),
        proposed_bytes=(admission.json.dumps(after,indent=2,allow_nan=False)+'\n').encode(),
        request_id=request['request_id'],intent_path=path.with_name(suffix+'.INTENT.private.json'),
        receipt_path=path.with_name(suffix+'.RECEIPT.private.json'),lock_path=path.with_suffix('.owner.lock'))
    return dict(once_receipt=receipt,recomputed_proof=proposal['recomputed_proof'],
        native_site_inputs=site,qsub_performed=False,native_or_scientific_admission=False)

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

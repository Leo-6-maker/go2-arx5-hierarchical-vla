"""Native scheduler/quota reads join a cold root before any model or CUDA work."""
import os
from pathlib import Path
import socket
import time

from scripts.pi0292.allocation_admission import owned_jobs,personal_quota
from scripts.pi0292.durable import read,write
from scripts.pi0294.fd1 import sha
from .allocation import validate_native
from .neural_scope import validate,inside


def admit(root,request_path):
    if not os.environ.get('PBS_JOBID'):
        raise PermissionError('PV1_NO_MODEL_ON_HOPPER_LOGIN_OR_WITHOUT_NATIVE_ALLOCATION')
    root=Path(root);m,b,inputs,stage,a=validate(root)
    request_path=Path(request_path)
    if request_path.is_symlink() or not request_path.resolve().is_relative_to((root/'pbs').resolve()) or request_path.name!='REQUEST.json':
        raise PermissionError('PV1_OWN_REGISTERED_NATIVE_REQUEST_PATH')
    r=read(request_path)
    if (request_path.stat().st_mode&0o222 or r['run_uuid']!=root.name
            or r['campaign_run_uuid']!=a['run_uuid'] or r['source_sha']!=m['code']['vla_commit']
            or r['run_manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
            or request_path.parent.parent.resolve()!=(root/'pbs').resolve()
            or request_path.parent.name!='alloc-'+str(r['submit_count']).zfill(2)
            or not 1<=r['submit_count']<=24 or r['new_bank_manifest_sha256']!=stage['new_bank_manifest_sha256']):
        raise PermissionError('PV1_FIXED_COLD_REQUEST_CHILD_SOURCE_MANIFEST_AND_BANK')
    for name,digest in r['sealed_inputs'].items():
        if sha(inside(root,name))!=digest:raise ValueError('PV1_NATIVE_REQUEST_INPUT_CHANGED:'+name)
    proof_path=request_path.parent/'RUNNING_PROJECT_ATTESTATION.json'
    if proof_path.is_symlink() or proof_path.stat().st_mode&0o222:
        raise PermissionError('PV1_FRESH_IMMUTABLE_RUNNING_JOB_PROJECT_ATTESTATION')
    queue=inside(root,r['queue_limit_relative'])
    if sha(queue)!=r['queue_limit_sha256'] or r['sealed_inputs'].get(r['queue_limit_relative'])!=sha(queue):
        raise PermissionError('PV1_NATIVE_QUEUE_EFFECTIVE_BOUND_PROOF_SEALED')
    user=os.environ['USER'];job_id=os.environ.get('PBS_JOBID')
    if not job_id:raise PermissionError('PV1_NO_MODEL_ON_HOPPER_LOGIN_OR_WITHOUT_NATIVE_ALLOCATION')
    ledger=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json')
    if ledger['pending_updates']:
        raise PermissionError('PV1_UNKNOWN_OPTIMIZER_EFFECT_REQUIRES_OWNER_RECONCILIATION')
    receipt=validate_native(now=time.time(),user=user,hostname=socket.gethostname(),job_id=job_id,
        jobs=owned_jobs(user),quota=personal_quota(user),adoption=a,request=r,ledger=ledger,
        project_proof=read(proof_path),queue_proof=read(queue),root_run_uuid=root.name)
    receipt.update(request_sha256=sha(request_path),manifest_sha256=sha(root/'RUN_MANIFEST.json'),
        source_sha=m['code']['vla_commit'],project_attestation_sha256=sha(proof_path),
        campaign_run_uuid=a['run_uuid'],run_uuid=root.name)
    out=request_path.parent/'ALLOCATION_ADMISSION.json'
    if out.exists():raise FileExistsError('PV1_PRESERVE_NATIVE_FIRST_ADMISSION_NO_BLIND_RESTART')
    write(out,receipt);out.chmod(0o400)
    return receipt

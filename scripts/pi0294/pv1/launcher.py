"""Original bounded PBS release pattern, with PV1 job-bound owner attestation."""
import argparse
import os
from pathlib import Path
import resource
import socket
import subprocess
import time

from scripts.pi0292.allocation_admission import owned_jobs
from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from .allocation import single_h200_select,native_request_binding
from .evaluation_ledger import for_request as ledger_for_request
from .neural_ledger import Ledger
from .neural_scope import validate
from .resources import pbs_limit


def native_bootstrap(root,request_path):
    root,request_path=Path(root),Path(request_path)
    if not os.environ.get('PBS_JOBID'):
        raise PermissionError('PV1_LAUNCHER_ONLY_IN_OWN_NATIVE_PBS_NO_LOGIN_MODELS')
    m,_,_,_,a=validate(root);r=read(request_path);directory=request_path.parent
    submission_limit=pbs_limit(read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json'))
    job_id=os.environ['PBS_JOBID'];host=socket.gethostname();user=os.environ['USER']
    if (directory.is_symlink() or directory.parent.resolve()!=(root/'pbs').resolve()
            or directory.name!='alloc-'+str(r['submit_count']).zfill(2) or not 1<=r['submit_count']<=submission_limit
            or request_path.is_symlink() or request_path.name!='REQUEST.json' or request_path.stat().st_mode&0o222
            or r['run_uuid']!=root.name or r['campaign_run_uuid']!=a['run_uuid']
            or r['source_sha']!=m['code']['vla_commit'] or r['run_manifest_sha256']!=sha(root/'RUN_MANIFEST.json')):
        raise PermissionError('PV1_BOOTSTRAP_OWN_IMMUTABLE_NATIVE_REQUEST')
    job=owned_jobs(user).get(job_id)
    if (not job or job['job_state']!='R' or job['Job_Owner'].split('@')[0]!=user
            or job['project']!=r['project'] or job['Job_Name']!=r['job_name']
            or int(job['Resource_List']['ngpus'])!=1 or not single_h200_select(job['Resource_List'],r['ncpus'])
            or not native_request_binding(job,r)
            or host.split('.')[0].startswith(('hopper-l','hopper-m'))
            or host.split('.')[0] not in {v.split('/')[0].split('.')[0] for v in job['exec_host'].split('+')}):
        raise PermissionError('PV1_BOOTSTRAP_FRESH_OWN_SELECTED_PROJECT_COMPUTE_JOB')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise PermissionError('PV1_NATIVE_PBS_GPU_VISIBILITY_MISSING_BEFORE_ENDURANCE_OR_MODEL')
    return m,r,directory,job_id,host


def run(root,request_path):
    root=Path(root);began=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN)
    m,r,directory,job_id,host=native_bootstrap(root,request_path)
    ledger=(ledger_for_request(root,request_path,final=r['stage']=='P5_FINAL')
        if 'platform_evaluation_version' in r else
        Ledger(root,final=True) if r['stage']=='P5_FINAL' else Ledger(root))
    ready=directory/'PBS_READY.json';release=directory/'START_NEURAL.json';status='FAILED_PRESERVED';code=None;error=None
    if ready.exists():raise FileExistsError('PV1_PRESERVE_FIRST_NATIVE_BOOTSTRAP_NO_RESTART')
    started=time.time();request_sha=sha(request_path)
    try:
        from .neural import stage_work
        stage_work(root,r)
        write(ready,dict(run_uuid=root.name,job_id=job_id,hostname=host,pid=os.getpid(),
            proc_start=Path('/proc/self/stat').read_text().split()[21],observed_epoch=started,
            request_sha256=request_sha,models_loaded=0,amgr_login_in_compute=False))
        while not release.exists():
            ledger.guard()
            if time.time()-started>=180:raise TimeoutError('PV1_OWNER_FRESH_PROJECT_RELEASE_MISSING_NO_IDLE_MODEL')
            time.sleep(.2)
        v=read(release);proof=directory/'RUNNING_PROJECT_ATTESTATION.json'
        if (release.is_symlink() or release.stat().st_mode&0o222 or v['run_uuid']!=root.name
                or v['job_id']!=job_id or v['request_sha256']!=request_sha or sha(request_path)!=request_sha
                or v['submission_nonce']!=r['submission_nonce'] or v['project_attestation_sha256']!=sha(proof)
                or not started<=v['observed_epoch']<=time.time() or time.time()-started>=180):
            raise PermissionError('PV1_EXACT_ONE_JOB_OWNER_RELEASE_WITH_FRESH_NATIVE_PROJECT_PROOF')
        code=subprocess.run([r['python'],'-m','scripts.pi0294.pv1.neural','--root',str(root),'--request',str(request_path)],check=False).returncode
        status='PV1_NATIVE_NEURAL_PROCESS_TERMINAL';return code
    except BaseException as exc:error=type(exc).__name__+': '+str(exc);raise
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN);self_cpu=time.process_time()-began
        child_cpu=after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime
        p=directory/'PBS_PROCESS_FINAL.json';native=read(p) if p.exists() else {}
        charged=native.get('CPU_s',0) if native.get('job_id')==job_id else 0
        additional=self_cpu+max(0,child_cpu-charged)+60
        receipt=dict(status=status,error=error,job_id=job_id,hostname=host,returncode=code,
            self_CPU_s=self_cpu,all_children_CPU_s=child_cpu,already_charged_neural_CPU_s=charged,
            additional_control_CPU_s=additional,bootstrap_control_upper_s=60,
            GPU_allocation_accounting='OWNER_NATIVE_PBS_TERMINAL_SETTLEMENT_REQUIRED',
            models_loaded_by_launcher=0,amgr_login_in_compute=False,completed_epoch=time.time())
        write(directory/'PBS_WRAPPER_FINAL.json',receipt)
        ledger.cpu(additional)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--request',required=True)
    args=parser.parse_args();raise SystemExit(run(args.root,args.request))

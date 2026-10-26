"""Actual PBS resource hook for the unchanged original model service body."""
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from . import mix_accounting as a, mix_vanda_eval_context as context
from . import mix_vanda_eval_identity as identity,mix_vanda_eval_admission as admission
from scripts.pi0294.dec1.vanda_campaign import collect_queue
from .evaluation_account_jobs import sealed_confirmation


def need(ok,why):
    if not ok:raise PermissionError('V5_VANDA_EVAL_NATIVE_'+why)


def phase(root,request_path):
    directory=Path(request_path).parent;ready=directory/'CORE_FUTURE_PHASE_READY.json'
    if ready.exists():
        v=read(context.member(Path(root),str(ready.relative_to(root))))
        need(v['request_sha256']==sha(request_path) and v['job_id']==os.environ.get('PBS_JOBID'), 'CURRENT_POST_QUALIFICATION_PHASE')
        return v['phase_started_epoch'],'POST_QUALIFICATION'
    p=directory/'NATIVE_CORE_PROCESS_STARTED.json';v=read(p)
    need(v['request_sha256']==sha(request_path) and v['job_id']==os.environ.get('PBS_JOBID'), 'CURRENT_NATIVE_START_PHASE')
    return v['observed_epoch'],'NATIVE_START'


def resource_inputs(root,request_path,process):
    root,request_path=Path(root),Path(request_path);m=read(root/'RUN_MANIFEST.json');r=read(request_path)
    started,label=phase(root,request_path)
    site=context.member(root,str((request_path.parent/('RUNNING_PROJECT_'+label+'.json')).relative_to(root)))
    need(site.is_file() and site.stat().st_uid==os.getuid() and not site.stat().st_mode&0o222,'IMMUTABLE_OWN_ACTUAL_PHASE_PROJECT_RECEIPT')
    queue=collect_queue();now=time.time()
    def raw(name):return context.member(root,name).read_bytes()
    historical=raw('plans/LAST_CONFIRMED_CORE_LOCAL_HISTORICAL.private.json')
    return dict(manifest_bytes=raw('RUN_MANIFEST.json'),request_bytes=request_path.read_bytes(),
        global_bytes=(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json').read_bytes(),historical_LOCAL_bytes=historical,
        training_capture_bytes=raw(context.CAPTURE),population_bytes=raw('plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'),
        plan_bytes=raw('plans/SOURCE_EVALUATION_PLAN.json'),prior_bytes=raw('plans/PRIOR_EVALUATION_OUTCOMES.json'),
        native_facts_bytes=a.raw(process),queue_bytes=a.raw(queue),site_bytes=site.read_bytes(),
        queue_limit_bytes=raw(r['queue_limit_relative']),independent_job_bytes=sealed_confirmation(root,m,r),
        phase_started_epoch=started,now_epoch=now)


def budget_now(root,request_path,ledger):
    root,request_path=Path(root),Path(request_path);r=read(request_path)
    process=context.observe_current(root,r)
    proof=admission.recompute(**resource_inputs(root,request_path,process))
    need(proof['shortfalls']=={} and proof['all_remaining_costs_fit'] is True,'FULL_FRESH_ALL_REMAINING_RESOURCE_JOIN')
    return proof,process


def admit(root,request_path):
    root,request_path=Path(root),Path(request_path);m=context.validate(root)[0]
    need(request_path==context.request_path(root) and request_path.parent.name=='alloc-01','ONE_CURRENT_REQUEST_NO_NAMESPACE_ALIAS')
    r=read(request_path);identity.request_identity(m,r,sha(root/'RUN_MANIFEST.json'))
    out=request_path.parent/'ALLOCATION_ADMISSION.json'
    need(not out.exists() and not out.is_symlink(),'NO_NATIVE_ADMISSION_REPLAY')
    proof,process=budget_now(root,request_path,None);value=proof['allocation']
    context.sealed_identity(m,r,value,run_uuid=root.name,manifest_sha256=sha(root/'RUN_MANIFEST.json'),request_sha256=sha(request_path))
    write(out,value);out.chmod(0o400)
    p=request_path.parent/'INITIAL_NATIVE_RESOURCE_PROOF.json';write(p,proof);p.chmod(0o400)
    return value

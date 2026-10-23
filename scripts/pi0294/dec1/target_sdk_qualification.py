"""Current standard-entry SDK build/constructor qualification, zero physics.

No native import/process on import. This finite component uses the already
registered owner's two actual reservations; it is neither a new scheduler nor
a task runner. The expired R4 run and closed counters are never called.
"""
import argparse
import base64
from dataclasses import asdict
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from . import target_sdk_resources as contract
from . import target_sdk_budget as budget
from . import target_sdk_build_plan as build_plan
from .target_sdk_client import SdkClient, SdkClientPins, CORE_SDK_SHA256
from .target_sdk_constraint_wire import MOBILE_SDK_SHA256, FROZEN_TASK_SHA256
from .target_zero_counter import read, filehash, put

need=budget.need
ENTRY_SHA='0c7973e0a3bc0e585e7ab96cfcc9033532a98f56e9c9410882282a291bb7082f'
DOCTOR_SHA='013e7cea41740b10fdd213e30f6d65b0fbcaee295cce44b624fc61e23fe6d861'
FROZEN_CLOCK_SHA='82004c3171cc8cd2a0486f391595ac39424bad96882d12c5936c4eca55fe3b08'
REQUIRED_CODE=('target_sdk_qualification.py','target_sdk_resources.py','target_sdk_budget.py',
    'target_sdk_build_plan.py','target_sdk_client.py','target_sdk_constraint_wire.py',
    'target_sdk_client.cpp','target_sdk_constraints.hpp','target_zero_counter.py','target_zero_counter_contract.py')


def validate(root):
    """Full source/entry/adoption/budget validation before any compiler/SDK."""
    root=Path(root);now=time.time();m=read(root/'RUN_MANIFEST.json');q=read(root/'REQUEST.json')
    b=read(root/'OUTPUT_BINDING.private.json');inputs=read(root/'INPUT_BINDINGS.private.json');e=m['execution']
    code=root/m['code']['capsule_relative'];parent=root.parent.parent
    need(root.is_absolute() and not root.is_symlink() and root.resolve()==Path(b['run_root']).resolve()
         and root.parent.name=='children' and parent.name==contract.CAMPAIGN
         and m['run_uuid']==q['child_run_uuid']==root.name,'SDK_CURRENT_REGISTERED_CHILD')
    need(not code.is_symlink() and code.resolve().is_relative_to(root.resolve())
         and Path(__file__).resolve()==(code/'scripts/pi0294/dec1/target_sdk_qualification.py').resolve(),
         'SDK_ACTUAL_IMMUTABLE_IMPORTED_SOURCE')
    need(m['schema']=='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1' and m['scope']==q['scope']==contract.ROLE
         and q['schema']==contract.REQUEST_SCHEMA and m['owner']=='EXISTING_LOCAL_PC'
         and q['campaign_run_uuid']==contract.CAMPAIGN and q['authority_release']==contract.common.AUTHORITY
         and m['authority']['protocol_ref']==contract.common.PARENT_AUTHORITY
         and m['authority']['amendments']==[contract.common.AUTHORITY,contract.common.VAP1_AUTHORITY],
         'SDK_CURRENT_ROLE_AND_ADOPTED_SCOPE_NOT_OLD_COUNTER_OR_PILOT')
    need(e['profile']=='4080-cpu-data' and e['host']==os.uname().nodename=='corelab-4080s'
         and e['uid']==os.getuid()==b['uid']==2009
         and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not os.environ.get('LD_PRELOAD')
         and not os.environ.get('LD_AUDIT') and 'PBS_JOBID' not in os.environ
         and os.environ.get('GR00T_WBC_PYTHON')==inputs['runtime_python']==sys.executable
         and os.environ.get('PV1_RUN_MANIFEST')==str(code.parent/'WORKER_RUN_MANIFEST.json')
         and read(code.parent/'WORKER_RUN_MANIFEST.json')==m,'SDK_ACTUAL_STANDARD_CPU_ENTRY')
    for name,pin in (('entry',ENTRY_SHA),('doctor',DOCTOR_SHA)):
        need(e[name+'_sha256']==filehash(e[name+'_path'])==pin,'SDK_EXISTING_ENTRY_AND_DOCTOR')
    need(e['sdk_initialization_mode']==q['sdk_initialization_mode']=='ORIGINAL_SDK_INITIALIZE'
         and q['original_T0_epoch']==m['limits']['PV1_T0_epoch']==contract.T0
         and m['limits']['PV1_hard_close_epoch']==contract.HARD_CLOSE
         and now<q['deadline_epoch']==m['limits']['operation_deadline_epoch']<=contract.STOP_NEW
         and 0<=now-q['observed_epoch']<120,'SDK_FRESH_ADMISSION_AND_ORIGINAL_CLOCK')
    need(all(type(q[k]) is int and q[k]==0 for k in
         ('GPU_submissions','training_updates','GR00T_queries','physics_ticks','policy_episodes'))
         and m['limits']['engine_starts']==2 and m['limits']['physics_ticks']==m['limits']['policy_episodes']==0,
         'SDK_NO_TASK_NEURAL_OR_PHYSICS_AUTHORITY')
    need(filehash(root/'OUTPUT_BINDING.private.json')==e['output_binding_sha256']
         and filehash(root/'INPUT_BINDINGS.private.json')==e['input_binding_sha256'],'SDK_EXACT_INPUT_OUTPUT_BINDINGS')
    for relative in ('DEC1/DEC1_NATIVE_ADOPTION_AND_BUDGET.json','VAP1/VAP1_ADOPTION_AND_DELTA_BUDGET.json'):
        need((root/'plans'/Path(relative).name).read_bytes()==(parent/'authority'/relative).read_bytes(),
             'SDK_ACTUAL_ADOPTED_AUTHORITY_BYTES')
    need(filehash(parent/'authority/VAP1/VAP1_ADOPTION_AND_DELTA_BUDGET.json')==contract.common.VAP1_SHA,
         'SDK_EXACT_NON_GPU_AMENDMENT')
    seal=read(root/m['code']['source_seal_relative']);ci=read(root/'plans/PUBLICATION_AND_CI.private.json')
    need(filehash(root/m['code']['source_seal_relative'])==m['code']['source_seal_sha256']
         and seal['source_sha']==m['code']['vla_commit']==ci['source']
         and ci['CI_conclusion']=='success' and ci['all_12_steps_success'] is True,'SDK_EXACT_QUALIFIED_SOURCE')
    for name in REQUIRED_CODE:
        path='scripts/pi0294/dec1/'+name
        need(seal['files'].get(path)==ci['full_readback_file_sha256'].get(path)
             and path in seal['files'],'SDK_CPP_HPP_AND_ALL_GUARDS_PUBLISHED_AND_SEALED')
    for name,pin in seal['files'].items():
        path=code/name
        need(path.resolve().is_relative_to(code.resolve()) and not path.is_symlink()
             and not path.stat().st_mode&0o222 and filehash(path)==pin,'SDK_EXACT_SOURCE_MEMBER')
    for name,pin in e['sealed_inputs'].items():
        path=root/name
        need(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
             and not path.stat().st_mode&0o222 and filehash(path)==pin,'SDK_EXACT_SEALED_INPUT')
    need({'REQUEST.json',contract.PROJECTION,contract.FORECAST,'plans/SDK_LOCAL_NAMESPACES.private.json'}
         <=set(e['sealed_inputs']),'SDK_ALL_CURRENT_ADMISSION_INPUTS_SEALED')
    need(filehash(inputs['original_task'])==FROZEN_TASK_SHA256
         and filehash(inputs['mobile_archive'])==MOBILE_SDK_SHA256
         and filehash(inputs['core_archive'])==CORE_SDK_SHA256
         and filehash(inputs['frozen_clock'])==FROZEN_CLOCK_SHA,
         'SDK_ORIGINAL_TASK_ARCHIVES_AND_EXPLICIT_FROZEN_CLOCK')
    for name,pin in inputs['original_readonly_files'].items():
        need(Path(name).is_absolute() and filehash(name)==pin,'SDK_PINNED_ORIGINAL_LINK_AND_ABI_INPUTS')
    out=root/'evidence/sdk-build';cache=root/'evidence/sdk-cache';io=root/'evidence/sdk-client'
    for path in (out,cache,io):
        need(path.is_dir() and not path.is_symlink() and not any(path.iterdir()),'SDK_NEW_EMPTY_OWNED_OUTPUT')
    plan=build_plan.plan(Path(inputs['flags_file']).read_bytes(),Path(inputs['link_file']).read_bytes(),
        sdk_root=inputs['sdk_root'],sealed_client_directory=str(code/'scripts/pi0294/dec1'),
        registered_build_directory=str(out))
    need(set(plan['existing_link_inputs'])<=set(inputs['original_readonly_files'])
         and inputs['mobile_archive'] in plan['existing_link_inputs']
         and inputs['core_archive'] in plan['existing_link_inputs'],'SDK_ALL_ACTUAL_LINK_INPUTS_PINNED')
    need(all(inputs['original_readonly_files'].get(path)==pin
             for path,pin in plan['dependent_shared_library_sha256'].items()),
         'SDK_EXACT_DELIVERED_TRANSITIVE_SHARED_LIBRARIES')
    cpus=m['limits']['cpus']
    need(len(cpus) in (1,2) and len(set(cpus))==len(cpus) and set(cpus)<=os.sched_getaffinity(0)
         and 0<m['limits']['memory_bytes']<=8*1024**3,'SDK_ORIGINAL_CPU_MEMORY_PARTITION')
    guard=resource_guard(root,m,q,b)
    guard()
    confirmation=read(root/'LOCAL_RESERVATION_CONFIRMED.private.json')
    parent_bytes=base64.b64decode(confirmation['parent_after_base64'],validate=True)
    core_bytes=(parent/'authority/DEC1/DEC1_RESOURCE_LEDGER.json').read_bytes()
    need(contract.common.sha(parent_bytes)==confirmation['parent_after_sha256'],'SDK_EXACT_ACTUAL_LOCAL_FULL_LEDGER')
    args=contract.unpack_forecast((root/contract.FORECAST).read_bytes());args.pop('now_epoch',None)
    value=budget.forecast(parent_bytes=parent_bytes,Core_bytes=core_bytes,namespace_proof=read(root/'plans/SDK_LOCAL_NAMESPACES.private.json'),
        now_epoch=now,**args)
    need(value['admissible_forecast'] is True,'SDK_FRESH_COMPLETE_REMAINING_RESOURCE_AND_CLOCK_RECOMPUTATION')
    return m,q,inputs,plan,guard


def resource_guard(root,m,q,b):
    parent=root.parent.parent;manifest_pin=filehash(root/'RUN_MANIFEST.json')
    def guard(closeout=False):
        need(filehash(root/'RUN_MANIFEST.json')==manifest_pin,'SDK_OWNER_MANIFEST_UNCHANGED')
        need(closeout or time.time()<q['deadline_epoch'],'SDK_FINITE_ORIGINAL_DEADLINE')
        need(closeout or not any(p.exists() for p in (root/'STOP.json',parent/'STOP.json')),'SDK_CURRENT_OWNER_STOP')
        role=contract.ROLE+'__'+root.name;lease=read(b['lease']);parent_binding=read(parent/'OUTPUT_BINDING.private.json')
        rows=[r for r in read(b['registry'])['entries'] if r.get('role')==role]
        need(b['role']==m['execution']['private_path_role']==role and len(rows)==1
             and rows[0]['run_uuid']==root.name and Path(rows[0]['path']).resolve()==root.resolve()
             and lease['status']=='ACTIVE' and lease['scope']==contract.ROLE and lease['run_uuid']==root.name
             and lease['deadline_epoch']==q['deadline_epoch']
             and lease['authority_release']==contract.common.PARENT_AUTHORITY
             and read(parent_binding['lease'])['status']=='ACTIVE_CONTROL_PLANE_ONLY'
             and read(parent_binding['lease'])['authority_release']==contract.common.PARENT_AUTHORITY,
             'SDK_CURRENT_REGISTERED_LEASES')
        current=read(parent/'authority/DEC1/DEC1_RESOURCE_LEDGER.json');rid=q['request_id']
        row=current[budget.ROWS][rid];confirm=read(root/'LOCAL_RESERVATION_CONFIRMED.private.json')
        need(current['unknown_effects']==[] and row['status']=='RESERVED' and row['schema']==budget.ROW_SCHEMA
             and row['scope']==contract.ROLE and row['manifest_sha256']==manifest_pin
             and row['source_sha']==m['code']['vla_commit'] and row['deadline_epoch']==q['deadline_epoch']
             and row['funded_work_id']==budget.WORK and row['component_id']==budget.COMPONENT
             and row['projection_sha256']==filehash(root/contract.PROJECTION)
             and row['request_sha256']==filehash(root/'REQUEST.json') and row['unknown_effects']==[]
             and row['remaining']==row['requested']==q['requested']
             and confirm['status']=='ACTUAL_LOCAL_PARENT_RESERVATION_CONFIRMED'
             and confirm['request_id']==rid and confirm['manifest_sha256']==manifest_pin
             and confirm['reservation']==row,'SDK_ACTUAL_BOTH_COMPLETE_RESERVATIONS')
    return guard


def phase(root,label,argv,*,guard,deadline,env=None):
    """A durable intent, one owned process group, finite wait, no restart."""
    guard();need(time.time()<deadline,'SDK_NO_BUILD_AFTER_FINITE_DEADLINE')
    put(root/('evidence/SDK_'+label+'_INTENT.json'),dict(argv=argv,once=True,no_retry=True))
    child=None;returned=False
    with (root/('evidence/SDK_'+label+'.log')).open('xb',buffering=0) as log:
        try:
            guard();need(time.time()<deadline,'SDK_FRESH_BUILD_DEADLINE_BEFORE_EXEC')
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                cwd=root/'evidence/sdk-build',start_new_session=True,env=env)
            put(root/('evidence/SDK_'+label+'_STARTED.json'),dict(pid=child.pid,
                proc_start=Path('/proc/%d/stat'%child.pid).read_text().rsplit(')',1)[1].split()[19]))
            while child.poll() is None:
                guard();need(time.time()<deadline,'SDK_FINITE_BUILD_WALL')
                need(log.tell()<16*1024**2,'SDK_BOUNDED_BUILD_LOG')
                time.sleep(.25)
            need(child.returncode==0,'SDK_BUILD_FAILED_NO_RETRY_'+label)
            put(root/('evidence/SDK_'+label+'_RETURNED.json'),dict(pid=child.pid,returncode=child.returncode))
            returned=True
        finally:
            if child is not None and (not returned or child.poll() is None):
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait(timeout=5)


def runtime_link(root,binary,*,guard,deadline):
    """Bounded loader inspection before the SDK constructor intent; no retry."""
    env=dict(os.environ,PATH='/usr/bin:/bin')
    for name in ('LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH'):env.pop(name,None)
    phase(root,'RUNTIME_LINK',['/usr/bin/ldd',binary],guard=guard,
          deadline=min(deadline,time.time()+15),env=env)
    log=(root/'evidence/SDK_RUNTIME_LINK.log').read_text()
    need('not found' not in log and '=>' in log,'SDK_ALL_DYNAMIC_DEPENDENCIES_RESOLVED_BEFORE_CONSTRUCTOR')


def run(root):
    root=Path(root);m,q,inputs,plan,guard=validate(root)
    put(root/'evidence/SDK_QUALIFICATION_INTENT.json',dict(request_id=q['request_id'],once=True,
        pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
    os.sched_setaffinity(0,set(m['limits']['cpus']));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(1200,1200));resource.setrlimit(resource.RLIMIT_AS,(m['limits']['memory_bytes'],)*2)
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    client=None;attempts=0;deadline=min(q['deadline_epoch']-20,time.time()+1200)
    result=dict(status='SDK_COMPONENT_ENGINEERING_FAILURE',target_task_native_qualified=False)
    try:
        phase(root,'COMPILE',plan['compile_argv'],guard=guard,deadline=deadline)
        phase(root,'LINK',plan['link_argv'],guard=guard,deadline=deadline)
        runtime_link(root,plan['binary'],guard=guard,deadline=deadline)
        pin=SdkClientPins(executable=plan['binary'],executable_sha256=filehash(plan['binary']),
            client_task=inputs['original_task'],urdf=inputs['urdf'],urdf_sha256=inputs['urdf_sha256'],
            mobile_archive=inputs['mobile_archive'],core_archive=inputs['core_archive'],
            private_readonly_cache=str(root/'evidence/sdk-cache'),cache_members=(),frames=tuple(inputs['frames']),
            owner_manifest=str(root/'RUN_MANIFEST.json'),owner_manifest_sha256=filehash(root/'RUN_MANIFEST.json'),
            owner_run_uuid=root.name,owner_source_sha=m['code']['vla_commit'],owner_request_id=q['request_id'],
            initialization_mode='ORIGINAL_SDK_INITIALIZE')
        put(root/'evidence/SDK_BUILT_CLIENT_PINS.json',asdict(pin));guard();attempts=1
        client=SdkClient(root/'evidence/sdk-client',pin,guard=guard,maximum_requests=80,startup_timeout_s=120)
        client.close()
        returned=read(root/'evidence/sdk-client/SDK_CLIENT_EXIT.json')
        need(returned['returncode']==0 and returned['failed_or_unknown'] is False,'SDK_ACTUAL_NATURAL_EXIT_AND_CACHE_SEAL')
        result['status']='PASS_ORIGINAL_SDK_BUILD_CONSTRUCTOR_AND_CACHE_ONLY'
    except Exception as error:
        result['error']=type(error).__name__+':'+str(error)
    finally:
        if client is not None and not client.closed:
            try:client.close()
            except Exception as error:result['close_error']=type(error).__name__+':'+str(error)
        self_end=resource.getrusage(resource.RUSAGE_SELF);child_end=resource.getrusage(resource.RUSAGE_CHILDREN)
        # Include validation/import CPU since THIS worker process started and
        # every waited compiler/SDK child, not only time after the first intent.
        cpu=self_end.ru_utime+self_end.ru_stime+child_end.ru_utime+child_end.ru_stime
        result.update(scope=contract.ROLE,request_id=q['request_id'],manifest_sha256=filehash(root/'RUN_MANIFEST.json'),
            source_sha=m['code']['vla_commit'],actual_worker_and_children_CPU_s=cpu,SDK_interface_constructor_attempts=attempts,
            MuJoCo_engine_constructors=0,physics_ticks=0,policy_episodes=0,GR00T_queries=0,
            actual_policy_prefix_samples=0,core64_cells=0,native_prefix_safety_qualified=False,
            physics_advance_authorized=False,constructor_and_CPU_settled=False,lease_closed=False)
        put(root/'evidence/SDK_QUALIFICATION_RESULT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    value=run(args.root);print(contract.common.encode(value).decode(),end='')
    raise SystemExit(0 if value['status'].startswith('PASS_') else 1)

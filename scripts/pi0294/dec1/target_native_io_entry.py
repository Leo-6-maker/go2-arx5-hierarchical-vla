"""Current zero-integration native IO entry, never the expired R4 runner.

All entry, sealed-source, lease and paired full-resource checks precede native
imports or process creation. The numerical controller and its assets stay
frozen. This component cannot release a positive quota or qualify a task.
"""
import argparse
import base64
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

from . import target_native_io_resources as contract
from . import target_native_io_budget as budget
from .target_paused_stop import FROZEN_CLOCK
from .target_sdk_qualification import ENTRY_SHA, DOCTOR_SHA, resource_guard
from .target_zero_counter import read, filehash, put

need=budget.need
REQUIRED_CODE=('target_native_io_entry.py','target_native_io_graph.py',
    'target_native_io_ros_environment.py',
    'target_native_io_predecessor.py',
    'target_native_io_resources.py','target_native_io_budget.py','target_zero_quota.py',
    'target_paused_stop.py','target_zero_observer.cpp','target_time_transaction.py',
    'target_mpc_prediction.py','target_sdk_qualification.py')
NAMESPACE='plans/NATIVE_IO_LOCAL_NAMESPACES.private.json'
SIM_SHA='809a46593feb7f6117ef482aa68ea74f0a6ffda02b4e6c3f807892254796644c'
NATIVE_PINS={
    'sim_binary':SIM_SHA,
    'controller_binary':'885042aee96eec2761e5cc9c1598194c94181bebe0de0548eb48b1110ddaa186',
    'solver_binary':'4a5f5aa6e0b9d4bde62f4d1da73135ef97e5268945a0cecf114d9190d3e04fd0',
    'original_task_observer':'3ed0d89dca49d242e9ebe44671bcad636a30b700d7f4f43b5931e70e7b172881',
    'compiled_model':'168b1232df2a50a163709c1c1e57d13d39aa3efb92a4c98005e797924142376d',
    'original_task':'0b69bd58db364a01e87331b0e4e777a9684fda18a4f67fa60aee55174f72dcca',
    'urdf':'403ce252c525437d66aa361a8a65b90bfe2b69dad0fcb1f7a17109582c7261bc'}
NATIVE_RELATIVE_PINS={
    'src/ctrl/rl_sar/policy/go2_x5/N1_s29_175632/policy.pt':'b07c2d494a47cdae2a97fd9b2471d8f6f243049741e949414a5d099fa5148e3a',
    'src/ctrl/rl_sar/policy/go2_x5/N1_s29_175632/config.yaml':'d39fbb039751f24f7fbae04e4402f803595fc009456e18a10bb7bce992b19855'}
CACHE_ONLY_TASK='d0c485bc0d4a4d420c2fcd621103358718f99980e4cde3b6c464e189b2896ad3'


def validate(root):
    root=Path(root);now=time.time()
    m=read(root/'RUN_MANIFEST.json');q=read(root/'REQUEST.json');e=m['execution']
    b=read(root/'OUTPUT_BINDING.private.json');inputs=read(root/'INPUT_BINDINGS.private.json')
    parent=root.parent.parent;code=root/m['code']['capsule_relative']
    need(root.is_absolute() and not root.is_symlink() and root.resolve()==Path(b['run_root']).resolve()
         and root.parent.name=='children' and parent.name==budget.CAMPAIGN
         and root.name==m['run_uuid']==q['child_run_uuid'],'IO_CURRENT_REGISTERED_CHILD')
    need(not code.is_symlink() and code.resolve().is_relative_to(root.resolve())
         and Path(__file__).resolve()==(code/'scripts/pi0294/dec1/target_native_io_entry.py').resolve(),
         'IO_ACTUAL_IMMUTABLE_IMPORTED_SOURCE')
    need(m['schema']=='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1' and m['scope']==q['scope']==budget.ROLE
         and m['owner']=='EXISTING_LOCAL_PC' and q['schema']==budget.REQUEST_SCHEMA
         and q['campaign_run_uuid']==budget.CAMPAIGN and q['authority_release']==budget.common.AUTHORITY
         and m['authority']['protocol_ref']==budget.common.PARENT_AUTHORITY
         and m['authority']['amendments']==[budget.common.AUTHORITY,budget.common.VAP1_AUTHORITY],
         'IO_CURRENT_AUTHORITY_NOT_OLD_PILOT_OR_COUNTER')
    need(e['profile']=='4080-simulation' and e['host']==os.uname().nodename=='corelab-4080s'
         and e['uid']==os.getuid()==b['uid']==2009 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
         and not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT') and 'PBS_JOBID' not in os.environ
         and os.environ.get('GR00T_WBC_PYTHON')==inputs['runtime_python']==sys.executable
         and os.environ.get('PV1_RUN_MANIFEST')==str(code.parent/'WORKER_RUN_MANIFEST.json')
         and read(code.parent/'WORKER_RUN_MANIFEST.json')==m,'IO_ACTUAL_STANDARD_SIMULATION_ENTRY')
    need(q['original_T0_epoch']==m['limits']['PV1_T0_epoch']==budget.T0
         and m['limits']['PV1_hard_close_epoch']==budget.HARD_CLOSE
         and now<q['deadline_epoch']==m['limits']['operation_deadline_epoch']<=budget.STOP_NEW
         and 0<=now-q['observed_epoch']<120,'IO_FRESH_ORIGINAL_CLOCK')
    need(budget.common.vector(q['requested'])==budget.REQUESTED
         and q['purpose']=='CURRENT_NON_OBJECT_BRIDGE_ZERO_INTEGRATION'
         and q['funded_work_id']==budget.WORK and q['component_id']==budget.COMPONENT
         and q['LOCAL_namespace']==budget.LOCAL_NAMESPACE
         and e['sdk_initialization_mode']==q['sdk_initialization_mode']=='SEALED_CACHE_LOAD'
         and type(q['maximum_SDK_requests']) is int and q['maximum_SDK_requests']==1
         and e['frozen_task_clock_sha256']==FROZEN_CLOCK
         and m['limits']['engine_starts']==8
         and all(type(m['limits'][k]) is int and m['limits'][k]==0 for k in ('physics_ticks','policy_episodes'))
         and all(type(q[k]) is int and q[k]==0 for k in
                 ('GPU_submissions','training_updates','GR00T_queries','policy_episodes','physics_ticks')),
         'IO_ZERO_INTEGRATION_ONLY_NO_TASK_OR_CACHE_REBUILD_AUTHORITY')
    for name,pin in (('entry',ENTRY_SHA),('doctor',DOCTOR_SHA)):
        need(e[name+'_sha256']==filehash(e[name+'_path'])==pin,'IO_ORIGINAL_STANDARD_ENTRY')
    need(filehash(root/'OUTPUT_BINDING.private.json')==e['output_binding_sha256']
         and filehash(root/'INPUT_BINDINGS.private.json')==e['input_binding_sha256'],'IO_SEALED_CURRENT_BINDINGS')
    for folder,name in (('DEC1','DEC1_NATIVE_ADOPTION_AND_BUDGET.json'),('VAP1','VAP1_ADOPTION_AND_DELTA_BUDGET.json')):
        need((root/'plans'/name).read_bytes()==(parent/'authority'/folder/name).read_bytes(),'IO_ACTUAL_ADOPTED_AUTHORITY_BYTES')
    need(filehash(parent/'authority/VAP1/VAP1_ADOPTION_AND_DELTA_BUDGET.json')==budget.common.VAP1_SHA,'IO_NON_GPU_AMENDMENT')
    seal=read(root/m['code']['source_seal_relative']);ci=read(root/'plans/PUBLICATION_AND_CI.private.json')
    need(filehash(root/m['code']['source_seal_relative'])==m['code']['source_seal_sha256']
         and seal['source_sha']==m['code']['vla_commit']==ci['source']
         and ci['CI_conclusion']=='success' and ci['all_12_steps_success'] is True,'IO_EXACT_QUALIFIED_SOURCE')
    for name in REQUIRED_CODE:
        name='scripts/pi0294/dec1/'+name
        need(name in seal['files'] and seal['files'][name]==ci['full_readback_file_sha256'].get(name),'IO_ALL_RUNTIME_GUARDS_QUALIFIED')
    for name,pin in budget.common.FROZEN_SOURCES.items():
        need(seal['files'].get(name)==pin,'IO_ORIGINAL_FROZEN_GEOMETRY_SOURCE')
    for name,pin in seal['files'].items():
        p=code/name
        need(p.resolve().is_relative_to(code.resolve()) and not p.is_symlink()
             and not p.stat().st_mode&0o222 and filehash(p)==pin,'IO_SEALED_SOURCE_MEMBER')
    for name,pin in e['sealed_inputs'].items():
        p=root/name
        need(p.resolve().is_relative_to(root.resolve()) and not p.is_symlink()
             and not p.stat().st_mode&0o222 and filehash(p)==pin,'IO_SEALED_ADMISSION_MEMBER')
    need({'REQUEST.json',contract.PROJECTION,contract.FORECAST,NAMESPACE,
          'INPUT_BINDINGS.private.json','plans/ZERO_IO_GEOMETRY.json'}<=set(e['sealed_inputs']),
         'IO_COMPLETE_EIGHT_DIMENSION_FORECAST_AND_GEOMETRY_SEALED')
    need(filehash(inputs['frozen_clock'])==FROZEN_CLOCK
         and inputs['original_readonly_files'][inputs['sim_binary']]==SIM_SHA
         and filehash(inputs['sim_binary'])==SIM_SHA,'IO_EXACT_FROZEN_CLOCK_AND_PAUSED_BINARY')
    required_assets=('compiler','xvfb_run','sim_binary','solver_binary','controller_binary',
        'original_task_observer','frozen_clock','original_task','urdf','compiled_model')
    need(all(inputs[k] in inputs['original_readonly_files'] for k in required_assets)
         and type(inputs['ROS_DOMAIN_ID']) is int and 1<=inputs['ROS_DOMAIN_ID']<=232,
         'IO_ALL_EXECUTED_BINARIES_AND_INPUTS_PINNED')
    need(all(inputs['original_readonly_files'].get(inputs[k])==h for k,h in NATIVE_PINS.items())
         and all(inputs['original_readonly_files'].get(str(Path(inputs['native_root'])/n))==h
             for n,h in NATIVE_RELATIVE_PINS.items()),'IO_ORIGINAL_FROZEN_CONTROLLER_MODEL_POLICY_AND_SOLVER_IDENTITIES')
    population=(root/'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json').read_bytes()
    geometry=budget.common.plan(population)
    need(read(root/'plans/ZERO_IO_GEOMETRY.json')==geometry
         and 'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json' in e['sealed_inputs'],
         'IO_ORIGINAL_PREDECLARED_NON_OBJECT_TRAIN_GEOMETRY')
    for name,pin in inputs['original_readonly_files'].items():
        need(Path(name).is_absolute() and not Path(name).is_symlink() and filehash(name)==pin,'IO_ORIGINAL_NATIVE_ASSET_OR_TOOLCHAIN_CHANGED')
    from .target_native_io_ros_environment import validate_binding
    validate_binding(inputs)
    # The entire current MuJoCo include closure, not just its first header, is
    # sealed before compiling OUR observer. No vendor/controller is compiled.
    include=Path(inputs['mujoco_include']);headers=inputs['mujoco_headers']
    actual={str(p.relative_to(include)) for p in include.rglob('*.h')}
    need(actual==set(headers) and headers and all(inputs['original_readonly_files'].get(str(include/n))==h
         for n,h in headers.items()),'IO_COMPLETE_ACTUAL_MUJOCO_HEADER_CLOSURE')
    need('#define mjVERSION_HEADER 327' in (include/'mujoco/mujoco.h').read_text(),'IO_MUJOCO_327_HEADER')
    cache=Path(inputs['codegen_directory']);members=inputs['codegen_members']
    need(cache.is_dir() and not cache.is_symlink() and cache.resolve().is_relative_to(root.resolve())
         and members and {str(p.relative_to(cache)) for p in cache.rglob('*') if p.is_file()}==set(members),
         'IO_OWNED_SEALED_ORIGINAL_MATH_CACHE_NO_NEW_COMPILATION')
    for name,pin in members.items():
        p=cache/name;relative=str(p.relative_to(root))
        need(not p.is_symlink() and p.resolve().is_relative_to(cache.resolve())
             and not p.stat().st_mode&0o222 and e['sealed_inputs'].get(relative)==filehash(p)==pin,
             'IO_ALL_ORIGINAL_SOLVER_MATH_CACHE_MEMBERS_SEALED')
    task=Path(inputs['runtime_cache_only_task'])
    need(task.resolve().is_relative_to(root.resolve()) and not task.is_symlink()
         and not task.stat().st_mode&0o222
         and e['sealed_inputs'].get(str(task.relative_to(root)))==filehash(task)==CACHE_ONLY_TASK,
         'IO_EXACT_CACHE_ONLY_TASK_DERIVATION_NOT_CONTROLLER_MATH_CHANGE')
    cpus=m['limits']['cpus']
    need(len(cpus) in (1,2) and len(set(cpus))==len(cpus) and set(cpus)<=os.sched_getaffinity(0)
         and type(m['limits']['memory_bytes']) is int and 0<m['limits']['memory_bytes']<=8*1024**3,
         'IO_ACTUAL_CPU_MEMORY_PARTITION')
    for name in ('graph','build'):
        p=root/'evidence'/name
        need(p.is_dir() and not p.is_symlink() and not any(p.iterdir()),'IO_NEW_EMPTY_OWNED_OUTPUT')
    current_contract=SimpleNamespace(ROLE=budget.ROLE,common=budget.common,PROJECTION=budget.PROJECTION)
    guard=resource_guard(root,m,q,b,contract=current_contract,budget=budget);guard()
    confirm=read(root/'LOCAL_RESERVATION_CONFIRMED.private.json')
    pb=base64.b64decode(confirm['parent_after_base64'],validate=True)
    cb=(parent/'authority/DEC1/DEC1_RESOURCE_LEDGER.json').read_bytes()
    paired,_,_=budget.rows(budget.common.obj(pb),budget.common.obj(cb))
    row=paired[q['request_id']]
    if 'predecessor_certificate' in row:
        packet=(root/budget.predecessor.PACK).read_bytes()
        need(e['sealed_inputs'].get(budget.predecessor.PACK)==filehash(root/budget.predecessor.PACK)
             ==q['closed_predecessor_packet_sha256']==row['predecessor_certificate']['packet_sha256']
             and base64.b64decode(row['predecessor_packet_base64'],validate=True)==packet,
             'IO_COLD_REPAIR_SEALED_PREDECESSOR_AND_ACTUAL_RESERVATIONS_IDENTICAL')
        original=parent/'children'/budget.predecessor.RUN
        original_binding=read(original/'OUTPUT_BINDING.private.json')
        need(inputs['known_zero_predecessor_native_lease']==original_binding['lease']
             and inputs['original_readonly_files'].get(original_binding['lease'])
                 ==filehash(original_binding['lease'])==budget.predecessor.ANCHORS['lease_bytes']
             and filehash(original/'evidence/NATIVE_IO_RESULT.json')==budget.predecessor.ANCHORS['result_bytes']
             and read(original_binding['lease'])['status']=='CLOSED',
             'IO_FRESH_ORIGINAL_NATIVE_CLOSED_LEASE_AND_RESULT_NOT_ONLY_PAST_PACKET')
    need(budget.common.sha(pb)==confirm['parent_after_sha256'],'IO_FULL_ACTUAL_LOCAL_RESERVED_LEDGER')
    args,db,closures=budget.unpack_forecast((root/budget.FORECAST).read_bytes());args.pop('now_epoch',None)
    forecast=budget.forecast(parent_bytes=pb,Core_bytes=cb,DEC1_bytes=db,SDK_closures=closures,
        namespace_proof=read(root/NAMESPACE),now_epoch=now,**args)
    need(forecast['admissible_forecast'] is True,'IO_FRESH_ALL_REMAINING_WORK_RESOURCES_AND_CLOCK')
    return m,q,inputs,guard


def run(root):
    root=Path(root);m,q,inputs,guard=validate(root)
    from .target_native_io_ros_environment import ensure_startup
    ensure_startup(root,root/m['code']['capsule_relative'],inputs,guard)
    # Native graph imports occur only after ALL admission above has passed.
    from .target_native_io_graph import run_graph
    put(root/'evidence/NATIVE_IO_OWNER_INTENT.json',dict(request_id=q['request_id'],
        manifest_sha256=filehash(root/'RUN_MANIFEST.json'),pid=os.getpid(),
        proc_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],once=True,no_retry=True))
    os.sched_setaffinity(0,set(m['limits']['cpus']));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(3000,3000))
    resource.setrlimit(resource.RLIMIT_AS,(m['limits']['memory_bytes'],m['limits']['memory_bytes']))
    return run_graph(root,m,q,inputs,guard)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    value=run(parser.parse_args().root)
    raise SystemExit(0 if value['status']=='ACTUAL_ZERO_IO_FEEDBACK_AND_STOP' else 1)

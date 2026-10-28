"""New successor CPU roles through the existing Core entry and counters.

Cold validation only; GPU process admission remains a separate native gate.
Historical role/lease clocks are never rewritten or passed an invented date.
"""
import hashlib,json,os,time
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import successor_cd as s

GEN_ROLE='V5_GEN1_STATIC_PREVIEW_CPU_V1'
EVAL_ROLE='V5_CD_CORE_POLICY_EVALUATION_V1'
ROLES=(GEN_ROLE,EVAL_ROLE)
SCHEMA='V5_CD_REGISTERED_CORE_CPU_CHILD_V1'
LEDGER_SCHEMA='V5_CD_CORE_POLICY_EFFECT_LEDGER_V1'
GEN_ALLOCATION='GEN1_STATIC24'
_BINDING_CACHE={}
ASSET_IDENTITY_SHA='27f19c63019704cb2e437c1f7b7b9df298e2059f5a6ad990036ebfe935ef1ce6'


def cpu_allocation(manifest,local,marker,*,now_epoch,closeout=False):
    """The existing whole-package escrow owns this finite CPU substage.

    A caller's manifest limits do not create another allowance. The actual
    canonical ledger must contain its unique once-bound run and manifest.
    """
    role=manifest.get('scope');need(role in ROLES,'ONLY_EXPLICIT_SUCCESSOR_CPU_ROLE')
    if role==EVAL_ROLE:need(manifest.get('candidate')in ('C50','D25'),'ONLY_FIXED_CD_CPU_EVAL_PEER')
    key=GEN_ALLOCATION if role==GEN_ROLE else 'EVAL:'+manifest.get('candidate','')+':N294101'
    row=local.get('V5_CD_Core_CPU_substages',{}).get(key)
    need(type(row)is dict and row.get('status')=='RESERVED_NOT_SPENT'
         and row.get('run_uuid')==manifest['run_uuid'] and row.get('role')==role
         and row.get('source_sha')==manifest['code']['vla_commit']
         and row.get('scope_grant_sha256')==s.GRANT_SHA,'ACTUAL_ONCE_CPU_SUBSTAGE_NOT_CALLER_BUDGET')
    limits=manifest['limits'];expected=dict(CPU_s=1200,physics_ticks=0,policy_episodes=0,engine_starts=30) if role==GEN_ROLE else dict(CPU_s=93900,physics_ticks=832000,policy_episodes=26,engine_starts=2108)
    from .mix_core_partial_resume import KEY,physical_limits
    if role==EVAL_ROLE and KEY in manifest:expected=physical_limits(manifest)
    need(all(type(limits.get(k))in(int,float) and type(limits[k])is not bool
             and limits[k]==v and row.get(k)==v for k,v in expected.items()),'EXACT_EXISTING_PACKAGE_SUBSTAGE_LIMITS')
    need(type(limits.get('cpus'))is list and 0<len(limits['cpus'])<=3
         and all(type(i)is int and i>=0 for i in limits['cpus'])
         and len(set(limits['cpus']))==len(limits['cpus']),'FINITE_UNIQUE_CPU_AFFINITY')
    need(type(closeout)is bool and marker['T_CD']<=now_epoch
         and limits['operation_deadline_epoch']<=marker['deadline_epoch']
         and (closeout or now_epoch<limits['operation_deadline_epoch']), 'CPU_STAGE_INSIDE_ONE_ACTUAL_WINDOW')
    return row


def need(ok,msg):
    if not ok:raise PermissionError('V5_CORE_'+msg)


def ledger_binding(root,*,closeout=False):
    root=Path(root);m=read(root/'RUN_MANIFEST.json');name='plans/V5_CD_CORE_LEDGER_BINDING.private.json'
    need(m['execution']['sealed_inputs'].get(name)==sha(root/name),'ACTUAL_CANONICAL_LOCAL_BINDING')
    b=read(root/name);path=Path(b['canonical_LOCAL_ledger']).resolve()
    need(path.is_relative_to((root.parent.parent/'control').resolve()) and path.name=='LOCAL_RESOURCE_LEDGER.private.json','EXISTING_CORE_LOCAL_LEDGER_NOT_ANOTHER_BUDGET')
    local_bytes=path.read_bytes();cache_key=str(root);bound_bytes=(root/name).read_bytes();manifest_bytes=(root/'RUN_MANIFEST.json').read_bytes()
    cached=_BINDING_CACHE.get(cache_key)
    if cached is not None and cached[:3]==(local_bytes,bound_bytes,manifest_bytes)and cached[3]==closeout:
        marker=cached[4];now=time.time()
        need(marker['T_CD']<=now and (closeout or now<min(marker['deadline_epoch'],m['limits']['operation_deadline_epoch'])),
             'CPU_STAGE_INSIDE_ONE_ACTUAL_WINDOW')
        return dict(marker),path
    value=json.loads(local_bytes);marker=s.validated(value)
    need(marker==read(root/'V5_CD_SCOPE_BINDING.json') and value['unknown_effects']==value['pending_updates']==value['pending_queries']==[],'SAME_ACTUAL_ADOPTED_SCOPE_AND_UNKNOWN_FENCE')
    need('V5_CD_execution_package'in value and value['V5_CD_execution_package']['scope_grant_sha256']==s.GRANT_SHA,'EXISTING_WHOLE_PAIR_ESCROW')
    row=cpu_allocation(m,value,marker,now_epoch=time.time(),closeout=closeout)
    need(row.get('run_manifest_sha256')==sha(root/'RUN_MANIFEST.json'),'ACTUAL_CPU_ROW_BINDS_EXACT_MANIFEST')
    from .mix_core_partial_resume import validate_inputs
    validate_inputs(root,m,value)
    if len(_BINDING_CACHE)>=4 and cache_key not in _BINDING_CACHE:_BINDING_CACHE.clear()
    _BINDING_CACHE[cache_key]=(local_bytes,bound_bytes,manifest_bytes,closeout,dict(marker))
    return marker,path


def deadline(root,manifest,*,closeout=False):
    marker,_=ledger_binding(root,closeout=closeout);now=time.time()
    need(marker['T_CD']<=now and (closeout or now<min(marker['deadline_epoch'],manifest['limits']['operation_deadline_epoch'])),'ACTUAL_SUCCESSOR_WINDOW_NO_HISTORICAL_NOW')
    return marker['deadline_epoch']


def validate_cpu_affinity(limits,actual):
    """The existing RPC child uses its separately registered control CPU.

    Parent/physical validation retains the previous full-grant check. Only
    the exact registered RPC singleton is accepted as a control alternative;
    unrelated partial grants remain rejected.
    """
    declared=limits.get('cpus');rpc=limits.get('rpc_cpu')
    need(type(declared)is list and 1<=len(declared)<=3
         and all(type(v)is int and v>=0 for v in declared)
         and len(set(declared))==len(declared),
         'FINITE_DECLARED_PHYSICAL_AND_RPC_CPUS')
    live=set(actual);allowed=set(declared)
    narrow=type(rpc)is int and rpc>=0 and live=={rpc}
    need(bool(live)and(allowed<=live or narrow),'FINITE_OWN_CPU_AFFINITY')
    return dict(declared_cpus=sorted(allowed),actual_cpus=sorted(live),
                registered_RPC_singleton=narrow,new_model_or_physics_permission=False)


def validate(root,profile,*,closeout=False):
    need(type(closeout)is bool,'EXPLICIT_CLOSEOUT')
    r=Path(root);need(r.is_absolute() and r.is_dir() and not r.is_symlink() and r.stat().st_uid==os.getuid()==2009 and os.uname().nodename=='corelab-4080s','REGISTERED_CORE_HOST_AND_UID')
    for name in ('RUN_MANIFEST.json','OUTPUT_BINDING.private.json'):
        p=r/name;need(p.is_file() and not p.is_symlink() and p.stat().st_uid==2009
                     and not p.stat().st_mode&0o222,'IMMUTABLE_OWN_ROOT_IDENTITY:'+name)
    m=read(r/'RUN_MANIFEST.json');b=read(r/'OUTPUT_BINDING.private.json');role=m.get('scope')
    need(role in ROLES and m.get('schema')==SCHEMA and profile=='4080-simulation' and m['execution']['profile']==profile,'DISTINCT_CPU_SCOPE_AND_EXISTING_PROFILE')
    need(m['authority']['protocol_ref']==s.TASK_SOURCE and m['authority']['scope_grant_sha256']==s.GRANT_SHA and m['run_uuid']==r.name and m['output_root']==str(r),'EXACT_ACCEPTED_SUCCESSOR_AUTHORITY')
    need(b['run_root']==str(r) and b['uid']==2009 and b['run_manifest_sha256']==sha(r/'RUN_MANIFEST.json') and r.parent.name=='children' and r.parent==Path(b['registered_parent']),'EXISTING_REGISTERED_CPU_OUTPUT_NAMESPACE')
    need(b['role']==role+'__'+r.name,'UNIQUE_NEW_ROLE_NOT_OLD_PV1_POPULATION')
    from .evaluation_core_scope import PARENT_BINDING_SHA
    need(sha(r.parent.parent/'OUTPUT_BINDING.private.json')==PARENT_BINDING_SHA,'EXACT_EXISTING_CORE_PARENT')
    rows=[x for x in read(b['registry'])['entries']if x.get('role')==b['role']]
    need(len(rows)==1 and rows[0]['path']==str(r) and rows[0]['run_uuid']==r.name,'ACTUAL_UNIQUE_REGISTRY_ENTRY')
    lease=read(b['lease']);end=deadline(r,m,closeout=closeout)
    need(lease['status']=='ACTIVE' and lease['run_uuid']==r.name and lease['authority_release']==s.TASK_SOURCE and lease['deadline_epoch']==end,'FRESH_OWN_SUCCESSOR_LEASE')
    code=r/m['code']['capsule_relative'];need(not code.is_symlink() and code.resolve().is_relative_to(r.resolve()) and Path(__file__).resolve()==(code/'scripts/pi0294/pv1/successor_core_context.py').resolve(),'ACTUAL_COLD_IMPORTED_SOURCE')
    need(os.environ.get('PV1_RUN_MANIFEST')==str(code.parent/'WORKER_RUN_MANIFEST.json') and read(code.parent/'WORKER_RUN_MANIFEST.json')==m,'EXISTING_STANDARD_WORKER_MANIFEST_ENTRY')
    seal=r/m['code']['source_seal_relative'];need(sha(seal)==m['code']['source_seal_sha256'],'EXACT_SOURCE_SEAL');pins=read(seal)
    need(pins['source_sha']==m['code']['vla_commit'],'ONE_ENGINEERING_SOURCE')
    for name,h in pins['files'].items():
        p=code/name;need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(code.resolve()) and sha(p)==h,'SEALED_SOURCE_MEMBER:'+name)
    for name,h in m['execution']['sealed_inputs'].items():
        p=r/name;need(p.is_file() and not p.is_symlink() and p.stat().st_uid==2009 and not p.stat().st_mode&0o222 and sha(p)==h,'SEALED_INPUT:'+name)
    from .evaluation_core_scope import ORIGINAL_NUMERICAL_TABLE_SHA
    frozen=m['code']['frozen_numerical_files'];need(len(frozen)==15 and hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()==ORIGINAL_NUMERICAL_TABLE_SHA,'EXACT_ORIGINAL_NUMERICAL_TABLE_NOT_CALLER_CHOSEN_HASHES')
    need(all(sha(code/n)==h for n,h in frozen.items()),'NO_PHYSICS_S1_SCORE_OR_CONTROLLER_REWRITE')
    need(m['execution']['sealed_inputs'].get('plans/ASSET_IDENTITY.private.json')==ASSET_IDENTITY_SHA
         and sha(r/'plans/ASSET_IDENTITY.private.json')==ASSET_IDENTITY_SHA,'EXACT_ORIGINAL_ASSET_MAP_NOT_CALLER_PINS')
    assets=read(r/'plans/ASSET_IDENTITY.private.json');need(len(assets['absolute_files'])==26 and all(sha(p)==h for p,h in assets['absolute_files'].items()),'ACTUAL_FROZEN_NATIVE_ASSETS')
    infra=Path(b['registry']).parent
    need(sha(infra/'enter.sh')==m['execution']['entry_sha256'] and sha(infra/'doctor.py')==m['execution']['doctor_sha256'],'REUSE_DEPLOYED_ENTRY_AND_DOCTOR')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'PBS_JOBID'not in os.environ,'NO_CPU_CHILD_NEURAL_OR_PBS_ALIAS')
    validate_cpu_affinity(m['limits'],os.sched_getaffinity(0))
    if not closeout:need(not any(p.exists()for p in (r/'STOP.json',r/'SOURCE_PHYSICS_STOP.json',r.parent.parent/'STOP.json')),'ORIGINAL_SAFETY_AND_PARENT_STOP')
    return m

"""One confirmation-only cold repair, retaining the original STOP and outcomes."""
from pathlib import Path
import re
import stat

from scripts.pi0292.durable import read
from scripts.pi0294.e0_scope import pbs_epoch
from scripts.pi0294.native_dataset import sha
from . import n1_limits as n1

SCHEMA='MP1_CONFIRMATION02_SAME_ROOT_COLD_REPAIR_V1'
KEY='confirmation_engineering_binding_relative'
BASE='plans/CONFIRM_ENGINEERING_ALLOCATION02/'
BINDING=BASE+'BOUNDARY.json'
PRIOR=BASE+'PRIOR_RUN_MANIFEST.json'
LEDGER=BASE+'PRIOR_HOPPER_LEDGER.json'
NATIVE=BASE+'NATIVE_TERMINAL.json'
PROCESSES=BASE+'PROCESS_CLOSURE.json'
DIAGNOSTIC=BASE+'TRANSPORT_REPAIR_RECEIPT.json'
CLOSURE='plans/CONFIRM_PRIOR_CLOSURE_ALLOCATION02.json'
OLD_STOP='RELAY_ENGINEERING_STOP.json'
NEW_STOP='RELAY_ENGINEERING_STOP_ALLOCATION02.json'
STOP_SHA='ab07d1c59567c0be0b705e5504e2063800a19f6e8e4f4016a8a07a1bc4093248'
OLD_SOURCE='6b291000858612423b67838275efa52caf8b633f'
PLAN_SHA='6e5a45e121aa493e1de32977c00d3f20ecc3c1f9cb7a6b9698438cbb16028e97'
CLOSURE_SHA='4ab4ec99fadf1ca4a596a9efe40f9b6afbb168d64a1f0a6f8b998b81c37cd67f'
NAMES=[BINDING,PRIOR,LEDGER,NATIVE,PROCESSES,DIAGNOSTIC,CLOSURE,OLD_STOP]
V2_BINDING=BASE+'V2/BOUNDARY.json'
V2_PROTOTYPE=BASE+'V2/CPU_PROTOTYPE_RUN_MANIFEST.json'
PROTOTYPE_SOURCE='b379ce202bd36cae22e873780d1b21753a7e4803'
PROTOTYPE_MANIFESTS={
    '4080-simulation':'9d5242bc65531c1398d4fc0fd249b67dde62ef64e138d2f81280ac9dbd850fe7',
    'hopper-neural':'0aa44c451689c9fc2439f5c912bbcadfa0c5efbd061d854003d1edfa7e8a7da1'}


def required_names(relative):
    if relative==BINDING:return list(NAMES)
    if relative==V2_BINDING:return [n.replace(BASE,BASE+'V2/',1) for n in NAMES[:6]]+NAMES[6:]+[V2_PROTOTYPE]
    raise PermissionError('ONLY_VERSIONED_ORIGINAL_ALLOCATION02_COLD_INPUTS')


def reader_authority(root,manifest):
    """A read-side clone keeps root authority; only its own binding hash is extra."""
    extra='session_visibility_binding_sha256'
    if extra not in manifest['execution']:return manifest
    actual=read(root/'RUN_MANIFEST.json')
    if (extra in actual['execution'] or actual['execution'].get(KEY) not in [BINDING,V2_BINDING]
            or manifest['execution']['profile']!='hopper-neural'
            or {k:v for k,v in manifest.items() if k not in ['code','execution']}
                !={k:v for k,v in actual.items() if k not in ['code','execution']}
            or {k:v for k,v in manifest['execution'].items() if k!=extra}!=actual['execution']
            or {k:v for k,v in manifest['code'].items() if k not in ['capsule_relative','source_seal_relative']}
                !={k:v for k,v in actual['code'].items() if k not in ['capsule_relative','source_seal_relative']}):
        raise PermissionError('ONLY_IDENTICAL_SOURCE_READONLY_VISIBILITY_WORKER_MANIFEST')
    code=root/manifest['code']['capsule_relative'];base=root/actual['code']['capsule_relative'];label=code.parent.name
    binding=root/'plans'/label/'METADATA_READER_BINDING.json'
    if (not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',label) or code!=base.parent/'visibility'/label/'code'
            or root/manifest['code']['source_seal_relative']!=code.parent/'SOURCE_SEAL.json'
            or code.is_symlink() or binding.is_symlink() or binding.stat().st_mode&0o222
            or sha(binding)!=manifest['execution'][extra]):
        raise PermissionError('EXACT_IMMUTABLE_SESSION_VISIBILITY_SUBCAPSULE_AND_BINDING')
    return actual


def validate(root,manifest=None):
    root=Path(root);m=read(root/'RUN_MANIFEST.json') if manifest is None else manifest
    m=reader_authority(root,m)
    relative=m['execution'].get(KEY)
    if relative is None:return None
    names=required_names(relative)
    if (root.name!='source-confirm-01' or root.parent.parent.name!=n1.PARENT_RUN
            or m['execution'].get('confirmation_phase')!='POLICY_FROZEN'
            or m['execution'].get('confirmation_n1_binding')!=dict(release=n1.RELEASE,adoption_delta_sha256=n1.ADOPTED_DELTA_SHA)):
        raise PermissionError('ONLY_ORIGINAL_CONFIRMATION_LAST_ALLOCATION_COLD_REPAIR')
    for name in names:
        p=root/name
        if (p.is_symlink() or p.stat().st_mode&0o222 or not p.resolve().is_relative_to(root.resolve())
                or m['execution']['sealed_inputs'].get(name)!=sha(p)):
            raise PermissionError('SEALED_CONFIRMATION_COLD_REPAIR_INPUTS')
    q,old,past,native,processes,diagnostic=[read(root/name) for name in names[:6]]
    if (q['schema']!=SCHEMA or q['authority_release']!=n1.RELEASE or q['run_uuid']!=root.name
            or q['parent_run_uuid']!=n1.PARENT_RUN or q['original_MP1_T0_epoch']!=n1.ORIGINAL_T0
            or q['N1_T0_epoch']!=1792264081 or q['target_allocation_number']!=2
            or q['prior_manifest_sha256']!=sha(root/names[1]) or q['new_source_sha']!=m['code']['vla_commit']
            or old['code']['vla_commit']!=OLD_SOURCE or q['old_STOP_sha256']!=STOP_SHA or sha(root/OLD_STOP)!=STOP_SHA
            or any(q[k] is not False for k in ['new_budget_or_clock','new_attempt_rights','old_STOP_removed'])
            or type(q['component_native_version_upper_bound']) is not int
            or not 1<=q['component_native_version_upper_bound']<=6):
        raise PermissionError('ORIGINAL_CONFIRMATION_AUTHORITY_CLOCK_SOURCE_AND_STOP')
    if ({k:v for k,v in m.items() if k not in ['code','execution']}
            !={k:v for k,v in old.items() if k not in ['code','execution']}):
        raise PermissionError('NO_CONFIRMATION_SCIENTIFIC_MANIFEST_CHANGE')
    if ({k:v for k,v in m['execution'].items() if k not in ['sealed_inputs',KEY]}
            !={k:v for k,v in old['execution'].items() if k!='sealed_inputs'}):
        raise PermissionError('NO_CONFIRMATION_ENTRY_OR_AUTHORITY_CHANGE')
    mutable={'vla_commit','capsule_relative','source_seal_relative','source_seal_sha256'}
    if ({k:v for k,v in m['code'].items() if k not in mutable}
            !={k:v for k,v in old['code'].items() if k not in mutable}):
        raise PermissionError('NO_CONFIRMATION_MODEL_DATA_OR_PROTECTED_CODE_CHANGE')
    expected=dict(old['execution']['sealed_inputs'],**{name:sha(root/name) for name in names})
    if m['execution']['sealed_inputs']!=expected:raise PermissionError('NO_LOST_OR_UNDECLARED_CONFIRMATION_INPUT')
    if sha(root/'plans/SOURCE_EVALUATION_PLAN.json')!=PLAN_SHA or sha(root/CLOSURE)!=CLOSURE_SHA:
        raise PermissionError('ONLY_ORIGINAL_FROZEN_PLAN_AND_ACTUAL_ALLOCATION01_CLOSURE')
    closure=read(root/CLOSURE)
    if (closure['prior_allocations']!=1 or closure['prior_GPU_s']!=8456 or closure['prior_generation_queries']!=930
            or closure['unknown_effects'] or not closure['all_recorded_processes_terminal']
            or len(closure['rows'])!=20 or any(r['live_native_failures'] for r in closure['rows'])
            or closure['core']['actual_executed_prefixes']!=917
            or closure['hopper']['cumulative_generation_queries']!=930):
        raise PermissionError('ALL_930_PRIOR_GENERATIONS_AND_ACTUAL_EFFECTS_CLOSED')
    if (past['run_uuid']!=root.name or past['GPU_allocated_s']!=8456 or past['GPU_reserved_s']
            or past['generation_queries']!=930 or past['optimizer_updates'] or past['pending_updates']
            or past.get('CPU_reserved_s',0) or len(past['PBS_submissions'])!=1):
        raise PermissionError('ALL_PRIOR_CONFIRMATION_NATIVE_COSTS_REMAIN')
    if relative==V2_BINDING:
        prototype=read(root/V2_PROTOTYPE);profile=m['execution']['profile']
        if (sha(root/V2_PROTOTYPE)!=PROTOTYPE_MANIFESTS.get(profile)
                or prototype['code']['vla_commit']!=PROTOTYPE_SOURCE or past['model_loads']!=5
                or any(prototype['execution']['sealed_inputs'].get(n)!=sha(root/n) for n in NAMES)
                or sha(root/prototype['code']['source_seal_relative'])!=prototype['code']['source_seal_sha256']):
            raise PermissionError('EXACT_CPU_ONLY_PROTOTYPE_RETAINED_NO_NEW_NATIVE_MODEL_OR_ATTEMPT')
    row=past['PBS_submissions'][0];job=native['native']
    wall=sum(int(v)*k for v,k in zip(job['resources_used']['walltime'].split(':'),(3600,60,1)))
    if (row['request_id']!='alloc-01' or row['status']!='NATIVE_TERMINAL'
            or row['job_id']!=native['job_id'] or row['job_id']!='673388.hopper-m-02'
            or job['job_state']!='F' or job['Exit_status']!=1 or job['project']!='CFP04-CF-007'
            or job['Job_Owner'].split('@')[0]!='e1192548'
            or row['GPU_conservative_charge_s']!=8456
            or max(wall,pbs_epoch(job['obittime'])-pbs_epoch(job['stime']))!=8456):
        raise PermissionError('ONLY_OWN_EXACT_TERMINAL_ALLOCATION01_WITH_SETTLED_GPU')
    if (processes['all_recorded_processes_terminal'] is not True or processes['unknown_effects']
            or not processes['records'] or any(p['terminal_verified'] is not True for p in processes['records'])):
        raise PermissionError('ALL_OLD_CONSUMERS_INCLUDING_DIAGNOSTIC_WORKERS_MUST_BE_TERMINAL')
    if (diagnostic['status']!='ACTUAL_CPU_ROUTE_DIAGNOSIS_AND_CONTROLLED_MECHANISM_REPAIR_PASS_NOT_NEW_NATIVE_QUALIFICATION'
            or diagnostic['actual_nominal80']['cycles']!=80 or diagnostic['actual_nominal80']['RPCs']!=160
            or not diagnostic['actual_nominal80']['all_actual_recorded_byte_checksums_match']
            or diagnostic['controlled_written_fsync_comparison']['actual_requests']!=2
            or diagnostic['controlled_written_fsync_comparison']['AFTER']['session_poisoned']
            or not diagnostic['controlled_written_fsync_comparison']['AFTER']['response_parsed']
            or diagnostic['resources']['new_GPU_PBS'] or diagnostic['resources']['new_model_queries']
            or diagnostic['resources']['new_physics'] or diagnostic['resources']['new_training_updates']):
        raise PermissionError('ACTUAL_BOUNDED_CPU_MECHANISM_EVIDENCE_REQUIRED_NOT_NATIVE_QUALIFICATION')
    old_seal=root/old['code']['source_seal_relative']
    if old_seal.is_symlink() or sha(old_seal)!=old['code']['source_seal_sha256']:
        raise PermissionError('ORIGINAL_CONFIRMATION_SOURCE_SEAL_RETAINED')
    stops={p.name for p in root.glob('RELAY_ENGINEERING_STOP*.json')}
    if stops-{NEW_STOP}!={OLD_STOP}:raise PermissionError('EVERY_ORIGINAL_CONFIRMATION_STOP_RETAINED')
    if any((root/name).exists() for name in ['SOURCE_PHYSICS_STOP.json','SOURCE_SAFETY_STOP_FROM_CORE.json']):
        raise PermissionError('CONFIRMATION_ENGINEERING_NEVER_RESOLVES_NATIVE_SAFETY_STOP')
    return q


def identity(path):
    s=path.lstat()
    if not stat.S_ISREG(s.st_mode):raise PermissionError('REGULAR_CONFIRMATION_BOUNDARY_MEMBER_REQUIRED')
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode


class Guard:
    def __init__(self,root,number):
        self.root,self.number=Path(root),number
        first=identity(self.root/'RUN_MANIFEST.json');m=read(self.root/'RUN_MANIFEST.json')
        names=required_names(m['execution'][KEY])+['RUN_MANIFEST.json'] if m['execution'].get(KEY) else ['RUN_MANIFEST.json']
        before={n:identity(self.root/n) for n in names};before['RUN_MANIFEST.json']=first
        self.binding=validate(root,m)
        self.identities={n:identity(self.root/n) for n in names}
        if before!=self.identities:raise PermissionError('CONFIRMATION_COLD_BINDING_CHANGED_DURING_ADMISSION')

    def blocked(self):
        if any(identity(self.root/n)!=v for n,v in self.identities.items()):
            raise PermissionError('CONFIRMATION_COLD_BINDING_CHANGED_AFTER_ADMISSION')
        if self.binding is None:return (self.root/OLD_STOP).exists()
        if self.number!=2 or (self.root/NEW_STOP).exists():return True
        return False


def stop_target(root,number):
    if number!=2 or validate(root) is None:raise PermissionError('ONLY_CURRENT_CONFIRMATION02_ENGINEERING_STOP')
    return Path(root)/NEW_STOP

"""Map eight logical slots into one real PBS allocation, without expanding it.

A new cold manifest must opt in explicitly. Old manifests retain literal CPU
IDs. This module neither reserves resources nor grants model/physics release.
"""
import os
from pathlib import Path

POLICY='V5_CD_PBS_GRANTED_EIGHT_LOGICAL_SLOTS_V1'
SCHEMA='V5_CD_REAL_PBS_CPU_SLOT_BINDING_V1'
SLOTS={'neural':(0,1,2,3),'physical':(4,5),'core_rpc':(6,),
       'neural_rpc':(7,),'owner':(6,7)}


def need(ok,why):
    if not ok:raise PermissionError('V5_VANDA_AFFINITY_'+why)


def mapping(granted):
    need(type(granted)in(set,list,tuple) and len(granted)>=8
         and all(type(n)is int and n>=0 for n in granted)
         and len(set(granted))==len(granted),'AT_LEAST_EIGHT_DISTINCT_GRANTED_CPU_IDS')
    ids=sorted(granted)[:8]
    return {role:[ids[i]for i in slots]for role,slots in SLOTS.items()}


def capture(*,job_id,request_id,source_sha,owner_manifest_sha256):
    """Read this process's full kernel grant before narrowing its affinity."""
    need(type(job_id)is str and job_id and os.environ.get('PBS_JOBID')==job_id,
         'REAL_CURRENT_PBS_JOB')
    granted=sorted(os.sched_getaffinity(0))
    return dict(schema=SCHEMA,policy=POLICY,job_id=job_id,request_id=request_id,
        source_sha=source_sha,owner_manifest_sha256=owner_manifest_sha256,
        hostname=os.uname().nodename,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cgroup=Path('/proc/self/cgroup').read_text(),granted_cpu_ids=granted,
        roles=mapping(granted),active_cpu_limit=8)


def checked(binding,owner):
    need(type(binding)is dict and binding.get('schema')==SCHEMA
         and binding.get('policy')==POLICY and binding.get('active_cpu_limit')==8,
         'EXPLICIT_NATIVE_BINDING')
    need(type(binding.get('roles'))is dict and set(binding['roles'])==set(SLOTS)
         and all(type(v)is list and all(type(n)is int for n in v)for v in binding['roles'].values())
         and binding['roles']==mapping(binding['granted_cpu_ids']), 'EXACT_DETERMINISTIC_DISJOINT_ROLE_MAP')
    for key in('job_id','request_id','source_sha','hostname','boot_id','cgroup'):
        need(binding[key]==owner[key], 'SAME_LIVE_OWNER_'+key)
    need(binding['owner_manifest_sha256']==owner['manifest_sha256'],'SAME_SEALED_OWNER_MANIFEST')
    return binding


def cpus(binding,role):
    need(role in SLOTS and binding['schema']==SCHEMA and binding['policy']==POLICY,
         'EXPLICIT_LOGICAL_ROLE')
    need(type(binding['roles'])is dict and set(binding['roles'])==set(SLOTS)
         and all(type(v)is list and all(type(n)is int for n in v)for v in binding['roles'].values())
         and binding['roles']==mapping(binding['granted_cpu_ids']), 'UNCHANGED_ROLE_MAP')
    return list(binding['roles'][role])


def physical_cpus(root,manifest):
    policy=manifest.get('execution',{}).get('CPU_slot_policy')
    if policy is None:return list(manifest['limits']['cpus'])
    from . import mix_vanda_eval_context as context,mix_vanda_eval_identity as identity
    from scripts.pi0294.native_dataset import sha
    need(policy==POLICY and manifest['scope']==identity.PHYSICAL_ROLE
         and manifest['limits']['cpus']==[4,5] and manifest['limits']['active_cores']==2,
         'REGISTERED_PHYSICAL_LOGICAL_SLOTS_ONLY')
    neural=Path(context.paths(root,manifest)['hopper']['root']);p=context.request_path(neural)
    from scripts.pi0292.durable import read
    owner=context.owner_current(root,read(p),request_sha256=sha(p))
    binding=checked(owner['CPU_slot_binding'],owner);actual=cpus(binding,'physical')
    need(set(actual)==os.sched_getaffinity(0),'ACTUAL_PHYSICAL_PROCESS_EXACT_TWO_BOUND_CPUS')
    return actual

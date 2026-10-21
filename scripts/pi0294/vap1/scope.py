"""New VAP1 authority; old CR2 guards and closed capsules remain unchanged."""
import json
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read
from scripts.pi0294.cr2.scope import ROLES, ZERO, sealed_file
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.fd1 import sha
from .plan import AUTH, CARD, R1

SCOPE = 'PI0294_VAP1_DATA_CPU_CHILD_V1'
LIMITS = dict(CoreLab_CPU_s=230400, episodes=72, live_ticks=1152000, shadow_ticks=1152000,
    physics_ticks=2800000, static_constructors=2000, dynamic_constructors=10000,
    output_peak_bytes=32*1024**3, transfer_bytes=24*1024**3,
    **{k: 0 for k in ZERO})


def authority(a, now, *, closeout=False):
    card = json.loads((Path(__file__).resolve().parents[3] / CARD).read_text())
    if (card['goal_id'] != 'VAP1_VANDA_POLICY_DATA_TEACHER_V1'
            or a.get('schema') != 'PI0294_VAP1_SOURCE_DATA_ADOPTION_V1'
            or a.get('owner') != 'EXISTING_LOCAL_PC' or a.get('protocol_release') != AUTH
            or a.get('limits') != LIMITS or a.get('R0_statistics_sha256') != STATS
            or a.get('R1_statistics_sha256') != R1 or a.get('prior_closed_scopes_reused') is not False
            or a.get('PV1_numerics_changed') is not False
            or len(a.get('worker_cpu_pool', [])) != 2
            or len(set(a['worker_cpu_pool'])) != 2
            or set(a['worker_cpu_pool']) & set(a.get('PV1_reserved_CPUs', []))
            or a['stop_dispatch_epoch'] != min(a['T0_epoch']+40*3600, 1792866596)
            or a['stop_compute_epoch'] != min(a['T0_epoch']+44*3600, 1792924196)
            or a['hard_close_epoch'] != min(a['T0_epoch']+48*3600, 1792952996)
            or not a['T0_epoch'] <= now < a['hard_close_epoch' if closeout else 'stop_compute_epoch']):
        raise PermissionError('VAP1_EXACT_NEW_AUTHORITY_WINDOW_AND_NONADDITIVE_LIMITS')
    return a


def priority(root, *, now=time.time):
    r = Path(root).parent.parent
    state = read(r/'PV1_PRIORITY_STATE.private.json')
    if (state.get('owner') != 'EXISTING_LOCAL_PC' or not state.get('job_id')
            or not 0 <= now()-state['observed_epoch'] <= 180
            or state.get('native_job_state') not in ('Q', 'NO_ACTIVE_PV1_GPU_ALLOCATION', 'R_COMPAT', 'R_TRAIN')
            or state.get('allow_future_data_dispatch') is not True
            or state.get('actual_or_reserved_Core_closed_loop') is not False):
        return False
    return not any((p/'STOP.json').exists() for p in (r, Path(state['PV1_native_root'])))


def validate(root, role):
    r = Path(root); parent = r.parent.parent
    if role not in ROLES:
        raise PermissionError('VAP1_EXACT_CPU_WORK_ROLE')
    for name in ('RUN_MANIFEST.json','OUTPUT_BINDING.private.json','INPUT_BINDINGS.private.json'):
        sealed_file(r/name)
    m, b, inputs = (read(r/n) for n in ('RUN_MANIFEST.json','OUTPUT_BINDING.private.json','INPUT_BINDINGS.private.json'))
    a = authority(read(parent/'VAP1_ADOPTION_AND_SCOPE.json'), time.time())
    if (r.is_symlink() or r.resolve() != Path(b['run_root']).resolve()
            or r.parent.resolve() != Path(b['registered_parent']).resolve()
            or b['uid'] != os.getuid() or b['host'] != os.uname().nodename
            or m.get('schema') != SCOPE or m.get('scope') != SCOPE or m.get('role') != role
            or m.get('owner') != 'EXISTING_LOCAL_PC' or m.get('run_uuid') != r.name
            or m['execution']['profile'] != ROLES[role] or m['execution']['host'] != os.uname().nodename
            or m['execution']['uid'] != os.getuid() or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or 'PBS_JOBID' in os.environ):
        raise PermissionError('VAP1_REGISTERED_HOST_CPU_SCOPE')
    if (m['authority']['protocol_release'] != AUTH
            or m['authority']['adoption_sha256'] != sealed_file(parent/'VAP1_ADOPTION_AND_SCOPE.json')
            or m['limits']['T0_epoch'] != a['T0_epoch']):
        raise PermissionError('VAP1_ORIGINAL_ADOPTED_CLOCK_AND_FIXED_AUTHORITY')
    lease = read(b['lease']); sealed_file(b['lease'])
    if (lease['status'] != 'ACTIVE' or lease['owner'] != 'EXISTING_LOCAL_PC'
            or lease['run_uuid'] != r.name or lease['protocol_release'] != AUTH
            or lease['deadline_epoch'] != m['limits']['operation_deadline_epoch']
            or not time.time() < lease['deadline_epoch'] <= a['stop_compute_epoch']):
        raise PermissionError('VAP1_ACTIVE_UNEXPIRED_CHILD_LEASE')
    matches = [x for x in read(b['registry'])['entries'] if x.get('role') == b['role']]
    if len(matches) != 1 or Path(matches[0]['path']).resolve() != r.resolve() or matches[0].get('scope') != SCOPE:
        raise PermissionError('VAP1_UNIQUE_CANONICAL_REGISTRY')
    code = r/m['code']['capsule_relative']; seal = r/m['code']['source_seal_relative']
    if (code.is_symlink() or not code.resolve().is_relative_to(r.resolve())
            or Path(__file__).resolve() != (code/'scripts/pi0294/vap1/scope.py').resolve()
            or os.environ.get('VAP1_RUN_MANIFEST') != str(code.parent/'WORKER_RUN_MANIFEST.json')
            or read(code.parent/'WORKER_RUN_MANIFEST.json') != m):
        raise PermissionError('VAP1_EXACT_IMMUTABLE_CAPSULE_NOT_OLD_SCOPE')
    sealed_file(seal, m['code']['source_seal_sha256']); contents = read(seal)
    if contents['source_sha'] != m['code']['vla_commit']:
        raise ValueError('VAP1_EXACT_SOURCE_RELEASE')
    for name, digest in contents['files'].items():
        if Path(name).is_absolute() or not (code/name).resolve().is_relative_to(code.resolve()):
            raise PermissionError('VAP1_SOURCE_ESCAPE')
        sealed_file(code/name, digest)
    if len(m['code']['frozen_numerical_files']) != 15:
        raise ValueError('VAP1_ALL_FIFTEEN_FROZEN_NUMERICAL_FILES')
    for name, digest in m['code']['frozen_numerical_files'].items():
        sealed_file(code/name, digest)
    for name, key in (('OUTPUT_BINDING.private.json','output_binding_sha256'),('INPUT_BINDINGS.private.json','input_binding_sha256')):
        sealed_file(r/name, m['execution'][key])
    for name, key in (('WORK_PLAN.private.json','plan_sha256'),('ASSET_IDENTITY.private.json','asset_identity_sha256'),
                      ('FIXED_EFFECTIVE_PARAMS.private.json','effective_params_ref_sha256')):
        sealed_file(r/'plans'/name, m['data'][key])
    for version, digest in (('R0',STATS),('R1',R1)):
        sealed_file(r/'plans'/(version+'_statistics.json'), digest)
    if m['interfaces'] != dict(action_schema='ORIGINAL_EEF10_H40_PREFIX8',R0_statistics_sha256=STATS,R1_statistics_sha256=R1):
        raise PermissionError('VAP1_FROZEN_PAIRED_CODEC_NO_PV1_INJECTION')
    assets = read(r/'plans/ASSET_IDENTITY.private.json')['absolute_files']
    if len(assets) != 26 or any(sha(p) != digest for p,digest in assets.items()):
        raise ValueError('VAP1_FROZEN_NATIVE_ASSETS')
    infra = Path(b['registry']).parent
    if any(sha(infra/name) != a[key] for name,key in (('enter.sh','entry_sha256'),('doctor.py','doctor_sha256'))):
        raise ValueError('VAP1_CANONICAL_ENTRY_CHANGED')
    proof = r/'plans/SOURCE_PUBLICATION_AND_CI.json'; sealed_file(proof,m['code']['publication_and_CI_sha256'])
    p = read(proof)
    if (p['source_sha'] != m['code']['vla_commit'] or p['CI_head_sha'] != p['source_sha']
            or p['CI_status'] != 'completed' or p['CI_conclusion'] != 'success' or not p['files_exact_readback']):
        raise PermissionError('VAP1_EXACT_PUBLISHED_SOURCE_AND_COMPLETE_CI')
    cpus = m['limits']['cpus']
    if (len(cpus) != 1 or not set(cpus) <= os.sched_getaffinity(0)
            or not set(cpus) <= set(a['worker_cpu_pool']) or set(cpus) & set(a['PV1_reserved_CPUs'])
            or not 30 < m['limits']['CPU_s'] <= a['limits']['CoreLab_CPU_s']
            or any(m['limits'].get(k) != 0 for k in ZERO)):
        raise PermissionError('VAP1_ONE_DISJOINT_CORE_ZERO_GPU_OR_NEURAL')
    if role in ('EXPORT','PACK') and any(m['limits'][k] != 0 for k in ('engine_starts','Teacher_episodes','physics_ticks')):
        raise PermissionError('VAP1_SAVED_READER_WITHOUT_NEW_PHYSICS')
    for path, digest in a['protected_files'].items():
        if sha(path) != digest:
            raise ValueError('VAP1_PROTECTED_CR2_OR_PV1_INPUT_CHANGED')
    if any((p/'STOP.json').exists() for p in (r,parent)):
        raise PermissionError('VAP1_STOP_PRESERVED')
    return m, inputs

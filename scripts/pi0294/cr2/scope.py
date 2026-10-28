"""Versioned authority for the fixed-R1 pilot, independent of closed COV1/PV1."""
import json
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read
from scripts.pi0294.data_cov1.scope import sealed_file
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.fd1 import sha
from .plan import AUTH, CARD, GO, R1

SCOPE = 'PI0294_CR2_FIXED_R1_CPU_CHILD_V1'
ROLES = dict(STATIC='4080-simulation', CAPTURE='4080-simulation', EXPORT='4080-cpu-data', PACK='4080-cpu-data')
ZERO = ('GPU', 'PBS', 'neural_forwards', 'queries', 'updates')


def authority(adoption, now, *, closeout=False):
    a = adoption
    expected = json.loads((Path(__file__).resolve().parents[3] / CARD).read_text())['limits']
    if (a.get('schema') != 'PI0294_CR2_FIXED_R1_ADOPTION_V1'
            or a.get('owner') != 'EXISTING_LOCAL_PC' or a.get('protocol_release') != AUTH
            or a.get('approval_ref') != GO or a.get('limits') != expected
            or a.get('R0_statistics_sha256') != STATS or a.get('R1_statistics_sha256') != R1
            or a.get('prior_closed_scopes_reused') is not False
            or a.get('PV1_budget_or_numerics_changed') is not False
            or any(expected[k] != 0 for k in ZERO)
            or len(a.get('worker_cpu_pool', [])) != 2
            or set(a['worker_cpu_pool']) & set(a.get('PV1_reserved_CPUs', []))
            or a['stop_dispatch_epoch'] != a['T0_epoch'] + 40 * 3600
            or a['stop_compute_epoch'] != a['T0_epoch'] + 44 * 3600
            or a['hard_close_epoch'] != a['T0_epoch'] + 48 * 3600
            or not a['T0_epoch'] <= now < a['hard_close_epoch' if closeout else 'stop_compute_epoch']):
        raise PermissionError('CR2_EXPLICIT_NEW_GO_AND_EXACT_INDEPENDENT_WINDOW')
    return a


def validate(root, role):
    r = Path(root)
    if role not in ROLES:
        raise PermissionError('CR2_EXACT_CPU_WORK_ROLE')
    for name in ('RUN_MANIFEST.json', 'OUTPUT_BINDING.private.json', 'INPUT_BINDINGS.private.json'):
        sealed_file(r / name)
    m, b, inputs = (read(r / n) for n in ('RUN_MANIFEST.json', 'OUTPUT_BINDING.private.json', 'INPUT_BINDINGS.private.json'))
    parent = r.parent.parent
    adoption = authority(read(parent / 'CR2_ADOPTION_AND_SCOPE.json'), time.time())
    if (r.is_symlink() or r.resolve() != Path(b['run_root']).resolve()
            or r.parent.resolve() != Path(b['registered_parent']).resolve()
            or b['uid'] != os.getuid() or b['host'] != os.uname().nodename
            or m.get('schema') != SCOPE or m.get('scope') != SCOPE or m.get('role') != role
            or m.get('owner') != 'EXISTING_LOCAL_PC' or m.get('run_uuid') != r.name
            or m['execution']['profile'] != ROLES[role] or m['execution']['host'] != os.uname().nodename
            or m['execution']['uid'] != os.getuid() or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or 'PBS_JOBID' in os.environ):
        raise PermissionError('CR2_ACTUAL_REGISTERED_HOST_CPU_SCOPE_NOT_PV1')
    if (m['authority']['protocol_release'] != AUTH or m['authority']['approval_ref'] != GO
            or m['authority']['adoption_sha256'] != sealed_file(parent / 'CR2_ADOPTION_AND_SCOPE.json')
            or m['limits']['T0_epoch'] != adoption['T0_epoch']):
        raise PermissionError('CR2_SINGLE_APPROVAL_AND_UNRESET_CLOCK')
    lease = read(b['lease']); sealed_file(b['lease'])
    if (lease['status'] != 'ACTIVE' or lease['owner'] != 'EXISTING_LOCAL_PC'
            or lease['run_uuid'] != r.name or lease['protocol_release'] != AUTH
            or lease['deadline_epoch'] != m['limits']['operation_deadline_epoch']
            or not time.time() < lease['deadline_epoch'] <= adoption['stop_compute_epoch']):
        raise PermissionError('CR2_EXACT_ACTIVE_BOUNDED_CHILD_LEASE')
    matches = [x for x in read(b['registry'])['entries'] if x.get('role') == b['role']]
    if len(matches) != 1 or Path(matches[0]['path']).resolve() != r.resolve() or matches[0].get('scope') != SCOPE:
        raise PermissionError('CR2_UNIQUE_CANONICAL_REGISTRY_ENTRY')
    code = r / m['code']['capsule_relative']; seal = r / m['code']['source_seal_relative']
    if (code.is_symlink() or not code.resolve().is_relative_to(r.resolve())
            or Path(__file__).resolve() != (code / 'scripts/pi0294/cr2/scope.py').resolve()
            or os.environ.get('CR2_RUN_MANIFEST') != str(code.parent / 'WORKER_RUN_MANIFEST.json')
            or read(code.parent / 'WORKER_RUN_MANIFEST.json') != m):
        raise PermissionError('CR2_ACTUAL_IMMUTABLE_SOURCE_NOT_OLD_LAUNCHER')
    sealed_file(seal, m['code']['source_seal_sha256']); contents = read(seal)
    if contents['source_sha'] != m['code']['vla_commit']:
        raise ValueError('CR2_EXACT_SOURCE_RELEASE')
    for name, digest in contents['files'].items():
        if Path(name).is_absolute() or not (code / name).resolve().is_relative_to(code.resolve()):
            raise PermissionError('CR2_SOURCE_PATH_ESCAPE')
        sealed_file(code / name, digest)
    for name, digest in m['code']['frozen_numerical_files'].items():
        sealed_file(code / name, digest)
    if len(m['code']['frozen_numerical_files']) != 15:
        raise ValueError('CR2_ALL_FIFTEEN_FROZEN_NUMERICAL_MEMBERS')
    for name, key in (('OUTPUT_BINDING.private.json', 'output_binding_sha256'),
                      ('INPUT_BINDINGS.private.json', 'input_binding_sha256')):
        sealed_file(r / name, m['execution'][key])
    sealed_file(r / 'plans/WORK_PLAN.private.json', m['data']['plan_sha256'])
    sealed_file(r / 'plans/FIXED_EFFECTIVE_PARAMS.private.json', m['data']['effective_params_ref_sha256'])
    for version, digest in (('R0', STATS), ('R1', R1)):
        sealed_file(r / 'plans' / (version + '_statistics.json'), digest)
    if m['interfaces'] != dict(action_schema='ORIGINAL_EEF10_H40_PREFIX8', R0_statistics_sha256=STATS, R1_statistics_sha256=R1):
        raise PermissionError('CR2_EXPLICIT_PAIRED_NUMERICAL_VERSION_NO_PV1_INJECTION')
    assets = r / 'plans/ASSET_IDENTITY.private.json'; sealed_file(assets, m['data']['asset_identity_sha256'])
    files = read(assets)['absolute_files']
    if len(files) != 26 or any(sha(path) != digest for path, digest in files.items()):
        raise ValueError('CR2_FROZEN_NATIVE_ASSETS_CHANGED')
    infra = Path(b['registry']).parent
    for name, key in (('enter.sh', 'entry_sha256'), ('doctor.py', 'doctor_sha256')):
        if sha(infra / name) != adoption[key]:
            raise ValueError('CR2_EXISTING_CANONICAL_ENTRY_CHANGED')
    proof_path = r / 'plans/SOURCE_PUBLICATION_AND_CI.json'
    sealed_file(proof_path, m['code']['publication_and_CI_sha256']); proof = read(proof_path)
    if (proof['source_sha'] != m['code']['vla_commit'] or proof['CI_head_sha'] != m['code']['vla_commit']
            or proof['CI_status'] != 'completed' or proof['CI_conclusion'] != 'success'
            or not proof['files_exact_readback']):
        raise PermissionError('CR2_EXACT_PUBLISHED_SUCCESSFUL_CI_REQUIRED')
    cpus = m['limits']['cpus']
    if (len(cpus) != 1 or not set(cpus) <= os.sched_getaffinity(0)
            or not set(cpus) <= set(adoption['worker_cpu_pool'])
            or set(cpus) & set(adoption['PV1_reserved_CPUs'])
            or not 30 < m['limits']['CPU_s'] <= adoption['limits']['CoreLab_CPU_s']):
        raise PermissionError('CR2_ONE_DISJOINT_CPU_AND_BOUNDED_RESERVATION')
    if any(m['limits'].get(k) != 0 for k in ZERO):
        raise PermissionError('CR2_NO_NEURAL_GPU_PBS_OR_TRAINING')
    if role in ('EXPORT', 'PACK') and any(m['limits'][k] != 0 for k in ('engine_starts', 'Teacher_episodes', 'physics_ticks')):
        raise PermissionError('CR2_SAVED_READER_AND_PACK_NO_NEW_PHYSICS')
    for name, old in adoption['protected_files'].items():
        if sha(name) != old:
            raise ValueError('CR2_PROTECTED_CLOSED_SCOPE_OR_PV1_INPUT_CHANGED')
    if any((p / 'STOP.json').exists() for p in (r, parent)):
        raise PermissionError('CR2_NATIVE_OR_INTEGRITY_STOP_PRESERVED')
    return m, inputs


def priority(root, *, now=time.time):
    parent = Path(root).parent.parent
    state = read(parent / 'PV1_PRIORITY_STATE.private.json')
    if (state.get('owner') != 'EXISTING_LOCAL_PC' or not state.get('job_id')
            or not 0 <= now() - state['observed_epoch'] <= 180
            or state['native_job_state'] not in ('Q', 'NO_ACTIVE_PV1_GPU_ALLOCATION')
            or state.get('allow_future_data_dispatch') is not True):
        return False
    pv1 = Path(state['PV1_native_root'])
    return not any((p / 'STOP.json').exists() for p in (pv1, pv1.parent.parent))

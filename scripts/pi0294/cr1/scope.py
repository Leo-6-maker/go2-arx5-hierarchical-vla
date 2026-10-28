"""Fail-closed native admission for a separately leased CR1 CPU capsule."""
import os
from pathlib import Path
import sys
import time

from scripts.pi0292.durable import read
from scripts.pi0294.data_cov1.scope import sealed_file
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.fd1 import sha
from .rules import AUTH

ROLE = 'PI0294_CR1_SAVED_TRAIN_CPU_ONLY'
ZERO = ('constructors', 'episodes', 'physics_ticks', 'GPU', 'PBS', 'neural_forwards', 'queries', 'updates')


def authority(adoption, now):
    a = adoption
    limits = a.get('limits', {})
    if (a.get('protocol_release') != AUTH or a.get('owner') != 'EXISTING_LOCAL_PC'
            or a.get('approval_ref') != 'HUMAN_REPLY_call_1c9b979dac7e4e92b967be93cc85083c_FULL_LIMITED_SCOPE'
            or not a['T0_epoch'] <= now < a['stop_compute_epoch'] <= a['T0_epoch']+22*3600
            or a['stop_new_batch_epoch'] != a['T0_epoch']+20*3600
            or a['hard_close_epoch'] != a['T0_epoch']+24*3600
            or limits.get('CoreLab_CPU_s') != 43200 or limits.get('local_CPU_s') != 7200
            or limits.get('max_workers') != 2 or limits.get('active_workers') != 1
            or len(limits.get('cpus', [])) != 1
            or set(limits['cpus']) & set(limits.get('PV1_reserved_CPUs', []))
            or limits.get('unique_parents') != 400 or limits.get('diagnostic_windows') != 32
            or limits.get('output_bytes') != 8*1024**3 or limits.get('network_bytes') != 2*1024**3
            or any(limits.get(k) != 0 for k in ZERO)):
        raise PermissionError('CR1_EXPLICIT_NEW_CPU_GO_WINDOW_BOUNDS_AND_PV1_DISJOINTNESS')


def validate(root):
    r = Path(root)
    for n in ('RUN_MANIFEST.json', 'OUTPUT_BINDING.private.json', 'INPUT_BINDINGS.private.json',
              'CR1_ADOPTION_AND_INPUTS.json', 'CR1_SPLIT_AND_RULE_SEAL.json', 'plans/INPUT_PLAN.private.json'):
        sealed_file(r/n)
    a, m, b, inputs = (read(r/n) for n in ('CR1_ADOPTION_AND_INPUTS.json', 'RUN_MANIFEST.json',
                                         'OUTPUT_BINDING.private.json', 'INPUT_BINDINGS.private.json'))
    authority(a, time.time())
    if (m['scope'] != ROLE or m['run_uuid'] != a['run_uuid'] or m['run_uuid'] != r.name
            or b['role'] != ROLE or b['uid'] != os.getuid() or b['host'] != os.uname().nodename
            or m['execution']['profile'] != '4080-cpu-data'
            or r.is_symlink() or r.resolve() != Path(b['run_root']).resolve()
            or r.parent.resolve() != Path(b['registered_parent']).resolve()
            or os.environ.get('CR1_RUN_MANIFEST') != str(r/'RUN_MANIFEST.json')
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'PBS_JOBID' in os.environ):
        raise PermissionError('CR1_REGISTERED_CPU_NATIVE_IDENTITY')
    for name, key in (('OUTPUT_BINDING.private.json', 'output_binding_sha256'),
                      ('INPUT_BINDINGS.private.json', 'input_binding_sha256'),
                      ('CR1_ADOPTION_AND_INPUTS.json', 'adoption_sha256'),
                      ('CR1_SPLIT_AND_RULE_SEAL.json', 'rules_sha256'),
                      ('plans/INPUT_PLAN.private.json', 'plan_sha256')):
        if sha(r/name) != m['execution'][key]:
            raise ValueError('CR1_SEALED_INPUT_OR_RULE_CHANGED')
    lease = read(b['lease']); sealed_file(b['lease'])
    if (lease['status'] != 'ACTIVE' or lease['owner'] != 'EXISTING_LOCAL_PC'
            or lease['run_uuid'] != r.name or lease['authority_release'] != AUTH
            or lease['deadline_epoch'] != a['hard_close_epoch']):
        raise PermissionError('CR1_INDEPENDENT_LEASE_NOT_OLD_DATA_OR_PV1')
    matches = [e for e in read(b['registry'])['entries'] if e.get('role') == ROLE]
    if len(matches) != 1 or Path(matches[0]['path']).resolve() != r.resolve():
        raise PermissionError('CR1_ONE_OWNER_REGISTERED_ROOT')
    infra = Path(b['registry']).parent
    for name, key in (('enter.sh', 'entry_sha256'), ('doctor.py', 'doctor_sha256')):
        if sha(infra/name) != inputs[key]: raise ValueError('CR1_DEPLOYED_ENTRY_CHANGED')
    old = Path(inputs['closed_DATA_root']); ob = read(old/'OUTPUT_BINDING.private.json')
    if read(ob['lease'])['status'] != 'CLOSED' or sha(ob['lease']) != inputs['prior_data_lease_sha256']:
        raise PermissionError('CR1_OLD_DATA_LEASE_MUST_STAY_CLOSED_UNCHANGED')
    for name, h in inputs['prior_data_files'].items(): sealed_file(old/name, h)
    bank = Path(inputs['PV1_root'])/'children/saved-stream-02/datasets/FROZEN_TRAIN_BANK/DATASET_MANIFEST.json'
    if sha(bank) != inputs['prior_PV1_bank_sha256']: raise ValueError('CR1_PV1_BANK_MUST_NOT_CHANGE')
    sealed_file(r/'plans/statistics.json', STATS)
    seal = r/m['code']['source_seal_relative']; sealed_file(seal, m['code']['source_seal_sha256'])
    code = r/m['code']['capsule_relative']; s = read(seal)
    if (s['source_sha'] != m['code']['vla_commit'] or code.is_symlink()
            or not code.resolve().is_relative_to(r.resolve())
            or Path(__file__).resolve() != (code/'scripts/pi0294/cr1/scope.py').resolve()):
        raise PermissionError('CR1_ACTUAL_IMMUTABLE_SOURCE_CAPSULE')
    for n, h in s['files'].items():
        if Path(n).is_absolute() or not (code/n).resolve().is_relative_to(code.resolve()):
            raise PermissionError('CR1_SOURCE_PATH_ESCAPE')
        sealed_file(code/n, h)
    proof = read(r/'plans/SOURCE_PUBLICATION_AND_CI.json')
    if (proof['source_sha'] != m['code']['vla_commit'] or proof['CI_head_sha'] != m['code']['vla_commit']
            or proof['CI_status'] != 'completed' or proof['CI_conclusion'] != 'success'
            or not proof['files_exact_readback']):
        raise PermissionError('CR1_EXACT_SOURCE_CI_AND_PUBLICATION_REQUIRED')
    sealed_file(r/'plans/SOURCE_PUBLICATION_AND_CI.json', m['code']['publication_sha256'])
    plan = read(r/'plans/INPUT_PLAN.private.json'); rules = read(r/'CR1_SPLIT_AND_RULE_SEAL.json')
    cases = plan['main']+plan['known']
    if (not 0 < len(cases) <= 400 or len({c['id'] for c in cases}) != len(cases)
            or rules['results_seen_before_seal'] is not False
            or rules['rules_code_sha256'] != sha(code/'scripts/pi0294/cr1/rules.py')
            or any(c['split'] != 'TRAIN' or c['nominal_clearance_m'] != .004 for c in cases)
            or any(c['role'] != 'KNOWN_CHALLENGE' for c in plan['known'])):
        raise PermissionError('CR1_ONLY_SEALED_UNIQUE_TRAIN_SPLIT_NO_CHALLENGE_FIT')
    if not set(a['limits']['cpus']) <= os.sched_getaffinity(0):
        raise PermissionError('CR1_NATIVE_CPU_AFFINITY_UNAVAILABLE')
    return a, m, inputs, plan


class NoPhysicsImport:
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in ('mujoco', 'mujoco_py', 'isaaclab', 'isaacsim')
                or fullname.startswith('gr00t.model.') and '.modeling_' in fullname):
            raise PermissionError('CR1_NO_NEW_PHYSICS_OR_NEURAL_MODULE_IMPORT')


def install_import_guard():
    sys.meta_path.insert(0, NoPhysicsImport())

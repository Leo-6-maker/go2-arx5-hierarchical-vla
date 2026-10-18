"""Cold PV1 CPU-child admission through the existing CoreLab entry."""
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read
from scripts.pi0294.fd1 import sha
from .decision import AUTHORITY


SCOPE = 'reports/PI0294/design/PV1_MULTITASK_POLICY_SCOPE_20261019.json'
PROTOCOL = 'docs/protocols/PI0294_PV1_MULTITASK_POLICY_RELEASE_20261019.md'
ROLE = 'PI0294_PV1_CORE_POPULATION_STATIC_V1'
CAPTURE_ROLE = 'PI0294_PV1_CORE_TRAIN_CAPTURE_V1'


def validate(root, role=ROLE):
    if role not in (ROLE, CAPTURE_ROLE):
        raise PermissionError('PV1_ONLY_VERSIONED_CPU_CHILD_ROLES')
    root = Path(root)
    m = read(root/'RUN_MANIFEST.json')
    b = read(root/'OUTPUT_BINDING.private.json')
    inputs = read(root/'INPUT_BINDINGS.private.json')
    code = root/m['code']['capsule_relative']
    if (code.is_symlink() or not code.resolve().is_relative_to(root.resolve())
            or Path(m['code']['capsule_relative']).is_absolute()
            or Path(m['code']['source_seal_relative']).is_absolute()
            or not (root/m['code']['source_seal_relative']).resolve().is_relative_to(root.resolve())):
        raise PermissionError('PV1_SOURCE_CAPSULE_MUST_STAY_IN_REGISTERED_CHILD')
    if (m['schema'] != 'PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1' or m['scope'] != role
            or m['owner'] != 'EXISTING_LOCAL_PC' or m['run_uuid'] != root.name
            or m['execution']['profile'] != '4080-simulation'
            or m['execution']['host'] != os.uname().nodename or m['execution']['uid'] != os.getuid()
            or os.environ.get('PV1_RUN_MANIFEST') != str(code.parent/'WORKER_RUN_MANIFEST.json')
            or read(code.parent/'WORKER_RUN_MANIFEST.json') != m):
        raise PermissionError('PV1_NATIVE_CPU_CHILD_IDENTITY_AND_MANIFEST')
    if (root.is_symlink() or root.resolve() != Path(b['run_root']).resolve()
            or root.parent.resolve() != Path(b['registered_parent']).resolve()
            or b['role'] != role or b['uid'] != os.getuid()
            or sha(root/'OUTPUT_BINDING.private.json') != m['execution']['output_binding_sha256']
            or sha(root/'INPUT_BINDINGS.private.json') != m['execution']['input_binding_sha256']):
        raise PermissionError('PV1_REGISTERED_CHILD_OUTPUT_NO_ESCAPE')
    parent = root.parent.parent
    adoption = read(parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')
    scope = read(code/SCOPE)
    if (adoption['authority_release'] != AUTHORITY or m['authority']['protocol_ref'] != AUTHORITY
            or m['limits']['PV1_T0_epoch'] != adoption['T0_epoch']
            or m['limits']['PV1_hard_close_epoch'] != adoption['hard_closeout_epoch']
            or adoption['bounds'] != scope['new_campaign_budget']
            or sha(code/SCOPE) != m['authority']['scope_sha256']
            or sha(code/PROTOCOL) != m['authority']['protocol_sha256']):
        raise PermissionError('PV1_EXACT_SEPARATE_AUTHORITY_AND_ORIGINAL_T0')
    lease = read(b['lease'])
    if (lease['status'] != 'ACTIVE' or lease['run_uuid'] != root.name
            or lease['owner'] != 'EXISTING_LOCAL_PC' or lease['authority_release'] != AUTHORITY
            or lease['deadline_epoch'] != m['limits']['operation_deadline_epoch']
            or not adoption['T0_epoch'] <= time.time() < min(lease['deadline_epoch'], adoption['stop_new_physics_epoch'])):
        raise PermissionError('PV1_FRESH_CHILD_LEASE_AND_TIME')
    registry = read(b['registry'])['entries']
    matches = [x for x in registry if x.get('role') == role and x.get('run_uuid') == root.name]
    if len(matches) != 1 or Path(matches[0]['path']).resolve() != root.resolve():
        raise PermissionError('PV1_UNIQUE_CHILD_OWNER_IN_EXISTING_REGISTRY')
    parent_binding = read(parent/'OUTPUT_BINDING.private.json')
    parent_lease = read(parent_binding['lease'])
    if (parent_binding['run_uuid'] != adoption['run_uuid']
            or parent_lease['status'] not in ('ACTIVE_CONTROL_PLANE_ONLY', 'ACTIVE')
            or parent_lease['authority_release'] != AUTHORITY):
        raise PermissionError('PV1_EXISTING_PARENT_CONTROL_OWNER')
    seal_path = root/m['code']['source_seal_relative']
    if sha(seal_path) != m['code']['source_seal_sha256']:
        raise ValueError('PV1_COLD_SOURCE_SEAL')
    seal = read(seal_path)
    if (seal['source_sha'] != m['code']['vla_commit']
            or any(sha(code/name) != h for name, h in seal['files'].items())
            or Path(__file__).resolve() != (code/'scripts/pi0294/pv1/scope.py').resolve()):
        raise ValueError('PV1_ACTUAL_IMPORTED_PINNED_SOURCE')
    frozen = m['code']['frozen_numerical_files']
    if len(frozen) != 15 or any(sha(code/name) != h for name, h in frozen.items()):
        raise ValueError('PV1_FROZEN_ORIGINAL_NUMERICAL_FILES')
    assets = read(root/'plans/ASSET_IDENTITY.private.json')
    if len(assets['absolute_files']) != 26 or any(sha(p) != h for p, h in assets['absolute_files'].items()):
        raise ValueError('PV1_FROZEN_26_NATIVE_ASSETS')
    infra = Path(b['registry']).parent
    if any(sha(infra/n) != m['execution'][key] for n,key in
           [('enter.sh','entry_sha256'),('doctor.py','doctor_sha256')]):
        raise ValueError('PV1_APPROVED_EXISTING_ENTRY_VERSION')
    data_file = 'EXCLUSIONS.json' if role == ROLE else 'CAPTURE_PLAN.json'
    if (sha(root/'plans'/data_file) != m['data']['manifest_sha256']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'PBS_JOBID' in os.environ):
        raise PermissionError('PV1_EXCLUSIONS_IDENTITY_CPU_ONLY_NO_PBS')
    if (parent/'STOP.json').exists():
        raise PermissionError('PV1_PARENT_SAFETY_OR_INTEGRITY_STOP')
    return m, inputs

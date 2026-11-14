"""Independent simulation ownership for a sealed Corelab expert cohort."""
from contextlib import closing, contextmanager
from pathlib import Path
import os

from .stable_task_scene_v1 import pinned

SCHEMA = 'CORELAB_INDEPENDENT_EXPERT_WORLDS_V3'
PLAN = 'f672fb1aa574a2dbac3c91e047346eaa441bc9f5'
DOMAIN_POOL = frozenset(range(192, 200))


def validate_layout(binding):
    worlds = binding['worlds']
    if (binding.get('schema') != SCHEMA or binding.get('plan_revision') != PLAN
            or len(worlds) not in (1, 2, 3, 4, 8) or len({w['root'] for w in worlds}) != len(worlds)
            or len({w['ROS_DOMAIN_ID'] for w in worlds}) != len(worlds)):
        raise PermissionError('EXACT_CORELAB_EXPERT_COHORT_REQUIRED')
    reserved_cpus = set(binding['reserved_cpus'])
    used_cpus = set()
    pool = Path(binding['pool_root']).resolve(strict=True)
    for world in worlds:
        root = Path(world['root'])
        cpus = set(world['cpus'])
        if (root.parent != pool or root.resolve() != root or root.is_symlink()
                or world['ROS_DOMAIN_ID'] not in DOMAIN_POOL
                or not cpus or len(cpus) != len(world['cpus'])
                or cpus & (reserved_cpus | used_cpus)):
            raise PermissionError('DISTINCT_WORLD_ROOT_ROS_DOMAIN_AND_CPU_AFFINITY_REQUIRED')
        used_cpus.update(cpus)
    return worlds


@contextmanager
def world_owner(spec, registry):
    from scripts.pi0292.durable import OwnerLock
    from .corelab_expert_semantics_v3 import ROLE
    from .task_learning_grasp_v1 import PHASE

    pin = spec['corelab_world_binding_v3']
    binding = pinned(pin)
    worlds = validate_layout(binding)
    root = Path(spec['registered_root'])
    campaign = Path(spec['campaign_root'])
    slot = next((world for world in worlds if world['root'] == str(root)), None)
    domain = spec['ROS_DOMAIN_ID']
    control = campaign / 'control'
    expected_owner = control / 'corelab-world-owners-v3' / f'ros-{domain}.lock'
    entry = next((row for row in registry['entries'] if row.get('path') == str(root)), None)
    if (slot is None or spec.get('task_learning_role') != ROLE or spec['phase'] != PHASE
            or spec.get('v3_plan_revision') != PLAN or spec.get('parallel_batch_spec') is not None
            or binding['campaign_root'] != str(campaign) or binding['source_sha'] != spec['source_sha']
            or slot['ROS_DOMAIN_ID'] != domain or slot['cpus'] != spec['cpus']
            or not set(slot['cpus']) <= os.sched_getaffinity(0)
            or spec['native_environment']['ROS_DOMAIN_ID'] != str(domain)
            or any(Path(spec[name]) != root / name for name in ('logs', 'tmp', 'cache', 'data'))
            or expected_owner.resolve() != expected_owner
            or not entry or entry.get('corelab_world_binding_sha256') != pin['sha256']
            or entry.get('run_uuid') != root.name or entry.get('role') != spec['role']
            or entry.get('source_sha') != spec['source_sha'] or entry.get('owner') != spec['owner']
            or entry.get('corelab_world_owner') != str(expected_owner)
            or entry.get('ROS_DOMAIN_ID') != domain):
        raise PermissionError('ACTUAL_REGISTERED_INDEPENDENT_SIMULATION_WORLD_REQUIRED')
    from .pilot1_storage import local_path
    local_path(spec, root)
    with closing(OwnerLock(expected_owner)):
        yield

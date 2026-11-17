"""Fresh project-authorized simulation scope; no effect on old capsules."""
import json
from pathlib import Path

from .storage import digest

VERSION = 'MFREE_SHIFT7_SIM_ONLY_V51'
SCOPE = 'SIM_ONLY_PROJECT_AUTHORIZED'
PLAN_COMMIT = '78206c6366cd57200db28a89218df5d2f07bfc4e'
STARTUP_TICKS = 5703
TASK_TICKS = 30000
TOTAL_TICKS = STARTUP_TICKS + TASK_TICKS
INTERFACE_TICKS = STARTUP_TICKS + 250
PACKAGE = 'ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb'
BINARIES = dict(
    native_binary='39e5136603f51c092b2f3e595245468dbb0c12dbca79818b0df08da8dd4af7dd',
    solver_binary='4a5f5aa6e0b9d4bde62f4d1da73135ef97e5268945a0cecf114d9190d3e04fd0',
    controller_binary='9e8ce46837d09bc39ae04d4b49f1ce4dd8c8c2d31bea39eac56c5806d108c1f3')


def validate(spec):
    """Read the actual user authorization and unchanged simulator identities.

    Site registry/lease/owner/source/guard validation remains in the existing
    mobile runner. This neither invents a site allocation nor launches anything.
    """
    pin = spec['sim_only_project_authorization']
    if digest(pin['path']) != pin['sha256']:
        raise PermissionError('SIM_ONLY_USER_AUTHORIZATION_BYTES_CHANGED')
    authority = json.loads(Path(pin['path']).read_bytes())
    condition = spec['mobile_condition']; role = spec['sim_only_role']
    limit = INTERFACE_TICKS if role == 'INTERFACE' else TOTAL_TICKS
    if (authority.get('schema') != 'USER_PROJECT_AUTHORIZED_ISOLATED_MUJOCO_V51'
            or authority.get('scope') != SCOPE
            or authority.get('source') != 'CURRENT_USER_EXPLICIT_V51_INSTRUCTION'
            or authority.get('new_training_updates') != 0
            or authority.get('hardware_allowed') is not False
            or authority.get('source_sha') != spec['source_sha']
            or spec.get('mobile_experiment_version') != VERSION
            or spec.get('plan_commit') != PLAN_COMMIT
            or spec.get('execution_clock_profile') != 'SIM_PAUSED_NATIVE'
            or spec.get('scope') != SCOPE or not spec['physics']
            or spec.get('native_backend') != 'MUJOCO_ONLY'
            or spec.get('hardware_connections_allowed') is not False
            or role not in ('INTERFACE', 'EXPERT', 'STUDENT')
            or condition.get('family') not in ('F1', 'F2') or condition.get('mode') != 'M_FREE'
            or spec['max_ticks'] != limit or spec['physics_dt_s'] != .002
            or spec['SHIFT7_package']['sha256'] != PACKAGE
            or spec.get('optimizer_updates') != 0 or spec.get('WBC_changes') is not False
            or spec['deadline_epoch'] > min(authority['deadline_epoch'], 1795027883.855488)):
        raise PermissionError('FRESH_BOUNDED_PROJECT_AUTHORIZED_SIMULATION_REQUIRED')
    expected = {'INTERFACE':'SIM_INTERFACE_HOLD', 'EXPERT':'EXPERT', 'STUDENT':'SHIFT7_SINGLE'}[role]
    if condition['method'] != expected:
        raise PermissionError('EXACT_SIMULATION_ROLE_AND_METHOD_REQUIRED')
    for key, sha in BINARIES.items():
        if spec['frozen_files'].get(spec[key]) != sha or digest(spec[key]) != sha:
            raise PermissionError('FIXED_SIMULATOR_SOLVER_CONTROLLER_ONLY')
    env = spec['native_environment']
    if (env.get('ROS_LOCALHOST_ONLY') != '1'
            or env.get('ROS_AUTOMATIC_DISCOVERY_RANGE') != 'LOCALHOST'
            or env.get('CUDA_VISIBLE_DEVICES') != ''
            or spec['ROS_DOMAIN_ID'] not in range(192, 200)):
        raise PermissionError('LOCALHOST_SIMULATION_DOMAIN_AND_CPU_RENDERING_REQUIRED')
    return dict(scope=SCOPE, role=role, native_requests_per_case=1,
                total_ticks_max=limit, initialization_ticks=STARTUP_TICKS,
                hardware_allowed=False, independent_signature_required=False)

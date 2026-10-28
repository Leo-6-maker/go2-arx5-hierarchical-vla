"""Inference-only E0 admission; independent clocks, original source D0 and S1."""
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import re
import socket
import time

from scripts.pi0292.durable import read, write
from .native_dataset import sha

RUN = '20261016T083821Z-localpc-e0'
AUTHORITY = '22e5f0baeed0af662a7cf1c97edae1f4060947fe'
T0 = 1792139901
ADMISSION_END = T0 + 86400
DEADLINE = T0 + 31 * 3600
STATS = 'fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7'
DATA = 'b8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df'
S1_PROOF = '927cf42bb51c6592c07c2e4cb8d4ba06c11ffa1e23aec90609a6866cebe06187'
LIMITS = dict(PBS_calls=2, GPU_allocated_s=21600, generation_queries=1056,
    physics_starts=16, physics_steps=256000, shadow_integrations=256000,
    rgb_pairs=10256, CoreLab_CPU_s=86400, Hopper_CPU_s=86400,
    CoreLab_cores=8, D0_worker_cores=4, Hopper_cores=4,
    CoreLab_new_peak_bytes=16*1024**3, Hopper_new_peak_bytes=24*1024**3,
    local_pc_new_peak_bytes=2*1024**3, training_updates=0,
    T0_epoch=T0, allocation_admission_cutoff_epoch=ADMISSION_END,
    deadline_epoch=DEADLINE, reset_on_resume=False)


def pbs_epoch(value):
    """Hopper PBS JSON uses either epoch seconds or site-local English dates."""
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r'[0-9]+', value):
        return int(value)
    if not isinstance(value, str):
        raise ValueError('E0_INVALID_PBS_TIMESTAMP')
    match = re.fullmatch(
        r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun) '
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) '
        r' ?([0-9]{1,2}) ([0-9]{2}):([0-9]{2}):([0-9]{2}) ([0-9]{4})', value)
    if not match:
        raise ValueError('E0_INVALID_PBS_TIMESTAMP')
    weekday, month, day, hour, minute, second, year = match.groups()
    # Do not inherit the login host's timezone or locale.
    months = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()
    parsed = datetime(int(year), months.index(month)+1, int(day), int(hour),
                      int(minute), int(second), tzinfo=timezone(timedelta(hours=8)))
    if 'Mon Tue Wed Thu Fri Sat Sun'.split()[parsed.weekday()] != weekday:
        raise ValueError('E0_PBS_TIMESTAMP_WEEKDAY_MISMATCH')
    return int(parsed.timestamp())


def inside(path, root):
    path, root = Path(path), Path(root)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise PermissionError('E0_REGISTERED_PATH_REQUIRED')
    return path


def validate_authority(manifest, now):
    if (manifest.get('schema') != 'RUN_MANIFEST_V1' or manifest.get('run_uuid') != RUN
            or manifest.get('owner') != 'EXISTING_LOCAL_PC' or manifest.get('pi') != 'PI0294'
            or manifest.get('authority', {}).get('protocol_ref') != AUTHORITY
            or manifest['authority'].get('approval') != 'DIRECT_HUMAN_E0_ADOPTION'
            or manifest.get('limits') != LIMITS
            or manifest.get('scope') != 'M2048_V30_EEF10_SOURCE_D0_ORIGINAL_S1_INFERENCE_ONLY'
            or not T0 <= now < DEADLINE):
        raise PermissionError('E0_EXACT_ADOPTION_SCOPE_CLOCK_AND_LIMITS_REQUIRED')
    if (manifest['data']['manifest_sha256'] != DATA
            or manifest['interfaces']['statistics_sha256'] != STATS):
        raise PermissionError('E0_NO_DATA_STATS_OR_SCHEMA_SWITCH')


def validate_entry(root, *, profile, manifest_path=None, now=None, closeout=False,
                   hostname=None, uid=None, verify_source=True):
    root = Path(root)
    manifest = read(inside(manifest_path or root/'RUN_MANIFEST.json', root))
    validate_authority(manifest, time.time() if now is None else now)
    host = socket.gethostname() if hostname is None else hostname
    owner_uid = os.getuid() if uid is None else uid
    execution = manifest['execution']
    compute = profile == 'hopper-neural' and os.environ.get('PBS_JOBID') and host.startswith('hopper-')
    if (root.name != RUN or execution['profile'] != profile or execution['uid'] != owner_uid
            or not (host in execution['hostnames'] or compute)):
        raise PermissionError('E0_EXACT_HOST_ROLE_AND_OWNER')
    binding_path = inside(root/'OUTPUT_BINDING.private.json', root)
    binding = read(binding_path)
    if sha(binding_path) != execution['output_binding_sha256']:
        raise PermissionError('E0_OUTPUT_BINDING_CHANGED')
    parent = Path(binding['registered_parent'])
    inside(root, parent)
    if (root.is_symlink() or parent.is_symlink() or root.stat().st_dev != parent.stat().st_dev
            or Path(binding['run_root']).resolve() != root.resolve()
            or binding['role'] != execution['private_path_role']):
        raise PermissionError('E0_REGISTERED_NODE_LOCAL_OUTPUT_ONLY')
    registry, lease = read(binding['registry']), read(binding['lease'])
    if not any(r.get('role') == binding['role'] and Path(r['path']).resolve() == root.resolve()
               and r.get('executor_node') in (host, 'hopper') for r in registry['entries']):
        raise PermissionError('E0_ACTUAL_REGISTRY_REQUIRED')
    if (lease.get('run_uuid') != RUN or lease.get('owner') != 'EXISTING_LOCAL_PC'
            or lease.get('status') != 'ACTIVE' or lease.get('protocol_ref') != AUTHORITY
            or lease.get('deadline_epoch') != DEADLINE):
        raise PermissionError('E0_ACTUAL_ACTIVE_LEASE_REQUIRED')
    adoption = read(root/'E0_ADOPTION_AND_BINDING.json')
    if (adoption['release'] != AUTHORITY or adoption['T_adopt_epoch'] != T0
            or adoption['user_explicit_adoption'] is not True):
        raise PermissionError('E0_SINGLE_HUMAN_ADOPTION_REQUIRED')
    if sha(root/'evidence/S1_TEACHER_QUALIFIED.json') != S1_PROOF:
        raise PermissionError('E0_ORIGINAL_S1_QUALIFICATION_REQUIRED')
    if verify_source:
        code = inside(root/manifest['code']['capsule_relative'], root)
        seal_path = inside(root/manifest['code']['source_seal_relative'], root)
        seal = read(seal_path)
        if (Path(__file__).resolve() != (code/'scripts/pi0294/e0_scope.py').resolve()
                or sha(seal_path) != manifest['code']['source_seal_sha256']
                or seal['source_sha'] != manifest['code']['vla_commit']
                or seal['protocol_release'] != AUTHORITY):
            raise PermissionError('E0_EXECUTE_EXACT_SEALED_CAPSULE')
        for name, digest in seal['files'].items():
            if sha(inside(code/name, code)) != digest:
                raise PermissionError('E0_SOURCE_CHANGED:'+name)
        for name, digest in manifest['code']['unchanged_numerical_files'].items():
            if seal['files'].get(name) != digest:
                raise PermissionError('E0_D0_S1_OR_NUMERICAL_DELTA:'+name)
    return manifest


def inference_files(identity):
    files = identity.get('files', {})
    required = {'config.json', 'model.safetensors.index.json', 'statistics.json',
        'processor/embodiment_id.json', 'processor/processor_config.json', 'processor/statistics.json'}
    allowed = required | {f'model-{i:05d}-of-00003.safetensors' for i in range(1, 4)}
    if (identity.get('global_step') != 2048 or set(files) != allowed
            or identity.get('statistics_sha256') != STATS):
        raise PermissionError('E0_EXACT_INFERENCE_ONLY_M2048_MEMBERS')
    return files


def verify_inputs(root, bindings):
    validate_entry(root, profile='hopper-neural')
    if sha(Path(bindings['dataset'])/'statistics.json') != STATS:
        raise PermissionError('E0_ORIGINAL_V30_STATISTICS_REQUIRED')
    if (bindings['source_commit'] != '51d4c89f72fda44cbf77285c6a8114b52676b8a1'
            or bindings['cosmos_revision'] != '9ce19a195e423419c349abfc86fd07178b230561'):
        raise PermissionError('E0_OFFICIAL_SOURCE_AND_COSMOS_PINS')


def admit(root, request_path):
    from scripts.pi0292.allocation_admission import owned_jobs, personal_quota
    root, request_path = Path(root), Path(request_path)
    manifest = validate_entry(root, profile='hopper-neural')
    adopted = read(root/'E0_ADOPTION_AND_BINDING.json')
    inputs = read(root/'evidence/E0_INPUT_PREPARATION_STATE.json')
    if (adopted.get('launch_allowed') is not True
            or inputs.get('status') != 'PASS_E0_INFERENCE_MEMBERS_ACTUAL_SHA_CHECKED_NO_MODEL_LOAD'):
        raise PermissionError('E0_ACTUAL_INPUTS_AND_FINAL_BINDING_REQUIRED_BEFORE_PBS_MODEL')
    request = read(inside(request_path, root))
    if (request['mode'] != 'resident' or request['stage'] != 'resident'
            or request['allowed_modes'] != ['resident'] or request['global_step'] != 2048
            or request['source_sha'] != manifest['code']['vla_commit']
            or request['run_manifest_sha256'] != sha(root/'RUN_MANIFEST.json')
            or request['bindings_sha256'] != sha(request['bindings'])
            or request['numerical_mode'] != 'DETERMINISTIC_CUDA_MATH_SDP_V1'
            or request['max_queries'] > 1056 or not 1 <= request['submit_count'] <= 2
            or request['walltime_s'] > min(14400 if request['submit_count'] == 1 else 21600,
                21600-request['prior_GPU_allocated_s'])):
        raise PermissionError('E0_RESIDENT_ONLY_NO_TRAINING_OR_EXTRA_RESOURCE')
    job_id, user, now = os.environ.get('PBS_JOBID'), os.environ['USER'], time.time()
    if not job_id:
        raise PermissionError('E0_NO_NEURAL_WITHOUT_PBS')
    jobs = owned_jobs(user)
    job = jobs.get(job_id)
    if (not job or job['job_state'] != 'R' or job.get('project') != 'CFP04-CF-007'
            or job['Job_Owner'].split('@')[0] != user
            or int(job['Resource_List']['ngpus']) != 1):
        raise PermissionError('E0_OWN_RUNNING_PROJECT_SINGLE_GPU_REQUIRED')
    nodes = {n.split('/')[0].split('.')[0] for n in job['exec_host'].split('+')}
    if socket.gethostname().split('.')[0] not in nodes:
        raise PermissionError('E0_NO_LOGIN_OR_FOREIGN_NODE_INFERENCE')
    wall = sum(int(x)*m for x,m in zip(job['Resource_List']['walltime'].split(':'), (3600,60,1)))
    started = pbs_epoch(job['stime'])
    if wall != request['walltime_s'] or not T0 <= started < ADMISSION_END:
        raise PermissionError('E0_ALLOCATION_ADMISSION_WINDOW_OR_WALLTIME')
    other_e0 = [k for k,v in jobs.items() if k != job_id and v.get('Job_Name','').startswith('pi0294_e0')]
    if other_e0:
        raise PermissionError('E0_ONE_ALLOCATION_ONLY')
    proof_path = inside(root/request['project_admission_relative'], root)
    if sha(proof_path) != request['project_admission_sha256']:
        raise PermissionError('E0_PROJECT_PROOF_CHANGED')
    proof = read(proof_path)
    project = proof['project']
    today = datetime.fromtimestamp(now, timezone(timedelta(hours=8))).date()
    if (proof['run_uuid'] != RUN or not 0 <= now-proof['observed_epoch'] <= 1800
            or project['name'] != 'CFP04-CF-007' or project['active'] is not True
            or user not in project['users'] or not date.fromisoformat(project['start_date']) <= today <= date.fromisoformat(project['end_date'])
            or not proof['balance_su'] > 0):
        raise PermissionError('E0_FRESH_PROJECT_MEMBERSHIP_BALANCE_REQUIRED')
    quota = personal_quota(user)
    if quota['hard_bytes'] <= quota['used_bytes'] or not 0 <= now-quota['report_epoch'] <= 900:
        raise PermissionError('E0_FRESH_PERSONAL_QUOTA_REQUIRED')
    end = min(started+wall, DEADLINE)
    result = dict(status='PASS_E0_OWN_PBS_INFERENCE_ONLY', job_id=job_id,
        observed_epoch=now, allocation_start_epoch=started, allocation_end_epoch=end,
        hostname=socket.gethostname(), project='CFP04-CF-007', allowed_modes=['resident'],
        request_sha256=sha(request_path))
    write(root/'NEURAL_AUTHORIZATION.json', result)
    write(request_path.parent/'ALLOCATION_ADMISSION.json', result)
    return result

"""Explicit eval-first authority; no rewrite of the closed parent run."""
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import shutil
import socket
import time
import uuid

from scripts.pi0292.durable import read
from .native_dataset import sha

RUN = '20261014T025730Z-localpc-pi0294-r1'
PARENT = '20261013T142457Z-localpc-pi0294'
AUTHORITY = 'a14ea2559d35bf05960e25bb653dcf09198bad08'
GO_COMMENT = 6159886188
T0, CUTOFF, DEADLINE = 1791946650, 1791979050, 1791982650
DATA = 'b8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df'
STATS = 'fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7'
PROJECT = 'CFP04-CF-007'
GIB = 1024**3
LIMITS = dict(T0_epoch=T0, new_work_cutoff_epoch=CUTOFF, deadline_epoch=DEADLINE,
              reset_on_resume=False, PBS_calls=3, GPU_allocated_s=28800,
              new_formal_updates=3584, new_engineering_updates=4,
              generation_queries=5200, physics_starts=96, physics_steps=1536000,
              rgb_pairs=61536, CoreLab_CPU_s=144000, CoreLab_cores=8,
              Hopper_CPU_s=57600, Hopper_new_peak_bytes=120*GIB,
              CoreLab_new_peak_bytes=32*GIB, local_pc_new_peak_bytes=GIB,
              public_checkpoint_count=2, public_checkpoint_bytes=60*GIB)
NUMERICAL_PATHS = (
    'scripts/pi0294/neural.py', 'scripts/pi0294/native_train.py',
    'scripts/pi0294/native_dataset.py', 'scripts/pi0294/sampling.py',
    'scripts/pi0294/numerics.py', 'scripts/pi0294/local_backbone.py',
    'scripts/pi0294/episode.py', 'scripts/pi0294/scene.py',
    'scripts/pi0294/scene_cards.py', 'scripts/pi0294/teacher.py',
    'scripts/pi0294/score.py', 'scripts/pi0294/relay_teacher.py',
    'scripts/pi0292/physics_relay.py', 'scripts/pi0292/relay_protocol.py',
    'scripts/pi0292/live_inputs.py', 'scripts/pi0292/eval_contract.py',
    'scripts/pi0292/train_adapter.py', 'scripts/pi0291_arx5_data/config.py',
)


def within(path, root):
    path, root = Path(path), Path(root)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise PermissionError('CONTINUATION_PATH_ESCAPE_OR_SYMLINK')
    return path


def validate_authority(manifest, now, *, closeout=False):
    authority = manifest.get('authority', {})
    if (manifest.get('schema') != 'RUN_MANIFEST_V1' or manifest.get('run_uuid') != RUN
            or manifest.get('pi') != 'PI0294' or manifest.get('owner') != 'EXISTING_LOCAL_PC'
            or authority.get('protocol_ref') != AUTHORITY
            or authority.get('go_comment') != GO_COMMENT
            or authority.get('approval') != 'USER_CHAT_COMPLETE_SECTION5_PACKAGE'
            or manifest.get('parent_run_uuid') != PARENT
            or manifest.get('limits') != LIMITS):
        raise PermissionError('EXACT_NEW_HUMAN_GO_PARENT_AND_INCREMENTAL_LIMITS_REQUIRED')
    if not T0 <= now < (DEADLINE if closeout else CUTOFF):
        raise PermissionError('CONTINUATION_WINDOW_CLOSED_NO_RESET')


def validate_entry(root, *, profile, manifest_path=None, now=None, closeout=False,
                   hostname=None, uid=None, verify_source=True):
    root = Path(root)
    m = read(within(manifest_path or root/'RUN_MANIFEST.json', root))
    validate_authority(m, time.time() if now is None else now, closeout=closeout)
    host = socket.gethostname() if hostname is None else hostname
    uid = os.getuid() if uid is None else uid
    e = m['execution']
    if root.name != RUN or e['profile'] != profile or e['uid'] != uid or host not in e['hostnames']:
        raise PermissionError('CONTINUATION_EXACT_HOST_PROFILE_OWNER')
    p = within(root/'OUTPUT_BINDING.private.json', root)
    b = read(p)
    parent = Path(b['registered_parent'])
    if (sha(p) != e['output_binding_sha256'] or b['run_uuid'] != RUN
            or Path(b['run_root']).resolve() != root.resolve()
            or b['role'] != e['private_path_role'] or root.is_symlink() or parent.is_symlink()
            or root.stat().st_dev != parent.stat().st_dev):
        raise PermissionError('CONTINUATION_EXACT_NODE_LOCAL_OUTPUT_BINDING')
    within(root, parent)
    registry, lease = read(b['registry']), read(b['lease'])
    if not any(r.get('role') == b['role'] and Path(r['path']).resolve() == root.resolve()
               and r.get('executor_node') in (host, 'hopper') for r in registry['entries']):
        raise PermissionError('CONTINUATION_ACTUAL_ROLE_REGISTRATION_REQUIRED')
    if (lease.get('run_uuid') != RUN or lease.get('owner') != 'EXISTING_LOCAL_PC'
            or lease.get('status') != 'ACTIVE' or lease.get('protocol_ref') != AUTHORITY
            or lease.get('deadline_epoch') != DEADLINE):
        raise PermissionError('CONTINUATION_ACTUAL_LEASE_REQUIRED')
    if m['data']['manifest_sha256'] != DATA or m['interfaces']['statistics_sha256'] != STATS:
        raise PermissionError('NO_V31_OR_STATISTICS_SWITCH')
    if not set(NUMERICAL_PATHS).issubset(m['code']['unchanged_numerical_files']):
        raise PermissionError('COMPLETE_NUMERICAL_DELTA_PROOF_REQUIRED')
    if verify_source:
        seal_path = within(root/m['code']['source_seal_relative'], root)
        seal, code = read(seal_path), within(root/m['code']['capsule_relative'], root)
        if (sha(seal_path) != m['code']['source_seal_sha256']
                or seal['source_sha'] != m['code']['vla_commit']
                or seal['protocol_release'] != AUTHORITY):
            raise PermissionError('CONTINUATION_SOURCE_SEAL_REQUIRED')
        for n, digest in seal['files'].items():
            if sha(within(code/n, code)) != digest:
                raise PermissionError('CONTINUATION_SEALED_SOURCE_CHANGED:'+n)
        for n, digest in m['code']['unchanged_numerical_files'].items():
            if seal['files'].get(n) != digest:
                raise PermissionError('NUMERICAL_PATH_NOT_PINNED_TO_PARENT:'+n)
    return m


def validate_allocation(*, now, user, job_id, jobs, quota, project_receipt,
                        request, hostname, initial=None):
    if not T0 <= now < CUTOFF:
        raise PermissionError('CONTINUATION_SCIENCE_WINDOW_CLOSED')
    if request['stage'] not in ('continuation', 'resident', 'main'):
        raise PermissionError('NO_OLD_SMOKE_BENCHMARK_ARCHIVE_OR_M6144_STAGE')
    if (type(request['submit_count']) is not int or not 1 <= request['submit_count'] <= 3
            or type(request['walltime_s']) is not int or not 0 < request['walltime_s'] <= 21600
            or request['prior_GPU_allocated_s'] < 0
            or request['prior_GPU_allocated_s']+request['walltime_s'] > 28800
            or request['numerical_mode'] != 'DETERMINISTIC_CUDA_MATH_SDP_V1'
            or request['microbatch'] != 2 or request['workers'] != 0):
        raise PermissionError('NEW_INCREMENTAL_ALLOCATION_OR_FROZEN_PROFILE_LIMIT')
    nonce = uuid.UUID(request['request_nonce'])
    if nonce.version != 4 or str(nonce) != request['request_nonce']:
        raise PermissionError('CONTINUATION_EXACT_NONCE')
    if request['stage'] == 'main' and (request['max_steps'] != 4096
            or request['stop_at'] not in (2048, 4096) or not request.get('resume')):
        raise PermissionError('ORIGINAL_FULL_STATE_RESUME_ONLY_NO_PRISTINE_OR_EXTENSION')
    job = jobs.get(job_id)
    if (job is None or job['job_state'] != 'R' or job['Job_Owner'].split('@')[0] != user
            or job.get('project') != PROJECT):
        raise PermissionError('CONTINUATION_OWN_RUNNING_PROJECT_JOB')
    r = job['Resource_List']
    wall = sum(int(v)*m for v,m in zip(r['walltime'].split(':'), (3600,60,1)))
    if int(r['ngpus']) != 1 or wall != request['walltime_s']:
        raise PermissionError('EXACT_SINGLE_GPU_WALLTIME')
    nodes = {p.split('/')[0].split('.')[0] for p in job['exec_host'].split('+')}
    if hostname.split('.')[0] not in nodes:
        raise PermissionError('NO_NEURAL_ON_LOGIN_OR_FOREIGN_NODE')
    peers = {k:v for k,v in jobs.items() if k != job_id and int(v.get('Resource_List',{}).get('ngpus',0)) > 0}
    if len(peers) > 1 or not set(peers).issubset(request['allowed_peer_ids']):
        raise PermissionError('MAX_TWO_AUTHORIZED_INDEPENDENT_OWN_ALLOCATIONS')
    if any(v['Job_Owner'].split('@')[0] != user or int(v['Resource_List']['ngpus']) != 1 for v in peers.values()):
        raise PermissionError('OWN_SINGLE_GPU_PEER_REQUIRED')
    project = project_receipt['project']
    day = datetime.fromtimestamp(now, timezone(timedelta(hours=8))).date()
    attestation_epoch = now if initial is None else initial['observed_epoch']
    if (project['name'] != PROJECT or project['active'] is not True or user not in project['users']
            or 'hopper-m-02' not in project['cluster']
            or not date.fromisoformat(project['start_date']) <= day <= date.fromisoformat(project['end_date'])
            or not 0 <= attestation_epoch-project_receipt['observed_epoch'] <= 1800
            or not 0 < project_receipt['balance_su'] < float('inf')):
        raise PermissionError('INITIAL_NATIVE_PROJECT_MEMBERSHIP_AND_BALANCE')
    remaining = request['predicted_remaining_peak_bytes']
    if (type(remaining) is not int or not 0 < remaining <= 120*GIB
            or quota['user'] != user or not 0 <= now-quota['report_epoch'] <= 900
            or quota['hard_bytes']-quota['used_bytes'] < remaining+20*GIB):
        raise PermissionError('FRESH_PERSONAL_QUOTA_NOT_SHARED_DF_REQUIRED')
    return dict(status='PASS_PI0294_R1_OWN_NATIVE_PBS_ADMISSION_NOT_MODEL_QUALIFICATION',
                job_id=job_id,hostname=hostname,project=PROJECT,observed_epoch=now,
                project_proof_mode='INITIAL_NATIVE_PROJECT_PROOF' if initial is None else 'SAME_LIVE_PARENT_INITIAL_PROJECT_PROOF')


def admit(root, request_path):
    from scripts.pi0292.allocation_admission import owned_jobs, personal_quota
    root, request_path = Path(root), Path(request_path)
    m = validate_entry(root, profile='hopper-neural')
    request = read(within(request_path, root))
    if (request['source_sha'] != m['code']['vla_commit']
            or request['run_manifest_sha256'] != sha(root/'RUN_MANIFEST.json')
            or request['code_seal_sha256'] != m['code']['source_seal_sha256']
            or sha(request['bindings']) != request['bindings_sha256']):
        raise PermissionError('CONTINUATION_REQUEST_SOURCE_OR_BINDING_CHANGED')
    project_path = request_path.parent/'PROJECT_ADMISSION.private.json'
    if sha(project_path) != request['project_admission_sha256']:
        raise PermissionError('CONTINUATION_INITIAL_PROJECT_PROOF_CHANGED')
    project = read(project_path)
    if project['run_uuid'] != RUN or project['request_nonce'] != request['request_nonce']:
        raise PermissionError('CONTINUATION_PROJECT_NONCE_AND_RUN_REQUIRED')
    job_id = os.environ.get('PBS_JOBID')
    if not job_id:
        raise PermissionError('NO_NEURAL_WITHOUT_PBS')
    initial = None
    if request['stage'] != 'continuation':
        parent = within(root/request['parent_request_relative'], root)
        if sha(parent) != request['parent_request_sha256'] or parent == request_path:
            raise PermissionError('EXACT_DISTINCT_PARENT_REQUEST_REQUIRED')
        original = read(parent)
        for k in ('source_sha','run_manifest_sha256','code_seal_sha256','bindings','bindings_sha256',
                  'request_nonce','project_admission_sha256','walltime_s','submit_count','prior_GPU_allocated_s',
                  'numerical_mode','microbatch','workers','allowed_peer_ids'):
            if request[k] != original[k]:
                raise PermissionError('CONTINUATION_PARENT_IDENTITY_CHANGED:'+k)
        initial = read(parent.parent/'ALLOCATION_ADMISSION.json')
        state = read(parent.parent/'continuation-session/STATE.json')
        if (initial['request_sha256'] != sha(parent) or initial['job_id'] != job_id
                or initial['status'] != 'PASS_PI0294_R1_OWN_NATIVE_PBS_ADMISSION_NOT_MODEL_QUALIFICATION'
                or initial['project_attestation_sha256'] != request['project_admission_sha256']
                or state['job_id'] != job_id or state['run_uuid'] != RUN
                or state['parent_request_sha256'] != sha(parent)
                or state.get('active_request_sha256') != sha(request_path)
                or state['status'] != 'RUNNING_'+request['stage'].upper()
                or not initial['observed_epoch'] <= time.time() < initial['allocation_end_epoch']):
            raise PermissionError('EXACT_ACTUAL_LIVE_PARENT_SUBSTAGE_REQUIRED')
        proc = Path('/proc')/str(state['pid'])
        fields = (proc/'stat').read_text().split()
        command = (proc/'cmdline').read_bytes().split(b'\0')
        if (proc.stat().st_uid != os.getuid() or fields[21] != state['proc_start']
                or fields[2] in ('Z','X') or b'scripts.pi0294.continuation_session' not in command
                or str(parent).encode() not in command):
            raise PermissionError('CONTINUATION_PARENT_PROCESS_NOT_LIVE')
    ledger = read(root/'HOPPER_RESOURCE_LEDGER.json')
    owned = sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    total = request['predicted_total_owned_peak_bytes']
    remaining = max(1, total-owned)
    if (not 0 < total <= 120*GIB or owned > total
            or shutil.disk_usage(root).free < remaining+20*GIB
            or not 512 <= ledger['main_updates'] <= 4096 or ledger['pending_updates']
            or len(ledger['PBS_submissions']) != request['submit_count']
            or ledger['GPU_allocated_s']+ledger['GPU_reserved_s'] > 28800):
        raise PermissionError('CONTINUATION_LIVE_LEDGER_AND_UNION_PEAK_REQUIRED')
    bound = dict(request, predicted_remaining_peak_bytes=remaining)
    jobs = owned_jobs(os.environ['USER'])
    result = validate_allocation(now=time.time(),user=os.environ['USER'],job_id=job_id,jobs=jobs,
        quota=personal_quota(os.environ['USER']),project_receipt=project,request=bound,
        hostname=socket.gethostname(),initial=initial)
    elapsed = jobs[job_id].get('resources_used',{}).get('walltime','00:00:00')
    elapsed_s = sum(int(v)*m for v,m in zip(elapsed.split(':'),(3600,60,1)))
    result.update(allocation_end_epoch=min(CUTOFF,time.time()+request['walltime_s']-elapsed_s),
                  request_sha256=sha(request_path),project_attestation_sha256=request['project_admission_sha256'],
                  current_owned_bytes=owned,predicted_remaining_peak_bytes=remaining)
    if initial is not None:
        result['allocation_end_epoch'] = min(result['allocation_end_epoch'],initial['allocation_end_epoch'])
    return result

"""Explicit C/D identity on the existing native Core process/device reader.

Pure request/receipt checks here do not start a process or grant model safety.
The original reader still observes the real unit, cgroup, lock and nvidia-smi.
"""
import re
from . import mix_core_accounting as c, mix_accounting as a, successor_cd as s

STATUS = 'PASS_V5_CD_OWN_NATIVE_CORE_ALLOCATION'


def need(ok, message):
    if not ok:
        raise PermissionError('V5_CD_CORE_IDENTITY_' + message)


def typed(value):
    return value.get('platform_evaluation_version') == c.VERSION or value.get('scope') == c.ROLE


def cold_profile(m):
    need(m.get('platform_evaluation_version') == c.VERSION and m.get('scope') == c.ROLE
         and m.get('execution', {}).get('profile') == c.PROFILE
         and 'platform_training_version' not in m, 'COMPLETE_NEW_NEURAL_ROLE')
    return True


def request_identity(m, r, manifest_sha256):
    cold_profile(m)
    need(m.get('schema') == 'PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1'
         and m.get('owner') == 'EXISTING_LOCAL_PC' and m.get('owner_user') == 'liuyu-nfs'
         and m.get('owner_uid') == 2009 and m['execution'].get('uid') == 2009
         and m['execution'].get('host') == 'corelab-4080s'
         and m['authority'].get('protocol_ref') == s.TASK_SOURCE
         and m['authority'].get('scope_grant_sha256') == s.GRANT_SHA,
         'EXISTING_STANDARD_MANIFEST_WITH_SUCCESSOR_AUTHORITY')
    need(r.get('schema') == 'V5_CD_CORE_NATIVE_GPU_REQUEST_V1'
         and r.get('platform_evaluation_version') == c.VERSION
         and r.get('native_platform') == 'CORELAB_RTX4080'
         and r.get('task') == s.TASK and r.get('authority_release') == s.TASK_SOURCE
         and r.get('scope_grant_sha256') == s.GRANT_SHA
         and r.get('run_uuid') == m.get('run_uuid')
         and r.get('run_manifest_sha256') == manifest_sha256
         and r.get('source_sha') == m.get('code', {}).get('vla_commit')
         and c.canonical_uuid(r.get('request_id')) and c.canonical_uuid(r.get('submission_nonce')),
         'SAME_EXACT_REGISTERED_NEW_REQUEST')
    forbidden = ('project', 'submit_count', 'job_name', 'PBS_JOBID', 'platform_training_version',
        'training_block', 'request_sha256', 'native_handle', 'status', 'charged_effects',
        'reserved_neural_queries', 'shared_GPU_grant_sha256', 'startup_recovery_sha256',
        'original_T0_epoch', 'hard_close_epoch')
    need(not any(n in r for n in forbidden) and not any(n.startswith('queue_') for n in r),
         'NO_LEDGER_AS_REQUEST_OR_OLD_AB_WINDOW_SHARING_EXCEPTION')
    arm = r.get('evaluation_candidate'); c.key(arm)
    expected = dict(stage='V5_MATCHED_DEV24', noise_bank=294101, remaining_conditions=24,
        checkpoint_step=1024, walltime_s=28800, CPU_upper_s=80000,
        declared_process_CPU_limit_s=79760, active_cpu_limit=4, requested_memory_bytes=48*1024**3,
        minimum_GPU_memory_bytes=30*1024**3, optimizer_updates=0, Teacher_takeover=False,
        normalizer_refit=False, evaluation_data_for_training=False, LOCKBOX_opened=False,
        numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1', statistics_sha256=a.STATS,
        parent_checkpoint_manifest_sha256=a.PARENT)
    need(all(r.get(n) == v and type(r[n]) is type(v) for n, v in expected.items()),
         'FROZEN_ORIGINAL_24_CONDITIONS_AND_ZERO_TRAINING')
    need(type(r.get('allocation_number')) is int and r['allocation_number'] == (1 if arm == 'C50' else 2)
         and r.get('native_unit') == r['run_uuid']+'-neural.service', 'ORIGINAL_NATIVE_UNIT_AND_EXPLICIT_CD_ORDINAL')
    for name, length in (('source_sha', 40), ('checkpoint_manifest_sha256', 64),
            ('training_handoff_sha256', 64), ('planned_panel_sha256', 64), ('cold_input_audit_sha256', 64)):
        need(type(r.get(name)) is str and re.fullmatch('[0-9a-f]{'+str(length)+'}', r[name]),
             'FULL_COLD_IDENTITY:' + name)
    # The marker's complete actual bytes are checked in the registered context.
    # This pure request check only rejects attempts to reset its two clocks.
    marker = s.scope(r.get('T_CD'))
    need(r.get('deadline_epoch') == marker['deadline_epoch']
         and m['limits'].get('T_CD') == marker['T_CD']
         and m['limits'].get('operation_deadline_epoch') == marker['deadline_epoch'],
         'SAME_ADOPTED_WINDOW_NOT_AN_OLD_PILOT_CLOCK')
    return r


def check_clock(r, *, started_epoch, now_epoch):
    start = a.number(started_epoch); now = a.number(now_epoch)
    created = a.number(r.get('created_epoch'))
    need(type(r.get('T_CD')) in (int, float) and
         r['deadline_epoch'] == s.scope(r['T_CD'])['deadline_epoch']
         and r['T_CD'] <= created and r['T_CD'] <= start and created-1 <= start
         and start <= now < r['deadline_epoch']
         and start < r['deadline_epoch']-28860,
         'REAL_PROCESS_START_INSIDE_COMPLETE_SUCCESSOR_WINDOW')


def sealed_identity(m, r, allocation, *, run_uuid, manifest_sha256, request_sha256):
    from . import evaluation_core_identity as original
    request_identity(m, r, manifest_sha256)
    p = allocation.get('native_process')
    need(type(p) is dict, 'REAL_NATIVE_PROCESS_REQUIRED')
    handle = c.process_handle(p.get('boot_id'), p.get('pid'), p.get('proc_start'))
    expected = dict(status=STATUS, platform_evaluation_version=c.VERSION,
        run_uuid=run_uuid, manifest_sha256=manifest_sha256, request_sha256=request_sha256,
        job_id=handle, native_handle=handle, model_loaded=False, new_optimizer_updates=0)
    need(m['run_uuid'] == r['run_uuid'] == run_uuid and
         all(allocation.get(n) == v and type(allocation[n]) is type(v) for n, v in expected.items())
         and not any(n in allocation for n in ('project','PBS_JOBID','submit_count',
             'platform_training_version','shared_GPU_grant_sha256','original_T0_epoch','hard_close_epoch')),
         'EXACT_ADMISSION_REQUEST_AND_NO_MODEL_PERMISSION')
    for name in ('source_sha', 'request_id', 'submission_nonce', 'stage', 'evaluation_candidate',
            'checkpoint_manifest_sha256', 'statistics_sha256', 'native_platform', 'allocation_number'):
        need(allocation.get(name) == r[name] and type(allocation[name]) is type(r[name]), 'SAME_BOUND_REQUEST:' + name)
    exact = dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_STARTED_V1', hostname='corelab-4080s', uid=2009,
        GPU_uuid=r['GPU_uuid'], request_id=r['request_id'], request_sha256=request_sha256,
        submission_nonce=r['submission_nonce'], run_manifest_sha256=manifest_sha256, source_sha=r['source_sha'],
        process_state='RUNNING', native_handle=handle, native_model_permission=False, physics_authorized=False)
    need(all(p.get(n) == v and type(p[n]) is type(v) for n, v in exact.items())
         and allocation.get('native_process_sha256') == c.digest(p)
         and allocation.get('allocation_start_epoch') == p['started_epoch']
         and allocation.get('allocation_end_epoch') == min(p['started_epoch']+28800, r['deadline_epoch'])
         and allocation.get('hostname') == p['hostname']
         and 0 <= p['observed_epoch']-p['started_epoch'] <= 120,
         'SAME_RAW_BOOT_PID_START_NOT_A_PREPARED_HANDLE')
    check_clock(r, started_epoch=p['started_epoch'], now_epoch=p['observed_epoch'])
    limits = p['kernel_limits']
    original.kernel_limits(str(limits['CPU_quota'])+' '+str(limits['CPU_period']),
        str(limits['memory_max_bytes']), str(limits['pids_max']))
    props = p['native_unit_properties']
    need(props.get('MainPID') == str(p['pid']) and props.get('ActiveState') == 'active'
         and props.get('RuntimeMaxUSec') == '8h' and bool(props.get('ControlGroup')), 'REAL_RAW_UNIT_LIMITS')
    raw = p['native_GPU_raw']
    need(set(raw) == {'devices', 'processes'} and all(type(v.get('returncode')) is int
         and v['returncode'] == 0 for v in raw.values()), 'SUCCESSFUL_COMPLETE_NATIVE_GPU_READS')
    need(original.parse_GPU_rows(raw['devices']['stdout'], raw['processes']['stdout'],
         GPU_uuid=r['GPU_uuid']) == p['device'], 'EXCLUSIVE_DEVICE_RAW_READ_MATCH_NO_OLD_SHARED_GRANT')
    return allocation


def require_device(request, allocation, *, visible, count, name, memory_bytes):
    r, v = request, allocation
    need(r.get('platform_evaluation_version') == v.get('platform_evaluation_version') == c.VERSION
         and v.get('status') == STATUS and r.get('stage') == v.get('stage') == 'V5_MATCHED_DEV24'
         and r.get('evaluation_candidate') in ('C50', 'D25')
         and v.get('evaluation_candidate') == r['evaluation_candidate']
         and r.get('checkpoint_step') == 1024
         and r.get('checkpoint_manifest_sha256') == v.get('checkpoint_manifest_sha256')
         and r.get('statistics_sha256') == v.get('statistics_sha256') == a.STATS
         and type(r.get('optimizer_updates')) is int and r['optimizer_updates'] == 0
         and visible == r.get('GPU_uuid') and type(count) is int and count == 1
         and name == 'NVIDIA GeForce RTX 4080 SUPER' and type(memory_bytes) is int and memory_bytes >= 30*1024**3
         and not any(n in obj for obj in (r, v) for n in
             ('platform_training_version', 'training_block', 'shared_GPU_grant_sha256')),
         'SINGLE_REAL_QUALIFIED_CD_DEVICE_NOT_A40_OLD_AB_OR_SHARED_B24')
    return True

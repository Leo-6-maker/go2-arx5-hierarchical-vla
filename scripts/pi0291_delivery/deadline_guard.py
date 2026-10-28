"""Bound delivery to original limits or an explicitly pinned CPU-only closeout."""
import hashlib
import json
import os
from pathlib import Path
import signal
import resource
import math
import time

DEADLINE = 1791851837.0
EXTERNAL_OUTPUT_UPPER_BYTES = 256 * 1024**2


def resource_bounds(root):
    root = Path(root)
    ledger = json.loads((root/'RESOURCE_ACCOUNTING.json').read_text())
    remaining = 48*3600-ledger['CPU_s']-ledger['unmetered_dependency_setup_CPU_s_upper']
    expected = os.environ.get('PI0291_CLOSEOUT_AUTHORIZATION_SHA256')
    if not expected:
        return DEADLINE, remaining
    raw = (root/'evidence/DATA_CLOSEOUT_AUTHORIZATION.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, 'Closeout authority bytes changed'
    authority = json.loads(raw)
    assert authority['run_uuid'] == root.name
    assert authority['source_sha'] == '57a02b4923e5001df5fd4b3b5958c372b9cff541'
    assert authority['original_deadline_epoch'] == DEADLINE
    assert authority['human_authorized'] and authority['physics_model_PBS_WBC_deletion_allowed'] is False
    assert authority['deadline_epoch']-authority['T0_epoch'] == 3600
    assert authority['new_CPU_budget_s'] == 14400 and authority['total_output_limit_bytes'] == 40*1024**3
    assert ledger['CPU_s'] >= authority['CPU_ledger_baseline_s']
    fresh_remaining = authority['new_CPU_budget_s']-(ledger['CPU_s']-authority['CPU_ledger_baseline_s'])-authority['auxiliary_unmetered_CPU_upper_s']
    return authority['deadline_epoch'], min(remaining,fresh_remaining)


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending-' + str(os.getpid()))
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def install_deadline(root, stage):
    deadline, cpu_remaining = resource_bounds(root)
    if os.environ.get('PI0291_CLOSEOUT_AUTHORIZATION_SHA256'):
        assert stage in ('audit_data','owner_final_raw_binding','export_dataset','qualify_loader','archive_transfer','readback_transfer')
    def expired(signum, frame):
        atomic_write(Path(root)/'evidence'/('DELIVERY_TIMEOUT_' + stage + '.json'),
            dict(status='STOP_BOUND_DELIVERY_DEADLINE', stage=stage, deadline_epoch=deadline,
                 observed_epoch=time.time(), remote_operation_result='UNKNOWN_IF_IN_FLIGHT',
                 recovery='Inspect atomic TRANSFER_PROGRESS and exact remote prefix; do not blindly reupload',
                 other_processes_signalled=0))
        os._exit(124)
    signal.signal(signal.SIGALRM, expired)
    remaining = deadline - time.time()
    assert remaining > 0, 'Bound delivery deadline expired'
    signal.setitimer(signal.ITIMER_REAL, remaining)
    ledger_path = Path(root)/'RESOURCE_ACCOUNTING.json'
    if ledger_path.is_file():
        assert cpu_remaining > 1, 'Original cumulative CPU budget exhausted'
        own = resource.getrusage(resource.RUSAGE_SELF)
        limit = math.ceil(own.ru_utime+own.ru_stime+cpu_remaining)
        def cpu_expired(signum, frame):
            atomic_write(Path(root)/'evidence'/('DELIVERY_CPU_STOP_'+stage+'.json'),
                dict(status='STOP_ORIGINAL_CPU_BUDGET',stage=stage,process_CPU_limit_s=limit,
                     other_processes_signalled=0))
            os._exit(125)
        signal.signal(signal.SIGXCPU,cpu_expired)
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        assert hard == resource.RLIM_INFINITY or hard >= limit
        resource.setrlimit(resource.RLIMIT_CPU,(limit,hard))

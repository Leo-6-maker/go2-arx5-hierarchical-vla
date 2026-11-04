"""Target reference clock contract. No controller, ROS or physics construction."""
import hashlib
import json
import math

SCHEMA = 'WM02_TARGET_REFERENCE_CONSUMPTION_V1'
TASK_START = 2503
ATOM_TICKS = 25
DT = .002


def deadline(source):
    if isinstance(source, bool) or not math.isfinite(source):
        raise ValueError('FINITE_ACTUAL_SOURCE_REQUIRED')
    return source + .4


def require_live_window(source, native_time, ticks):
    if (type(ticks) is not int or ticks <= 0 or ticks % ATOM_TICKS
            or not math.isfinite(native_time) or native_time < source):
        raise ValueError('ACTUAL_SOURCE_AND_COMPLETE_RELATIVE_ATOMS_REQUIRED')
    end = native_time + ticks*DT
    # The receiver expires at >=deadline-1e-10, not 400ms after install.
    if end >= deadline(source)-1e-10:
        raise ValueError('WINDOW_REACHES_REFERENCE_DEADLINE_RENEW_FIRST')
    return dict(source_mpc_s=source, native_start_s=native_time,
                native_end_s=end, deadline_s=deadline(source),
                remaining_at_install_s=deadline(source)-native_time,
                expiry_margin_s=deadline(source)-end)


def require_install_clock(source, native_time):
    deadline(source)
    if not math.isfinite(native_time) or not 0 <= native_time-source <= .025:
        raise ValueError('INSTALL_REQUIRES_FRESH_MEASURED_SOURCE')


def trajectory_identity(times, states, inputs):
    value=dict(times=list(times), poses=[list(x) for x in states],
               inputs=[list(x) for x in inputs])
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def dynamic_limit(phase, previous):
    """Versioned finite diagnostic, never accept arbitrary metadata limits."""
    table = {
        'g2-hold-short': ('g2-reference-qualification', TASK_START+25),
        'g2-hold-renew': ('g2-hold-short', TASK_START+400),
        'g2-eef-x': ('g2-hold-renew', TASK_START+200),
        'g2-eef-y': ('g2-eef-x', TASK_START+200),
    }
    if phase not in table:
        raise PermissionError('UNKNOWN_FINITE_DYNAMIC_PHASE')
    parent, limit = table[phase]
    if (previous is None or previous.get('phase') != parent
            or previous.get('status') != 'NATIVE_STAGE_PASSED'
            or not previous.get('shutdown_clean')
            or not previous.get('owned_process_exits')
            or any(p['exit_code'] != 0 for p in previous['owned_process_exits'])):
        raise PermissionError('QUALIFIED_FINITE_DYNAMIC_PREDECESSOR_REQUIRED')
    return limit

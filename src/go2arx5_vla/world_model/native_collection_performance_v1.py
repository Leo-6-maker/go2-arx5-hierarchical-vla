"""Opt-in timing and synchronization for new construction episodes only.

This does not admit training data or change the native controller. Inclusive
and self times are separate; callback CPU is the calling thread's CPU, not
the simulator/solver processes' CPU. Unmarked capsules retain their waits.
"""
from contextlib import contextmanager, nullcontext
from functools import wraps
import json
import math
import time

VERSION = 'TASK_COLLECTION_PERFORMANCE_CONSTRUCTION_V1'
MODES = ('profile', 'events', 'camera_cache', 'events_camera_cache')


def configuration(spec):
    config = spec.get('collection_performance')
    if config is None:
        return None
    from .task_learning_grasp_v1 import CONSTRUCTION_ROLE
    if not (isinstance(config, dict) and set(config) == {'version', 'mode'}
            and config['version'] == VERSION and config['mode'] in MODES
            and spec.get('phase') == 'collect-target-task-learning-grasp-v1'
            and spec.get('task_learning_role') == CONSTRUCTION_ROLE):
        raise ValueError('PERFORMANCE_FRESH_TASK_CONSTRUCTION_ONLY')
    return config


class PerformanceTrace:
    def __init__(self, *, wall=time.perf_counter_ns, cpu=time.thread_time_ns, sample_limit=4096):
        self.wall, self.cpu = wall, cpu
        self.sample_limit = sample_limit
        self.stack, self.stats, self.events = [], {}, []

    @contextmanager
    def span(self, label, **metadata):
        frame = [self.wall(), self.cpu(), 0, 0]
        self.stack.append(frame)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            end_wall, end_cpu = self.wall(), self.cpu()
            wall, cpu = end_wall-frame[0], end_cpu-frame[1]
            self.stack.pop()
            if self.stack:
                self.stack[-1][2] += wall
                self.stack[-1][3] += cpu
            row = self.stats.setdefault(label, dict(count=0, failures=0,
                wall_ns=0, thread_cpu_ns=0, self_wall_ns=0, self_thread_cpu_ns=0,
                maximum_wall_ns=0, samples_wall_ns=[]))
            row['count'] += 1
            row['failures'] += failed
            for key, value in (('wall_ns', wall), ('thread_cpu_ns', cpu),
                               ('self_wall_ns', wall-frame[2]), ('self_thread_cpu_ns', cpu-frame[3])):
                row[key] += value
            row['maximum_wall_ns'] = max(row['maximum_wall_ns'], wall)
            if len(row['samples_wall_ns']) < self.sample_limit:
                row['samples_wall_ns'].append(wall)
            if metadata and len(self.events) < self.sample_limit:
                self.events.append(dict(label=label, start_ns=frame[0], end_ns=end_wall,
                    failed=failed, **metadata))

    def report(self):
        rows = {}
        for label, raw in self.stats.items():
            row = {k: v for k, v in raw.items() if k != 'samples_wall_ns'}
            samples = sorted(raw['samples_wall_ns'])
            row.update(quantiles_first_n=len(samples),
                median_sample_wall_ns=samples[len(samples)//2],
                p95_sample_wall_ns=samples[min(len(samples)-1, math.ceil(.95*len(samples))-1)])
            rows[label] = row
        return dict(schema=VERSION, spans=rows, events=self.events,
            events_limit=self.sample_limit, active_spans=len(self.stack),
            inclusive_spans_overlap=True, self_spans_exclude_instrumented_children=True,
            CPU_scope='PYTHON_CALLING_THREAD_ONLY_NOT_NATIVE_CHILDREN',
            quantiles_scope='FIRST_BOUNDED_SAMPLES_COUNTS_AND_TOTALS_ARE_COMPLETE')


def measure(owner, label, **metadata):
    trace = getattr(owner, 'performance', None)
    return trace.span(label, **metadata) if trace is not None else nullcontext()


def profiled(label):
    def decorate(function):
        @wraps(function)
        def call(owner, *args, **kwargs):
            with measure(owner, label):
                return function(owner, *args, **kwargs)
        return call
    return decorate


def settle_atom(adapter, *, event_start, before, after, clock, submitted):
    """Wait for the original evidence, with no additional ENTER or tick grant.

    Startup and preaction settling keep their original 150ms wait. The new
    readiness condition is deliberately stronger than a bare callback drain.
    Original consumption, gripper and standing checks still run afterwards.
    """
    config = configuration(adapter.spec)
    if (config is None or config['mode'] not in ('events', 'events_camera_cache')
            or before['step'] < adapter.spec['task_learning_grasp']['start_step']):
        with measure(adapter, 'atom.fixed_spin_150ms', native_step=before['step']):
            return adapter.spin(.15)
    from scripts.pi0294.task_clock import actual_source_age
    from .fusion_reference_qualification_native import consumed_window
    from .fusion_trajectory_qualification_native import confirm_gripper

    # The quota is already paused, so wrapper consumption is now immutable.
    # A mismatch is a failure, not a condition to conceal until timeout.
    with measure(adapter, 'atom.gripper_consumption'):
        confirm_gripper(adapter, submitted, adapter.measured_task_state())

    def ready():
        packet = adapter.observe()
        try:
            ack = json.loads((adapter.output/'pause_ack.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        if not (packet['step'] == after['step'] == ack.get('step')
                and packet['time_s'] == after['time_s']
                and abs(ack.get('time_s', float('inf'))-after['time_s']) <= 1e-8
                and ack.get('running') is False and ack.get('clock_paused') is True):
            raise RuntimeError('PERFORMANCE_EXACT_NATIVE_PAUSE_LOST')
        if adapter.latest['mode'].data != 'WBC':
            raise RuntimeError('DYNAMIC_WBC_MODE_LOST')
        if any(r.get('reason') == 'PREFIX_TTL_EXPIRED' for r in adapter.reference_events()):
            raise RuntimeError('PERFORMANCE_ORIGINAL_REFERENCE_TTL_EXPIRED')
        fresh = {row['topic_key'] for row in adapter.ros_evidence[event_start:]
                 if row['topic_key'] in ('ee', 'observation')
                 and 0 <= actual_source_age(after['time_s'], float(row['message']['time'])) <= .025}
        if fresh != {'ee', 'observation'}:
            return False
        with measure(adapter, 'atom.solver_consumption_join'):
            result = consumed_window(adapter, event_start, before, after, clock)
        return bool(result['solver_policies_with_exact_installed_target'] and result['arm_request_count'])

    # Keep the old allowance as a timeout; success may now return earlier.
    with measure(adapter, 'atom.event_wait', native_step=before['step']):
        if not adapter.spin(.15, ready):
            raise RuntimeError('PERFORMANCE_REQUIRED_ATOM_FEEDBACK_TIMEOUT_NO_RETRY')


def close_resources(adapter):
    try:
        session = getattr(adapter, '_source_camera_session', None)
        if session is not None:
            with measure(adapter, 'camera.session_close'):
                session.close()
    finally:
        trace = getattr(adapter, 'performance', None)
        if trace is not None:
            from scripts.pi0292.durable import write
            write(adapter.output/'COLLECTION_PERFORMANCE.private.json', dict(trace.report(),
                source_sha=adapter.spec['source_sha'], mode=configuration(adapter.spec)['mode']))

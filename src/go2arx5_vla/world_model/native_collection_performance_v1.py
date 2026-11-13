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
EVENT_READ_WAIT_VERSION = 'PAUSED_EVENT_WALL_READ_WAIT_V3'


def configuration(spec):
    config = spec.get('collection_performance')
    if config is None:
        return None
    from .task_learning_grasp_v1 import CONSTRUCTION_ROLE
    role=spec.get('task_learning_role')
    expert=role=='CORELAB_EXPERT_TRAIN4_SEMANTICS_NOT_BC_V3'
    if expert:
        from .native_body_camera_capture_v1 import session_configuration
        if session_configuration(spec) is None:
            raise ValueError('V3_EXPERT_PERFORMANCE_REQUIRES_BODY_SESSION')
    if not (isinstance(config, dict) and set(config) == {'version', 'mode'}
            and config['version'] == VERSION and config['mode'] in MODES
            and spec.get('phase') == 'collect-target-task-learning-grasp-v1'
            and (role == CONSTRUCTION_ROLE or expert)):
        raise ValueError('PERFORMANCE_FRESH_TASK_CONSTRUCTION_ONLY')
    return config


def event_read_seconds(spec):
    value = spec.get('collection_event_read_wait_v3')
    if value is None:return .15
    from .native_body_memory_v3 import configuration as memory_configuration
    config = configuration(spec)
    if (value != dict(version=EVENT_READ_WAIT_VERSION,max_wall_s=2.)
            or config is None or config['mode'] not in ('events','events_camera_cache')
            or memory_configuration(spec) is None):
        raise ValueError('FRESH_CPU_MEMORY_EVENT_WALL_READ_ONLY')
    return 2.


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


def reset_render_scene(renderer, model):
    """Give a cached 3.2.7 renderer the same CPU scene as a fresh capture.

    mjv_updateScene places infinite planes using the previous scene camera.
    Reusing that scene changes pixels even with identical model/data arrays.
    Retain the expensive OpenGL context, but reset this per-capture history.
    Renderer.scene is read-only in the pinned MuJoCo version.
    """
    import mujoco
    if mujoco.__version__ != '3.2.7':
        raise ValueError('CAMERA_SCENE_RESET_REQUIRES_QUALIFIED_MUJOCO_3_2_7')
    renderer._scene = mujoco.MjvScene(model, maxgeom=renderer.scene.maxgeom)


def settle_atom(adapter, *, event_start, before, after, clock, submitted):
    """Wait for the original evidence, with no additional ENTER or tick grant.

    Legacy startup keeps its original150ms wait. Explicit new tmpfs experts
    also join the stationary-prefix receipts before another atom is granted.
    Original consumption, gripper and standing checks still run afterwards.
    """
    config = configuration(adapter.spec)
    read_wait = event_read_seconds(adapter.spec)
    from .native_live_control_v3 import configuration as live_configuration, acknowledgement_wait
    live_control = live_configuration(adapter.spec) is not None
    if live_control:
        read_wait = acknowledgement_wait(adapter.spec,read_wait)
    if (config is None or config['mode'] not in ('events', 'events_camera_cache')
            or (before['step'] < adapter.spec['task_learning_grasp']['start_step'] and not live_control)):
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
        from .native_live_control_v3 import live_path
        packet = adapter.observe()
        try:
            ack = json.loads(live_path(adapter,'pause_ack.json').read_text())
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

    # Read only while the exact native quota remains paused. A fresh memory
    # run may allow delayed callbacks longer, with the identical 25ms source
    # age, installed-policy/gripper, WBC and TTL predicates; no new clock key.
    with measure(adapter, 'atom.event_wait', native_step=before['step']):
        if not adapter.spin(read_wait, ready):
            raise RuntimeError('PERFORMANCE_REQUIRED_ATOM_FEEDBACK_TIMEOUT_NO_RETRY')


def close_resources(adapter):
    try:
        try:
            memory = getattr(adapter, '_body_memory_capture', None)
            if memory is not None:memory.close()
        finally:
            body = getattr(adapter, '_body_camera_session', None)
            if body is not None:
                with measure(adapter, 'camera.body_session_close'):
                    body.close()
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

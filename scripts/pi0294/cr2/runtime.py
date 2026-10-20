"""New-scope admission around the unchanged native static and Teacher engines."""
import os
from pathlib import Path
import resource
import signal
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.data_cov1.runtime import Meter as CovMeter, rendering_identity
from scripts.pi0294.fd1 import put, sha
from scripts.pi0294.pv1.capture import ensure_legacy_attempt_plan, qualify_counter, run_cell
from scripts.pi0294.scene_cards import digest, near_duplicate_report, task_card
from .plan import cards, validate_group
from .scope import priority, sealed_file, validate


def wait_priority(root, guard):
    while not priority(root):
        guard()
        time.sleep(60)


def bind_cpu(m):
    os.sched_setaffinity(0, set(m['limits']['cpus']))
    resource.setrlimit(resource.RLIMIT_CPU, (m['limits']['CPU_s'], m['limits']['CPU_s']))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.nice(15)


def static(root):
    import mujoco
    from scripts.pi0294.scene import make_engine, qualify_engine
    r = Path(root); lock = OwnerLock(r / 'CR2_WORKER.lock')
    m, inputs = validate(r, 'STATIC'); work = read(r / 'plans/WORK_PLAN.private.json')
    sealed_file(work['candidate_pool_path'], work['candidate_pool_sha256'])
    sealed_file(work['exclusions_path'], work['exclusions_sha256'])
    pool, exclusions = read(work['candidate_pool_path']), read(work['exclusions_path'])
    if pool['task_results_seen'] is not False or len(pool['slots']) != 12:
        raise PermissionError('CR2_SEALED_PRERESULT_TWELVE_SLOT_POOL_REQUIRED')
    ledger = read(r / 'RESOURCE_LEDGER.json'); assert ledger['status'] == 'RESERVED'
    accepted = []; rejected = []; receipts = []; starts = 0; began = time.process_time(); failure = None
    graphics = None; bind_cpu(m)
    def guard(*unused):
        if (time.time() >= m['limits']['operation_deadline_epoch'] or time.process_time() - began >= m['limits']['CPU_s'] - 30
                or any((p / 'STOP.json').exists() for p in (r, r.parent.parent))):
            raise TimeoutError('CR2_STATIC_FINITE_BOUND_OR_STOP')
    def forbidden(*a, **k):
        raise PermissionError('CR2_STATIC_ZERO_INTEGRATION_ONLY')
    original = [mujoco.mj_step, mujoco.mj_step1, mujoco.mj_step2]
    try:
        mujoco.mj_step = mujoco.mj_step1 = mujoco.mj_step2 = forbidden
        ledger['status'] = 'RUNNING'; write(r / 'RESOURCE_LEDGER.json', ledger)
        for slot in pool['slots']:
            wait_priority(r, guard)
            for group in slot['candidates']:
                guard(); validate_group(group)
                conflicts = near_duplicate_report([group], [*exclusions, *accepted])['conflicts']
                if conflicts:
                    rejected.append(dict(group_id=group['group_id'], candidate=group['layout_candidate_index'],
                                         reason='GEOMETRIC_NEAR_DUPLICATE', conflicts=conflicts))
                    continue
                pair = []
                for task in ('T1', 'T2'):
                    wait_priority(r, guard)
                    if starts >= m['limits']['engine_starts']:
                        raise PermissionError('CR2_STATIC_CONSTRUCTOR_BOUND')
                    starts += 1; ledger.update(engine_starts=starts, worker_CPU_s=time.process_time() - began)
                    write(r / 'RESOURCE_LEDGER.json', ledger)
                    card = task_card(group, task); engine = None
                    try:
                        engine = make_engine(card, inputs['asset_root'])
                        q = qualify_engine(engine, render=True, guard=guard)
                        if graphics is None:
                            graphics = rendering_identity()
                            put(r / 'evidence/ACTUAL_RENDERER_IDENTITY.json', graphics)
                    except ValueError as exc:
                        if not str(exc).startswith('Initial native IK unreachable:'):
                            raise
                        q = dict(status='STATIC_FAIL', reason=str(exc), integration_steps=0,
                                 task_success_claim=False, object_pose_modified=False)
                    finally:
                        if engine is not None:
                            engine.close()
                    if q['integration_steps'] or q['task_success_claim'] or q['object_pose_modified']:
                        raise RuntimeError('CR2_STATIC_CANNOT_BE_DYNAMIC_OR_SUCCESS_FILTER')
                    q.update(scene_group=group['group_id'], task=task, card_sha256=digest(card),
                             candidate_index=group['layout_candidate_index'])
                    path = r / 'evidence/static' / ('S' + str(len(receipts)).zfill(5) + '.json')
                    path.parent.mkdir(exist_ok=True); put(path, q)
                    receipts.append(dict(scene_group=group['group_id'], task=task, status=q['status'],
                        receipt_relative=str(path.relative_to(r)), receipt_sha256=sha(path)))
                    pair.append(q)
                if all(q['status'] == 'STATIC_PASS_PENDING_LIVE_CONTACT' for q in pair):
                    accepted.append(group)
                    break
                rejected.append(dict(group_id=group['group_id'], candidate=group['layout_candidate_index'],
                                     reason='ORIGINAL_STATIC_IK_COLLISION_OR_FOV_FAILURE'))
            else:
                raise PermissionError('CR2_STATIC_POOL_EXHAUSTED_NO_MORE_THAN_32')
        full_cards = cards(accepted)
        result = dict(schema='PI0294_CR2_STATIC_POPULATION_V1', status='SEALED_STATIC_QUALIFIED_BEFORE_TASK_RESULTS',
            groups=accepted, cards=full_cards, static_receipts=receipts, rejected_candidates=rejected,
            candidate_pool_sha256=work['candidate_pool_sha256'], exclusions_sha256=work['exclusions_sha256'],
            Teacher_episodes=0, integrations=0, renderer=graphics)
        put(r / 'evidence/STATIC_RESULT.private.json', result)
        return dict(status=result['status'], groups=len(accepted), constructors=starts, integrations=0)
    except BaseException as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
        put(r / 'evidence/STATIC_FAILURE.private.json', dict(error=failure, accepted=len(accepted), integrations=0))
        raise
    finally:
        mujoco.mj_step, mujoco.mj_step1, mujoco.mj_step2 = original
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_STATIC', CPU_s=time.process_time() - began,
            CPU_reserved_s=0, engine_starts=starts, episode_starts=0, live_ticks=0, shadow_ticks=0, ticks=0, failure=failure)
        write(r / 'RESOURCE_LEDGER.json', ledger); lock.close()


class Meter(CovMeter):
    def __init__(self, root):
        self.root = Path(root); self.started = time.process_time(); self.children = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.m, self.inputs = validate(root, 'CAPTURE')
        self.ledger = read(self.root / 'RESOURCE_LEDGER.json'); self.plan = read(self.root / 'plans/WORK_PLAN.private.json')
        sealed_file(self.plan['static_plan_path'], self.plan['static_plan_sha256'])
        static = read(self.plan['static_plan_path'])
        if static['status'] != 'SEALED_STATIC_QUALIFIED_BEFORE_TASK_RESULTS':
            raise PermissionError('CR2_ORIGINAL_STATIC_SEAL_REQUIRED')
        full = {c['id']: c for c in cards(static['groups'])}
        if (self.ledger['status'] != 'RESERVED' or self.ledger['unknown_effects'] or self.ledger['attempted_cells']
                or not self.plan['cards'] or self.plan['cards_sha256'] != digest(self.plan['cards'])
                or any(full.get(c['id']) != c for c in self.plan['cards'])
                or len({c['id'] for c in self.plan['cards']}) != len(self.plan['cards'])):
            raise PermissionError('CR2_EXACT_FIRST_ATTEMPTS_FROM_STATIC_NEW_NAMESPACE')
        self.ticks = self.live_ticks = self.shadow_ticks = self.starts = 0
        self.geometry_rows = []; self.render_times = []
        ensure_legacy_attempt_plan(self.root, self.plan['cards']); bind_cpu(self.m)
        self.ledger['status'] = 'RUNNING'; self.progress()


def capture(root):
    import numpy as np
    from scripts.pi0294.e1.artifacts import atomic_npz
    r = Path(root); lock = OwnerLock(r / 'CR2_WORKER.lock'); meter = None; rows = []; failure = None
    try:
        meter = Meter(r); wait_priority(r, meter.guard); qualify_counter(meter)
        for card in meter.plan['cards']:
            wait_priority(r, meter.guard); meter.guard()
            if (card['id'] in meter.ledger['attempted_cells'] or meter.ticks + 32000 > meter.m['limits']['physics_ticks']
                    or meter.starts + 88 > meter.m['limits']['engine_starts']):
                raise PermissionError('CR2_ONE_FIRST_ATTEMPT_FULL_RESERVE_NO_PHYSICAL_REPLACEMENT')
            meter.ledger['attempted_cells'].append(card['id']); meter.progress(current_cell=card['id'])
            meter.geometry_rows = []; meter.render_times = []
            try:
                row = run_cell(meter, card, engine_observer=meter.instrument_renderer)
            finally:
                path = r / 'evidence/geometry' / (card['id'] + '.npz'); path.parent.mkdir(exist_ok=True)
                atomic_npz(path, rows=np.asarray(meter.geometry_rows, dtype=float).reshape(-1, 5)); path.chmod(0o400)
                put(path.with_suffix('.json'), dict(id=card['id'], rows=len(meter.geometry_rows), sha256=sha(path),
                    source='ORIGINAL_NATIVE_PAD_FIELDS_NO_EXTRA_FORWARD_OR_STEP',
                    metric_is_vertical_plane_gap_not_full_geometric_distance=True))
            rows.append(row); meter.progress(completed_cells=len(rows), current_cell=None)
            if (r / 'STOP.json').exists():
                break
        put(r / 'evidence/CAPTURE_RESULT.private.json', dict(status='TERMINAL_CR2_CAPTURE', rows=rows,
            planned=len(meter.plan['cards']), attempted=len(rows), physical_replacements=0,
            fixed_R1_not_used_to_control_teacher=True, policy_training=False))
        return dict(completed=len(rows), task_successes=sum(bool(row['raw_task_score_success']) for row in rows))
    except BaseException as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
        put(r / 'evidence/CAPTURE_FAILURE.private.json', dict(error=failure, completed=len(rows)))
        raise
    finally:
        if meter is not None:
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            child_cpu = after.ru_utime + after.ru_stime - meter.children.ru_utime - meter.children.ru_stime
            meter.progress(status='FAILED_PRESERVED' if failure else 'TERMINAL_CAPTURE',
                CPU_s=time.process_time() - meter.started + child_cpu, CPU_reserved_s=0,
                completed_cells=len(rows), failure=failure)
        lock.close()

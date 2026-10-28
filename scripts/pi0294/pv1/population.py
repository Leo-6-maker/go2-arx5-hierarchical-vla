"""One bounded static-only population seal, without Teacher success screening."""
import os
from pathlib import Path
import resource
import signal
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.fd1 import put, sha
from scripts.pi0294.scene_cards import digest, near_duplicate_report, task_card, validate_groups
from .plan import COUNTS, candidate_pool, evaluation_cards, slots
from .scope import validate


def run(root):
    began = time.process_time()
    import mujoco
    from scripts.pi0294.scene import make_engine, qualify_engine
    root = Path(root)
    lock = OwnerLock(root/'PV1_POPULATION_OWNER.lock')
    m, inputs = validate(root)
    ledger = read(root/'RESOURCE_LEDGER.json')
    if (ledger['status'] != 'RESERVED' or ledger['unknown_effects']
            or ledger['CPU_reserved_s'] != 7200 or ledger['engine_starts_reserved'] != 8192):
        raise PermissionError('PV1_EXACT_STATIC_RESERVATION_ONLY')
    if (root/'plans/CANDIDATE_POOL.json').exists():
        raise PermissionError('PV1_POPULATION_ONE_ATTEMPT_NO_RESULT_DEPENDENT_REDRAW')
    exclusions = read(root/'plans/EXCLUSIONS.json')
    pool = candidate_pool(exclusions)
    put(root/'plans/CANDIDATE_POOL.json', pool)
    ledger.update(status='RUNNING',pool_sha256=sha(root/'plans/CANDIDATE_POOL.json'))
    write(root/'RESOURCE_LEDGER.json',ledger)
    starts = 0
    accepted, rejected, receipts = [], [], []
    failure = None
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU,(7200,7200))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def guard(*args):
        if (time.time() >= m['limits']['operation_deadline_epoch']
                or time.process_time()-began >= 7100 or (root/'STOP.json').exists()):
            raise TimeoutError('PV1_BOUNDED_STATIC_QUALIFICATION')
    def forbidden(*args,**kwargs):
        raise PermissionError('PV1_STATIC_STAGE_FORBIDS_ANY_INTEGRATION')
    def expire(*args):
        raise TimeoutError('PV1_BOUNDED_STATIC_QUALIFICATION_WALL')
    originals = [mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2]
    mujoco.mj_step = mujoco.mj_step1 = mujoco.mj_step2 = forbidden
    signal.signal(signal.SIGALRM,expire)
    signal.alarm(max(1,int(m['limits']['operation_deadline_epoch']-time.time())))
    try:
        for slot in pool['slots']:
            for group in slot['candidates']:
                guard()
                conflicts = near_duplicate_report([group],[*exclusions,*accepted])['conflicts']
                if conflicts:
                    rejected.append(dict(group_id=group['group_id'],candidate_index=group['layout_candidate_index'],
                        reason='GEOMETRIC_NEAR_DUPLICATE',conflicts=conflicts))
                    continue
                pair = []
                for task in ('T1','T2'):
                    guard()
                    if starts >= ledger['engine_starts_reserved']:
                        raise PermissionError('PV1_ALL_STATIC_CONSTRUCTORS_RESERVED')
                    starts += 1
                    ledger.update(engine_starts=starts,CPU_s=time.process_time()-began)
                    write(root/'RESOURCE_LEDGER.json',ledger)
                    card = task_card(group,task)
                    engine = None
                    try:
                        engine = make_engine(card,inputs['asset_root'])
                        q = qualify_engine(engine,render=True,guard=guard)
                    except ValueError as exc:
                        if not str(exc).startswith('Initial native IK unreachable:'):
                            raise
                        q = dict(status='STATIC_FAIL',reason=str(exc),integration_steps=0,
                            task_success_claim=False,object_pose_modified=False)
                    finally:
                        if engine is not None:engine.close()
                    if q['integration_steps'] != 0 or q['task_success_claim'] or q['object_pose_modified']:
                        raise RuntimeError('PV1_STATIC_QUALIFIER_CHANGED_TASK_OR_STEPPED')
                    q.update(scene_group=group['group_id'],task=task,card_sha256=digest(card),
                        candidate_index=group['layout_candidate_index'])
                    file=root/'evidence/static'/('S'+str(len(receipts)).zfill(5)+'.json')
                    file.parent.mkdir(exist_ok=True)
                    put(file,q)
                    record=dict(scene_group=group['group_id'],task=task,status=q['status'],
                        card_sha256=q['card_sha256'],receipt_relative=str(file.relative_to(root)),receipt_sha256=sha(file),
                        geometry_passed=q.get('geometry_passed',False),
                        field_of_view_passed=bool(q.get('field_of_view') and q['field_of_view']['passed']))
                    receipts.append(record)
                    pair.append(record)
                if all(q['status']=='STATIC_PASS_PENDING_LIVE_CONTACT' for q in pair):
                    accepted.append(group)
                    break
                rejected.append(dict(group_id=group['group_id'],candidate_index=group['layout_candidate_index'],
                    reason='ORIGINAL_STATIC_IK_COLLISION_OR_FOV_FAIL',receipts=pair))
            else:
                raise PermissionError('PV1_FINITE_PREFROZEN_POOL_EXHAUSTED_NO_NEW_SEED_SEARCH')
            ledger.update(accepted_groups=len(accepted),rejected_candidates=len(rejected))
            write(root/'RESOURCE_LEDGER.json',ledger)
        if len(accepted) != len(slots()):
            raise ValueError('PV1_COMPLETE_128_GROUP_POPULATION_REQUIRED')
        uniqueness=validate_groups(accepted,exclusions)
        plan=dict(schema='PI0294_PV1_DATA_PLAN_AND_SPLIT_SEAL_V1',status='SEALED_STATIC_QUALIFIED_BEFORE_TASK_RESULTS',
            groups=accepted,counts=COUNTS,static_receipts=receipts,rejected_candidates=rejected,
            candidate_pool_sha256=sha(root/'plans/CANDIDATE_POOL.json'),exclusions_sha256=sha(root/'plans/EXCLUSIONS.json'),
            uniqueness=uniqueness,DEV_cards=evaluation_cards(accepted,'DEV'),
            LOCKBOX_cards=evaluation_cards(accepted,'LOCKBOX'),
            evaluation_success_screening=False,original_generator_and_static_rules=True,
            Teacher_episodes=0,GR00T_queries=0,training_updates=0,integrations=0,
            LOCKBOX_execution_requires_unique_candidate_freeze=True)
        put(root/'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json',plan)
        return dict(status=plan['status'],groups=len(accepted),plan_sha256=sha(root/'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'),
            engine_starts=starts,integrations=0)
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        put(root/'evidence/FAILURE.json',dict(error=failure,accepted_groups=len(accepted),engine_starts=starts,
            integrations=0,unknown_effects=[]))
        raise
    finally:
        signal.alarm(0)
        mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2=originals
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_STATIC_PLAN_SEALED',
            CPU_s=time.process_time()-began,CPU_reserved_s=0,engine_starts=starts,engine_starts_reserved=0,
            integrations=0,Teacher_episodes=0,GR00T_queries=0,training_updates=0,failure=failure,
            accepted_groups=len(accepted))
        write(root/'RESOURCE_LEDGER.json',ledger)
        lock.close()

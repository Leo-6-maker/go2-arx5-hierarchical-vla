"""Common-prefix contact decision data; measured returns are restart aggregates.

Both candidates must have actually executed from the declared close56 prefix.
The student sees nine past observations and one of the two predeclared goals,
never parent IDs, program IDs, future outcomes or teacher action values.
"""
import numpy as np
from .contact_features import state_feature
from .pilot2_contact_model import observed_events, planned_openings
from .pilot2_supported import GOALS, score

PROGRAMS = ('CLOSE_KEEP', 'CLOSE_RELEASE')
DECISION_ATOM = 112
ANCHOR_INDEX = DECISION_ATOM-1


def observed_history(rows):
    if len(rows) < DECISION_ATOM:
        raise ValueError('REAL_COMMON_SUPPORTED_PREFIX_REQUIRED')
    last = rows[ANCHOR_INDEX]
    if last['native_end_step'] != 2503+25*DECISION_ATOM:
        raise ValueError('EXACT_PAUSED_SUPPORTED_DECISION_CLOCK_REQUIRED')
    for row in rows[ANCHOR_INDEX-8:ANCHOR_INDEX+1]:
        memory = row['actual_gripper_internal_memory']
        if memory['status'] != 'ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY' or memory['values']['steps'] != row['native_end_step']:
            raise ValueError('ACTUAL_SAME_PAUSE_CONTACT_POLICY_MEMORY_REQUIRED')
    return np.concatenate([np.r_[state_feature(row), observed_events(row)]
        for row in rows[ANCHOR_INDEX-8:ANCHOR_INDEX+1]])


def features(history, goal):
    if goal not in GOALS:
        raise ValueError('ONE_OF_TWO_PREDECLARED_SUPPORTED_GOALS_REQUIRED')
    return np.r_[np.asarray(history), np.eye(2)[GOALS.index(goal)]]


def candidate_plans(episode):
    if episode['close_start_atom'] != 56:
        raise ValueError('QUALIFIED_CLOSE56_COMMON_PREFIX_REQUIRED')
    values = []
    for program in PROGRAMS:
        plan = planned_openings(dict(episode, command_program=program))
        future = plan[DECISION_ATOM:DECISION_ATOM+64]
        values.append(np.pad(future, (0, 64-len(future)), constant_values=plan[-1]))
    return np.asarray(values)


def dataset(episodes, *, teacher='GOAL_FSM', wm_arrays=None):
    if teacher not in ('GOAL_FSM', 'MEASURED_PARENT_MEAN_RETURN', 'PARTIAL_WM_EXPECTED_EVENT'):
        raise ValueError('DECLARED_CONTACT_POLICY_TEACHER_REQUIRED')
    parents, runs, eligible, returns = {}, set(), [], {}
    for ep in episodes:
        parent, split, e = ep['parent'], ep['split'], ep['episode']
        if split not in ('TRAIN', 'DEV') or parent in parents and parents[parent] != split:
            raise ValueError('WHOLE_PARENT_TRAIN_DEV_NO_TEST_REQUIRED')
        parents[parent] = split
        if ep['run_id'] in runs: raise ValueError('DUPLICATE_ACTUAL_SUPPORTED_EPISODE')
        runs.add(ep['run_id'])
        if e['command_program'] not in PROGRAMS or e['close_start_atom'] != 56:
            continue
        rows = ep['samples']
        if len(rows) != 168 or any(row['stage']=='EARLY_RELEASE' for row in rows[:DECISION_ATOM]):
            continue  # preserve these failures in the enclosing bundle, not as expert prefixes
        events = [event for row in rows for event in row['native_contact_events']]
        if not score(events, dict(e, task_goal=GOALS[0]))['qualified_supported_grasp_hold']:
            continue
        history = observed_history(rows)
        reward = np.array([float(score(events, dict(e, task_goal=goal))['goal_success']) for goal in GOALS])
        action = PROGRAMS.index(e['command_program'])
        returns.setdefault((parent, action), []).append(reward)
        eligible.append((ep, history, action, reward))
    if not eligible or any((parent, action) not in returns for parent in {x[0]['parent'] for x in eligible} for action in range(2)):
        raise ValueError('BOTH_REAL_QUALIFIED_CLOSE56_PROGRAMS_PER_PARENT_REQUIRED')
    x, q, partition, identities, physical = [], [], [], [], []
    for ep, history, action, reward in eligible:
        parent = ep['parent']
        for g, goal in enumerate(GOALS):
            if teacher=='GOAL_FSM':
                values = np.eye(2)[g]
            elif teacher=='MEASURED_PARENT_MEAN_RETURN':
                # These are different physical restarts, never an exact fork oracle.
                values = np.array([np.mean(returns[(parent, a)], axis=0)[g] for a in range(2)])
            else:
                if wm_arrays is None: raise ValueError('FROZEN_NUMPY_PARTIAL_WM_REQUIRED')
                from .pilot2_contact_numpy import predict
                inputs = np.c_[np.repeat(history[None], 2, axis=0), candidate_plans(ep['episode'])]
                raw = predict(wm_arrays, inputs)
                # 32 atoms are within the 56-atom remaining native program.
                # This is a declared event surrogate, not full terminal return.
                p = 1/(1+np.exp(-np.clip(raw[:, 4, 6:], -50, 50)))
                values = p[:, 1]*p[:, 2] if g==0 else p[:, 3]
            logits = (values-values.max())/.1; probabilities = np.exp(logits); probabilities /= probabilities.sum()
            x.append(features(history, goal)); q.append(probabilities); partition.append(ep['split'])
            physical.append(reward[g])
            identities.append(dict(run_id=ep['run_id'], parent=parent, goal=goal, executed_action=action))
    return dict(x=np.asarray(x, np.float32), labels=np.asarray(q, np.float32),
        split=np.asarray(partition), identities=identities, measured_executed_reward=np.asarray(physical),
        teacher=teacher, exact_fork_oracle=False, decision_atom=DECISION_ATOM,
        synthetic_teacher_values_are_not_physical_rewards=teacher!='MEASURED_PARENT_MEAN_RETURN')

"""Fixed ADAPT5K batch, physical event sampling and phase clocks."""
import math

import numpy as np

from .scale20k_contract import DOMAINS, bank_sha256, cursors as scale_cursors

PLAN = '5734a73cbda5c333cc3d7d7e0f032e679a4c08b6'
UPDATES = 5000
BATCH = dict(zip(DOMAINS, (64, 32, 32), strict=True))
FM_WEIGHTS = dict(zip(DOMAINS, (2., 1., 1.), strict=True))
FORMAL_PARENTS = (500, 2000, 5000)
SWITCH_PHASE = 2000
APPROACH = frozenset(('PREGRASP', 'DESCEND'))
RELEASE_OR_RECOVERY = frozenset(('LOWER', 'OPEN', 'WITHDRAW', 'RELEASE', 'RETREAT', 'RECOVER', 'RETRY'))


def phase_rates():
    phase = np.arange(UPDATES, dtype=np.float64)
    return np.asarray(1e-6 + .5*(5e-6-1e-6)*(1+np.cos(math.pi*phase/(UPDATES-1))), dtype='<f8')


def progress(parent_step, phase_step, stage_start=0):
    if (type(parent_step) is not int or parent_step not in FORMAL_PARENTS or type(phase_step) is not int
            or stage_start not in (0, SWITCH_PHASE) or not stage_start <= phase_step <= UPDATES):
        raise ValueError('ADAPT5K_REQUIRES_ONE_FORMAL_PARENT_AND_OPTIONAL_COLD_2K_SWITCH')
    ancestor = scale_cursors('SCALE-MIX20K', parent_step)
    return dict(parent_global_step=parent_step, phase_step=phase_step, lineage_total_updates=parent_step+phase_step,
                stage_start=stage_start, phase_cursors={d: phase_step*BATCH[d] for d in DOMAINS},
                lineage_cursors={d: ancestor[d]+phase_step*BATCH[d] for d in DOMAINS},
                stage_local_cursors={d: (phase_step-stage_start)*BATCH[d] for d in DOMAINS})


def auxiliary_slots(phase_before_update):
    if type(phase_before_update) is not int or not 0 <= phase_before_update < UPDATES:
        raise ValueError('ADAPT5K_REQUIRES_A_VALID_PRE_UPDATE_PHASE_STEP')
    start = 4*(phase_before_update % 16)
    return tuple(range(start, start+4))


def due_close(label, consumed, previous_width):
    """Detect a closing crossing supported by an actual P10 command, including its seam."""
    width = .002 + .040*np.asarray(label)[:, 9]
    timeline = np.r_[previous_width, width]
    crossing = (timeline[:-1] >= .022) & (timeline[1:] < .022)
    return bool(np.any(crossing[:10] & np.asarray(consumed, dtype=bool)[:10]))


def grouped_episodes(dataset):
    """Expose task, episode and stage without adding these fields to observations."""
    episodes = {}
    for index, (episode, anchor) in enumerate(dataset.rows):
        eid = episode['id']
        label = dataset.labels[eid]
        phase = str(label['phase'][anchor])
        is_due = due_close(label['nominal_policy_H40'][anchor], label['consumed_mask'][anchor],
                           dataset.reference(index)['width_m'])
        item = episodes.setdefault(eid, dict(task=episode['task'], new=index >= dataset.original_windows, phases={}))
        category = (phase, is_due or phase in APPROACH or phase in RELEASE_OR_RECOVERY)
        item['phases'].setdefault(category, []).append(index)
    return list(episodes.values())


def target_bank(dataset, *, stage_start, phase_stop, seed):
    if ((stage_start, phase_stop) not in ((0, 2000), (0, UPDATES), (2000, UPDATES))
            or dataset.new_episodes < 1):
        raise ValueError('ADAPT5K_TARGET_BANK_NEEDS_REAL_NEW_DATA_AND_THE_FROZEN_STAGE_RANGE')
    rng = np.random.Generator(np.random.PCG64(seed))
    episodes = grouped_episodes(dataset)
    new_quota = min(48, 4*dataset.new_episodes)
    prepared = {}
    for is_new in (False, True):
        for critical in (False, True):
            tasks = {}
            for episode in episodes:
                if episode['new'] != is_new:
                    continue
                eligible = [indices for (_, is_critical), indices in episode['phases'].items()
                            if not critical or is_critical]
                if eligible:
                    tasks.setdefault(episode['task'], []).append(eligible)
            prepared[is_new, critical] = [tasks[task] for task in sorted(tasks)]
        if not prepared[is_new, False]:
            raise ValueError('ADAPT5K_REQUIRES_NEW_AND_REPLAY_TARGET_POPULATIONS')

    bank = np.empty((phase_stop-stage_start, BATCH['target']), dtype='<i8')
    counts = dict(new_visits=0, old_visits=0, critical_requested=0, critical_from_event=0, critical_backfilled=0)
    for block in bank:
        offset = 0
        for is_new, quota in ((False, BATCH['target']-new_quota), (True, new_quota)):
            counts['new_visits' if is_new else 'old_visits'] += quota
            for critical in (False, True):
                groups = prepared[is_new, critical]
                if critical:
                    counts['critical_requested'] += quota//2
                    counts['critical_from_event' if groups else 'critical_backfilled'] += quota//2
                groups = groups or prepared[is_new, False]
                for _ in range(quota//2):
                    task = groups[rng.integers(len(groups))]
                    ep = task[rng.integers(len(task))]
                    stage = ep[rng.integers(len(ep))]
                    block[offset] = stage[rng.integers(len(stage))]
                    offset += 1
        rng.shuffle(block)
    return bank.reshape(-1), dict(counts, sha256=bank_sha256(bank), new_quota_per_update=new_quota,
                                  independent_new_episodes=dataset.new_episodes, stage_start=stage_start,
                                  phase_stop=phase_stop, seed=seed)


def joint_gradient(fm, auxiliary, *, tree_map):
    """Combine separate FM means and target auxiliary gradients before the single Adam clip."""
    target, robot, libero = (fm[domain] for domain in DOMAINS)
    tw, rw, lw = (FM_WEIGHTS[domain] for domain in DOMAINS)
    shared = tree_map(lambda t, r, l, a: tw*t + rw*r + lw*l + a,
                      target['shared'], robot['shared'], libero['shared'], auxiliary['shared'])
    target_io = tree_map(lambda t, a: tw*t + a, target['private'], auxiliary['private'])
    return {'shared': shared, 'target': target_io, 'robodojo': robot['private'], 'libero': libero['private']}

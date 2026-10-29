"""Four-task episode-first sampling using the existing reader's row indices.

Metadata stays outside model input. This sampler never admits demonstrations,
fits statistics, creates recovery labels or starts the official Trainer.
"""
import hashlib
import json

import numpy as np

TASKS = ('T1', 'T2', 'DOOR', 'BOOK')
EVENTS = {
    'T1': {'grasp', 'lift', 'release'},
    'T2': {'contact_onset', 'push_stop', 'detach'},
    'DOOR': {'handle_grasp', 'turn_onset', 'regrasp', 'release'},
    'BOOK': {'grasp', 'reorient', 'insert_onset', 'release'},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class SharedTaskSampler:
    """Task -> normal/recovery -> episode -> uniform/event -> window.

    Every batch32 visits each task eight times. First round uses 4 uniform +
    4 event visits per task. The explicit recovery round uses 6 normal +
    2 recovery, with 50/50 uniform/event within each. Episodes are uniform
    before windows, so a long stationary episode gains no episode weight.
    A missing task, event bank or recovery bank fails rather than resampling.
    """
    def __init__(self, dataset, *, seed, start=0, stop=10000*32,
                 recovery_round=False, positioned=True):
        if (type(seed) is not int or seed < 0 or type(start) is not int or type(stop) is not int
                or not 0 <= start <= stop <= 320000 or start % 32 or stop % 32
                or type(recovery_round) is not bool or type(positioned) is not bool
                or not sha(dataset.manifest_sha) or not sha(dataset.statistics_sha)):
            raise ValueError('EXPLICIT_FROZEN_INPUTS_SEED_AND_BATCH32_SESSION_REQUIRED')
        self.dataset, self.seed, self.start, self.stop = dataset, seed, start, stop
        self.positioned, self.recovery_round = positioned, recovery_round
        self.bank, self.metadata = {}, {}
        seen = set()
        for index, (row, anchor) in enumerate(dataset.rows):
            task, episode = row.get('task'), row.get('id')
            lane = row.get('learning_lane')
            if (task not in TASKS or not isinstance(episode, str) or not episode
                    or row.get('split') != 'TRAIN' or lane not in ('normal', 'recovery')
                    or row.get('domain') != 'SOURCE_CONTACT_PHYSICS'
                    or row.get('admitted_success') is not True or not sha(row.get('sha256'))
                    or type(anchor) is not int or anchor < 0 or (episode, anchor) in seen):
                raise ValueError('UNIQUE_ADMITTED_SOURCE_TRAIN_WINDOWS_REQUIRED')
            seen.add((episode, anchor))
            events = row.get('event_anchors')
            if (type(events) is not dict or not set(events) <= EVENTS[task]
                    or any(type(values) is not list or any(type(a) is not int or a < 0 for a in values)
                           or len(set(values)) != len(values) for values in events.values())):
                raise ValueError('TASK_SPECIFIC_MEASURED_EVENT_ANCHORS_REQUIRED')
            if lane == 'recovery':
                r = row.get('recovery', {})
                if (not recovery_round or r.get('origin') != 'SOURCE_STUDENT_ACTUAL_VISIT'
                        or not all(sha(r.get(k)) for k in ('failure_state_sha256', 'restored_state_sha256',
                                                         'student_policy_sha256', 'expert_execution_sha256'))
                        or r['failure_state_sha256'] != r['restored_state_sha256']
                        or r.get('expert_replanned_from_actual_state') is not True
                        or r.get('recovery_physically_verified') is not True):
                    raise ValueError('RECOVERY_REQUIRES_RESTORED_ACTUAL_VISIT_AND_VERIFIED_EXPERT_CONTINUATION')
            metadata = dict(task=task, lane=lane, events=events, sha256=row['sha256'],
                            recovery=row.get('recovery'))
            if episode in self.metadata and self.metadata[episode] != metadata:
                raise ValueError('ONE_IMMUTABLE_EPISODE_IDENTITY_REQUIRED')
            self.metadata[episode] = metadata
            bank = self.bank.setdefault((task, lane), {}).setdefault(episode, {'uniform': [], 'event': []})
            bank['uniform'].append((anchor, index))
            if any(anchor in anchors for anchors in events.values()):
                bank['event'].append((anchor, index))
        for task in TASKS:
            for lane in ('normal', 'recovery') if recovery_round else ('normal',):
                if not self.bank.get((task, lane)):
                    raise ValueError('REQUIRED_TASK_LANE_NOT_READY:'+task+'/'+lane)
        for key, episodes in self.bank.items():
            for episode, windows in episodes.items():
                for values in windows.values():
                    values.sort()
                legal = {a for a, _ in windows['uniform']}
                declared = {a for values in self.metadata[episode]['events'].values() for a in values}
                if not declared or not declared <= legal or not windows['event']:
                    raise ValueError('EVENTS_MUST_BE_NONEMPTY_LEGAL_ACTUAL_READER_WINDOWS')
        self.support_sha = digest({f'{t}/{l}': {e: {k: [a for a, _ in v] for k, v in w.items()}
            for e, w in sorted(episodes.items())} for (t, l), episodes in sorted(self.bank.items())})
        self.metadata_sha = digest(self.metadata)

    def __len__(self):
        return self.stop-self.start

    def selection_at(self, cursor):
        if type(cursor) is not int or not self.start <= cursor < self.stop:
            raise IndexError('CURSOR_OUTSIDE_FROZEN_SESSION')
        task = TASKS[cursor % 4]
        ordinal = (cursor % 32)//4
        lane = 'recovery' if self.recovery_round and ordinal >= 6 else 'normal'
        mode = 'event' if ordinal % 2 else 'uniform'
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, cursor])))
        bank = self.bank[(task, lane)]
        episode = sorted(bank)[int(rng.integers(len(bank)))]
        windows = bank[episode][mode]
        anchor, index = windows[int(rng.integers(len(windows)))]
        return dict(task=task, lane=lane, episode_id=episode, mode=mode, anchor=anchor, index=index)

    def __iter__(self):
        for cursor in range(self.start, self.stop):
            index = self.selection_at(cursor)['index']
            yield (cursor, index) if self.positioned else index

    def index_at(self, cursor):
        return self.selection_at(cursor)['index']

    def state_dict(self, consumed):
        if type(consumed) is not int or not self.start <= consumed <= self.stop or consumed % 32:
            raise ValueError('ONLY_ACTUAL_COMMITTED_BATCH32_BOUNDARIES')
        return dict(algorithm='SHARED_FOUR_TASK_EPISODE_EVENT_RECOVERY_V1', seed=self.seed,
                    cursor=consumed, scientific_horizon=320000, recovery_round=self.recovery_round,
                    manifest_sha256=self.dataset.manifest_sha, statistics_sha256=self.dataset.statistics_sha,
                    support_sha256=self.support_sha, metadata_sha256=self.metadata_sha,
                    metadata_not_model_input=True)

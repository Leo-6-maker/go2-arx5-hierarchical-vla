"""Fixed-quota visit planning; no tensor loads, model calls or job admission.

Domain counters identify a shared visit bank. M75 and SEQ therefore present
the same T/S samples and random tags in different orders. These tags require
their own GPU input-bank qualification before use by a learner.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import random


VERSION = 'SLA_TYPED_DOMAIN_VISIT_BANK_V4'
BATCH = 32
DOMAINS = ('T', 'S_cmd')
QUOTAS = {'T100_3K': (32, 0), 'M75_3K': (24, 8),
          'M50_3K': (16, 16), 'S_ONLY_3K': (0, 32)}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class Catalog:
    """Only references to a fixed TRAIN snapshot; scientific admission is external."""
    def __init__(self, domain, snapshot_sha256, rows):
        if domain not in DOMAINS or len(snapshot_sha256) != 64:
            raise ValueError('EXPLICIT_T_OR_SOURCE_COMMAND_SNAPSHOT_REQUIRED')
        try:
            int(snapshot_sha256, 16)
        except ValueError as exc:
            raise ValueError('EXACT_SNAPSHOT_SHA256_REQUIRED') from exc
        self.domain, self.snapshot_sha256 = domain, snapshot_sha256
        self.rows = []
        unique = set()
        tree = {}
        for index, row in enumerate(rows):
            if (row.get('split') != 'TRAIN' or any(not isinstance(row.get(k), str)
                    or not row[k] for k in ('task', 'episode', 'phase'))
                    or type(row.get('anchor')) is not int or row['anchor'] < 0):
                raise ValueError('TRAIN_TASK_EPISODE_PHASE_ANCHOR_REQUIRED')
            identity = (row['task'], row['episode'], row['anchor'])
            if identity in unique:
                raise ValueError('DUPLICATE_EPISODE_ANCHOR')
            unique.add(identity)
            entry = {key: row[key] for key in ('task', 'episode', 'phase', 'anchor')}
            self.rows.append(dict(entry, row_index=index))
            tree.setdefault(row['task'], {}).setdefault(row['episode'], {}).setdefault(
                row['phase'], []).append(len(self.rows) - 1)
        if not tree:
            raise ValueError('NONEMPTY_SNAPSHOT_CATALOG_REQUIRED')
        self.tree = {task: {episode: {phase: tuple(sorted(anchors,
            key=lambda i: self.rows[i]['anchor'])) for phase, anchors in sorted(phases.items())}
            for episode, phases in sorted(episodes.items())} for task, episodes in sorted(tree.items())}
        # Input row order is part of the reader binding even though sampling
        # choices use canonical task/episode/phase/anchor order.
        self.sha256 = _digest(dict(domain=domain, snapshot=snapshot_sha256, rows=self.rows))

    def sample(self, seed, ordinal):
        key = _digest([VERSION, seed, self.domain, ordinal])
        rng = random.Random(int(key, 16))
        task = rng.choice(tuple(self.tree))
        episodes = self.tree[task]
        episode = rng.choice(tuple(episodes))
        phases = episodes[episode]
        phase = rng.choice(tuple(phases))
        row = self.rows[rng.choice(phases[phase])]
        return Visit(domain=self.domain, snapshot_sha256=self.snapshot_sha256,
                     within_domain_visit=ordinal, **row,
                     rng_tag_sha256=_digest([VERSION, seed, self.domain, row['task'],
                         row['episode'], row['phase'], row['anchor'], ordinal]))


@dataclass(frozen=True)
class Visit:
    domain: str
    snapshot_sha256: str
    within_domain_visit: int
    task: str
    episode: str
    phase: str
    anchor: int
    row_index: int
    rng_tag_sha256: str


def quota(arm, update):
    if type(update) is not int or update < 0:
        raise ValueError('NONNEGATIVE_UPDATE_REQUIRED')
    if arm == 'SEQ_3K':
        return (0, 32) if update < 750 else (32, 0)
    try:
        return QUOTAS[arm]
    except KeyError as exc:
        raise ValueError('EXPLICIT_V4_ARM_REQUIRED') from exc


def counts_before(arm, update):
    quota(arm, update)
    if arm == 'SEQ_3K':
        return max(0, update - 750) * BATCH, min(update, 750) * BATCH
    a, b = quota(arm, update)
    return update * a, update * b


class MixedVisits:
    def __init__(self, arm, catalogs, *, seed, updates=3000):
        quota(arm, 0)
        if (type(seed) is not int or not 0 <= seed < 2**64
                or type(updates) is not int or updates not in (3000, 4000)
                or (updates == 4000 and arm not in ('T100_3K', 'M75_3K'))):
            raise ValueError('DECLARED_SEED_AND_3K_OR_MATCHED_4K_ENDPOINT_REQUIRED')
        self.arm, self.seed, self.updates = arm, seed, updates
        self.catalogs = dict(catalogs)
        needed = {domain for domain, count in zip(DOMAINS, counts_before(arm, updates)) if count}
        if (set(self.catalogs) != needed or any(c.domain != d for d, c in self.catalogs.items())):
            raise ValueError('EXACT_REQUIRED_DOMAIN_CATALOGS_ONLY')
        self.cursor = 0

    def batch(self, update):
        if type(update) is not int or not 0 <= update < self.updates:
            raise ValueError('FINITE_COMPLETE_BATCH_BOUNDARY_REQUIRED')
        offsets = counts_before(self.arm, update)
        rows = [self.catalogs[domain].sample(self.seed, start + i)
                for domain, start, count in zip(DOMAINS, offsets, quota(self.arm, update))
                for i in range(count)]
        random.Random(int(_digest([VERSION, self.seed, self.arm, 'order', update]), 16)).shuffle(rows)
        return tuple(rows)

    def next_batch(self):
        rows = self.batch(self.cursor)
        self.cursor += 1
        return rows

    def state_dict(self):
        return dict(schema=VERSION, arm=self.arm, seed=self.seed, updates=self.updates,
                    cursor=self.cursor, domain_visits=dict(zip(DOMAINS, counts_before(self.arm, self.cursor))),
                    catalogs={d: c.sha256 for d, c in self.catalogs.items()})

    def restore(self, value):
        cursor = value.get('cursor')
        if type(cursor) is not int or not 0 <= cursor <= self.updates:
            raise ValueError('FINITE_SAVED_UPDATE_CURSOR_REQUIRED')
        expected = self.state_dict()
        expected.update(cursor=cursor, domain_visits=dict(zip(DOMAINS, counts_before(self.arm, cursor))))
        if value != expected:
            raise ValueError('SAME_ARM_SEED_SNAPSHOT_AND_DOMAIN_CURSORS_REQUIRED')
        self.cursor = cursor


def visit_record(visit):
    """Audit metadata stays outside the student's observation dict."""
    return asdict(visit)

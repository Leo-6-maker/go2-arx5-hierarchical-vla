"""Charged-before-operation finite resource journal, guarded by the owner lease."""
import json
from pathlib import Path
import time


CAPS = dict(env_episode_starts=168, physics_integrations=2000000,
            WBC_actor_forwards=600000, teacher_queries=70000, RGB_pairs=120000)


class Budget:
    def __init__(self, root):
        self.path = Path(root) / 'BUDGET.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.totals = {k: 0 for k in CAPS}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                for key, value in json.loads(line)['charges'].items(): self.totals[key] += value

    def charge(self, **values):
        for key, value in values.items():
            if key not in CAPS or type(value) is not int or value < 0: raise ValueError('Invalid charge')
            if self.totals[key] + value > CAPS[key]: raise RuntimeError('BUDGET_STOP ' + key)
        with self.path.open('a') as stream:
            stream.write(json.dumps(dict(wall_epoch=time.time(), charges=values)) + '\n')
            stream.flush()
        for key, value in values.items(): self.totals[key] += value

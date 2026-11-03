"""Episode-first causal replay with TRAIN-only normalization."""
import json
from pathlib import Path
import numpy as np
from .contracts import macro_transition
from .features import macro_arrays, ROTATIONS, STATE_DIM
from .population import validate_parent_splits
from .storage import digest, load_private


class Replay:
    def __init__(self, root, *, split, horizon=4):
        if split not in ('TRAIN', 'CALIB', 'DEV') or horizon != 4:
            raise ValueError('TEST is unavailable to training/diagnostic replay')
        root = Path(root)
        if root.is_file():
            self._load(root, split)
            return
        manifest = json.loads((root/'RESULTS.json').read_text())
        self.source_sha = manifest['source_sha']
        self.manifest_sha256 = digest(root/'RESULTS.json')
        sequences, keys, stages, events_seen = [], [], [], []
        rejected_states, rejected_actions, rejected_goals = [], [], []
        self.complete_macros = 0
        self.partial_macros = 0
        self.episodes = len(manifest['episodes'])
        self.atoms = 0
        all_rows = []
        for result in manifest['episodes']:
            if result['split'] != split:
                raise ValueError('Replay manifest split mismatch')
            folder = root/result['episode_id']
            path = folder/'transitions.pkl.gz'
            if digest(path) != result['transition_sha256']:
                raise ValueError('Immutable replay member changed')
            data = load_private(path)
            # A pre-execution rejection is a real episode termination event,
            # not an executed command or a physical state transition. Only the
            # termination logit gets this label; no future state/contact target.
            from .features import encode
            from .contracts import command_labels
            for rejection in sorted(folder.glob('rejected-S1-*.pkl.gz')):
                event = load_private(rejection)
                if event['forecast']['decision'] != 'REJECT':
                    raise ValueError('Rejection event identity mismatch')
                saved = event['state']
                raw, valid = command_labels(event['plan']['poses'][:8], event['plan']['opening'][:8], saved['cached_state']['tcp'])
                if valid:
                    rejected_states.append(encode(saved['cached_state'], saved['controller'], saved['native']))
                    rejected_actions.append(raw.ravel())
                    rejected_goals.append([*saved['card']['goal_xy'], .46])
            self.atoms += len(data['atoms'])
            all_rows.extend({k:r[k] for k in ('parent_id', 'scene_id', 'split')} for r in data['atoms'])
            macros = data['macros']
            for m in macros:
                checked = macro_transition(m['atoms'])
                if checked['complete_state_target']:
                    self.complete_macros += 1
                    events_seen.append(macro_arrays(m)[3])
                else:
                    self.partial_macros += 1
            for start in range(len(macros)-horizon+1):
                window = macros[start:start+horizon]
                if not all(m['complete_state_target'] for m in window):
                    continue
                if any(a['terminal'] or a['truncated'] or abs(a['t1']-b['t0']) > 1e-8
                       for a, b in zip(window, window[1:])):
                    continue
                arrays = [macro_arrays(m) for m in window]
                sequences.append(arrays)
                keys.append((result['episode_id'], start))
                stages.append(window[0]['atoms'][0]['analysis_teacher_stage'])
        validate_parent_splits(all_rows)
        if not sequences:
            raise ValueError('No complete causal four-macro sequences')
        self.states = np.asarray([[a[0] for a in s] for s in sequences], dtype=np.float32)
        self.actions = np.asarray([[a[1] for a in s] for s in sequences], dtype=np.float32)
        self.targets = np.asarray([[a[2] for a in s] for s in sequences], dtype=np.float32)
        self.events = np.asarray([[a[3] for a in s] for s in sequences], dtype=np.float32)
        # Goal is an environment constant already known in this TRAIN/CALIB scene.
        card_by_id = {r['episode_id']: json.loads((root/r['episode_id']/'CARD.json').read_text())
                      for r in manifest['episodes']}
        self.goals = np.asarray([[*card_by_id[k[0]]['goal_xy'], .46] for k in keys], dtype=np.float32)
        self.keys, self.stages, self.split = keys, stages, split
        self.event_counts = np.asarray(events_seen).sum(0).astype(int).tolist()
        self.rejected_states = np.asarray(rejected_states, dtype=np.float32).reshape(-1,STATE_DIM)
        self.rejected_actions = np.asarray(rejected_actions, dtype=np.float32).reshape(-1,56)
        self.rejected_goals = np.asarray(rejected_goals, dtype=np.float32).reshape(-1,3)

    def export(self, path):
        """A minimal private numerical training view; native originals stay put."""
        path = Path(path)
        with path.open('xb') as stream:
            np.savez_compressed(stream, states=self.states, actions=self.actions, targets=self.targets,
                events=self.events, goals=self.goals, keys=np.asarray(self.keys, dtype=str),
                stages=np.asarray(self.stages), rejected_states=self.rejected_states,
                rejected_actions=self.rejected_actions, rejected_goals=self.rejected_goals,
                metadata=json.dumps(dict(schema='WM01_TYPED_REPLAY_V1',
                    split=self.split, source_sha=self.source_sha, manifest_sha256=self.manifest_sha256,
                    complete_macros=self.complete_macros, partial_macros=self.partial_macros,
                    episodes=self.episodes, atoms=self.atoms, event_counts=self.event_counts)))

    def _load(self, path, split):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata['schema'] != 'WM01_TYPED_REPLAY_V1' or metadata['split'] != split:
                raise ValueError('Typed replay split/schema mismatch')
            for key in ('states', 'actions', 'targets', 'events', 'goals'):
                setattr(self, key, data[key].copy())
            for key in ('rejected_states','rejected_actions','rejected_goals'):
                setattr(self,key,data[key].copy())
            self.keys, self.stages = data['keys'].tolist(), data['stages'].tolist()
        if (self.states.shape != (len(self.keys), 4, STATE_DIM)
                or self.targets.shape != self.states.shape or self.actions.shape != (len(self.keys), 4, 56)
                or self.events.shape != (len(self.keys), 4, 4) or self.goals.shape != (len(self.keys), 3)
                or not all(np.isfinite(v).all() for v in (self.states, self.actions, self.targets, self.events, self.goals))):
            raise ValueError('Invalid typed replay shape/value')
        for key in ('source_sha', 'manifest_sha256', 'complete_macros', 'partial_macros', 'episodes', 'atoms', 'event_counts', 'split'):
            setattr(self, key, metadata[key])

    def normalizer(self):
        if self.split != 'TRAIN':
            raise ValueError('Normalization is TRAIN-only')
        # Nonoverlapping macro starts, rather than repeated windows, determine statistics.
        values = self.states[:, 0]
        mean, scale = values.mean(0), values.std(0).clip(.001)
        for sl in ROTATIONS:
            mean[sl], scale[sl] = 0., 1.
        return mean, scale

    def sample(self, rng, count):
        ids = rng.integers(0, len(self.keys), size=count)
        return tuple(v[ids] for v in (self.states, self.actions, self.goals, self.targets, self.events)), ids

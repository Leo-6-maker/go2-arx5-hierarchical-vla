"""Count actual committed training visits, not the unique-window histogram."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, pin=None):
    path = Path(path)
    if pin is not None and sha(path) != pin:
        raise ValueError('SEALED_TRAINING_EVIDENCE_CHANGED')
    return json.loads(path.read_bytes())


def run(owner):
    owner = Path(owner)
    result = load(owner / 'RESULT.private.json')
    lease = load(owner / 'control/LEASE.json')
    if (result['schema'] != 'SLA_CORE_FRESH_REFERENCE_N20_FM3K_RGB_ACTUAL_V6'
            or lease['status'] != 'CLOSED' or not result['exit_code'] == lease['worker_exit_code'] == 0):
        raise ValueError('CLOSED_REFERENCE3K_ONLY')
    candidate = owner / 'candidate3000'
    spec = load(candidate / 'EXECUTION_SPEC.json')
    manifest = load(Path(spec['dataset']) / 'MANIFEST.private.json', spec['dense_manifest_sha256'])
    checkpoint = load(Path(result['last_checkpoint']['path']) / 'COMPLETE.json',
                      result['last_checkpoint']['sha256'])
    rows = [(episode['id'], anchor, episode['expert_stage_audit_only'][anchor])
            for episode in manifest['episodes'] for anchor in range(61)]
    indices, steps = [], 0
    log = candidate / 'work/updates.private.jsonl'
    with log.open() as stream:
        for line in stream:
            row = json.loads(line)
            steps += 1
            assert row['step'] == steps and row['cursor'] == steps * 32
            batch = [i for micro in row['micro'] for i in micro['indices']]
            assert len(batch) == 32
            indices.extend(batch)
    actual = np.asarray(indices, dtype='<i8')
    pin = hashlib.sha256(actual.tobytes()).hexdigest()
    assert steps == result['actual']['optimizer_updates'] == checkpoint['step'] == 3000
    assert len(actual) == result['actual']['completed_training_visits'] == checkpoint['cursor'] == 96000
    assert pin == checkpoint['bindings']['visit_map_sha256']
    phase = Counter(rows[i][2] for i in actual)
    unique_phase = Counter(row[2] for row in rows)
    repeated = Counter(map(int, actual))
    episode_counts = Counter(rows[i][0] for i in actual)
    return dict(schema='SLA_REFERENCE3K_ACTUAL_COMMITTED_EXPOSURE_V7',
        updates=steps, actual_target_visits=len(actual), source_visits=0,
        actual_visit_sequence_matches_complete_binding=True, visit_map_sha256=pin,
        training_log_sha256=sha(log), checkpoint_sha256=result['last_checkpoint']['sha256'],
        unique_windows=len(rows), visited_unique_windows=len(repeated),
        unique_window_phase_counts=dict(unique_phase), actual_visit_phase_counts=dict(phase),
        actual_visit_phase_fractions={k: v / len(actual) for k, v in phase.items()},
        episodes=len(episode_counts), actual_episode_visits=dict(episode_counts),
        per_window_repeats_min=min(repeated.values()), per_window_repeats_max=max(repeated.values()),
        sampling=checkpoint['bindings']['visit_bank']['sampling'],
        new_training_visits=0, optimizer_updates=0, model_loads=0, physical_requests=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('owner', type=Path)
    print(json.dumps(run(parser.parse_args().owner), allow_nan=False))

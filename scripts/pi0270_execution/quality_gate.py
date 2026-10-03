"""Shared actual final-training admission for modern and native quality entries."""
import hashlib
import json
from pathlib import Path

BASE = Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT = BASE/'PI0270'


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


def admitted():
    seal = read(ROOT/'design/QUALITY_EXECUTION_FILESET.json')
    assert seal['status'] == 'FROZEN_PI0270_BEFORE_NEW_QUALITY'
    for row in seal['files']: assert sha(row['path']) == row['sha256'], row['path']
    audit = read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json')
    assert audit['status'] == 'PASS_AGG4096_REUSED_CONTROL_TRAINING_INTEGRITY' and audit['optimizer_updates'] == 4096
    checkpoints = audit['complete_checkpoints']
    assert {(r['arm'], r['step']) for r in checkpoints} == {('AGG', 2048), ('AGG', 4096)} and len(checkpoints) == 2
    for checkpoint in checkpoints:
        assert checkpoint['complete_trainer']
        assert {'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'model.safetensors.index.json'} <= {r['path'] for r in checkpoint['entries']}
    assert audit['all4096_COMMON_input_label_bit_identities'] and audit['all8192_actual_random_packets_equal_control']
    assert len(audit['NEW_FIT192_actual_consumption']) == 192 and all(r['actual_seen'] for r in audit['NEW_FIT192_actual_consumption'])
    reuse = read(ROOT/'evidence/EVALUATION_EXACT_REUSE.json')
    assert reuse['status'] == 'PASS_EXACT_CLOSED_1936_OFFLINE_128_MAIN_64_COMPONENT_REUSE'
    assert reuse['counts'] == dict(offline=1936, virtual=128, component=64)
    assert read(ROOT/'evidence/AGG4096/FROZEN_SCOPE.json')['status'] == 'PASS_FIXED_FINAL_MODEL_VIEW_PRECISION_FROZEN_SCOPE'

"""Shared final-only admission for both modern inference and native virtual entries."""
import hashlib
import json
from pathlib import Path

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0273'

def read(p):return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def admitted():
    seal=read(ROOT/'design/QUALITY_EXECUTION_FILESET.json')
    assert seal['status']=='FROZEN_PI0273_BEFORE_NEW_QUALITY'
    for row in seal['files']:assert sha(row['path'])==row['sha256'],row['path']
    audit=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json')
    assert audit['status']=='PASS_PI0273_BOTH_ACTUAL_5120_MATCHED_TRAINING'
    assert audit['actual_random_packets_matched'] and audit['actual_inputs_match_CPU_plan'] and audit['actual_loss_gradient_mode_checked']
    assert audit['actual_generated_nodes']==dict(BROAD=4,TARGET=4)
    assert len(audit['complete_checkpoints'])==4
    assert {(r['arm'],r['step']) for r in audit['complete_checkpoints']}=={(a,s) for a in ('BROAD','TARGET') for s in (2560,5120)}
    for cp in audit['complete_checkpoints']:
        assert cp['complete_trainer']
        assert {'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','model.safetensors.index.json','PI0273_READOUT_CONTRACT.json'}<={r['path'] for r in cp['entries']}
    assert len(audit['NEW_FIT192_actual_consumption'])==192 and all(r['actual_seen'] for r in audit['NEW_FIT192_actual_consumption'])
    for point in ('BROAD5120','TARGET5120'):
        assert read(ROOT/'evidence'/point/'FROZEN_SCOPE.json')['status']=='PASS_FIXED_FINAL_MODEL_VIEW_PRECISION_FROZEN_SCOPE'
    resolution=read(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json')
    assert resolution['status']=='RESOLVED_BY_EXPLICIT_SCIENTIFIC_AUTHORITY' and resolution['launch_allowed']

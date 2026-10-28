"""Separate closed BROAD field admission from the new final-only EQUIL admission."""
import hashlib,importlib.util,json
from pathlib import Path
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0274';RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0274');ACTIVE_POINT=None
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def admitted(point=None):
    point=point or ACTIVE_POINT or 'EQUIL5120';assert point in ('BROAD5120','EQUIL5120')
    seal=read(ROOT/'design/QUALITY_EXECUTION_FILESET.json');assert seal['status']=='FROZEN_PI0274_BEFORE_NEW_QUALITY'
    for row in seal['files']:assert sha(row['path'])==row['sha256'],row['path']
    archive=read(ROOT/'evidence/RESULT_ARCHIVE_data.json');assert archive['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS' and archive['revision']=='a786d91e501d241eb8af29d9666f6b8fd3c7a6be'
    assert read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')['status']=='PASS_PI0274_NEW_INPUTS_AND_REUSED8192_BASE'
    if point=='BROAD5120':
        p=BASE/'PI0273/code/quality_gate.py';spec=importlib.util.spec_from_file_location('pi0273_closed_gate',p);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.admitted()
        binding=read(ROOT/'model_bindings/BROAD5120.json');assert binding==read(BASE/'PI0273/model_bindings/BROAD5120.json')
        assert binding['source_revision']=='leoyuer/pi0246-gate-pass-campaign@5bed9a0d209ccaf23cfcd8bb28fb5a4c41a699c4/arms/PI0273/BROAD/checkpoint-5120'
    else:
        audit=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json');assert audit['status']=='PASS_PI0274_ACTUAL5120_MATCHED_BROAD_TRAINING'
        assert audit['actual_random_packets_matched'] and audit['actual_inputs_match_CPU_plan'] and audit['actual_loss_gradient_mode_checked']
        assert {(r['arm'],r['step']) for r in audit['complete_checkpoints']}=={('EQUIL',2560),('EQUIL',5120)}
        for cp in audit['complete_checkpoints']:
            assert cp['complete_trainer']
            assert {'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','model.safetensors.index.json','PI0274_READOUT_CONTRACT.json'}<={r['path'] for r in cp['entries']}
        assert read(ROOT/'evidence/EQUIL5120/FROZEN_SCOPE.json')['status']=='PASS_FIXED_FINAL_MODEL_VIEW_PRECISION_FROZEN_SCOPE'
    resolution=read(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json');assert resolution['status']=='RESOLVED_BY_EXPLICIT_SCIENTIFIC_AUTHORITY' and resolution['launch_allowed']

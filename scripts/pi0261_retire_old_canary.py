"""Retire one superseded R3 technical canary checkpoint to fit PI0261 canaries."""
import json
import shutil
from pathlib import Path

R3 = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0260/R3')
ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0261')
TARGET = R3 / 'Base/canary-deterministic-cont/training/checkpoint-8'


def main():
    assert json.loads((R3 / 'Base/canary-deterministic-cont_compare.json').read_text())['status'] == 'PASS_R3_REAL_FULL_TRAINER_CANARY'
    assert json.loads((R3 / 'evidence/Base/checkpoint-6000.full_readback.json').read_text())['status'] == 'PASS_FULL_FILE_READBACK'
    assert TARGET.resolve().is_relative_to((R3 / 'Base/canary-deterministic-cont/training').resolve())
    assert json.loads((TARGET / 'trainer_state.json').read_text())['global_step'] == 8
    files = [path for path in TARGET.rglob('*') if path.is_file()]
    assert files and all(not path.is_symlink() and path.stat().st_nlink == 1 for path in files)
    assert not any(path.is_symlink() for path in TARGET.rglob('*'))
    total = sum(path.stat().st_size for path in files)
    shutil.rmtree(TARGET)
    receipt = dict(status='RETIRED_ONE_SUPERSEDED_R3_TECHNICAL_CANARY_CHECKPOINT',
                   path=str(TARGET), logical_bytes=total, files=len(files),
                   reason='R3 canary comparison PASS and final Base6000 fixed HF full readback PASS; capacity for PI0261 real canaries',
                   preserved='R3 raw logs, comparison receipt, final Base/G full HF Trainer and model results')
    output = ROOT / 'R3_TECHNICAL_CANARY_LOCAL_RETIREMENT.json'
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

"""Materialize only final model files on Corelab from a fixed PI0264 HF archive."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0264')
GPU = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0264')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main(arm):
    point = arm + '2048'
    readback = ROOT / 'evidence' / arm / 'checkpoint-2048.full_readback.json'
    source_path = readback if readback.exists() else ROOT / 'evidence' / arm / 'checkpoint-2048.archive_receipt.json'
    source = json.loads(source_path.read_text())
    assert (source['arm'],source['step'])==(arm,2048)
    if readback.exists():
        assert source['status']=='PASS_FULL_FILE_READBACK' and source['complete_trainer']
        source_files=source['files']
    else:
        # A verified model-only download can overlap full optimizer readback; no Trainer retirement.
        evidence=ROOT/'training_evidence'/arm
        inputs=json.loads((evidence/'main-2048.actual_input_audit.json').read_text())
        graph=json.loads((evidence/'main-2048.graph_loss_gradient_audit.json').read_text())
        assert inputs['status']=='PASS_PI0264_ACTUAL_SOURCE_TEMPLATE_LABEL_MASK_RNG_LOG_MATCH'
        assert (inputs['arm'],inputs['phase'],inputs['units'],inputs['microbatches'])==(arm,'main',2048,4096)
        assert graph['status']=='PASS_PI0264_SAVED_FULL_S4_LOSS_AND_GRADIENT_AUDIT'
        assert (graph['arm'],graph['updates'],graph['microbatches'])==(arm,2048,4096) and source['status']=='UPLOADED'
        source_files=source['entries']
        assert len(source_files)>=16 and all(n in {r['path'] for r in source_files} for n in
            ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'))
    names = ('config.json', 'model-00001-of-00002.safetensors',
             'model-00002-of-00002.safetensors', 'model.safetensors.index.json')
    entries = {row['path']: row for row in source_files}
    assert all(name in entries for name in names)
    download_root = GPU / 'models' / point
    root = download_root / source['prefix']
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = root / name
        if not target.exists() or sha(target) != entries[name]['sha256']:
            downloaded = Path(hf_hub_download(repo_id=source['repo_id'],
                filename=source['prefix'] + '/' + name, revision=source['revision'], local_dir=download_root))
            assert downloaded == target
        assert target.stat().st_size == entries[name]['bytes'] and sha(target) == entries[name]['sha256']
    binding = dict(schema='PI0264_FIXED_EVAL_MODEL_BINDING_V1', point=point, view_type='model-only', view=str(root),
                   source_revision=f"{source['repo_id']}@{source['revision']}/{source['prefix']}",
                   checkpoint_tree_sha256=source['tree_sha256'],
                   model_files_sha256={name: entries[name]['sha256'] for name in names})
    target = ROOT / 'model_bindings' / (point + '.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(binding, indent=2, sort_keys=True) + '\n'
    if target.exists():
        assert target.read_text() == rendered
    else:
        target.write_text(rendered)
    print(json.dumps(dict(status='PASS_PI0264_FIXED_FINAL_MODEL_VIEW', point=point,
                          model_view=str(root), binding=str(target))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('B', 'L'), required=True)
    main(parser.parse_args().arm)

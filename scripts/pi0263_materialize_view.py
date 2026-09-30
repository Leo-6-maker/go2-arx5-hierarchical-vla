"""Materialize only final model files on Corelab from a fixed PI0263 HF archive."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0263')
GPU = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0263')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main(arm):
    point = arm + '2000'
    source = json.loads((ROOT / 'evidence' / arm / 'checkpoint-2000.full_readback.json').read_text())
    assert (source['status'], source['arm'], source['step'], source['complete_trainer']) == (
        'PASS_FULL_FILE_READBACK', arm, 2000, True)
    names = ('config.json', 'model-00001-of-00002.safetensors',
             'model-00002-of-00002.safetensors', 'model.safetensors.index.json')
    entries = {row['path']: row for row in source['files']}
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
    binding = dict(schema='PI0263_FIXED_EVAL_MODEL_BINDING_V1', point=point, view_type='model-only', view=str(root),
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
    print(json.dumps(dict(status='PASS_PI0263_FIXED_FINAL_MODEL_VIEW', point=point,
                          model_view=str(root), binding=str(target))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('F', 'X'), required=True)
    main(parser.parse_args().arm)

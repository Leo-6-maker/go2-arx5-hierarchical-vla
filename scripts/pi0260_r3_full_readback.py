"""Fixed-revision full-file HF readback of a PI0260-R3 Trainer archive."""
import argparse
import hashlib
import json
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=('Base', 'G'), required=True)
    parser.add_argument('--step', type=int, choices=(1000, 3000, 6000), required=True)
    args = parser.parse_args()
    source = json.loads(args.archive_receipt.read_text())
    revision = source['manifest_commit_id']
    prefix = f'arms/PI0260/R3/{args.arm}/checkpoint-{args.step}'
    assert source['repo_id'] == 'leoyuer/pi0246-gate-pass-campaign' and source['path'] == prefix
    assert source['status'] == 'DURABLE_VERIFIED' and revision
    local = source['local']
    checkpoint = Path(local['root'])
    assert checkpoint.resolve().is_relative_to(Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0260/R3').resolve())
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert state['global_step'] == args.step and state['max_steps'] == 6000
    assert all((checkpoint / name).is_file() for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'model.safetensors.index.json'))
    assert HfApi().model_info(source['repo_id'], revision=revision).private is False
    checked = []
    for item in local['entries']:
        url = hf_hub_url(source['repo_id'], filename=prefix + '/' + item['path'], revision=revision)
        digest, size = hashlib.sha256(), 0
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            for block in response.iter_content(8 << 20):
                digest.update(block); size += len(block)
        assert size == item['bytes'] and digest.hexdigest() == item['sha256'], item['path']
        checked.append(dict(path=item['path'], bytes=size, sha256=digest.hexdigest()))
        print(json.dumps(dict(status='R3_HF_MEMBER_READBACK', arm=args.arm, step=args.step, path=item['path'], bytes=size)), flush=True)
    result = dict(schema='PI0260_R3_FULL_TRAINER_HF_READBACK_V1', status='PASS_FULL_FILE_READBACK',
                  arm=args.arm, step=args.step, repo_id=source['repo_id'], revision=revision, prefix=prefix,
                  source_tree_sha256=local['tree_sha256'], files=checked, complete_trainer=True,
                  archive_receipt_sha256=hashlib.sha256(args.archive_receipt.read_bytes()).hexdigest())
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        assert args.out.read_text() == rendered
    else:
        args.out.write_text(rendered)
    print(json.dumps(dict(status=result['status'], arm=args.arm, step=args.step, revision=revision, files=len(checked), tree=local['tree_sha256'])), flush=True)


if __name__ == '__main__':
    main()

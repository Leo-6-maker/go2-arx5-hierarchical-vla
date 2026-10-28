#!/usr/bin/env python3
"""Restore original CALIB and comparison model; zero project model queries."""
import argparse
import json
import os
from pathlib import Path
import tarfile

from pi0253_full_screen_v2 import ROOT, sha, read, write
from pi0253_full_live import admission, STEPS
import pi0253_full_calib as calib
import pi0228_h200_large_adapt_v1 as inventory
from pi0245_restore_inference_model import validate_model_index


def main(receipt_path):
    closed = admission()
    for step in STEPS:
        assert read(ROOT / f'live/FULL{step}/cpu_replay.json')['status'] == 'PASS_INDEPENDENT_CPU_REPLAY'
    assert read(ROOT / 'evidence/FULL/FULL_LIVE_PIPELINE_STATE.json')['status'] == 'COMPLETE'
    receipt = read(receipt_path)
    assert receipt['status'] == 'PASS_FULL_FILE_READBACK' and receipt['public'] is True
    entries = {r['path']: r for r in receipt['local']['entries']}
    assert {r['path']: (r['bytes'], r['readback_sha256']) for r in receipt['readback']} == {
        p: (r['bytes'], r['sha256']) for p, r in entries.items()}
    store = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0253/original_calib_inputs')
    store.mkdir(parents=True, exist_ok=True)
    assert __import__('shutil').disk_usage(store).free >= sum(r['bytes'] for r in entries.values()) + (10 << 30)
    import requests
    from huggingface_hub import hf_hub_url
    token = Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip()

    def fetch(name):
        row = entries[name]; path = store / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.partial')
            with requests.get(hf_hub_url(receipt['repo_id'], filename=receipt['prefix']+'/'+name,
                    revision=receipt['revision']), headers={'Authorization': 'Bearer '+token},
                    stream=True, timeout=(30,180)) as response:
                response.raise_for_status()
                with temporary.open('wb') as out:
                    for block in response.iter_content(8 << 20): out.write(block)
                    out.flush(); os.fsync(out.fileno())
            assert temporary.stat().st_size == row['bytes'] and sha(temporary) == row['sha256']
            os.replace(temporary, path)
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        print(json.dumps(dict(status='PINNED_ORIGINAL_INPUT', path=name)), flush=True)
        return path

    transfer = read(fetch('TRANSFER_INVENTORY.json'))
    seal = fetch('pi0228_train_calib_seal.json'); authority = fetch('pi0228_train_calib_target_authority.json')
    baseline_seal = read(fetch('pi0232_baseline_checkpoint_tree_seal.json'))
    assert baseline_seal['tree_sha256'] == transfer['baseline_tree_sha256'] == '12c83e6695c28be3b4a91ef6df201b18f6a1a2877d3e80f7db15ef656176ab3b'
    assert baseline_seal['entries'] == transfer['baseline_entries']
    model_files = [r for r in baseline_seal['entries'] if r['path'] in {'config.json', 'model.safetensors.index.json',
        'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'}]
    assert len(model_files) == 4
    for row in model_files:
        assert entries['baseline1000/'+row['path']]['sha256'] == row['sha256']
        fetch('baseline1000/'+row['path'])
    validate_model_index(store/'baseline1000', model_files)
    archive = fetch('train_calib_dataset.tar'); assert sha(archive) == transfer['tar_sha256']
    dataset = store/'train_calib_dataset'
    if not dataset.exists():
        temporary = store/'train_calib_dataset.partial'; temporary.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            assert len(members) == len({m.name for m in members}) == len(transfer['CALIB_entries'])
            assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
            assert {m.name for m in members} == {r['path'] for r in transfer['CALIB_entries']}
            tar.extractall(temporary, filter='data')
        observed, _ = inventory.tree_sha256(temporary, transfer['CALIB_tree_prefix'])
        assert observed == transfer['CALIB_tree_sha256'] == read(seal)['train_calib']['tree_sha256']
        os.replace(temporary, dataset)
    bindings = dict(schema='PI0253_FULL_CALIB_BINDINGS_V1', closed_family=closed, expected_forward_count=1088,
        selected_checkpoint='NONE', runner_sha256=sha(calib.__file__),
        profile_validator_sha256=sha(calib.validation.__file__), selector_sha256=sha(calib.selector.__file__),
        seal=dict(path=str(seal), sha256=sha(seal)), authority=dict(path=str(authority), sha256=sha(authority)),
        calib_dataset=dict(path=str(dataset), source_root=transfer['CALIB_source_root'],
            tree_prefix=transfer['CALIB_tree_prefix'], tree_sha256=transfer['CALIB_tree_sha256']),
        baseline=dict(path=str(store/'baseline1000'), model_files=model_files, complete_trainer_resume=False,
            role='ORIGINAL_ADAPTED1000_COMPARISON_ONLY', source_tree_sha256=transfer['baseline_tree_sha256']),
        original_inputs_HF_readback_sha256=sha(receipt_path), original_inputs_HF_revision=receipt['revision'])
    path = ROOT/'evidence/FULL/FULL_CALIB_BINDINGS.json'; write(path, bindings)
    _, _, rows, _, _, _ = calib.setup(path)
    assert len(rows) == 1088
    write(ROOT/'evidence/FULL/CALIB_RUNTIME_INTEGRATION.json', dict(status='PASS_ZERO_MODEL_RUNTIME_BINDING',
        bindings_sha256=sha(path), preparation_runner_sha256=sha(__file__), model_queries=0, optimizer_updates=0))
    output = ROOT/'calib'; local_output = store.parent/'calib'
    local_output.mkdir(exist_ok=True)
    if output.exists(): assert output.is_symlink() and output.resolve() == local_output
    else: output.symlink_to(local_output, target_is_directory=True)
    print('PASS_ORIGINAL_CALIB_RUNTIME_ADMISSION_ZERO_MODEL_QUERIES', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--receipt', type=Path, required=True)
    main(parser.parse_args().receipt)

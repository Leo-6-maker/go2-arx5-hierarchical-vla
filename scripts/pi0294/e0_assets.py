"""Bounded E0 inference members only; no model/pickle load or historical deletion."""
import argparse
import json
import os
from pathlib import Path
import time

from scripts.pi0292.durable import OwnerLock, read, write
from .e0_scope import DEADLINE, inference_files, validate_entry
from .native_dataset import sha


def prepare(root):
    root = Path(root)
    validate_entry(root, profile='hopper-neural')
    owner = OwnerLock(root/'E0_INPUTS_OWNER.lock')
    began = time.process_time()
    config = read(root/'E0_INPUT_PREPARATION.private.json')
    inventory = read(root/'evidence/HF_PUBLIC_M2048_RECEIPT.json')
    identity = read(root/'evidence/M2048_INFERENCE_IDENTITY.json')
    inference_files(identity)
    if (config['repo_id'] != inventory['repo_id'] or config['revision'] != inventory['revision']
            or config['prefix'] != inventory['prefix']):
        raise PermissionError('E0_EXACT_FIXED_M2048_DOWNLOAD_SOURCE')
    destination = root/'model_inputs'/'M2048'/config['prefix']
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    state = dict(status='VERIFYING_INFERENCE_ASSETS', pid=os.getpid(),
        proc_start=Path('/proc/self/stat').read_text().split()[21], started_epoch=time.time(),
        expected_bytes=sum(inventory['files'][n]['bytes'] for n in identity['files']),
        files=rows, model_loaded=False, optimizer_loaded=False, selected='NONE', deployment='NONE')
    if state['expected_bytes'] > 24*1024**3:
        raise PermissionError('E0_NEW_INPUT_STORAGE_CAP')
    try:
        from huggingface_hub import hf_hub_download
        for name, digest in identity['files'].items():
            if time.time() >= DEADLINE or time.process_time()-began > 7200:
                raise TimeoutError('E0_BOUNDED_INPUT_PREPARATION')
            target = destination/name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise PermissionError('E0_MODEL_MEMBER_PATH_ESCAPE')
            if target.exists():
                if sha(target) != digest:
                    raise ValueError('E0_PRESERVE_MISMATCHED_EXISTING_MODEL_MEMBER:'+name)
                mode = 'EXISTING_E0_FIXED_MEMBER_REUSED'
            else:
                cached = None
                for candidate_root in config['registered_read_only_candidates']:
                    candidate = Path(candidate_root)/name
                    if candidate.is_file() and candidate.stat().st_size == inventory['files'][name]['bytes'] and sha(candidate) == digest:
                        cached = candidate.resolve()
                        break
                if cached is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(cached, target)
                    mode = 'EXISTING_REGISTERED_IDENTICAL_MEMBER_READ_ONLY'
                else:
                    path = Path(hf_hub_download(repo_id=config['repo_id'], repo_type='dataset',
                        revision=config['revision'], filename=config['prefix']+'/'+name,
                        local_dir=root/'model_inputs'/'M2048'))
                    if path != target:
                        raise ValueError('E0_FIXED_DOWNLOAD_DESTINATION_CHANGED')
                    mode = 'MISSING_INFERENCE_MEMBER_FIXED_REVISION_ONLY'
            if target.stat().st_size != inventory['files'][name]['bytes'] or sha(target) != digest:
                raise ValueError('E0_ACTUAL_MODEL_MEMBER_SIZE_OR_SHA_MISMATCH:'+name)
            rows.append(dict(name=name, sha256=digest, bytes=target.stat().st_size, mode=mode))
            write(root/'evidence/E0_INPUT_PREPARATION_STATE.json', state)
        index = read(destination/'model.safetensors.index.json')
        if set(index['weight_map'].values()) != {n for n in identity['files'] if n.endswith('.safetensors')}:
            raise ValueError('E0_MODEL_SHARD_INDEX_COVERAGE')
        from .public_checkpoint import public_config
        original = Path(config['original_checkpoint'])
        if public_config(read(original/'config.json')) != read(destination/'config.json'):
            raise ValueError('E0_PUBLIC_CONFIG_IS_NOT_ORIGINAL_COSMOS_PATH_ONLY_RELOCATION')
        bindings = read(root/'NEURAL_BINDINGS.private.json')
        if sha(Path(bindings['dataset'])/'statistics.json') != identity['statistics_sha256']:
            raise ValueError('E0_DATASET_STATISTICS_CHANGED')
        identity['checkpoint'] = str(destination)
        write(root/'evidence/M2048_INFERENCE_IDENTITY.json', identity)
        state.update(status='PASS_E0_INFERENCE_MEMBERS_ACTUAL_SHA_CHECKED_NO_MODEL_LOAD',
            checkpoint=str(destination), identity_sha256=sha(root/'evidence/M2048_INFERENCE_IDENTITY.json'),
            completed_epoch=time.time())
        return state
    except BaseException as error:
        state.update(status='INPUT_PREPARATION_FAILURE_PRESERVED',
            error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        state['CPU_s'] = time.process_time()-began
        write(root/'evidence/E0_INPUT_PREPARATION_STATE.json', state)
        owner.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    print(json.dumps(prepare(p.parse_args().root)), flush=True)

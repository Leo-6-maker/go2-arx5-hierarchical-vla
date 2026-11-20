"""Receive a pinned public MIX checkpoint and read its exact next TRAIN batch."""
import argparse
import math
import os
from pathlib import Path
import re
import shutil
import time

from go2arx5_vla.sla_adapter import scale20k_training
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import BATCH, DOMAINS, bank_sha256, cursors
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.goal_v2_mix_a40_restore import ARM, archived_inputs, job_row, read_data, require_scope
from scripts.sla.publish_scale_checkpoint_hf import REPO, inventory, sha
from scripts.wm02.multisource_d4_vanda import GIB, allocation_window


TRANSFER_CAP = 8 * GIB
PUBLIC_JOB_WALL = 1800


def download(config, root, binding, guard):
    from huggingface_hub import HfApi, hf_hub_url
    from requests import Session

    publication = config['publication']
    step, revision, digest = (publication[key] for key in ('step', 'revision', 'checkpoint_sha256'))
    if (type(step) is not int or not 1000 <= step < 20000
            or not re.fullmatch(r'[0-9a-f]{40}', revision)
            or not re.fullmatch(r'[0-9a-f]{64}', digest)):
        raise ValueError('MIX_PUBLIC_CUSTODY_REQUIRES_EXACT_REVISION_AND_CHECKPOINT_DIGEST')
    prefix = f'{ARM}/step-{step:05d}-{digest[:12]}'
    names = ('COMPLETE.json', *(f'{i:06d}.npy' for i in range(140)))
    public = HfApi(token=False)
    if public.repo_info(REPO, repo_type='dataset', revision=revision, timeout=20).private is not False:
        raise PermissionError('MIX_PUBLIC_CUSTODY_SOURCE_MUST_BE_PUBLIC')
    files = {item.path: item for item in public.list_repo_tree(REPO, path_in_repo=prefix,
        repo_type='dataset', revision=revision, recursive=True) if hasattr(item, 'size')}
    if set(files) != {f'{prefix}/{name}' for name in names}:
        raise ValueError('MIX_PUBLIC_CUSTODY_SOURCE_FILE_SET_CHANGED')
    total = sum(item.size for item in files.values())
    if (not 0 < total <= TRANSFER_CAP
            or not 0 < files[f'{prefix}/COMPLETE.json'].size <= 131072
            or shutil.disk_usage(root).free < total + 24*GIB):
        raise ValueError('MIX_PUBLIC_CUSTODY_TRANSFER_OR_DISK_BOUND')
    pending = root/'checkpoint.pending'
    pending.mkdir()
    with Session() as http:
        http.trust_env = False
        expected = digest
        for index, name in enumerate(names):
            guard()
            key = f'{prefix}/{name}'
            record = files[key]
            url = hf_hub_url(REPO, key, repo_type='dataset', revision=revision)
            if index:
                expected = state['arrays'][index-1]['sha256']
                if record.lfs and record.lfs.sha256 != expected:
                    raise ValueError('MIX_PUBLIC_CUSTODY_REMOTE_MEMBER_DIGEST_CHANGED')
            path = pending/name
            with http.get(url, stream=True, timeout=(15, 60)) as response, path.open('xb') as out:
                response.raise_for_status()
                count = 0
                for block in response.iter_content(chunk_size=4*1024**2):
                    guard()
                    count += len(block)
                    if count > record.size:
                        raise ValueError('MIX_PUBLIC_CUSTODY_REMOTE_MEMBER_EXCEEDED_SIZE')
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            if count != record.size or sha(path) != expected:
                raise ValueError('MIX_PUBLIC_CUSTODY_RECEIVED_MEMBER_HASH_OR_SIZE_CHANGED:'+name)
            if not index:
                state = read_pin(dict(path=str(path), sha256=digest))
                if (state['binding'] != binding or state['step'] != step or state['cursors'] != cursors(ARM, step)
                        or [row['file'] for row in state['arrays']] != list(names[1:])):
                    raise ValueError('MIX_PUBLIC_CUSTODY_ORIGINAL_TRAINING_BINDING_CHANGED')
            if index in (0, 25, 50, 75, 100, 125, 140):
                write(root/'PROGRESS.private.json', dict(stage='ANONYMOUS_DOWNLOAD', members_verified=index+1))
    inventory(pending, digest, step=step)
    guard()
    checkpoint = root/'checkpoint'
    pending.rename(checkpoint)
    return dict(path=str(checkpoint), sha256=digest), state, total


def run(config):
    # The existing original-worker scope still validates the 500-step reference.
    root = require_scope(config)
    observed = job_row(config['job_id'])
    resources = observed['Resource_List']
    hosts = {value.split('/')[0] for value in observed['exec_host'].split('+')}
    if (os.getuid() != config['uid'] or observed['Job_Owner'].split('@')[0] != config['user']
            or observed['project'] != config['project'] or observed['job_state'] != 'R'
            or os.uname().nodename.split('.')[0] not in hosts or int(resources.get('ngpus', 0))
            or Path(__file__).resolve() != Path(config['receiver']['path']).resolve()
            or sha(Path(__file__)) != config['receiver']['sha256']):
        raise PermissionError('MIX_PUBLIC_CUSTODY_REQUIRES_OWN_CPU_PBS_AND_PINNED_RECEIVER')
    allocation = allocation_window(observed, campaign_deadline=config['deadline_epoch'])
    deadline = allocation['deadline_epoch']
    result = dict(job_id=config['job_id'], allocation=allocation, source_sha=config['source_sha'],
                  publication=config['publication'], optimizer_updates=0, optimization_visits=0,
                  model_generations=0, native_requests=0, native_ticks=0, exit_code=1,
                  verified_files=0, data_only_window_reads=dict.fromkeys(DOMAINS, 0))
    lock = OwnerLock(root/'PBS_OWNER.lock')

    def guard():
        if time.time()+75 >= deadline or any(Path(path).exists() for path in config['stop_files']):
            raise TimeoutError('MIX_PUBLIC_CUSTODY_STOP_OR_SITE_WINDOW')

    try:
        import jax

        if (allocation['PBS_wall_s'] > PUBLIC_JOB_WALL or jax.default_backend() != 'cpu'
                or not math.isfinite(deadline)):
            raise PermissionError('MIX_PUBLIC_CUSTODY_REQUIRES_FINITE_CPU_ONLY_BACKEND')
        guard()
        _, training, reference = archived_inputs(config)
        checkpoint, state, received = download(config, root, reference['binding'], guard)
        result.update(checkpoint=checkpoint, bytes_verified=received, verified_files=141)
        datasets, banks, tokenizer = read_data(training, state['binding'])
        batch_digests = {}
        for domain in DOMAINS:
            start = state['cursors'][domain]
            indices = banks[domain][start:start+BATCH[ARM][domain]]
            if len(indices) != BATCH[ARM][domain]:
                raise ValueError('MIX_PUBLIC_CUSTODY_ORIGINAL_NEXT_BATCH_TRUNCATED')
            batch_digests[domain] = bank_sha256(indices)
            for cursor, index in enumerate(indices, start=start):
                guard()
                observation, action = scale20k_training.host_batch(datasets[domain], tokenizer, domain, [cursor], [index])
                jax.block_until_ready((observation, action))
                result['data_only_window_reads'][domain] += 1
        result.update(exit_code=0, status='MIX_PUBLIC_CUSTODY_AND_EXACT_NEXT_TRAIN_BATCH_VERIFIED',
                      batch_index_sha256=batch_digests)
        return 0
    except Exception as error:
        result.update(status='MIX_PUBLIC_CUSTODY_FAILED',
                      first_failure=f'{type(error).__name__}: {error}' if type(error).__module__ == 'builtins'
                      else type(error).__name__)
        return 1
    finally:
        write(root/'RESULT.private.json', result)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    os.umask(0o077)
    raise SystemExit(run(read_pin(dict(path=args.spec, sha256=args.sha256))))

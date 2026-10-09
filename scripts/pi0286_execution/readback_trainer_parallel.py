"""Four bounded HTTP ranges, ordered full-file SHA, fixed uploaded revision only."""
import argparse
import hashlib
import importlib.util
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286')
BLOCK = 64 << 20
THREADS = 4
LOCAL = threading.local()


def fetch(url, start, end, total):
    import requests
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    for attempt in range(3):
        try:
            with LOCAL.session.get(url, headers={'Range': f'bytes={start}-{end}'},
                                   stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                assert response.status_code == 206
                assert response.headers['Content-Range'] == f'bytes {start}-{end}/{total}'
                assert int(response.headers['Content-Length']) == end-start+1
                data = bytearray()
                for block in response.iter_content(8 << 20):
                    data.extend(block)
                    assert len(data) <= end-start+1
                assert len(data) == end-start+1
                return data
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2*(attempt+1))


def run(arm):
    assert arm in ('C0_REPLAY1024', 'C1_CORRECT1024')
    slug = 'c0' if arm.startswith('C0_') else 'c1'
    folder = ROOT / 'archive' / f'{slug}_full_trainer'
    target = folder / f'archive_{slug}_trainer.py'
    spec = importlib.util.spec_from_file_location('archive_wrapper', target)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    wrapper.configure()
    mature = wrapper.helper()
    boundary = wrapper.ready()
    assert boundary is not None
    evidence = ROOT / 'evidence' / arm
    archive = json.loads((evidence / 'checkpoint-1024.archive_receipt.json').read_text())
    audit = json.loads((ROOT / 'evidence' / f'{arm}-1024_TRAINING_CPU.json').read_text())
    assert archive['entries'] == audit['entries'] and len(archive['entries']) == 14
    assert archive['prefix'] == f'arms/PI0286/{arm}/checkpoint-1024'
    assert archive['repo_id'] == wrapper.REPO and len(archive['revision']) == 40
    reused = {}
    log = folder / 'WORKER_CPU.log'
    for line in log.read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if value.get('status') == 'PI0286_HF_MEMBER_READBACK' and value.get('arm') == arm:
            reused[value['path']] = value['bytes']
    for name, size in reused.items():
        assert any(row['path'] == name and row['bytes'] == size for row in archive['entries'])
    wrapper.write(folder / 'REUSE_MATURE_PASS_MEMBERS.json', dict(
        status='REUSE_ONLY_ACTUAL_MATURE_FULL_FILE_SHA_ASSERTED_PASS_MEMBERS',
        members=reused, helper_sha256=wrapper.HELPER_SHA,
        original_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        revision=archive['revision'], entries_equal_native_audit=True))
    checked = []
    for row in archive['entries']:
        if row['path'] in reused:
            checked.append(row)
            continue
        url = f"https://huggingface.co/{archive['repo_id']}/resolve/{archive['revision']}/{archive['prefix']}/{row['path']}"
        digest, count, started = hashlib.sha256(), 0, time.time()
        # Small trailing metadata needs one ordinary complete anonymous GET.
        if row['bytes'] <= BLOCK:
            import requests
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                for block in response.iter_content(8 << 20):
                    digest.update(block)
                    count += len(block)
        else:
            ranges = iter((start, min(start+BLOCK, row['bytes'])-1)
                          for start in range(0, row['bytes'], BLOCK))
            with ThreadPoolExecutor(max_workers=THREADS) as pool:
                pending = []
                for _ in range(THREADS):
                    interval = next(ranges, None)
                    if interval is not None:
                        pending.append(pool.submit(fetch, url, *interval, row['bytes']))
                while pending:
                    data = pending.pop(0).result()
                    digest.update(data)
                    count += len(data)
                    del data
                    print(json.dumps(dict(status='FIXED_REVISION_ORDERED_RANGE_PROGRESS',
                                          arm=arm, path=row['path'], verified_ordered_bytes=count,
                                          expected_bytes=row['bytes'], elapsed_s=time.time()-started)), flush=True)
                    interval = next(ranges, None)
                    if interval is not None:
                        pending.append(pool.submit(fetch, url, *interval, row['bytes']))
        assert count == row['bytes'] and digest.hexdigest() == row['sha256'], row['path']
        checked.append(row)
        print(json.dumps(dict(status='PI0286_HF_MEMBER_READBACK', arm=arm, step=1024,
                              path=row['path'], bytes=count)), flush=True)
    result = dict(schema='PI0286_FULL_TRAINER_HF_READBACK_V1', status='PASS_FULL_FILE_READBACK',
                  arm=arm, step=1024, repo_id=archive['repo_id'], prefix=archive['prefix'],
                  revision=archive['revision'], tree_sha256=archive['tree_sha256'],
                  files=checked, complete_trainer=True)
    mature['save'](evidence / 'checkpoint-1024.full_readback.json', result)
    wrapper.write(folder / 'COMPLETE.json', dict(
        status=f'PASS_{slug.upper()}_PUBLIC_FULL14_STREAMED_BYTE_READBACK_LOCAL_TRAINER_RETAINED',
        revision=result['revision'], tree_sha256=result['tree_sha256'], prefix=result['prefix'],
        repo_id=result['repo_id'], CPU_binding=boundary, public_readback_anonymous=True,
        full_readback_sha256=hashlib.sha256((evidence/'checkpoint-1024.full_readback.json').read_bytes()).hexdigest(),
        local_full_trainer_retained=True, full_trainer_copied_to_PC_or_Core=False,
        concurrent_HTTP_ranges=4, ordered_range_window_bytes=4*BLOCK,
        reused_actual_mature_PASS_members=len(reused), duplicate_upload=False,
        model_queries=0, optimizer_updates=0, physics_calls=0, epoch=time.time()))
    print(json.dumps(dict(status=result['status'], revision=result['revision'], members=len(checked))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    run(parser.parse_args().arm)

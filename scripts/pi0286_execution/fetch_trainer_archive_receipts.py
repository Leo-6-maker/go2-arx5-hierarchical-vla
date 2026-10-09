"""Fetch only original small receipts with explicit arm/destination/revision binding."""
import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

ARMS = {
    'C0_REPLAY1024': ('C0', 'c0_full_trainer', 'b0b8c65de8d5bbdf0946ccdda9ecb23348cf16c9',
                      'e61e2d5288dab9a0408a67bd88583d022e3c85d42aef5ebfbe699957bcb3d7ab'),
    'C1_CORRECT1024': ('C1', 'c1_full_trainer', 'd35dcda18def3d0691b55957ce461fe6b8449437',
                       'df2bfeedc443e68fe8f06d60f2786e02cd128e5e2dc881011d3ebfed37f8ad5f'),
}


def fetch(arm):
    slug, remote_folder, revision, expected_tree = ARMS[arm]
    root = Path(__file__).resolve().parents[2]
    destination = root / 'reports/PI0286/archival' / slug
    sys.path.insert(0, str(root/'scripts/pi0276_execution'))
    from coordinator import remote
    output = remote('hop', f'''import base64,hashlib,json,time
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286');f=r/'archive'/{remote_folder!r}
paths=[f/name for name in ['WORKER_START.json','WORKER_DONE.json','ONLINE_PREFLIGHT.json','ACTUAL_{slug}_BINDING.json','COMPLETE.json','AUDIT_ARCHIVE_BINDING.json','REUSE_MATURE_PASS_MEMBERS.json','PARALLEL_READBACK_SWITCH.json','PARALLEL_RANGE_PREFLIGHT.json','DIRECT_PARALLEL_READINESS.json','DIRECT_PARALLEL_PREP.json']]
paths += [r/'evidence'/{(arm+'-1024_TRAINING_CPU.json')!r},r/'evidence'/{arm!r}/'checkpoint-1024.archive_receipt.json',r/'evidence'/{arm!r}/'checkpoint-1024.full_readback.json']
files=[dict(path=str(p),name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),blob=base64.b64encode(p.read_bytes()).decode()) for p in paths if p.exists()]
print(json.dumps(dict(epoch=time.time(),files=files)))
''')
    result = json.loads(output)
    payload = {}
    for row in result['files']:
        blob = base64.b64decode(row.pop('blob'))
        assert len(blob) == row['bytes'] and hashlib.sha256(blob).hexdigest() == row['sha256']
        assert row['name'] not in payload and Path(row['name']).name == row['name']
        payload[row['name']] = blob
    archive = json.loads(payload['checkpoint-1024.archive_receipt.json'])
    readback = json.loads(payload['checkpoint-1024.full_readback.json'])
    native_blob = payload[f'{arm}-1024_TRAINING_CPU.json']
    native = json.loads(native_blob)
    binding = json.loads(payload['AUDIT_ARCHIVE_BINDING.json'])
    complete = json.loads(payload['COMPLETE.json'])
    assert archive['arm'] == readback['arm'] == binding['arm'] == arm
    assert native['point'] == f'{arm}-1024'
    assert native['status'] == 'PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE'
    assert readback['status'] == 'PASS_FULL_FILE_READBACK' and readback['complete_trainer']
    assert archive['revision'] == readback['revision'] == binding['revision'] == complete['revision'] == revision
    assert archive['entries'] == readback['files'] == native['entries']
    assert len(archive['entries']) == 14 and len({row['path'] for row in archive['entries']}) == 14
    tree = hashlib.sha256()
    for row in archive['entries']:
        tree.update(f"PI0286/{arm}/checkpoint-1024/{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    assert archive['tree_sha256'] == readback['tree_sha256'] == binding['tree_sha256'] == complete['tree_sha256'] == tree.hexdigest() == expected_tree
    assert archive['prefix'] == readback['prefix'] == complete['prefix'] == f'arms/PI0286/{arm}/checkpoint-1024'
    assert complete['CPU_binding']['training_CPU_sha256'] == hashlib.sha256(native_blob).hexdigest()
    assert complete['full_readback_sha256'] == hashlib.sha256(payload['checkpoint-1024.full_readback.json']).hexdigest()
    assert destination.name == slug and destination.parent == root/'reports/PI0286/archival'
    destination.mkdir(parents=True, exist_ok=True)
    for name, blob in payload.items():
        path = destination/name
        path.write_bytes(blob)
        assert path.read_bytes() == blob
    result.update(status='PASS_EXPLICIT_ARM_DESTINATION_REVISION_NATIVE14_TREE_AND_ORIGINAL_BYTES',
                  arm=arm, local_folder=str(destination), revision=revision, tree_sha256=expected_tree,
                  remote_science_upload_readback_rerun=False)
    (destination/'ACTUAL_RECEIPTS_INDEX.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    fetch(parser.parse_args().arm)

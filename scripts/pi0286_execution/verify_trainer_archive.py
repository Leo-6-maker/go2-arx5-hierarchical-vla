"""Bind completed PUBLIC full-byte archival to the actual native CPU audit."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286')


def verify(arm):
    assert arm in ('C0_REPLAY1024', 'C1_CORRECT1024')
    evidence = ROOT / 'evidence'
    audit_path = evidence / f'{arm}-1024_TRAINING_CPU.json'
    audit = json.loads(audit_path.read_text())
    archive = json.loads((evidence / arm / 'checkpoint-1024.archive_receipt.json').read_text())
    readback = json.loads((evidence / arm / 'checkpoint-1024.full_readback.json').read_text())
    folder = ROOT / 'archive' / ('c0_full_trainer' if arm.startswith('C0_') else 'c1_full_trainer')
    complete = json.loads((folder / 'COMPLETE.json').read_text())
    assert audit['status'] == 'PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE'
    assert audit['point'] == f'{arm}-1024' and audit['chunks'] == 8192
    assert readback['status'] == 'PASS_FULL_FILE_READBACK' and readback['complete_trainer']
    entries = sorted(audit['entries'], key=lambda entry: entry['path'])
    assert len(entries) == 14 and len({entry['path'] for entry in entries}) == 14
    assert entries == archive['entries'] == readback['files']
    prefix = f'arms/PI0286/{arm}/checkpoint-1024'
    assert archive['prefix'] == readback['prefix'] == complete['prefix'] == prefix
    assert archive['repo_id'] == readback['repo_id'] == complete['repo_id'] == 'leoyuer/pi0246-gate-pass-campaign'
    revision = archive['revision']
    assert len(revision) == 40 and readback['revision'] == complete['revision'] == revision
    tree = hashlib.sha256()
    for entry in entries:
        tree.update(f"PI0286/{arm}/checkpoint-1024/{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode())
    assert archive['tree_sha256'] == readback['tree_sha256'] == complete['tree_sha256'] == tree.hexdigest()
    assert complete['CPU_binding']['training_CPU_sha256'] == hashlib.sha256(audit_path.read_bytes()).hexdigest()
    checkpoint = ROOT / 'learning' / arm / 'main/checkpoint-1024'
    assert checkpoint.is_dir() and str(checkpoint) == audit['checkpoint']
    for entry in entries:
        path = checkpoint / entry['path']
        assert path.is_file() and not path.is_symlink() and path.stat().st_size == entry['bytes']
    value = dict(status='PASS_NATIVE_CPU_AUDIT_ALL14_EXACT_PUBLIC_BYTE_READBACK_BOUND',
                 arm=arm, prefix=prefix, revision=revision, tree_sha256=tree.hexdigest(),
                 members=14, total_bytes=sum(entry['bytes'] for entry in entries),
                 local_full_trainer_retained=True, full_trainer_copied_to_PC_or_Core=False,
                 model_queries=0, optimizer_updates=0, physics_calls=0)
    target = folder / 'AUDIT_ARCHIVE_BINDING.json'
    content = json.dumps(value, indent=2, sort_keys=True) + '\n'
    if target.exists():
        assert target.read_text() == content
    else:
        target.write_text(content)
    print(json.dumps(value))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    verify(parser.parse_args().arm)

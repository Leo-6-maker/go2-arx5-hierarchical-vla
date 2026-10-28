"""Package completed Hopper journals for the existing Corelab summary archive."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0268')
SUFFIXES = {'.json', '.jsonl', '.csv', '.md', '.py', '.pbs', '.log', '.ER', '.OU'}


def files(root):
    folders = ['code', 'contracts', 'design', 'evidence', 'logs', 'OLD', 'ROT',
               'runtime/failed-643647', 'runtime/failed-643670']
    paths = list(root.glob('*.json')) + list(root.glob('*.jsonl'))
    for name in folders:
        paths.extend((root / name).rglob('*'))
    selected = []
    for path in sorted(set(paths)):
        relative = path.relative_to(root)
        # Full checkpoint payloads have their own fixed-revision archives.
        if any(part.startswith('checkpoint-') for part in relative.parts[:-1]):
            continue
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        assert not path.is_symlink() and root.resolve() in path.resolve().parents
        name = relative.as_posix()
        if path.suffix in {'.ER', '.OU'}:
            name += '.log'  # Preserve scheduler logs through the summary's suffix filter.
        selected.append((path, name))
    assert len({name for _, name in selected}) == len(selected)
    return selected


def main():
    def read(path):
        return json.loads((ROOT / path).read_text())
    assert read('evidence/MAIN_TRAINING_AUDIT.json')['status'] == 'PASS_BOTH_2048_MATCHED_TRAINING_INTEGRITY'
    assert read('evidence/optimization/OPTIMIZATION_REPORT.json')['status'] == 'PASS_ACTUAL_INPUT_LOSS_GRADIENT_JOIN'
    for arm in ('OLD', 'ROT'):
        for step in (1024, 2048):
            assert read(f'evidence/{arm}/checkpoint-{step}.full_readback.json')['status'] == 'PASS_FULL_FILE_READBACK'
    out = ROOT / 'runtime/archive/training-evidence.tar.gz'
    assert not out.exists(), 'Reuse the existing sealed package; do not overwrite it'
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with tarfile.open(out.with_suffix('.partial'), 'w:gz') as tar:
        for path, name in files(ROOT):
            before = path.stat()
            payload = path.read_bytes()
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
            entries.append(dict(source=path.relative_to(ROOT).as_posix(), path=name,
                                bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
            info = tarfile.TarInfo('training_hopper/' + name)
            info.size = len(payload)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))
        manifest = dict(status='SEALED_COMPLETED_TRAINING_EVIDENCE', files=entries,
                        source_root=str(ROOT), model_queries=0, gradient_queries=0)
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
        info = tarfile.TarInfo('training_hopper/TRAINING_EVIDENCE_MANIFEST.json')
        info.size = len(payload)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(payload))
    out.with_suffix('.partial').replace(out)
    print(json.dumps(dict(package=str(out), files=len(entries), source_bytes=sum(x['bytes'] for x in entries),
                          sha256=hashlib.sha256(out.read_bytes()).hexdigest())), flush=True)


if __name__ == '__main__':
    import argparse
    import tempfile
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    if parser.parse_args().self_check:
        parent = ROOT / 'runtime/archive'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            root = Path(folder)
            for name in ['OLD/main/training/sample_audit.jsonl', 'OLD/main/training/checkpoint-1024/optimizer.pt',
                         'OLD/main/training/checkpoint-1024/config.json', 'runtime/hf_online/token.json', 'logs/job.ER']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
            assert {name for _, name in files(root)} == {'OLD/main/training/sample_audit.jsonl', 'logs/job.ER.log'}
        result = dict(status='PASS_SMALL_TRAINING_EVIDENCE_SELECTION_NO_MODELS_OR_HF_CACHE',
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      checked='Journal inclusion, scheduler-log preservation, checkpoint and HF-cache exclusion',
                      final_package_created=False, model_queries=0, gradient_queries=0)
        (ROOT / 'evidence/TRAINING_EVIDENCE_PACK_CPU_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
        print(result['status'], flush=True)
    else:
        main()

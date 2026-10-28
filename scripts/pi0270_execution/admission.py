"""Close the serialized predecessor gate from actual receipts and saved arrays."""
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import numpy as np
from scipy.spatial.transform import Rotation

BASE = Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT = BASE / 'PI0270'
OLD = BASE / 'PI0269'
RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0269')
RELEASE = '7a6eea5ae38854bec7b80782213a92ca02fd0318'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    text = json.dumps(value, indent=2, sort_keys=True) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, ('IMMUTABLE_CONFLICT', str(path))
    else:
        path.write_text(text)


def main():
    destination = ROOT / 'evidence/A0_PREDECESSOR_CLOSEOUT.json'
    if destination.exists():
        prior = read(destination)
        for row in prior['sources']:
            assert sha(row['path']) == row['sha256'], row['path']
        print(prior['status']); return
    sources = []

    def record(path):
        value = read(path)
        sources.append(dict(path=str(path), sha256=sha(path), status=value.get('status')))
        return value

    close = record(OLD / 'evidence/final-publication/FINAL_CLOSEOUT_RECEIPT.json')
    assert close['status'] == 'PI0269_PREFIX_LEARNING_STUDY_COMPLETE'
    assert close['execution'] == 'COMPLETE' and close['missing_required_work'] == []
    assert (close['logical_offline_rows'], close['main_episodes'], close['component_valid'], close['full_Trainer_checkpoints']) == (2904, 192, 96, 4)
    assert close['all_checkpoint_and_result_fixed_revision_readbacks'] == 'PASS'
    canonical = record(OLD / 'evidence/final-publication/FINAL_CANONICAL_RECEIPT.json')
    assert canonical['comment_id'] == 5970279132 and canonical['status'] == 'PASS_FINAL_CANONICAL_EXACT_BODY_READBACK'
    assert sha(OLD / 'evidence/final-publication/FINAL_CANONICAL_BODY.txt') == canonical['body_sha256']
    actual = record(ROOT / 'design/PREDECESSOR_CANONICAL_GITHUB.json')
    assert actual['id'] == canonical['comment_id']
    assert hashlib.sha256(actual['body'].encode()).hexdigest() == canonical['body_sha256']
    authority = record(ROOT / 'design/AUTHORIZATION_GITHUB.json')
    assert authority['id'] == 5969189182 and RELEASE in authority['body']
    audit = record(OLD / 'evidence/PRIMARY_COMPLETION_AUDIT.json')
    assert audit['status'] == 'PASS_COMPLETE_PRIMARY_ARTIFACTS_AND_CUMULATIVE_BUDGET_RECONCILIATION'
    diagnosis = record(OLD / 'evidence/SAVED_STATE_PREFIX_AUDIT.json')
    assert diagnosis['status'] == 'PASS_BOUNDED_SAME_STATE_LABEL_DIAGNOSIS_NO_NEW_CORRECTNESS_DEFECT'
    cpu = record(RAW / 'delivery/CPU_REPLAY.json')
    assert len(cpu['predictions']) == 16656 and cpu['scorer_shared']
    verdict = record(RAW / 'delivery/PRIMARY_VERDICT.json')
    descriptive = record(RAW / 'delivery/FINAL_DESCRIPTIVE_ANALYSIS.json')
    binding = record(OLD / 'model_bindings/PREFIX4096.json')
    initializer = record(OLD / 'evidence/DONOR_EXACT_REUSE.json')['binding']
    assert binding['checkpoint_tree_sha256'] == '12aff760d05dd05e0e11a024be3353fb45a5cb98a8b110ae7c7a0da7199609a5'
    assert initializer['checkpoint_tree_sha256'] == '3db80d35d5e476fe5a90d540ff8959fb5edf3da43a988fde7f88ce9f54b9b147'
    for model, receipt in ((binding, OLD / 'evidence/PREFIX/checkpoint-4096.full_readback.json'),
                           (initializer, BASE / 'PI0268/R1/evidence/training_hopper/evidence/ROT/checkpoint-2048.full_readback.json')):
        archive = record(receipt)
        assert archive['status'] == 'PASS_FULL_FILE_READBACK' and archive['complete_trainer'] and len(archive['files']) == 18
        assert archive['tree_sha256'] == model['checkpoint_tree_sha256'] and archive['revision'] in model['source_revision']
        for name, digest in model['model_files_sha256'].items():
            assert sha(Path(model['view']) / name) == digest, name
        write(ROOT / 'model_bindings' / (model['point'] + '.json'), model)
    assert sha(OLD / 'evidence/PREFIX/checkpoint-4096.full_readback.json') == '08773cd8e8a85c4da439a9519be78b3c6a5291c3f35ca6b31fdb8eaa52480a7f'
    stats = OLD / 'contracts/ROT/source/dataset/statistics.json'
    assert sha(stats) == binding['statistics_sha256'] == initializer['statistics_sha256'] == '0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    assert read(Path(binding['view']) / 'config.json') == read(Path(initializer['view']) / 'config.json')
    with (RAW / 'delivery/VIRTUAL192.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['point'] == 'PREFIX4096']
    assert len(rows) == 64
    groups = []
    for task in ('M1', 'M3'):
        for backend in ('E0', 'E1'):
            own = [r for r in rows if (r['task'], r['backend']) == (task, backend)]
            assert len(own) == 16
            groups.append(dict(task=task, backend=backend, success=sum(r['success'] == 'True' for r in own),
                               commands=[sum(r['success'] == 'True' for r in own if int(r['command']) == c) for c in (0, 1)],
                               reliable=sum(r['reliable'] == 'True' for r in own)))
    for row in rows:
        case = f"{row['family']}-{row['task']}-c{row['command']}-{row['backend']}"
        folder = RAW / 'virtual/PREFIX4096' / case
        result = read(folder / 'result.json')
        assert result['validity'] == 'VALID' and result['success'] == (row['success'] == 'True')
        assert sha(folder / 'trajectory.npz') == result['arrays_sha256']
        with np.load(folder / 'trajectory.npz') as z:
            spec = json.loads(str(z['spec_json'])); times = z['time']; select = times >= 35 - 1e-10
            assert len(times) == 2001 and select.sum() == 251 and abs(times[-1] - 40) < 1e-10
            xyz = np.linalg.norm(z['actual_p'][select] - spec['goal'], axis=-1)
            so3 = Rotation.from_matrix(z['actual_r'][select] @ np.array(spec['goal_rotation']).T).magnitude()
        retention = json.loads(row['retention'])
        assert abs(float(xyz.max()) - retention['final5s_max_xyz_m']) < 1e-10
        assert abs(float(so3.max()) - retention['final5s_max_so3_rad']) < 1e-10
        reliable = bool(retention['original_completed_by35'] and retention['full_episode_safe'] and np.all(xyz <= .035) and np.all(so3 <= .10))
        assert reliable == retention['reliable'] == (row['reliable'] == 'True')
    original = verdict['ORIGINAL_VIRTUAL_QUALIFICATION']['PREFIX4096']
    reliable = verdict['RELIABLE_VIRTUAL_QUALIFICATION']['PREFIX4096']
    assert original['full64_safe'] and reliable['full64_safe']
    status = 'PI0270_LEARNING_NOT_NEEDED_PREFIX_MAIN_RELIABLE' if original['PASS'] and reliable['PASS'] else 'PASS_CONTINUE_PI0270_A1'
    storage = {str(p): dict(zip(('total', 'used', 'free'), shutil.disk_usage(p))) for p in (BASE, RAW.parent)}
    assert all(v['free'] > 10 * 2**30 for v in storage.values())
    old_pids = subprocess.run(['pgrep', '-af', '/PI0269/code/'], capture_output=True, text=True).stdout.strip()
    assert not old_pids, ('PREDECESSOR_PROCESS_STILL_ACTIVE', old_pids)
    value = dict(schema='PI0270_SERIAL_A0_ACTUAL_CLOSEOUT_V1', status=status, release=RELEASE, canonical=5969189182,
                 observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), sources=sources,
                 predecessor_complete=True, prefix_groups=groups, original=original, reliable=reliable,
                 legacy=verdict['legacy_retention_vs_donor'], component=descriptive['component_groups'],
                 collector=binding, initializer=initializer, statistics_sha256=sha(stats), storage=storage,
                 quota='UNKNOWN_no_current_user_quota_report', gpu_snapshot=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used', '--format=csv,noheader'], text=True),
                 old_processes=old_pids, new_model_teacher_render_optimizer_calls=0,
                 selected_checkpoint='NONE', deployment_selected_checkpoint='NONE', learning_authorized=status == 'PASS_CONTINUE_PI0270_A1')
    write(destination, value)
    print(json.dumps(dict(status=status, prefix_groups=groups, predecessor_complete=True)), flush=True)


if __name__ == '__main__':
    main()

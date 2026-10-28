"""Bind admitted new data and the existing initializer without model copies."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0270')
CONTROL = ROOT.with_name('PI0269')


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True); text = json.dumps(value, indent=2, sort_keys=True)+'\n'
    if p.exists(): assert p.read_text() == text
    else: p.write_text(text)


def quota_bytes(text):
    row = re.search(r'^\s*e1192548\s*\|\|\s*([\d.]+)(GB|TB)\s*\|\s*([\d.]+)(GB|TB)\s*\|\|', text, re.M)
    if row is None: return None
    used, u, hard, h = row.groups(); scale = dict(GB=2**30, TB=2**40)
    # The usage column is rounded to one decimal; reserve an extra 0.1 unit.
    return max(0, int(float(hard)*scale[h]-(float(used)+.1)*scale[u]))


def main():
    staged = read(ROOT/'evidence/TRAINING_DATA_LOCAL_READBACK.json')
    assert staged['status'] == 'PASS_NEW_FIXED_DATA_PACKAGE_AND_ALL_MEMBERS_STAGED'
    assert read(ROOT/'evidence/DATA_COVERAGE_ADMISSION.json')['status'] == 'PASS_ALL96_FIXED_NEW_SUPPORT_CELLS'
    assert read(ROOT/'evidence/NEWFIT_ADMISSION.json')['status'] == 'PASS_FIXED_NEWFIT192_AVAILABLE'
    assert sha(ROOT/'PAIRING_PLAN.json') == 'fca9a29e69e4b63ec35ab995bacc60a4867e4b86a11e2d7a3c6397d4d316f5ef'
    stats = ROOT/'contracts/ROT/source/dataset/statistics.json'
    assert sha(stats) == '0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    prior = read(CONTROL/'DONOR_LOCAL_READBACK.json'); cp = Path(prior['path'])
    assert prior['status'] == 'PASS_PI0269_FIXED_ROT2048_COMPLETE_TRAINER_BINDING'
    assert prior['revision'] == '97d8bf027f07241e2206b05e07597e4719c891e4'
    assert prior['tree_sha256'] == '3db80d35d5e476fe5a90d540ff8959fb5edf3da43a988fde7f88ce9f54b9b147'
    for row in prior['files']:
        p = cp/row['path']; assert p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes']
    # The existing full Trainer/readback remains protected; only model bytes are consumed anew.
    checked = []
    for row in prior['files']:
        if row['path'] not in ('config.json','model.safetensors.index.json','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors'): continue
        assert sha(cp/row['path']) == row['sha256']; checked.append(row)
    assert len(checked) == 4 and read(cp/'statistics.json') == read(stats)
    target = ROOT/'DONOR_LOCAL_READBACK.json'
    if target.exists(): assert target.read_bytes() == (CONTROL/'DONOR_LOCAL_READBACK.json').read_bytes()
    else: target.write_bytes((CONTROL/'DONOR_LOCAL_READBACK.json').read_bytes())
    write(ROOT/'evidence/DONOR_REUSE_BINDING.json', dict(status='PASS_REUSED_FULL_TRAINER_RECEIPT_AND_CURRENT_MODEL_BYTES',
          source_receipt_sha256=sha(CONTROL/'DONOR_LOCAL_READBACK.json'), inherited_full_tree=prior['tree_sha256'],
          source_revision=prior['revision'], freshly_rehashed_model_files=checked,
          no_new_HF_historical_downloads=True, optimizer_used_for_initialization=False))
    for name in ('runtime', 'logs', 'design'): (ROOT/name).mkdir(exist_ok=True)
    storage = ROOT/'STORAGE_ADMISSION.json'
    if not storage.exists():
        unique = {}
        for p in ROOT.rglob('*'):
            if p.is_file() and not p.is_symlink():
                s = p.stat(); unique[s.st_dev,s.st_ino] = s.st_size
        df = int(subprocess.check_output(['/usr/bin/df','-B1','--output=avail','/scratch/e1192548'], text=True).splitlines()[-1])
        quota = subprocess.run(['/usr/bin/hpc','quota'], capture_output=True, text=True, timeout=30)
        text = quota.stdout+quota.stderr; q = quota_bytes(text); available = min(df, q) if q is not None else df
        size = prior['complete_trainer_bytes']; required = 5*size+10*2**30
        assert available >= required, ('ACTUAL_CANARY_MAIN_SAVE_OVERLAP_AND_MARGIN', available, required)
        write(storage, dict(available_bytes_at_freeze=available, owned_bytes_at_freeze=sum(unique.values()),
              quota_scope_path='/scratch/e1192548', quota_report=text if q is not None else 'UNKNOWN_'+text,
              quota_returncode=quota.returncode, parsed_quota_headroom_bytes=q, df_available_bytes=df,
              full_trainer_bytes=size, required_five_overlap_and_10GiB=required,
              canary_plan='Retire only discarded compared canary weights after durable evidence; keep both formal Trainers'))
    write(ROOT/'design/APPROVED_RUN_CARD.json', dict(schema='PI0270_ONE_FIXED_AGG_ARM_CARD_V1',
          release='7a6eea5ae38854bec7b80782213a92ca02fd0318', canonical=5969189182, launch_allowed=True,
          arms=['AGG'], updates_per_arm=4096, save_steps=[2048,4096], quality_steps=[4096],
          source_plan_sha256=sha(ROOT/'PAIRING_PLAN.json'), statistics_sha256=sha(stats),
          recovery_mapping_sha256=sha(ROOT/'RECOVERY_MAPPING.json'), data_revision=staged['revision'],
          donor_revision=prior['revision'], prefix_points=list(range(9)), prefix_transitions=list(range(8)), prefix_mass=.8, suffix_mass=.2,
          coefficients=dict(native=1,position=.4,increment=.1,rotation=.1,speed=.1),
          optimizer=dict(name='AdamW',learning_rate=1e-5,warmup=10,cosine_steps=4096,weight_decay=1e-5,clip=1.,accumulation=2),
          raw_update_cap=4368,raw_training_head_cap=80000, actual_CPU_runtime_and_real_canary_still_required=True,
          selected_checkpoint='NONE', deployment_selected_checkpoint='NONE'))
    print('PASS_CURRENT_MODEL_DATA_AND_SAVE_STORAGE_BINDINGS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--self-check', action='store_true')
    if p.parse_args().self_check:
        assert quota_bytes(' e1192548 || 648.4GB | 1.0TB || 519460 | unlimited') == int(2**40-648.5*2**30)
        assert quota_bytes('quota tool failed') is None
        print('PASS_ACTUAL_QUOTA_FORMAT_AND_UNKNOWN_FIXTURE_NO_RUNTIME_ADMISSION')
    else: main()

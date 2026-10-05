"""One finite owner; receipts describe executed work, never merely available code."""
import argparse, fcntl, hashlib, json, os, subprocess, sys
from pathlib import Path

RELEASE = '57fd8a9c53ce241c05a8d5b351034bed7ba56b14'
UPSTREAM = '62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3'
ROOT = Path(os.environ.get('PI0276_ROOT', '/home/liuyu-nfs/Projects/go2-arx5/PI0276'))
CODE = Path(__file__).resolve().parent

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n'); os.replace(tmp, p)

def audit():
    old = ROOT.parent / 'PI0274/evidence/QUALITY_QUERY_LANES_COMPLETE.json'
    assert json.loads(old.read_text())['status'] == 'ALL_FIXED_NEW_QUALITY_QUERY_LANES_COMPLETE'
    handoff = json.loads((ROOT/'evidence/HANDOFF.json').read_text())
    assert handoff['status'] == 'SUPERSEDED_BEFORE_NATIVE_RUN_BY_PI0276'
    conditions = {s: list(range(a, a+n)) for s,a,n in [('DEV',0,8),('TRAIN',10000,96),('VAL',20000,16),('TEST',30000,32)]}
    card = dict(release=RELEASE, upstream=UPSTREAM, task='StackCube-v1', robot='panda_wristcam',
        conditions=conditions, TEST_locked_until_final=True, instruction='Stack the red cube on the green cube and release it.',
        sim_freq=100, control_freq=20, control_steps=640, cameras=['base_camera','hand_camera'],
        policy_frame='PI0276_MANISKILL_WORLD', tcp='PI0276_PANDA_HAND_TCP', selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    path = ROOT/'design/CONDITIONS.json'
    if path.exists(): assert json.loads(path.read_text()) == card
    else: write(path, card)
    write(ROOT/'evidence/AUDIT.json', dict(status='PASS_CURRENT_HANDOFF_AND_FROZEN_CONDITIONS', card_sha256=sha(path), model_queries=0, physics_starts=0))

def setup():
    import importlib.metadata as md
    src = ROOT/'ManiSkill'
    assert subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip() == UPSTREAM
    assert not subprocess.check_output(['git','-C',str(src),'status','--porcelain'],text=True).strip()
    names = ['mani_skill','sapien','torch','numpy','mplib','pytorch_kinematics','Pillow']
    versions = {n: md.version(n) for n in names}
    assert versions['sapien'] == '3.0.3'
    files = list((src/'mani_skill').rglob('*.py')) + list((src/'mani_skill/assets/robots/panda').rglob('*'))
    files += list((src/'mani_skill/utils/scene_builder/table/assets').rglob('*'))
    files += list((ROOT/'project-src').rglob('*.py'))
    files = [p for p in files if p.is_file()] + list(CODE.glob('*.py'))
    inventory = [dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    assert any(x['path'].endswith('panda_v3.urdf') for x in inventory)
    write(ROOT/'design/SOURCE_FILESET.json', dict(upstream=UPSTREAM, files=inventory))
    write(ROOT/'evidence/SETUP.json', dict(status='PASS_ISOLATED_PINNED_SOURCE_AND_PACKAGES',
        python=sys.executable, versions=versions, dependency_lock_sha256=sha(ROOT/'evidence/DEPENDENCIES.lock.txt'),
        resize='Pillow.Image.Resampling.BILINEAR: native128x128RGB->96x96',
        source_fileset_sha256=sha(ROOT/'design/SOURCE_FILESET.json'), physics_starts=0, model_queries=0))

def native():
    from source_worker import native_fixtures
    native_fixtures()

def source():
    from source_worker import collect
    collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['all','audit','setup','native','source','bridge','dataset','admit','canary','train','evaluate','replay','archive','closeout'])
    parser.add_argument('--release', required=True); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); assert args.release == RELEASE
    ROOT.mkdir(exist_ok=True); (ROOT/'runtime').mkdir(exist_ok=True)
    # ponytail: one owner for this one experiment; flock releases on process exit.
    with (ROOT/'runtime/owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(ROOT/'runtime/owner.json', dict(pid=os.getpid(), command=sys.argv, python=sys.executable))
        stages = ['audit','setup','native','source','bridge','dataset','admit','canary','train','evaluate','replay','archive','closeout'] if args.stage=='all' else [args.stage]
        for stage in stages:
            print('PI0276_STAGE', stage, flush=True)
            operation = globals().get(stage)
            if operation is None: raise RuntimeError('STAGE_IMPLEMENTATION_PENDING: '+stage)
            operation()

if __name__ == '__main__': main()

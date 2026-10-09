"""Finite SSH staging/readback through the existing Corelab identities."""
import argparse
import base64
import json
from pathlib import Path
import subprocess

ROOT = '/home/liuyu-nfs/Projects/go2-arx5/PI0287'
WORKTREE = Path(__file__).resolve().parents[2]


def remote(code):
    args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'corelab-amd-server',
            'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'corelab-4080s', 'python3']
    p = subprocess.run(args, input=code, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stdout[-3000:] + p.stderr[-3000:])
    return p.stdout


def stage():
    files = [p for folder in ('scripts/pi0287_target_sim', 'src/go2arx5_vla/target_sim',
                              'tests/target_sim', 'reports/PI0287/design')
             for p in (WORKTREE / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for folder in ('src/go2arx5_vla/contracts', 'src/go2arx5_vla/trajectory'):
        files += [p for p in (WORKTREE / folder).rglob('*.py')]
    files += [WORKTREE / 'docs/protocols/PI0287_GO2_ARX5_TARGET_SIM_AND_EXPERT_DATA_V1.md',
              WORKTREE / 'src/go2arx5_vla/__init__.py']
    payload = {str(p.relative_to(WORKTREE)).replace('\\', '/'): base64.b64encode(p.read_bytes()).decode() for p in files}
    config = dict(wbc_root='/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare',
                  controller_run='/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare/runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018',
                  checkpoint_id='039999', controller_class='HISTORICAL_EXECUTOR_BASELINE',
                  original_release='7a9f0c02b55222646601307ccd7922e9297e2ab5', canonical=6084617266)
    print(remote(f"""from pathlib import Path
import base64,json,os
r=Path({ROOT!r})
for p in Path('/proc').iterdir():
 if p.name.isdigit() and int(p.name)!=os.getpid():
  try:
   c=(p/'cmdline').read_bytes()
   if b'/PI0287/' in c and b'run.py' in c:raise RuntimeError('Healthy owned sim runtime: do not restage')
  except (FileNotFoundError,PermissionError):pass
for folder in ['runtime/tmp','runtime/cache','runtime/torch_extensions','evidence','private']:(r/folder).mkdir(parents=True,exist_ok=True)
for name,data in {payload!r}.items():
 p=r/'project-src'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
p=r/'private/config.json'
if not p.exists():p.write_text(json.dumps({config!r},indent=2)+'\\n')
print('STAGED_PUBLIC_CODE_AND_DECLARED_HISTORICAL_CONFIG',len({payload!r}))
"""))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['stage', 'doctor']); a = p.parse_args()
    if a.mode == 'stage': stage()
    else:
        print(remote(f"""import subprocess
from pathlib import Path
r=Path({ROOT!r})
v=subprocess.run(['/bin/bash',str(r/'project-src/scripts/pi0287_target_sim/corelab_environment.sh'),str(r/'project-src/scripts/pi0287_target_sim/doctor.py'),'--config',str(r/'private/config.json'),'--output',str(r/'evidence/RUNTIME_MANIFEST.json')],capture_output=True,text=True)
print(v.returncode,v.stdout,v.stderr[-3000:])
"""))

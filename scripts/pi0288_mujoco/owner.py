"""Reuse the established SSH route; stage only original public integration code."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys

WORKTREE=Path(__file__).resolve().parents[2]
ROOT='/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0288'
CANDIDATE='/var/cache/users/liuyu-nfs/Projects/go2-arx5/executor-candidates/wbc-mujoco-6dd13098'
sys.path.insert(0,str(WORKTREE/'scripts/pi0287_target_sim'))
from owner import remote


def stage():
    paths=[p for folder in ['scripts/pi0288_mujoco','src/go2arx5_vla/target_mujoco','src/go2arx5_vla/trajectory','src/go2arx5_vla/contracts','src/go2arx5_vla/target_sim','tests/target_mujoco']
           for p in (WORKTREE/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths+=[WORKTREE/'src/go2arx5_vla/__init__.py']
    payload={str(p.relative_to(WORKTREE)).replace('\\','/'):base64.b64encode(p.read_bytes()).decode() for p in paths}
    parent=json.loads((WORKTREE/'reports/executor_candidates/OCS2_MUJOCO_20261010/NATIVE_REPRODUCTION_RESULT.json').read_text())
    hashes={x['path']:x['sha256'] for x in parent['identities']}
    hashes.update({str(Path(x['path']).relative_to(CANDIDATE+'/wbc_mujoco_minimal')).replace('\\','/'):x['sha256'] for x in parent['linkage']['executables']})
    config=dict(private_root=CANDIDATE+'/wbc_mujoco_minimal',core_hashes=hashes,release='ede6d93b0ced2b0d409f992e7168b066da1f2c32',canonical=6087400691)
    print(remote(f"""from pathlib import Path
import json,base64,os,subprocess
r=Path({ROOT!r})
for p in Path('/proc').iterdir():
 if p.name.isdigit() and int(p.name)!=os.getpid():
  try:
   c=(p/'cmdline').read_bytes()
   if b'/PI0288/' in c and b'run.py' in c:raise RuntimeError('Owned runtime exists: no restage')
  except (FileNotFoundError,PermissionError):pass
for name,data in {payload!r}.items():
 p=r/'project-src'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
for name in ['runtime','runtime/tmp','runtime/ros-logs','private','evidence']:(r/name).mkdir(exist_ok=True)
(r/'private/config.json').write_text(json.dumps({config!r},indent=2)+'\\n')
private=Path({CANDIDATE!r})/'wbc_mujoco_minimal'
lib=private/'src/ctrl/rl_sar/library/mujoco'
cmd=['g++','-shared','-fPIC','-O2','-std=c++17','-I'+str(lib/'include'),str(r/'project-src/src/go2arx5_vla/target_mujoco/observer.cpp'),'-L'+str(lib/'lib'),'-Wl,-rpath,'+str(lib/'lib'),'-lmujoco','-ldl','-o',str(r/'runtime/observer.so')]
p=subprocess.run(cmd,capture_output=True,text=True);print('BUILD_OBSERVER',p.returncode,p.stdout,p.stderr);p.check_returncode()
cmd=['g++','-O2','-std=c++17','-I'+str(lib/'include'),str(r/'project-src/src/go2arx5_vla/target_mujoco/audit_model.cpp'),'-L'+str(lib/'lib'),'-Wl,-rpath,'+str(lib/'lib'),'-lmujoco','-o',str(r/'runtime/audit_model')]
p=subprocess.run(cmd,capture_output=True,text=True);print('BUILD_CPU_AUDIT',p.returncode,p.stderr);p.check_returncode()
p=subprocess.run([str(r/'runtime/audit_model'),str(private/'src/ctrl/rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml')],capture_output=True,text=True);p.check_returncode()
(r/'evidence/DEFAULT_MODEL_CPU_AUDIT.json').write_text(p.stdout)
print(p.stdout)
print('STAGED',len({payload!r}))
"""))


def launch(mode,condition=None):
    args=['--config',ROOT+'/private/config.json','--root',ROOT,'--mode',mode]
    if condition:args+=['--condition',condition]
    import shlex
    env=f'''unset CONDA_PREFIX PYTHONHOME PYTHONPATH LD_PRELOAD
export WBC_ROS_DOMAIN_ID=228 ROS_LOCALHOST_ONLY=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST CUDA_VISIBLE_DEVICES=0
export TMPDIR={ROOT}/runtime/tmp ROS_LOG_DIR={ROOT}/runtime/ros-logs
source {CANDIDATE}/wbc_mujoco_minimal/env.sh
export PATH={CANDIDATE}/dependency-root/usr/bin:$PATH
export LD_LIBRARY_PATH={CANDIDATE}/dependency-root/opt/ros/jazzy/lib/x86_64-linux-gnu:{CANDIDATE}/dependency-root/opt/ros/jazzy/lib:{CANDIDATE}/dependency-root/usr/lib/x86_64-linux-gnu:{CANDIDATE}/wbc_mujoco_minimal/install/lib:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONPATH={ROOT}/project-src/src:$PYTHONPATH
exec /usr/bin/python3 {ROOT}/project-src/scripts/pi0288_mujoco/run.py {shlex.join(args)}'''
    print(remote(f"""import subprocess
p=subprocess.run(['/bin/bash','-c',{env!r}],capture_output=True,text=True)
print(p.stdout);print(p.stderr);print('EXIT',p.returncode)
"""))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['stage','interface-tests','native-reference']);p.add_argument('--condition');a=p.parse_args()
    if a.mode=='stage':stage()
    else:launch(a.mode,a.condition)

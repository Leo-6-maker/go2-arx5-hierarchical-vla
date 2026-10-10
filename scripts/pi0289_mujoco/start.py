"""One bounded original native stack; use the established environment and SSH route."""
import argparse
import runpy
from pathlib import Path

WORKTREE=Path(__file__).resolve().parents[2]
old=runpy.run_path(str(WORKTREE/'scripts/pi0288_mujoco/initialization_start.py'),run_name='pi0289_environment')
env=old['env'].replace('/go2-arx5/PI0288','/go2-arx5/PI0289').replace('/initialization-repair/','/startup-v1/')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['interface-tests','native-reference'],default='interface-tests');p.add_argument('--condition',default='hold');p.add_argument('--variant',type=int,choices=[1,2],default=1);a=p.parse_args()
    env=env.replace('/startup-v1/','/startup-v'+str(a.variant)+'/')
    env=env.replace('--mode interface-tests',f'--mode {a.mode} --condition {a.condition}')
    print(old['remote'](f'''import subprocess
p=subprocess.run(['/bin/bash','-c',{env!r}],capture_output=True,text=True)
print(p.stdout);print(p.stderr);print('EXIT',p.returncode)
'''))

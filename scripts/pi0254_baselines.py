"""Serial fixed baseline worker; no change to the sealed query plan."""
from pathlib import Path
import fcntl
import json
import subprocess
import sys

root=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0254')
lock=(root/'baselines.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
freeze=json.loads((root/'prequery_publication.json').read_text())
assert freeze['status']=='PREQUERY_FREEZE_PUBLISHED' and freeze['canonical_issue_comment']>5882151489
for point in ('G2000','FULL10000'):
    for extra in ([],['--cpu']):
        subprocess.run([sys.executable,'-u',str(root/'execution/pi0254_eval.py'),'--point',point,*extra],check=True)
subprocess.run([sys.executable,'-u',str(root/'execution/pi0254_compare.py'),'--p1'],check=True)

"""Finite remaining fixture sequence; recorded launches are never repeated."""
import json, os, subprocess, sys, time
from pathlib import Path
from budget import guard

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0281')
CONDITIONS = [('native','monotonic'), ('native','return'), ('repair','hold'),
              ('repair','monotonic'), ('repair','return')]

def main():
    guard()
    receipt = ROOT/'wbc/BATCH_START.json'
    assert not receipt.exists(), 'Adopt the existing batch; never duplicate starts'
    receipt.write_text(json.dumps(dict(pid=os.getpid(),begin=time.time(),conditions=CONDITIONS))+'\n')
    rows=[]
    for version, fixture in CONDITIONS:
        out=ROOT/'wbc'/(version+'_'+fixture)
        assert not (out/'LAUNCH.json').exists(), 'Recorded launch requires inspection'
        argv=[sys.executable,'-u',str(ROOT/'code/wbc_launch.py'),version,fixture]
        with (ROOT/'wbc/BATCH.log').open('ab') as log:
            code=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
        done=json.loads((out/'DONE.json').read_text()) if (out/'DONE.json').exists() else None
        rows.append(dict(condition=version+'_'+fixture,returncode=code,done=done))
        (ROOT/'wbc/BATCH_PROGRESS.json').write_text(json.dumps(rows,indent=2)+'\n')
        if code or not done or done['returncode']:
            raise RuntimeError('Engineering failure retained; inspect before recovery: '+json.dumps(rows[-1]))
    (ROOT/'wbc/BATCH_DONE.json').write_text(json.dumps(dict(status='FIXED_FIVE_EXECUTED_NOT_QUALITY_PASS',end=time.time(),rows=rows),indent=2)+'\n')

if __name__=='__main__': main()

"""Exactly three same-condition technical replacements for the unused alias."""
import json, os, subprocess, sys, time
from pathlib import Path
from budget import guard
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0281')
def main():
    guard()
    assert (ROOT/'wbc/REPAIR_INSTALLATION_DEFECT.json').exists()
    start=ROOT/'wbc/RECOVERY_START.json'
    assert not start.exists(),'Adopt recorded replacements; never duplicate'
    start.write_text(json.dumps(dict(pid=os.getpid(),begin=time.time()))+'\n')
    rows=[]
    for fixture in ('hold','monotonic','return'):
        folder=ROOT/'wbc'/('repair_'+fixture+'-engineering-001')
        assert not (folder/'LAUNCH.json').exists()
        with (ROOT/'wbc/RECOVERY.log').open('ab') as log:
            code=subprocess.run([sys.executable,'-u',str(ROOT/'code/wbc_launch.py'),'repair',fixture,'--technical-replacement'],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
        done=json.loads((folder/'DONE.json').read_text()) if (folder/'DONE.json').exists() else None
        rows.append(dict(fixture=fixture,returncode=code,done=done))
        (ROOT/'wbc/RECOVERY_PROGRESS.json').write_text(json.dumps(rows,indent=2)+'\n')
        if code or not done or done['returncode']:raise RuntimeError(json.dumps(rows[-1]))
    (ROOT/'wbc/RECOVERY_DONE.json').write_text(json.dumps(dict(status='THREE_TECHNICAL_REPLACEMENTS_EXECUTED',rows=rows,end=time.time()),indent=2)+'\n')
if __name__=='__main__':main()

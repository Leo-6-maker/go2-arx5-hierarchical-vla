"""Finite serial native renderer dispatcher; no scheduler, training or physics."""
import json
import subprocess
import sys
from pathlib import Path
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267'
sys.path[:0]=[str(BASE/'PI0256/runtime'),str(BASE/'PI0250/runtime/reference-317a/scripts')]
import pi0255_run
pi0255_run.ROOT=ROOT
py,env=pi0255_run.environment();env['PYTHONPATH']=str(ROOT/'code')+':'+str(BASE/'PI0266/code')+':'+env['PYTHONPATH']
if '--fixtures' in sys.argv:
    subprocess.run([str(py),str(ROOT/'code/pi0267_virtual_renderer.py')],env=env,check=True)
else:
    card=json.loads((ROOT/'design/SCENE_CARD.json').read_text())
    for family in card['families']:
        if family['split']!='FEEDBACK_TRAIN':continue
        fid=family['family_id'];raw=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1/S2')/fid
        if (raw/'COMPLETE.json').exists():continue
        with (ROOT/'R1/evidence'/('S2-'+fid+'.log')).open('ab') as log:
            subprocess.run([str(py),str(ROOT/'code/pi0267_virtual_data.py'),fid],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(fid+' COMPLETE',flush=True)

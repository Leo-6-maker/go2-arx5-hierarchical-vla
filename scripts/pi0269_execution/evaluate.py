"""One finite, serialized evaluation; reuse the mature per-family process boundary."""
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from infer_worker import BASE,ROOT,read,sha,admitted

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0269')


def main():
    # ponytail: one evaluator owns the append-only query budgets, no scheduler needed.
    with (ROOT/'runtime/evaluation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        admitted()
        modern=BASE/'PI0250/runtime/venv/bin/python'
        env=dict(os.environ);env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
        env['PYTHONPATH']=str(ROOT/'code')
        for point in ('UNIFORM4096','PREFIX4096'):
            with (ROOT/'evidence'/('offline-'+point+'.log')).open('ab') as stream:
                subprocess.run([str(modern),'-u',str(ROOT/'code/offline.py'),'--point',point],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        sys.path.insert(0,str(BASE/'PI0256/runtime'))
        import pi0255_run
        pi0255_run.ROOT=BASE/'PI0267';native,env=pi0255_run.environment()
        env['PYTHONPATH']=':'.join([str(ROOT/'code'),str(BASE/'PI0267/code'),str(BASE/'PI0266/code'),env['PYTHONPATH']])
        env['PYTHONFAULTHANDLER']='1'
        families=[f for f in read(BASE/'PI0267/design/SCENE_CARD.json')['families'] if f['split']=='NEW_SIM_EVAL']
        assert len(families)==8
        for point in ('UNIFORM4096','PREFIX4096'):
            for family in families:
                fid=family['family_id']
                paths=[RAW/'virtual'/point/f'{fid}-{task}-c{c}-{backend}'/'result.json' for task in ('M1','M3') for c in (0,1) for backend in ('E0','E1')]
                if all(p.exists() for p in paths):
                    for p in paths:
                        result=read(p)
                        assert result['validity']=='VALID' and result['point']==point
                        assert result['evaluation_card_sha256']==sha(ROOT/'design/EVALUATION_CARD.json')
                        assert result['arrays_sha256']==sha(p.parent/'trajectory.npz')
                    continue
                admitted()
                with (ROOT/'evidence'/('virtual-'+point+'-'+fid+'.log')).open('ab') as stream:
                    subprocess.run([str(native),'-u',str(ROOT/'code/virtual_episode.py'),'--point',point,'--family',fid],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                assert all(p.exists() and read(p)['validity']=='VALID' for p in paths)
                print(json.dumps(dict(point=point,family=fid,status='EIGHT_VALID_COMPLETE')),flush=True)
        component=read(ROOT/'design/COMPONENT_CARD.json')
        for point in ('ROT2048','UNIFORM4096','PREFIX4096'):
            for family in dict.fromkeys(r['family'] for r in component['rows']):
                rows=[r for r in component['rows'] if r['family']==family]
                paths=[RAW/'component'/point/r['case']/'result.json' for r in rows]
                if all(p.exists() for p in paths):
                    for p in paths:
                        record=read(p);assert record['validity'] in ('VALID','PREQUERY_EXCLUDED') and record['point']==point
                        assert record['evaluation_card_sha256']==sha(ROOT/'design/EVALUATION_CARD.json')
                        if record['validity']=='VALID':assert record['arrays_sha256']==sha(p.parent/'trajectory.npz')
                    continue
                with (ROOT/'evidence'/('component-'+point+'-'+family+'.log')).open('ab') as stream:
                    subprocess.run([str(native),'-u',str(ROOT/'code/component_episode.py'),'--point',point,'--family',family],
                        env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                assert all(p.exists() for p in paths)
        print('FIXED_OFFLINE_2904_MAIN192_COMPONENT96_COMPLETE',flush=True)


if __name__=='__main__':main()

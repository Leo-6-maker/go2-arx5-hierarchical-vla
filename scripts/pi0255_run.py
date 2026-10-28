"""Finite expert pilot dispatcher; validation is the default, no training path."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/reference-317a/src')
from pi0255_collect import ROOT, REF, WBC, AUTH


def environment():
    native=Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/stage13c/runtime_env')
    runtime=ROOT/'runtime'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',MAX_JOBS='2')
    for name,folder in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('MPLCONFIGDIR','matplotlib'),('TORCH_EXTENSIONS_DIR','torch_extensions')]:
        (runtime/folder).mkdir(exist_ok=True);env[name]=str(runtime/folder)
    env['PYTHONPATH']=':'.join(map(str,[ROOT.parent/'PI0252/runtime/PI0226_pytorch3d_shim',WBC,native/'public_native/isaacgym/python',REF/'scripts',REF/'src',runtime]))
    env['LD_PRELOAD']='/lib/x86_64-linux-gnu/libstdc++.so.6'
    env['LD_LIBRARY_PATH']=':'.join(map(str,[native/'public_native/isaacgym/python/isaacgym/_bindings/linux-x86_64','/lib/x86_64-linux-gnu','/usr/lib/x86_64-linux-gnu',native/'public_native/python_env/lib/python3.8/site-packages/torch/lib',native/'native_py38_4080/lib',native/'public_native/python_env/lib']))
    env['PATH']=':'.join(map(str,[native/'native_py38_4080/bin',native/'public_native/python_env/bin','/usr/local/cuda/bin','/usr/bin','/bin']))
    return native/'native_py38_4080/bin/python',env


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--authorization')
    parser.add_argument('--episode');parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    registry=json.loads((ROOT/'design/task_registry_v1.json').read_text())
    py,env=environment();code=Path(__file__).resolve().parent
    # Run the real native entry point's harmless default path before any physics.
    subprocess.run([str(py),str(code/'pi0255_collect.py')],env=env,check=True)
    if not (args.pilot or args.episode):
        print(json.dumps(dict(status='VALIDATED_NO_PHYSICS',GR00T_loads=0)));return
    assert not args.validate_only and args.authorization==AUTH
    rows=[r for r in registry['episodes'] if r['split']=='PILOT' and (not args.episode or r['episode_id']==args.episode)]
    assert rows
    for row in rows:
        existing=list((ROOT/'episodes'/row['episode_id']).glob('attempt-*/result.json'))
        if any(json.loads(p.read_text())['validity']=='VALID' for p in existing):continue
        log=ROOT/'logs'/(row['episode_id']+'-%02d.log'%len(existing))
        with log.open('wb') as stream:
            done=subprocess.run([str(py),str(code/'pi0255_collect.py'),'--episode',row['episode_id'],'--authorization',AUTH],env=env,stdout=stream,stderr=subprocess.STDOUT)
        if done.returncode:
            print(json.dumps(dict(status='ENGINEERING_ATTENTION',episode=row['episode_id'],exit_code=done.returncode,log=str(log))),flush=True)
            raise SystemExit(done.returncode)
        print(json.dumps(dict(status='FIRST_VALID_RETAINED',episode=row['episode_id'],log=str(log))),flush=True)


if __name__=='__main__':main()

"""Run finite native WBC phases in its existing, isolated Python 3.8 runtime."""
import argparse,base64,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote
ROOT='/home/liuyu-nfs/Projects/go2-arx5/PI0280'

def main(phase):
    payload=base64.b64encode(Path(__file__).with_name('wbc.py').read_bytes()).decode()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    print(remote('core',f'''import base64,hashlib,json,os,subprocess
from pathlib import Path
r=Path({ROOT!r});base=r.parent;runtime=r/'runtime';code=r/'code/wbc.py'
assert json.loads((base/'PI0279/runtime/SCIENTIFIC_PAUSE.json').read_text())['launch_allowed'] is False
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:assert str(code).encode() not in (p/'cmdline').read_bytes().split(bytes([0])),'WBC worker active'
 except OSError:pass
data=base64.b64decode({payload!r});sha=hashlib.sha256(data).hexdigest()
history=r/'design/wbc-implementation';history.mkdir(exist_ok=True)
(history/(sha+'.py')).write_bytes(data);code.write_bytes(data)
(r/'design/WBC_IMPLEMENTATION.json').write_text(json.dumps(dict(commit={commit!r},sha256=sha,training_updates=0)))
n=Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/stage13c/runtime_env')
w='/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare'
ref=base/'PI0250/runtime/reference-317a'
env=dict(os.environ,CUDA_VISIBLE_DEVICES={'' if phase=='bind' else '0'!r},PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MAX_JOBS='2',
 PYTHONPATH=':'.join([str(base/'PI0252/runtime/PI0226_pytorch3d_shim'),w,str(n/'public_native/isaacgym/python'),str(ref/'scripts'),str(ref/'src'),str(r/'code')]),
 LD_PRELOAD='/lib/x86_64-linux-gnu/libstdc++.so.6',
 LD_LIBRARY_PATH=':'.join([str(n/'public_native/isaacgym/python/isaacgym/_bindings/linux-x86_64'),'/lib/x86_64-linux-gnu','/usr/lib/x86_64-linux-gnu',str(n/'public_native/python_env/lib/python3.8/site-packages/torch/lib'),str(n/'native_py38_4080/lib'),str(n/'public_native/python_env/lib')]),
 PATH=':'.join([str(n/'native_py38_4080/bin'),str(n/'public_native/python_env/bin'),'/usr/local/cuda/bin','/usr/bin','/bin']))
for key in ['TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_EXTENSIONS_DIR']:
 p=runtime/key.lower();p.mkdir(exist_ok=True);env[key]=str(p)
with (runtime/('wbc-'+{phase!r}+'.log')).open('ab') as log:
 child=subprocess.Popen([str(n/'native_py38_4080/bin/python'),'-u',str(code),{phase!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(runtime/('wbc-'+{phase!r}+'.start.json')).write_text(json.dumps(dict(pid=child.pid,sha256=sha,phase={phase!r})))
print(json.dumps(dict(pid=child.pid,phase={phase!r})))
'''))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['bind','hold','vertical']);main(parser.parse_args().phase)

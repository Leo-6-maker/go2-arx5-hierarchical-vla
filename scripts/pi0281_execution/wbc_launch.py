"""Finite remote native WBC launcher, with a completion/resource receipt."""
import hashlib, json, os, resource, subprocess, sys, time
from pathlib import Path
from budget import guard

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0281')


def main(version, fixture, technical=False):
    guard()
    seal=json.loads((ROOT/'design/IMPLEMENTATION.json').read_text())
    for name in ('wbc.py','wbc_repair.py','wbc_cpu.py','wbc_readout.py','wbc_launch.py'):
        assert seal['files'].get(name)==hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest(),('WBC_NOT_IN_FROZEN_PUBLICATION',name)
    assert version in ('native','repair') and fixture in ('hold','monotonic','return')
    if technical:assert version=='repair' and (ROOT/'wbc/REPAIR_INSTALLATION_DEFECT.json').exists()
    out = ROOT/'wbc'/(version+'_'+fixture+('-engineering-001' if technical else ''));out.mkdir(parents=True,exist_ok=True)
    done=out/'DONE.json';start=out/'LAUNCH.json'
    assert not start.exists(), 'Adopt recorded attempt; no quality retry'
    runtime=Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/stage13c/runtime_env')
    private='/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare'
    ref=ROOT.parent/'PI0250/runtime/reference-317a'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MAX_JOBS='2',
      PYTHONPATH=':'.join([str(ROOT.parent/'PI0252/runtime/PI0226_pytorch3d_shim'),private,str(runtime/'public_native/isaacgym/python'),str(ref/'scripts'),str(ref/'src'),str(ROOT/'code')]),
      LD_PRELOAD='/lib/x86_64-linux-gnu/libstdc++.so.6',
      LD_LIBRARY_PATH=':'.join([str(runtime/'public_native/isaacgym/python/isaacgym/_bindings/linux-x86_64'),'/lib/x86_64-linux-gnu','/usr/lib/x86_64-linux-gnu',str(runtime/'public_native/python_env/lib/python3.8/site-packages/torch/lib'),str(runtime/'native_py38_4080/lib'),str(runtime/'public_native/python_env/lib')]),
      PATH=':'.join([str(runtime/'native_py38_4080/bin'),str(runtime/'public_native/python_env/bin'),'/usr/local/cuda/bin','/usr/bin','/bin']))
    for key in ('TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_EXTENSIONS_DIR'):
        p=ROOT/'wbc'/'runtime'/key.lower();p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
    argv=[str(runtime/'native_py38_4080/bin/python'),'-u',str(ROOT/'code/wbc.py'),version,fixture]
    if technical:argv.append('--technical-replacement')
    begin=time.time();monotonic=time.monotonic();status='UNKNOWN';returncode=None
    start.write_text(json.dumps(dict(pid=os.getpid(),argv=argv,begin=begin))+'\n')
    try:
        with (out/'worker.log').open('ab') as log:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            try:
                returncode=child.wait(timeout=3600)
                status='PROCESS_COMPLETE' if returncode==0 else 'PROCESS_ERROR'
            except subprocess.TimeoutExpired:
                child.terminate()
                try:returncode=child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();returncode=child.wait()
                status='TECHNICAL_TIMEOUT'
    finally:
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt=dict(status=status,returncode=returncode,wall_s=time.monotonic()-monotonic,begin=begin,end=time.time(),
          child_cpu_user_s=usage.ru_utime,child_cpu_system_s=usage.ru_stime,child_maxrss_kib=usage.ru_maxrss,
          child_minor_faults=usage.ru_minflt,child_major_faults=usage.ru_majflt,
          child_block_input=usage.ru_inblock,child_block_output=usage.ru_oublock)
        done.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    assert len(sys.argv) in (3,4)
    technical=len(sys.argv)==4
    if technical:assert sys.argv[3]=='--technical-replacement'
    main(sys.argv[1],sys.argv[2],technical)

"""Keep one foreground Hopper SSH session alive for a CPU archival worker."""
import argparse
import json
import sys
from pathlib import Path


def run(arm, readback_only=False):
    assert arm in ('C0_REPLAY1024', 'C1_CORRECT1024')
    slug = 'c0' if arm.startswith('C0_') else 'c1'
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'scripts/pi0276_execution'))
    from coordinator import remote
    script = f'''import fcntl,hashlib,json,os,resource,subprocess,time
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286')
folder=r/'archive/{slug}_full_trainer';target=r/'archive/readback_trainer_parallel.py' if {readback_only!r} else folder/'archive_{slug}_trainer.py'
python='/scratch/e1192548/stage22_h200_runtime_deb920_20260902/runtime/venv/bin/python'
with (r/'archive/ONE_CPU_ARCHIVE_FOREGROUND.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (folder/'COMPLETE.json').exists(),'Archive already complete; use verifier'
 started=time.time();argv=[python,'-u',str(target)]+(['--arm',{arm!r}] if {readback_only!r} else [])
 with (folder/'WORKER_CPU.log').open('ab') as log:
  child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  value=dict(status='STARTED_SOLE_CPU_ARCHIVE_FOREGROUND_SSH_KEEPALIVE',pid=child.pid,wrapper_pid=os.getpid(),argv=argv,started_epoch=started,wrapper_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),max_CPU_threads=4,memory_limit_bytes=16*1024**3,model_queries=0,optimizer_updates=0,physics_calls=0,training_code_seal_changed=False)
  (folder/'WORKER_START.json').write_text(json.dumps(value,indent=2)+'\\n')
  code=child.wait()
 usage=resource.getrusage(resource.RUSAGE_CHILDREN)
 done=dict(status='PASS_ARCHIVE_WORKER_EXIT' if code==0 else 'ARCHIVE_CPU_FAILURE_OR_PARTIAL_RETAIN_LOCAL_TRAINER',returncode=code,pid=child.pid,ended_epoch=time.time(),user_CPU_s=usage.ru_utime,system_CPU_s=usage.ru_stime,model_queries=0,optimizer_updates=0,physics_calls=0)
 (folder/'WORKER_DONE.json').write_text(json.dumps(done,indent=2)+'\\n')
 print(json.dumps(done))
 if code==0:
  subprocess.run([python,str(r/'archive/verify_trainer_archive.py'),'--arm',{arm!r}],check=True)
 else:raise SystemExit(code)
'''
    output = remote('hop', script)
    folder = root / 'reports/PI0286/archival' / slug.upper()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'FOREGROUND_FINAL_OUTPUT.txt').write_text(output)
    print(output, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--readback-only', action='store_true')
    args = parser.parse_args()
    run(args.arm, args.readback_only)

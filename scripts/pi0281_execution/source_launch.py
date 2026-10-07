"""One finite source run with waited children and actual process resource receipts."""
import json,os,resource,subprocess,time,fcntl
from pathlib import Path
from source import BASE,ROOT,OUT,guard,write

def usage(value):return {key:getattr(value,key) for key in ['ru_utime','ru_stime','ru_maxrss','ru_inblock','ru_oublock','ru_nvcsw','ru_nivcsw']}

def main():
    card=guard();runtime=OUT/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    with (runtime/'finite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (runtime/'start.json').exists(),'Adopt/audit recorded attempt; do not relaunch'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
            PYTHONPATH=':'.join([str(ROOT/'code'),str(BASE/'PI0276/ManiSkill'),str(BASE/'PI0276/project-src'),str(BASE/'PI0276/code')]))
        for key in ['TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR']:
            p=runtime/key.lower();p.mkdir(exist_ok=True);env[key]=str(p)
        before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic();wall_start=time.time_ns()
        with (runtime/'finite.log').open('ab') as log:
            child=subprocess.Popen(card['actual_run_command'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            write(runtime/'start.json',dict(pid=child.pid,wrapper_pid=os.getpid(),argv=card['actual_run_command'],unix_ns=wall_start,
                source_card_sha256=__import__('hashlib').sha256((OUT/'design/SOURCE_CARD.json').read_bytes()).hexdigest(),RUSAGE_CHILDREN_before=usage(before)))
            result=child.wait()
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        write(runtime/'done.json',dict(returncode=result,pid=child.pid,wrapper_pid=os.getpid(),elapsed_s=time.monotonic()-started,unix_start_ns=wall_start,unix_end_ns=time.time_ns(),
            RUSAGE_CHILDREN_before=usage(before),RUSAGE_CHILDREN_after=usage(after),user_CPU_s=after.ru_utime-before.ru_utime,system_CPU_s=after.ru_stime-before.ru_stime,
            waited_driver_and_policy_children=True,training_updates=0,backward=0,optimizer=0))
        assert result==0,'Finite source phase failed; actual attempt preserved'

if __name__=='__main__':main()

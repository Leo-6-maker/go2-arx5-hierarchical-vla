"""One finite, waited source panel; no background monitor or implicit relaunch."""
import argparse,fcntl,json,os,resource,subprocess,time
from pathlib import Path
from source import BASE,ROOT,OUT,guard,write,digest

def main(point,panel):
    card=guard(point,panel);runtime=OUT/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    name=point+'-'+panel
    with (runtime/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=runtime/(name+'.start.json');done=runtime/(name+'.done.json')
        assert not start.exists(),'Adopt the actual recorded process/result; do not repeat a panel'
        command=[str(BASE/'PI0276/venv/bin/python'),'-u',str(ROOT/'code/source.py'),'run',point,panel]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
            PYTHONPATH=':'.join([str(ROOT/'code'),str(BASE/'PI0276/ManiSkill'),str(BASE/'PI0276/project-src'),str(BASE/'PI0276/code')]))
        for key in ('TMPDIR','XDG_CACHE_HOME','MPLCONFIGDIR'):
            path=runtime/key.lower();path.mkdir(exist_ok=True);env[key]=str(path)
        before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic();utc=time.time_ns()
        with (runtime/(name+'.log')).open('ab') as log:
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            write(start,dict(pid=child.pid,wrapper_pid=os.getpid(),argv=command,utc_ns=utc,source_card_sha256=digest(ROOT/'design/SOURCE_CARD.json')))
            result=child.wait()
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        write(done,dict(returncode=result,pid=child.pid,wrapper_pid=os.getpid(),elapsed_s=time.monotonic()-started,utc_start_ns=utc,utc_end_ns=time.time_ns(),
            user_CPU_s=after.ru_utime-before.ru_utime,system_CPU_s=after.ru_stime-before.ru_stime,waited_driver_and_resident_worker=True,
            GPU_allocation_time_conservative=True,new_WBC_calls=0))
        assert result==0,'Actual engineering attempt retained; no automatic same-condition retry'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('point');parser.add_argument('panel');args=parser.parse_args();main(args.point,args.panel)

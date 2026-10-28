"""Finite continuation waits for the existing source owner, then admits its data."""
import fcntl,json,os,subprocess,sys
from run import ROOT,write,sha

def main():
    with (ROOT/'runtime/owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        write(ROOT/'runtime/owner.json',dict(pid=os.getpid(),command=sys.argv,stage='SOURCE_TO_DATA_ADMISSION'))
        assert json.loads((ROOT/'evidence/SOURCE.json').read_text())['status']=='PASS_FIXED_SOURCE_AND_EEF_TASK_GATES'
        paths=[ROOT/'code'/n for n in ('run.py','source_worker.py','replay.py','dataset.py','admit.py','finish_source.py')]
        write(ROOT/'evidence/POST_SOURCE_FILESET.json',dict(files=[dict(path=str(p),sha256=sha(p)) for p in paths]))
        from replay import main as replay
        from dataset import main as dataset
        replay();dataset()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott')
        subprocess.run(['/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python','-u',str(ROOT/'code/admit.py')],env=env,check=True)
        write(ROOT/'evidence/SOURCE_TO_DATA_COMPLETE.json',dict(status='PASS_ACTUAL_SOURCE_TO_DATA_ADMISSION',
            data_admission_sha256=sha(ROOT/'evidence/DATA_ADMISSION.json'),canary='NOT_YET_RUN',training='NOT_YET_RUN',goal_terminal=False))

if __name__=='__main__':
    try:main()
    except BaseException as err:
        write(ROOT/'evidence/SOURCE_TO_DATA_FAILURE.json',dict(error=repr(err),status='SOURCE_TO_DATA_INCOMPLETE',launch_allowed=False))
        raise

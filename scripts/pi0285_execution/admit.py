"""One finite CPU dependency: adopt collection, audit original arrays, freeze native bank."""
import json,os,resource,subprocess,sys,time
from pathlib import Path
from contract import ROOT,guard,sha,write


def run():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';guard()
    start=json.loads((ROOT/'runtime/COLLECTION_PROCESS.json').read_text());pid=start['pid']
    deadline=start['start_epoch']+4*3600;completed=ROOT/'runtime/COLLECTION_COMPLETE.json'
    # ponytail: wait for this one existing dependency; no scheduler or recurring monitor.
    while not completed.exists():
        assert (Path('/proc')/str(pid)).exists(),'Original collection exited without its completion receipt'
        assert time.time()<deadline,'Finite collection dependency timeout; inspect original process'
        time.sleep(10)
    value=json.loads(completed.read_text());assert value['returncode']==0,value
    before=resource.getrusage(resource.RUSAGE_CHILDREN);began=time.time()
    for script,arguments in [('audit_source.py',['--split','TRAIN']),('audit_source.py',['--split','DEV']),('data.py',['--root',str(ROOT/'data')]),('codec_check.py',['--root',str(ROOT/'data')])]:
        name=script.replace('.py','')+'-'+(arguments[-1] if script=='audit_source.py' else 'BANK')
        done=ROOT/'runtime'/(name+'.CPU_COMPLETE.json')
        if done.exists():assert json.loads(done.read_text())['returncode']==0;continue
        with (ROOT/'runtime'/(name+'.CPU.log')).open('ab') as log:
            phase=time.time();result=subprocess.run([sys.executable,'-u',str(ROOT/'code'/script),*arguments],stdout=log,stderr=subprocess.STDOUT)
        write(done,dict(returncode=result.returncode,script=script,arguments=arguments,started_epoch=phase,ended_epoch=time.time(),CUDA_VISIBLE_DEVICES='',new_neural_queries=0,new_physics=0))
        assert result.returncode==0,('Actual CPU contract not closed; original failure retained',name)
    import numpy as np
    from data import Dataset
    dataset=Dataset(ROOT/'data',ROOT/'data/PLACE4096.jsonl');row=dataset.rows[0];base,wrist,state,_=dataset.raw(row)
    path=ROOT/'data/CANARY_INPUT.npz'
    if path.exists():
        with np.load(path) as f:assert np.array_equal(f['actual_state'],state) and np.array_equal(f['base_raw'],base) and np.array_equal(f['wrist_raw'],wrist) and str(f['instruction'])==row['instruction']
    else:np.savez_compressed(path,actual_state=state,base_raw=base,wrist_raw=wrist,instruction=np.asarray(row['instruction']))
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    write(ROOT/'runtime/DATA_ADMISSION_COMPLETE.json',dict(status='PASS_ACTUAL_BANK_CPU_READBACK_FULL_CODEC_TWO_WORKERS_CANARY_INPUT',collection_complete_sha256=sha(completed),data_codec_sha256=sha(ROOT/'data/DATA_CODEC_ADMISSION.json'),canary_source_UID=row['uid'],canary_input_sha256=sha(path),elapsed_wall_s=time.time()-began,user_CPU_s=after.ru_utime-before.ru_utime,system_CPU_s=after.ru_stime-before.ru_stime,new_model_queries=0,new_optimizer_updates=0,new_physics=0,selected='NONE',deployment='NONE'))


if __name__=='__main__':run()

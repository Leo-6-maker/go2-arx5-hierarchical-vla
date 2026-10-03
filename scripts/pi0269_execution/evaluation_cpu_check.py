"""Finite actual condition identity and original executor/gate CPU checks."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from infer_worker import BASE,ROOT,read,sha,write,build,batch,identity

def main():
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30))
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    native_call=torch.nn.Module.__call__
    def guard(self,*a,**kw):
        if type(self).__module__.startswith('torchvision.transforms'):return native_call(self,*a,**kw)
        raise RuntimeError('NO_LEARNED_CPU_QUERY')
    torch.nn.Module.__call__=guard
    def forbidden(*a,**kw):raise RuntimeError('NO_CPU_OPTIMIZER')
    torch.optim.Optimizer.__init__=forbidden
    card=read(ROOT/'design/FIXED_OFFLINE_CARD.json');proc=build(read(ROOT/'contracts/ROT/source/dataset/statistics.json'),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    rows=card['fit_rows'][::2]+card['factor_rows'];checks=[]
    for row in rows:
        assert sha(row['input'])==row['input_sha256']
        with np.load(row['input']) as z:
            packet=dict(body_rgb=z['body_rgb'],wrist_rgb=z['wrist_rgb'],state_v1=z['state_v1'],language=row['language']);raw=z['raw_action']
        train=batch(proc,packet,raw)['inputs'];condition=batch(proc,packet)['inputs']
        assert 'action' not in condition and 'action_mask' not in condition
        assert identity(condition)==identity({k:v for k,v in train.items() if k not in ('action','action_mask')})
        if row['kind']=='fit':assert identity(condition)==row['processed']['ROT']['condition_sha256']
        checks.append(dict(uid=row['uid'],condition_sha256=identity(condition),train_eval_condition_equal=True))
    assert len(checks)==416
    sys.path.insert(0,str(BASE/'PI0256/runtime'));import pi0255_run
    pi0255_run.ROOT=BASE/'PI0267';native,env=pi0255_run.environment()
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    env['PYTHONPATH']=':'.join([str(ROOT/'code'),str(BASE/'PI0267/code'),str(BASE/'PI0266/code'),env['PYTHONPATH']])
    run=subprocess.run([str(native),str(ROOT/'code/virtual_episode.py'),'--self-check'],env=env,capture_output=True,text=True)
    assert run.returncode==0,(run.stdout,run.stderr)
    assert 'PASS_E0_E1_NONIDENTITY_PREFIX_AND_TIMESTAMPED_ORIGINAL_GATE_ZERO_QUERY_ZERO_PHYSICS' in run.stdout
    write(ROOT/'evidence/EVALUATION_CPU_CHECK.json',dict(status='PASS_PI0269_ACTUAL_SHARED_ROT_INPUTS_AND_ORIGINAL_EXECUTOR_GATE',checks=checks,
        original_executor_cpu_stdout=run.stdout,source_virtual_sha256=sha(ROOT/'code/virtual_episode.py'),
        script_sha256=sha(__file__),model_queries=0,optimizer_updates=0,renderer_calls=0,physics_calls=0))
    print('PASS_PI0269_416_ACTUAL_CONDITIONS_AND_ORIGINAL_EXECUTOR',flush=True)

if __name__=='__main__':main()

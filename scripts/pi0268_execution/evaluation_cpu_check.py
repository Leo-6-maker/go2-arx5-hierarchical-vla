"""Policy-free train/eval processor boundary and fixed-factor input checks."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import json
import resource
import numpy as np
import torch
from freeze_contracts import ROOT,BASE,read,sha,write
from pi0260_r3_inputs import build,batch,identity
from reliability import self_check

def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));torch.set_num_threads(4);torch.set_num_interop_threads(1)
    native=torch.nn.Module.__call__
    def no_model(self,*a,**kw):
        if type(self).__module__.startswith('torchvision.transforms'):return native(self,*a,**kw)
        raise RuntimeError('EVALUATION_CPU_CHECK_NO_NEURAL_QUERY')
    torch.nn.Module.__call__=no_model
    def no_optimizer(*a,**kw):raise RuntimeError('EVALUATION_CPU_CHECK_NO_OPTIMIZER')
    torch.optim.Optimizer.__init__=no_optimizer
    self_check()
    card=read(ROOT/'design/FIXED_OFFLINE_CARD.json');rows=card['fit_rows'][::2]+card['factor_rows'];assert len(rows)==416
    checks=[];shared={}
    for arm in ('OLD','ROT'):
        proc=build(read(ROOT/'contracts'/arm/'source/dataset/statistics.json'),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
        for row in rows:
            assert sha(row['input'])==row['input_sha256']
            with np.load(row['input']) as z:
                packet=dict(body_rgb=z['body_rgb'],wrist_rgb=z['wrist_rgb'],state_v1=z['state_v1'],language=row['language'])
                raw=z['raw_action']
            train=batch(proc,packet,raw)['inputs'];policy=batch(proc,packet)['inputs']
            assert 'action' not in policy and 'action_mask' not in policy
            condition={k:v for k,v in train.items() if k not in ('action','action_mask')}
            key=identity(policy);assert key==identity(condition),'TRAIN_EVAL_CONDITION_MISMATCH'
            if row['kind']=='fit':assert key==row['processed'][arm]['condition_sha256']
            unmodified=identity({k:v for k,v in policy.items() if k!='state'})
            if arm=='OLD':shared[row['uid']]=unmodified
            else:assert unmodified==shared[row['uid']]
            checks.append(dict(arm=arm,uid=row['uid'],condition_sha256=key,train_eval_equal=True,pixels_tokens_equal=True))
        print('PASS_ACTUAL_EVAL_INPUTS',arm,len(rows),flush=True)
    write(ROOT/'evidence/EVALUATION_CPU_CHECK.json',dict(status='PASS_ACTUAL_TRAIN_EVAL_CONDITIONS_FACTOR_INPUTS_AND_RELIABILITY_BOUNDARIES',
        checks=checks,model_queries=0,gradient_queries=0,optimizer_updates=0,new_renders=0,new_physics=0,code_sha256=sha(__file__)))

if __name__=='__main__':main()

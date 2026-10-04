"""Serialized fixed requests, first-valid reuse, four exact direct repeats."""
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from quality_worker import BASE,ROOT,read,sha,write,admitted,count
sys.path.insert(0,str(BASE/'PI0267/code'))
from pi0267_offline import message


def main():
    admitted();card_path=ROOT/'design/EVALUATION_CARD.json';card=read(card_path)
    with (ROOT/'evidence/inference.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for point in ('S4FIT1024','DIRECT0','DIRECT1024'):
            rows=[r for r in card['rows'] if r['point']==point]
            if all(Path(r['output']).exists() and Path(r['output']).with_suffix('.receipt.json').exists() for r in rows):continue
            env=dict(os.environ);env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
            env.update(PYTHONPATH=str(ROOT/'code'),CUDA_VISIBLE_DEVICES='0')
            with (ROOT/'evidence'/f'worker-{point}.log').open('ab') as log:
                worker=subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'),str(ROOT/'code/quality_worker.py'),'--point',point],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
                try:
                    assert message(worker)==dict(ready=True,point=point)
                    for row in rows:
                        worker.stdin.write(json.dumps(row)+'\n');worker.stdin.flush();answer=message(worker)
                        assert answer['status'] in ('VALID','VALID_REUSED') and answer['output']==row['output']
                        print('FIXED_OUTPUT',point,row['panel'],row['input_id'],flush=True)
                    worker.stdin.close();assert worker.wait()==0
                finally:
                    if worker.poll() is None:worker.terminate();worker.wait()
        outputs=[];repeats=[]
        for q in card['rows']:
            p=Path(q['output']);receipt=read(p.with_suffix('.receipt.json'))
            assert receipt['status']=='VALID' and receipt['frozen_request']==q and receipt['output_sha256']==sha(p)
            assert receipt['binding']==read(ROOT/'model_bindings'/f"{q['point']}.json")
            assert receipt['input_sha256']==q['expected_condition_sha256'] and sha(q['input'])==q['input_sha256']
            assert receipt['heads']==(4 if q['readout']=='S4FIT' else 1) and receipt['execution_phase']=='quality'
            if q['readout']=='DIRECT':assert receipt['direct_generation_RNG_unchanged']
            if q['repeatability_only']:
                original=next(r for r in card['rows'] if r['point']==q['point'] and r['input_id']==q['input_id'] and r['panel']=='support')
                old=read(Path(original['output']).with_suffix('.receipt.json'))
                assert old['direct_generation_RNG_unchanged']
                assert old['direct_generation_RNG_identity']==receipt['direct_generation_RNG_identity']
                with np.load(p,allow_pickle=False) as a,np.load(original['output'],allow_pickle=False) as b:
                    for name in ('raw','decoded','zero_query','state'):
                        assert a[name].dtype==b[name].dtype and a[name].shape==b[name].shape and a[name].tobytes()==b[name].tobytes(),(q['input_id'],name)
                repeats.append(dict(input_id=q['input_id'],output=str(p),original=original['output'],exact_bytes=True,RNG_unchanged=True))
            outputs.append(dict(ordinal=q['ordinal'],point=q['point'],panel=q['panel'],input_id=q['input_id'],
                j=q['noise']['j'] if q['noise'] else None,path=str(p),sha256=sha(p),receipt_sha256=sha(p.with_suffix('.receipt.json'))))
        assert len(outputs)==2148 and len(repeats)==4
        attempts=count(ROOT/'evidence/model_attempts.jsonl');heads=count(ROOT/'evidence/model_heads.jsonl')
        assert 2148<=attempts<=2560 and 6228<=heads<=7168
        write(ROOT/'evidence/QUERY_OUTPUTS_COMPLETE.json',dict(status='ALL_FIXED_PI0272_OUTPUTS_COMPLETE',
            evaluation_card_sha256=sha(card_path),rows=outputs,repeatability=repeats,actual_attempts=attempts,actual_heads=heads,
            selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))


if __name__=='__main__':main()

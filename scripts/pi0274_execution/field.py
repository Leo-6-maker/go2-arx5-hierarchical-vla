"""Registered local poses and exact two noise banks; no new renders or teacher calls."""
import argparse,fcntl,json,os,subprocess
from pathlib import Path
import numpy as np
from quality_gate import BASE,ROOT,RAW,read,sha,admitted
from quality_offline import message,physical_score,write
def main(point):
    admitted(point);card=read(ROOT/'design/POSE_CARD.json');inputs={r['pose_id']:r for r in read(ROOT/'design/FITTING_CARD.json')['field_inputs']}
    noises={(r['cell'],r['j']):r for r in card['noise']};source_sha=sha(ROOT/'design/POSE_CARD.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(ROOT/'code'),OPENBLAS_CORETYPE='Prescott');env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    # ponytail: finite field and endpoint evaluators share one budget owner at a time.
    with (ROOT/'runtime/quality.lock').open('a') as lock,(ROOT/'evidence'/('field-worker-'+point+'.log')).open('ab') as log:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        worker=subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'),str(ROOT/'code/quality_worker.py'),'--point',point],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
        try:
            assert message(worker)==dict(ready=True,point=point)
            count=0
            for row in card['rows']:
                item=inputs[row['id']];sample=Path(item['path']);assert sha(sample)==item['sample_sha256']
                with np.load(sample,allow_pickle=False) as z:state=z['state_v1'].copy();expert=z['raw_action'].copy()
                for j in (0,1):
                    noise=noises[row['cell'],j];folder=RAW/'field'/point/row['id']/f'j{j}';folder.mkdir(parents=True,exist_ok=True);record=folder/'record.json';prediction=folder/'prediction.npz'
                    if record.exists():
                        prior=read(record);assert prior['row']==row and prior['point']==point and prior['j']==j and prior['pose_card_sha256']==source_sha and sha(prediction)==prior['tensor_sha256'];count+=1;continue
                    request=dict(input=str(sample),output=str(prediction),language=row['field_language'],expected_condition_sha256=item['condition_sha256'],noise_file=noise['file'],noise_name=noise['name'],noise_registry='FIELD_QUERY_CARD.json')
                    worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();answer=message(worker);assert answer['status'] in ('VALID','VALID_REUSED') and answer['output']==str(prediction)
                    with np.load(prediction,allow_pickle=False) as z:decoded=z['decoded'].copy()
                    metrics,gate=physical_score(decoded,expert,state,'PI0274-FIELD-'+point+'-'+row['id']+'-'+str(j))
                    write(record,dict(status='VALID',point=point,row=row,j=j,metrics=metrics,gate=gate,condition_sha256=item['condition_sha256'],labels_masks_sha256=item['labels_masks_sha256'],tensor=str(prediction),tensor_sha256=sha(prediction),pose_card_sha256=source_sha,
                        TRAIN_constructed_diagnosis=True,blind_TEST=False,actual_episode_executed=False,teacher_used_for_control=False))
                    count+=1
                if count%208==0:print('REGISTERED_FIELD',point,count,flush=True)
            assert count==2496;worker.stdin.close();assert worker.wait()==0
            write(ROOT/'evidence'/('FIELD_COMPLETE_'+point+'.json'),dict(status='PASS_ALL2496_REGISTERED_FIELD_OUTPUTS',point=point,rows=count,pose_card_sha256=source_sha,teacher_calls=0,renders=0,physics=0))
        finally:
            if worker.poll() is None:worker.terminate();worker.wait()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('BROAD5120','EQUIL5120'),required=True);main(p.parse_args().point)

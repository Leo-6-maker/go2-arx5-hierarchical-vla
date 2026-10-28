"""Exactly the original 296 support conditions and j0/j1; no new noise."""
import argparse
import json
from pathlib import Path
import numpy as np
from quality_worker import ROOT,read,sha,write
from quality_offline import message,physical_score
import newfit

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0273')

def evaluate(point,worker):
    path=ROOT/'design/SUPPORT_CARD.json';rows=read(path)['late'];assert len(rows)==296
    for row in rows:
        assert sha(row['input'])==row['input_sha256'] and [n['j'] for n in row['noise']]==[0,1]
        with np.load(row['input'],allow_pickle=False) as z:expert=z['raw_action'].copy();state=z['state_v1'].copy()
        for noise in row['noise']:
            folder=RAW/'late'/point/row['input_id']/f"j{noise['j']}";folder.mkdir(parents=True,exist_ok=True)
            record=folder/'record.json';tensor=folder/'prediction.npz'
            if record.exists():
                old=read(record);assert old['row']==row and old['point']==point and old['j']==noise['j'] and old['card_sha256']==sha(path)
                assert old['tensor_sha256']==sha(tensor);continue
            request=dict(input=row['input'],output=str(tensor),language=row['language'],noise_file=noise['file'],
                noise_name='float32',noise_registry='LATE_QUERY_CARD.json',expected_condition_sha256=row['condition_sha256'])
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();answer=message(worker)
            assert answer['status'] in ('VALID','VALID_REUSED') and answer['output']==str(tensor)
            with np.load(tensor,allow_pickle=False) as z:decoded=z['decoded'].copy()
            metrics,gate=physical_score(decoded,expert,state,'PI0273-LATE-'+point+'-'+row['input_id'])
            write(record,dict(status='VALID',point=point,domain='late',row=row,j=noise['j'],metrics=metrics,gate=gate,
                tensor=str(tensor),tensor_sha256=sha(tensor),card_sha256=sha(path),TRAIN_support_diagnosis=True,blind_TEST=False))
    print('FIXED_LATE592_COMPLETE',point,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('BROAD5120','TARGET5120'),required=True)
    newfit.evaluate=evaluate;newfit.main(p.parse_args().point)

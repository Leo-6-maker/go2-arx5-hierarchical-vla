"""Registered query clients; only already-frozen conditions can reach the model."""
import argparse,json,os,subprocess
from pathlib import Path
import numpy as np
from evaluation_card import ROOT,OLD,RAW,read,sha,write,freeze,packet

class Server:
    def __init__(self,point):
        self.point=point;(ROOT/'runtime').mkdir(exist_ok=True)
        self.errors=(ROOT/'runtime'/f'infer-{point}.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott')
        self.process=subprocess.Popen(['/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python','-u',str(ROOT/'code/infer.py'),'--point',point],
            env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.errors,text=True,bufsize=1)
        while True:
            line=self.process.stdout.readline();assert line,'Model server exited before ready'
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if row.get('ready'):assert row['point']==point;break
    def query(self,value,noise,folder):
        folder.mkdir(parents=True,exist_ok=True);inp=folder/'input.npz';out=folder/'output.npz'
        if inp.exists():
            with np.load(inp,allow_pickle=False) as z:assert set(z.files)==set(value) and all(np.array_equal(z[k],v) for k,v in value.items())
        else:np.savez(inp,**value)
        request=dict(input=str(inp),output=str(out),noise_file=noise['file'],noise_name=noise['name'],noise_sha256=noise['sha256'])
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        while True:
            line=self.process.stdout.readline();assert line,'Model server terminated during counted request'
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if 'output' in row:assert row['output']==str(out);break
        with np.load(out,allow_pickle=False) as z:return z['decoded'][0].copy(),json.loads(str(z['receipt_json']))
    def close(self):
        self.process.stdin.close();code=self.process.wait();self.errors.close();assert code==0

def query_fixed(point):
    finished=ROOT/'evidence'/f'FIXED_QUERIES_{point}.json'
    if finished.exists():
        for row in read(finished)['results']:assert sha(row['output'])==row['sha256']
        print('REUSE_COMPLETE_FIXED_QUERIES',point,flush=True);return
    card=read(ROOT/'design/EVALUATION_CARD.json');noises={(r['lane'],r['condition'],r['index']):r for r in read(ROOT/'design/EVALUATION_NOISE.json')['rows']}
    lanes=['FIT'] if point=='TASK8192' else ['OFFLINE','FIT']
    server=Server(point);results=[]
    try:
        for lane in lanes:
            rows=card['fit'] if lane=='FIT' else card['offline']
            for row in rows:
                value=packet(row)
                for index in ((0,1) if lane=='FIT' else (0,)):
                    folder=RAW/'quality'/point/lane/row['uid']/f'n{index}'
                    pred,receipt=server.query(value,noises[lane,row['uid'],index],folder)
                    results.append(dict(point=point,lane=lane,uid=row['uid'],noise_index=index,output=str(folder/'output.npz'),sha256=sha(folder/'output.npz'),receipt=receipt))
            print('FIXED_QUERY_LANE_COMPLETE',point,lane,len(results),flush=True)
    finally:server.close()
    expected=128 if point=='TASK8192' else 320;assert len(results)==expected
    freeze(ROOT/'evidence'/f'FIXED_QUERIES_{point}.json',dict(status='PASS_ALL_REGISTERED_FIXED_QUERIES_RETAINED',point=point,results=results,physics_steps=0,
        quality='PENDING_CPU_METRICS_NOT_A_TASK_PASS',selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))

if __name__=='__main__':
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('TASK8192','CONTROL4096','GRIP4096'),required=True);a=p.parse_args()
    (ROOT/'runtime').mkdir(exist_ok=True)
    with (ROOT/'runtime/quality_owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);query_fixed(a.point)

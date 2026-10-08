"""Finite frozen parent study with exact reuse and one charged wrapper parity."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from prepare_core import write,sha

def reply(server,key):
    while True:
        line=server.stdout.readline()
        if not line:raise RuntimeError(f'Native worker ended before {key}; exit={server.poll()}')
        try:v=json.loads(line)
        except ValueError:continue
        if key in v:return v

def run(root):
    done=root/'diagnostics/RESULT.json'
    assert not done.exists(),'Reuse complete sampler study'
    hf_home=json.loads(Path('/home/liuyu-nfs/Projects/go2-arx5/PI0283/design/SOURCE_CARD.json').read_text())['hf_home']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HOME=hf_home,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1',PYTHONPATH=str(root/'code')+':/home/liuyu-nfs/Projects/go2-arx5/PI0280/official:/home/liuyu-nfs/Projects/go2-arx5/PI0276/project-src')
    worker=[sys.executable,'-u',str(root/'code/native_worker.py'),'--root',str(root)]
    if not (root/'design/NOISE_PACKETS.json').exists():subprocess.run(worker+['--prepare-noise'],env=env,check=True)
    data=json.loads((root/'design/PRE_SCORE_INPUTS.json').read_text());pairs=json.loads((root/'design/NATURAL_PAIRS.json').read_text())['pairs']
    started=time.monotonic();records=[];root.joinpath('runtime').mkdir(exist_ok=True)
    with (root/'runtime/native-worker.log').open('ab') as log:
        server=subprocess.Popen(worker,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        try:
            ready=reply(server,'ready');write(root/'runtime/PARENT_LOAD_ACTUAL.json',ready)
            first=data['rows'][0];parity=root/'diagnostics/NATIVE_WRAPPER_PARITY.npz'
            if not parity.exists():
                request=dict(K=4,input=first['input'],noise=first['packet0'],output=str(parity))
                server.stdin.write(json.dumps(request)+'\n');server.stdin.flush();reply(server,'output')
            with np.load(parity) as a,np.load(first['reused_K4_output']) as b:
                assert all(np.array_equal(a[n],b[n]) for n in ('raw','decoded','noise_uint16','actual_state')),'WRAPPER_NATIVE_PARITY'
            write(root/'runtime/WRAPPER_PARITY.json',dict(status='PASS_ACTUAL_PARENT_NATIVE_K4_EXACT_REUSED_PACKET_OUTPUT',new_charged_queries=1,heads=4,parity_sha256=sha(parity),prior_sha256=sha(first['reused_K4_output'])))
            for row in data['rows']:
                for packet in range(4):
                    for k in (4,16):
                        target=Path(row['input']).parent/f'K{k}-packet{packet}.npz';reused=k==4 and packet==0
                        if reused:target=Path(row['reused_K4_output'])
                        elif not target.exists():
                            request=dict(K=k,input=row['input'],noise=str(Path(row['input']).parent/f'noise-{packet}.npy'),output=str(target))
                            server.stdin.write(json.dumps(request)+'\n');server.stdin.flush();reply(server,'output')
                        with np.load(target) as saved:
                            assert str(saved['source_sha256'])==sha(row['input'])
                            assert np.array_equal(saved['noise_uint16'],np.load(Path(row['input']).parent/f'noise-{packet}.npy'))
                            if not reused:assert int(saved['K'])==k and saved['raw_states_uint16'].shape==(k+1,1,40,132)
                        records.append(dict(uid=row['uid'],index=row['index'],split=row['split'],phase=row['phase'],K=k,packet=packet,path=str(target),sha256=sha(target),reused=reused,intermediate_trace='UNOBSERVED_PRIOR_REUSE' if reused else 'SAVED_ALL_EULER_STATES'))
                        write(root/'diagnostics/PROGRESS.json',dict(rows=records))
            write(done,dict(status='COMPLETE_PAIRED_K4_K16_ALL48_FOUR_NOISE',records=records,rows=len(records),new_study_queries=336,prior_reused_queries=48,elapsed_s=time.monotonic()-started))
            from score import run as score
            chosen=score(root);natural=[]
            for index,pair in enumerate(pairs):
                for side in ('a','b'):
                    for packet in (0,1):
                        target=root/'natural'/f'pair-{index}-{side}-packet{packet}-K{chosen["K"]}.npz'
                        if not target.exists():
                            request=dict(K=chosen['K'],input=pair[side]['input'],noise=str(root/'natural'/f'pair-{index}-noise-{packet}.npy'),output=str(target))
                            server.stdin.write(json.dumps(request)+'\n');server.stdin.flush();reply(server,'output')
                        natural.append(dict(pair=index,side=side,packet=packet,path=str(target),sha256=sha(target)))
            write(root/'natural/RESULT.json',dict(status='COMPLETE_FIXED_AVAILABLE_NATURAL_PAIRS',K=chosen['K'],pairs=len(pairs),records=natural,weak_response_evidence='INCONCLUSIVE_TOO_FEW_PAIRS' if len(pairs)<6 else 'CPU_RESPONSE_RECOMPUTE_REQUIRED'))
        finally:
            server.stdin.close();code=server.wait(timeout=120)
            assert code==0,('NATIVE_WORKER_FAILED',code)
    print(json.dumps(dict(status='COMPLETE_PAIRED_K4_K16',rows=len(records),new_study_queries=336)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(Path(a.root))

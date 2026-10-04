"""Execute the sealed fit/factor/retention union, retaining first valid outputs."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from quality_worker import BASE,ROOT,R1,read,sha,write,admitted
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0273')
OLD_RAW=RAW.parent/'PI0267/R1'
sys.path[:0]=[str(BASE/'PI0267/code'),str(BASE/'PI0264/code'),str(BASE/'PI0263/code'),str(BASE/'PI0250/runtime/reference-317a/scripts')]
from pi0267_offline import message

def physical_score(decoded,expert,sv,chunk_id,legacy=False):
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,rot6d_rows_to_matrix
    from go2arx5_vla.providers import GR00TProviderConfig,GR00TTrajectoryProvider
    if not np.isfinite(decoded).all():return None,dict(decision='REJECT',reason='NONFINITE_POLICY_OUTPUT')
    try:rot6d_rows_to_matrix(decoded[0,:,3:9].astype(float))
    except ValueError:return None,dict(decision='REJECT',reason='INVALID_ROT6D_POLICY_OUTPUT')
    metrics=stage43._screen_metrics(decoded[0].astype(float),expert.astype(float),None)
    if legacy:chunk=selector._build_chunk(decoded,chunk_id=chunk_id)
    else:
        provider=GR00TTrajectoryProvider(decoded,config=GR00TProviderConfig(source_frame='PI0267_STATIC_POLICY_ANCHOR_WORLD_AXES',
            tcp_semantic_id='PI0267_VIRTUAL_TOOL_TCP_V1',action_sample_period_s=.05,timestamp_provenance='PI0267_FIXED_OFFLINE'),observation_timestamp=0.,chunk_id=chunk_id)
        chunk=provider.next_chunk(0.)
    gate=TrajectorySafetyGateV1().evaluate(chunk,now=0.,achieved_position=sv[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(sv[None,3:9]))[0]).as_dict()
    return metrics,gate


def evaluate(point,worker):
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    import pi0264_freeze_eval as legacy
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,rot6d_rows_to_matrix
    from go2arx5_vla.providers import GR00TProviderConfig,GR00TTrajectoryProvider
    card=read(R1/'design/RUNTIME_OFFLINE_CARD.json');card_sha=sha(R1/'design/RUNTIME_OFFLINE_CARD.json')
    arm='ROT'
    for domain,rows in [('fit',card['fit_rows']),('factor',card['factor_rows']),('legacy',card['legacy_rows'])]:
        for row in rows:
            folder=RAW/'offline'/point/domain/row['uid'];folder.mkdir(parents=True,exist_ok=True);record=folder/'record.json'
            if record.exists():
                prev=read(record);assert prev['row']==row and prev['point']==point and prev['card_sha256']==card_sha
                if prev.get('tensor'):assert sha(prev['tensor'])==prev['tensor_sha256']
                continue
            if domain=='factor' and not row['admitted']:
                write(record,dict(status='PREQUERY_EXCLUDED_KEEP_DENOMINATOR',point=point,row=row,card_sha256=card_sha,reasons=row['reasons']));continue
            if domain=='legacy':
                packet,expert=legacy.source(row);sv=packet['state_v1'];text=packet['language'];ip=folder/'input.npz'
                if not ip.exists():np.savez_compressed(ip,**{k:v for k,v in packet.items() if k!='language'})
                noise_file=str(OLD_RAW/'noise'/('legacy-'+row['noise']+'.npz'));noise_name='float32'
            else:
                ip=Path(row['input']);assert sha(ip)==row['input_sha256']
                with np.load(ip) as z:expert=z['raw_action'].copy();sv=z['state_v1'].copy()
                text=row['language'];noise_file=row['noise_file'];noise_name=row['noise_name']
            tensor=folder/'prediction.npz';request=dict(input=str(ip),output=str(tensor),language=text,noise_file=noise_file,noise_name=noise_name,noise_registry='RUNTIME_OFFLINE_CARD.json')
            if domain=='fit':request['expected_condition_sha256']=row['processed'][arm]['condition_sha256']
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();answer=message(worker)
            assert answer['status'] in ('VALID','VALID_REUSED') and answer['output']==str(tensor)
            with np.load(tensor) as z:decoded=z['decoded'].copy()
            metrics,gate=physical_score(decoded,expert,sv,'PI0273-'+point+'-'+row['uid'],legacy=domain=='legacy')
            write(record,dict(status='VALID',point=point,domain=domain,row=row,metrics=metrics,gate=gate,tensor=str(tensor),tensor_sha256=sha(tensor),
                card_sha256=card_sha,donor_exact_cache_reused=False,reference_semantics='ORIGINAL_TIME_PACED' if domain=='legacy' else 'ARCLENGTH_EXP_CORRECTION_V1'))
        print('OFFLINE_DOMAIN_COMPLETE',point,domain,len(rows),flush=True)

def main(point):
    admitted();env=dict(os.environ);env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    env['PYTHONPATH']=str(ROOT/'code')
    with (R1/'evidence'/('offline-worker-'+point+'.log')).open('ab') as log:
        worker=subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'),str(ROOT/'code/quality_worker.py'),'--point',point],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
        try:
            assert message(worker)==dict(ready=True,point=point)
            evaluate(point,worker);worker.stdin.close();assert worker.wait()==0
        finally:
            if worker.poll() is None:worker.terminate();worker.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('BROAD5120','TARGET5120'),required=True);main(p.parse_args().point)

"""Frozen 256 new-domain +440 OLD-contract rows per point, first valid only."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
sys.path[:0]=[str(BASE/'PI0266/code'),str(BASE/'PI0250/runtime/reference-317a/src')]
from pi0255_motion3 import write,sha


def read(p):return json.loads(Path(p).read_text())
def message(worker):
    while True:
        line=worker.stdout.readline()
        if not line:raise RuntimeError('INFERENCE_WORKER_EXITED_WITHOUT_RESULT')
        try:return json.loads(line)
        except ValueError:continue


def render_new():
    import isaacgym
    from scipy.spatial.transform import Rotation
    from pi0267_virtual_episode import language
    from pi0267_virtual_renderer import Renderer
    from pi0267_virtual_data import state
    from pi0267_expert import ArclengthCorrection
    audit=read(R1/'evidence/MAIN_TRAINING_AUDIT.json');assert audit['status']=='PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY' and len(audit['complete_checkpoints'])==4
    card=read(R1/'design/FIXED_OFFLINE_CARD.json');source=read(ROOT/'design/SCENE_CARD.json');reg=source['registry'];anchor=np.array(source['policy_anchor_world_m'])
    for f in source['families']:
        own=[r for r in card['new_rows'] if r['family']==f['family_id']]
        if not own:continue
        renderer=None;teachers={}
        try:
            for row in own:
                out=RAW/'offline/source'/row['uid'];out.mkdir(parents=True,exist_ok=True)
                receipt=out/'record.json'
                if receipt.exists():
                    saved=read(receipt);assert saved['row']==row and saved['arrays_sha256']==sha(out/'input.npz');continue
                key=row['task'],row['command']
                if key not in teachers:teachers[key]=ArclengthCorrection(reg,f,*key)
                teacher=teachers[key];p,r=teacher.at_length(row['ell_fraction']*teacher.ell[-1]);p,r=p[0],r[0]
                variant=row['variant']
                if variant:
                    axis=(variant-1)//2;sign=1 if variant%2 else -1;p[axis]+=.010*sign
                    r=Rotation.from_rotvec(np.eye(3)[axis]*.060*sign).as_matrix()@r
                if renderer is None:renderer=Renderer(reg,f)
                frames,info=renderer.place(p,r,0.);lp,lr,label=teacher.label(p,r)
                action=np.c_[lp-anchor,lr[:,:2,:].reshape(40,6),np.full(40,.5)].astype(np.float32)
                np.savez_compressed(out/'input.npz',body_rgb=frames['body'],wrist_rgb=frames['wrist'],state_v1=state(p,r,anchor),expert=action,
                    position_world=p,rotation_world=r,label_position_world=lp,label_rotation_world=lr)
                np.savez_compressed(out/'debug.npz',**{k:frames[k] for k in ('body_segmentation','wrist_segmentation','body_depth','wrist_depth')})
                write(receipt,dict(row=row,language=language(reg,f,*key),label_metadata=label,camera=info,arrays_sha256=sha(out/'input.npz'),
                    EVAL_post_freeze=True,physics_steps=0,teacher_not_in_policy_condition=True))
        finally:
            if renderer:renderer.close()
    print('PASS_FIXED_256_OFFLINE_SOURCE_GENERATED_POST_BOTH_FINALS',flush=True)


def evaluate(point,worker):
    sys.path[:0]=[str(BASE/'PI0250/runtime/reference-317a/scripts'),str(BASE/'PI0264/code'),str(BASE/'PI0263/code')]
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,rot6d_rows_to_matrix
    from go2arx5_vla.providers import GR00TProviderConfig,GR00TTrajectoryProvider
    sys.path.insert(0,str(BASE/'PI0264/code'));import pi0264_freeze_eval as old
    card=read(R1/'design/FIXED_OFFLINE_CARD.json');anchor=np.array(read(ROOT/'design/SCENE_CARD.json')['policy_anchor_world_m'])
    for domain,rows in (('new',card['new_rows']),('legacy',card['legacy_rows'])):
        for row in rows:
            folder=RAW/'offline'/point/domain/row['uid'];folder.mkdir(parents=True,exist_ok=True);record=folder/'record.json'
            if record.exists():
                saved=read(record);assert saved['row']==row and saved['point']==point and saved['card_sha256']==sha(R1/'design/FIXED_OFFLINE_CARD.json')
                assert saved['tensor_sha256']==sha(saved['tensor']);continue
            donor_reuse=domain=='legacy' and point=='donor'
            if domain=='new':
                source=RAW/'offline/source'/row['uid'];meta=read(source/'record.json');input_path=source/'input.npz'
                z=np.load(input_path);expert=z['expert'];sv=z['state_v1'];text=meta['language']
                noise_file,noise_name=row['noise_file'],row['noise_name']
            else:
                packet,expert=old.source(row);sv=packet['state_v1'];text=packet['language'];input_path=folder/'input.npz'
                if not input_path.exists():np.savez_compressed(input_path,**{k:v for k,v in packet.items() if k!='language'})
                noise_file=str(RAW/'noise'/('legacy-'+row['noise']+'.npz'));noise_name='float32'
            if donor_reuse:
                from pathlib import Path
                source=Path(row['PI0267_donor_cell']);original=read(source/'record.json');tensor=source/'tensors.npz'
                assert sha(tensor)==row['PI0267_donor_tensor_sha256'] and sha(source/'record.json')==row['PI0267_donor_record_sha256']
                values=np.load(tensor);assert np.array_equal(values['expert'],expert) and np.array_equal(values['state'],sv)
                decoded=values['decoded'];metrics=original['metrics'];gate=original['gate']
            else:
                tensor=folder/'prediction.npz';request=dict(input=str(input_path),output=str(tensor),language=text,noise_file=noise_file,noise_name=noise_name,
                    noise_registry='FIXED_OFFLINE_CARD.json',processor_contract='OLD' if domain=='legacy' else 'COMMON_VIRTUAL')
                if domain=='legacy':request['expected_condition_sha256']=row['identities']['condition_sha256']
                worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=message(worker)
                assert response['status'] in ('VALID','VALID_REUSED') and response['output']==str(tensor)
                decoded=np.load(tensor)['decoded']
                if np.isfinite(decoded).all():
                    metrics=stage43._screen_metrics(decoded[0].astype(float),expert.astype(float),None)
                    if domain=='legacy':
                        gate=TrajectorySafetyGateV1().evaluate(selector._build_chunk(decoded,chunk_id='PI0267-'+point+'-'+row['uid']),now=0.,
                            achieved_position=sv[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(sv[None,3:9]))[0]).as_dict()
                    else:
                        provider=GR00TTrajectoryProvider(decoded,config=GR00TProviderConfig(source_frame='PI0267_STATIC_POLICY_ANCHOR_WORLD_AXES',
                            tcp_semantic_id='PI0267_VIRTUAL_TOOL_TCP_V1',action_sample_period_s=.05,timestamp_provenance='PI0267_FIXED_OFFLINE'),
                            observation_timestamp=0.,chunk_id='PI0267-'+point+'-'+row['uid'])
                        gate=TrajectorySafetyGateV1().evaluate(provider.next_chunk(0.),now=0.,achieved_position=sv[:3],
                            achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(sv[None,3:9]))[0]).as_dict()
                else:metrics=None;gate=dict(decision='REJECT',reason='NONFINITE_POLICY_OUTPUT')
            write(record,dict(status='VALID',point=point,row=row,domain=domain,metrics=metrics,gate=gate,tensor=str(tensor),tensor_sha256=sha(tensor),
                card_sha256=sha(R1/'design/FIXED_OFFLINE_CARD.json'),donor_cache_reused=donor_reuse,
                reference_semantics='ORIGINAL_TIME_PACED' if domain=='legacy' else 'ARCLENGTH_EXP_CORRECTION_V1'))
            print(json.dumps(dict(point=point,domain=domain,uid=row['uid'])),flush=True)


def main(point):
    import os,subprocess
    from pi0267_infer_worker import admitted
    admitted();assert len(list((RAW/'offline/source').glob('*/record.json')))==256
    env=dict(os.environ);env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None);env['PYTHONPATH']=str(ROOT/'code')
    log=(R1/'evidence'/('offline-worker-'+point+'.log')).open('ab')
    worker=subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'),str(ROOT/'code/pi0267_infer_worker.py'),'--point',point],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
    try:
        assert message(worker)==dict(ready=True,point=point);evaluate(point,worker);worker.stdin.close();assert worker.wait()==0
    finally:
        if worker.poll() is None:worker.terminate();worker.wait()
        log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('donor','N4096','R4096'));p.add_argument('--render',action='store_true');a=p.parse_args()
    if a.render:assert a.point is None;render_new()
    else:assert a.point;main(a.point)

"""Fixed first-two-TRAIN expert/model WBC transfer through the mature physical episode."""
import isaacgym
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
sys.path[:0]=[str(R1/'code'),str(ROOT/'code'),str(BASE/'PI0266/code')]
spec=importlib.util.spec_from_file_location('pi0267_physical_control',R1/'code/pi0266_control.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
from pi0267_expert import ArclengthCorrection
from pi0267_metrics_v3 import score
from pi0267_virtual_episode import language,message
from pi0255_motion3 import write,sha
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix


def read(p):return json.loads(Path(p).read_text())


def run(point,case_id):
    audit=read(R1/'evidence/MAIN_TRAINING_AUDIT.json');assert audit['status']=='PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY'
    card=read(R1/'design/TEACHER_CARD.json');case=next(c for c in card['diagnostics'] if c['case_id']==case_id)
    out=RAW/'diagnostics'/case_id
    prior=list(out.glob('attempt-*/result.json'))
    valid=[read(p) for p in prior if read(p)['validity']=='VALID']
    if valid:assert len(valid)==1;print('VALID_SECONDARY_REUSED',case_id,flush=True);return
    assert len(prior)==len(list(out.glob('attempt-*'))),'UNRESOLVED_SECONDARY_ATTEMPT'
    worker=None;query=None;log=None
    assert case['point']==point
    if point!='expert':
        env=dict(os.environ);env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None);env['PYTHONPATH']=str(ROOT/'code')
        log=(R1/'evidence'/('WBC-worker-'+case_id+'.log')).open('ab')
        worker=subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'),str(ROOT/'code/pi0267_infer_worker.py'),'--point',point],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
        assert message(worker)==dict(ready=True,point=point)
        ordinal=0;anchor=np.array(read(R1/'design/SCENE_CARD.json')['policy_anchor_world_m']);family=case['family']
        def query(state,body,wrist,p,r,now,folder,tick):
            nonlocal ordinal
            assert ordinal<100
            frames=folder/'model';frames.mkdir(exist_ok=True);ip=frames/('%03d-input.npz'%ordinal);op=frames/('%03d-prediction.npz'%ordinal)
            np.savez_compressed(ip,body_rgb=body,wrist_rgb=wrist,state_v1=state,position_world=p,rotation_world=r,time=now)
            request=dict(input=str(ip),output=str(op),language=language(None,family,case['task'],case['command']),
                noise_registry='SECONDARY_CARD.json',noise_file=str(RAW/'noise'/('WBC-'+family['family_id']+'.npz')),noise_name=f"{case['task']}_E0_{ordinal:03d}")
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();answer=message(worker)
            assert answer['status'] in ('VALID','VALID_REUSED') and answer['output']==str(op)
            d=np.load(op)['decoded'][0];ordinal+=1;metadata=dict(valid_label=True,model_point=point,actual_request=request,prediction_sha256=sha(op),
                teacher_future_not_used=True,task_phase_not_input=True,non_realtime_lockstep=True)
            if not np.isfinite(d).all():metadata.update(valid_label=False,reason='NONFINITE_POLICY_OUTPUT');return None,None,d,metadata
            try:rr=rot6d_rows_to_matrix(d[:,3:9].astype(float))
            except ValueError:metadata.update(valid_label=False,reason='INVALID_ROT6D_POLICY_OUTPUT');return None,None,d,metadata
            return d[:,:3].astype(float)+anchor,rr,d,metadata
    control.ROOT=R1;control.RAW=RAW;control.ObservableCorrection=ArclengthCorrection;control.score=score
    try:control.run(case_id,policy_query=query,geometry_worker_path=ROOT/'code/pi0267_geometry_worker.py')
    finally:
        if worker:
            if worker.poll() is None:worker.stdin.close();worker.wait(timeout=30)
        if log:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('expert','donor','N4096','R4096'),required=True);p.add_argument('--case',required=True);a=p.parse_args();run(a.point,a.case)

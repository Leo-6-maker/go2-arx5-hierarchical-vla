"""CPU-only replay of saved fresh observations, frozen noise and installed prefixes."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
sys.path[:0]=[str(ROOT/'code'),str(BASE/'PI0266/code'),str(BASE/'PI0250/runtime/reference-317a/src')]
from pi0255_motion3 import sha,write
from pi0267_cpu import interpolate
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix


def read(p):return json.loads(Path(p).read_text())


def main(point):
    card=read(R1/'design/EVALUATION_CARD.json');scene=read(ROOT/'design/SCENE_CARD.json')
    families=[f['family_id'] for f in scene['families'] if f['split']=='NEW_SIM_EVAL']
    expected={f'{f}-{t}-c{c}-{b}' for f in families for t in ('M1','M3') for c in (0,1) for b in ('E0','E1')}
    paths=sorted((RAW/'virtual'/point).glob('*/result.json'))
    assert len(expected)==64 and {p.parent.name for p in paths}==expected,'INCOMPLETE_FIXED_POINT'
    source=R1/'runtime/virtual_family/pi0267_virtual_episode.py'
    advance=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='advance')
    namespace=dict(np=np,Rotation=Rotation,interpolate=interpolate)
    exec(compile(ast.Module(body=[advance],type_ignores=[]),str(source),'exec'),namespace)
    noise={};checks=[];anchor=np.asarray(scene['policy_anchor_world_m'])
    for path in paths:
        record=read(path);assert record['validity']=='VALID' and record['point']==point
        assert record['evaluation_card_sha256']==sha(R1/'design/EVALUATION_CARD.json')
        assert record['arrays_sha256']==sha(path.parent/'trajectory.npz')
        trajectory=np.load(path.parent/'trajectory.npz');gates={g['replan']:g['result']['decision'] for g in record['gates']}
        worst=0.;executed=0
        for ordinal,query in enumerate(record['queries']):
            assert query['replan']==ordinal
            request=query['query'];ip=Path(request['input']);op=Path(request['output']);tick=20*ordinal
            assert ip==path.parent/'queries'/f'{ordinal:03d}'/'observation.npz'
            assert op==ip.parent/'prediction.npz' and sha(op)==query['prediction_sha256']
            if 'observation_sha256' in query:assert sha(ip)==query['observation_sha256']
            observed=np.load(ip);prediction=np.load(op);receipt=read(op.with_suffix('.receipt.json'))
            assert receipt['point']==point and receipt['output_sha256']==sha(op)
            p,r=trajectory['actual_p'][tick],trajectory['actual_r'][tick]
            assert float(observed['timestamp'])==ordinal*.4
            assert np.array_equal(observed['position_world'],p) and np.array_equal(observed['rotation_world'],r)
            state=np.r_[p-anchor,r[:2,:].reshape(6),.5].astype(np.float32)
            assert np.array_equal(state,observed['state_v1']) and np.array_equal(state,prediction['state'])
            nf=request['noise_file'];key=request['noise_name']
            assert Path(nf)==RAW/'noise'/(record['family']+'.npz')
            assert key==f"{record['task']}_{record['backend']}_{ordinal:03d}"
            if nf not in noise:
                entry=next(n for n in card['noise'] if n['file']==nf);assert sha(nf)==entry['sha256']
                with np.load(nf) as z:noise[nf]={k:z[k].copy() for k in z.files}
            assert np.array_equal(prediction['noise'],noise[nf][key])
            assert receipt['noise_sha256']==hashlib.sha256(noise[nf][key].tobytes()).hexdigest()
            if gates.get(ordinal)!='ACCEPT':
                assert ordinal==len(record['queries'])-1 and tick==len(trajectory['time'])-1
                continue
            decoded=prediction['decoded'][0];lp=decoded[:,:3].astype(float)+anchor
            lr=rot6d_rows_to_matrix(decoded[:,3:9].astype(float))
            arrays=namespace['advance'](p,r,lp,lr,record['backend'])
            length=min(20,len(trajectory['time'])-tick-1);assert length>0
            for name,array in zip(('actual_p','actual_r','installed_p','installed_r'),arrays):
                error=float(np.max(abs(array[:length]-trajectory[name][tick+1:tick+1+length])))
                assert error<=1e-12,(path,ordinal,name,error)
                worst=max(worst,error)
            executed+=length
        assert executed==len(trajectory['time'])-1
        checks.append(dict(case=path.parent.name,queries=len(record['queries']),executed_ticks=executed,
            maximum_prefix_replay_abs=worst,result_sha256=sha(path),trajectory_sha256=record['arrays_sha256']))
    write(R1/'evidence'/f'QUERY_CHAIN_REPLAY_{point}.json',dict(status='PASS_SAVED_OBSERVATION_NOISE_AND_EXECUTION_CHAIN',
        point=point,episodes=checks,queries=sum(c['queries'] for c in checks),source_sha256=sha(Path(__file__)),
        execution_source_sha256=sha(source),executor_replay_shared=True,renderer_rerun=False,
        scope='Saved observation identity and actual frozen noise; original shared executor prefix replay, not independent physics or model recomputation',
        model_queries=0,new_renders=0,new_physics_steps=0))
    print(point,len(checks),sum(c['queries'] for c in checks),'PASS_QUERY_CHAIN',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('point',choices=('donor','N4096','R4096'));main(parser.parse_args().point)

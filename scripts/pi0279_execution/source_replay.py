"""Independent saved-predicate, input, teacher and paired-state verification."""
import ast,json
from pathlib import Path
import numpy as np
from pilot import ROOT,RAW,OLD_RAW,read,sha,freeze,differences,teacher


def main():
    source=ROOT.with_name('PI0276')/'code/replay.py'
    node=next(n for n in ast.parse(source.read_text()).body if getattr(n,'name',None)=='reconstruct');scope=dict(np=np)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope);reconstruct=scope['reconstruct'];rows=[]
    verdict=read(ROOT/'evidence/PILOT_VERDICT.json')
    for slot in verdict['rows']:
        seed,fork=slot['seed'],slot['fork'];pair={}
        for kind,qualification in slot['branches'].items():
            folder=Path(qualification['path']);rec=read(folder/'RESULT.json')
            assert sha(folder/'RESULT.json')==qualification['result_sha256']
            for name,h in rec['files'].items():assert sha(folder/name)==h,(folder,name)
            with np.load(folder/'physics.npz') as z:
                assert all(np.isfinite(z[k]).all() for k in z.files)
                actual=reconstruct(z);assert all(np.array_equal(v,z[k]) for k,v in actual.items())
                success=actual['success'][::5];persistent=len(success)>=601 and bool(success[560:601].all());assert persistent==rec['persistent_success']
                ids=np.arange(fork*5-40,fork*5+1);pair[kind]={k:z[k][ids].copy() for k in z.files}
                with np.load(folder/'observations.npz') as obs:
                    n=len(obs['state_v1']);tcp=z['tcp'][::5][:n];q=z['qpos'][::5][:n]
                    expected=np.c_[tcp[:,:3,3],tcp[:,:2,:3].reshape(n,6),(q[:,-2:].sum(1)-.00004)/.08].astype(np.float32)
                    assert np.max(np.abs(obs['state_v1']-expected))<=1e-6 and np.max(np.abs(obs['time_s']-np.arange(n)*.05))<=1e-9
                    assert all(obs[k].dtype==np.uint8 and obs[k].shape==(n,96,96,3) for k in ('base_camera_96','hand_camera_96'))
                measures=None
                if kind=='OPEN_CLEAR' and rec['control_calls']==640:
                    t0=np.asarray(read(folder/'TEACHER_ANCHOR.json')['actual_world_TCP'])
                    with np.load(folder/'commands.npz') as c:
                        assert np.array_equal(c['action'][fork:],teacher(t0,np.arange(640-fork)))
                        assert int(c['ik_hold'].sum())==rec['ik_holds']
                    sample=np.arange(2800,3001,5);width=(z['qpos'][sample,-2:].sum(axis=1)-.00004);height=z['tcp'][sample,2,3]-t0[2,3]
                    measures=dict(opening_min_m=float(width.min()),clearance_min_m=float(height.min()))
                    positive=bool(persistent and width.min()>=.075 and height.min()>=.06 and rec['ik_holds']==0)
                    assert positive==qualification['positive']
                rows.append(dict(seed=seed,fork=fork,kind=kind,path=str(folder),persistent_success=persistent,positive=qualification['positive'],measures=measures,result_sha256=sha(folder/'RESULT.json'),input_state_from_actual_physics=True,native_predicates_exact=True))
        assert differences(pair['OPEN_CLEAR'],pair['RECORDED'])['pass_']
        with np.load(OLD_RAW/'episodes'/f'TRAIN_PHYSICS-{seed}-COVER934096'/'attempt-000/physics.npz') as z:
            reference={k:z[k][ids] for k in z.files}
        assert all(differences(v,reference)['pass_'] for v in pair.values())
    freeze(ROOT/'evidence/PILOT_CPU_REPLAY.json',dict(status='PASS_SAVED_NATIVE_PAIRED_STATES_ACTUAL_INPUTS_AND_FIXED_TEACHER',rows=rows,source_sha256=sha(source),pilot_verdict_sha256=sha(ROOT/'evidence/PILOT_VERDICT.json'),new_physics=0,new_neural_queries=0))
    print('PASS_INDEPENDENT_SAVED_PILOT_CPU_REPLAY',len(rows),flush=True)


if __name__=='__main__':main()

"""Saved S4/decode arithmetic and fixed offline errors; zero neural queries."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott')
import ast,csv,json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from run import ROOT,sha,write
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0276');BASE=ROOT.parent;OLD=RAW.parent/'PI0260/R2'
sys.path[:0]=[str(OLD/'R3'),str(OLD/'runtime'),str(OLD/'qa'),str(BASE/'PI0250/runtime/reference-317a/src'),'/home/liuyu-nfs/Projects/isaac-gr00t',str(ROOT/'project-src')]

def metrics(pred,target):
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    if not np.isfinite(pred).all():return {k:None for k in ('XYZ_RMS_m','SO3_RMS_rad','prefix8_XYZ_RMS_m','prefix8_SO3_RMS_rad','gripper_MAE','prefix8_gripper_MAE','OPEN_MAE','CLOSE_MAE')}
    xyz=np.linalg.norm(pred[:,:3]-target[:,:3],axis=1)
    try:angle=Rotation.from_matrix(rot6d_rows_to_matrix(pred[:,3:9])@rot6d_rows_to_matrix(target[:,3:9]).transpose(0,2,1)).magnitude()
    except ValueError:return metrics(np.full_like(pred,np.nan),target)
    grip=np.abs(pred[:,9]-target[:,9]);rms=lambda x:float(np.sqrt(np.mean(x*x)))
    selected=lambda mask:float(grip[mask].mean()) if mask.any() else None
    return dict(XYZ_RMS_m=rms(xyz),SO3_RMS_rad=rms(angle),prefix8_XYZ_RMS_m=rms(xyz[:8]),prefix8_SO3_RMS_rad=rms(angle[:8]),
        gripper_MAE=float(grip.mean()),prefix8_gripper_MAE=float(grip[:8].mean()),OPEN_MAE=selected(np.isclose(target[:,9],1)),CLOSE_MAE=selected(np.isclose(target[:,9],0)))

def main():
    import resource,torch,pi0260_r2_codec as codec
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('SAVED_ARRAYS_NO_NEURAL_QUERY_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden;assert not torch.cuda.is_available()
    source=BASE/'PI0268/R1/code/report.py';text=source.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='replay')
    scope=dict(np=np,sha=sha);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    proc=codec.processor(json.loads((ROOT/'contracts/TASK/source/dataset/statistics.json').read_text()));proof=[]
    gate_source=ROOT/'code/source_worker.py'
    node=next(n for n in ast.parse(gate_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='pose_gate')
    gate_scope=dict(ROOT=ROOT,np=np)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(gate_source),'exec'),gate_scope)
    outputs=sorted((RAW/'quality').rglob('output.npz'))
    assert len(outputs)==len(set(outputs))
    for path in outputs:
        rec=json.loads(path.with_suffix('.receipt.json').read_text());assert rec['status']=='VALID' and rec['output_sha256']==sha(path)
        assert rec['observed_query_dtypes']==['torch.bfloat16']*4
        with np.load(path) as z:assert hashlib.sha256(z['noise'].tobytes()).hexdigest()==rec['noise_sha256']
        proof.append(scope['replay'](path,proc,torch,codec))
    panel=json.loads((ROOT/'design/OFFLINE_CARD.json').read_text())['rows'];rows=[]
    for point in ('BASE0','TASK8192'):
        for row in panel:
            p=RAW/'quality'/point/'OFFLINE'/row['uid'];j=row['anchor']
            with np.load(Path(row['episode'])/'commands.npz') as z:target=z['action'][j:j+40]
            with np.load(Path(row['episode'])/'observations.npz') as z:
                expected=dict(body_rgb=z['base_camera_96'][j],wrist_rgb=z['hand_camera_96'][j],state_v1=z['state_v1'][j])
            with np.load(p/'input.npz') as z:
                assert set(z.files)==set(expected) and all(np.array_equal(z[k],v) for k,v in expected.items())
            with np.load(p/'output.npz') as z:pred=z['decoded'][0]
            if not np.isfinite(pred).all():gate=dict(decision='REJECT',reason='NONFINITE_POLICY_OUTPUT')
            else:
                try:gate=gate_scope['pose_gate'](pred,expected['state_v1'],j*.05,row['uid'])
                except ValueError as error:gate=dict(decision='REJECT',reason='INVALID_ROT6D_POLICY_OUTPUT',error=str(error))
            rows.append(dict(point=point,uid=row['uid'],split=row['split'],seed=row['seed'],anchor=j,group=row['group'],finite=bool(np.isfinite(pred).all()),
                gate=gate['decision'],gate_evidence=json.dumps(gate,sort_keys=True),**metrics(pred,target)))
    assert len(rows)==384
    for path in (RAW/'episodes').glob('*/attempt-*/QUERIES.json'):
        queries=json.loads(path.read_text())
        if not queries:continue
        with np.load(path.parent/'observations.npz') as z:
            for q in queries:
                index=8*q['index'];assert q['time_s']==index*.05
                with np.load(Path(q['output']).with_name('input.npz')) as inp:
                    assert np.array_equal(inp['state_v1'],z['state_v1'][index])
                    assert np.array_equal(inp['body_rgb'],z['base_camera_96'][index])
                    assert np.array_equal(inp['wrist_rgb'],z['hand_camera_96'][index])
    dst=RAW/'delivery';dst.mkdir(exist_ok=True)
    with (dst/'OFFLINE384.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    attempts=[json.loads(l) for l in (ROOT/'evidence/model_attempts.jsonl').open()];heads=[json.loads(l) for l in (ROOT/'evidence/model_heads.jsonl').open()]
    assert len(attempts)<=8192 and len(heads)<=32768
    assert {a['request']['output'] for a in attempts}=={str(p) for p in outputs},'Unresolved counted query or unregistered output'
    write(ROOT/'evidence/MODEL_CPU_REPLAY.json',dict(status='PASS_PI0276_ALL_SAVED_S4_DECODE_AND_OFFLINE_OUTPUTS',predictions=proof,
        offline_rows=384,attempts=len(attempts),heads=len(heads),shared_native_BF16_Euler_and_qualified_codec=True,
        replay_helper_path=str(source),replay_helper_sha256=sha(source),new_model_queries=0,physics_steps=0))

if __name__=='__main__':main()

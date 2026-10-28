"""Independent saved-array S4/decode replay and task-fit diagnostics, no neural calls."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott')
import argparse,ast,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from evaluation_card import ROOT,OLD,RAW,OLD_RAW,read,sha,write,freeze,packet

def function(path,name,scope):
    text=Path(path).read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]

def transitions(g):
    closed=np.asarray(g)<=.2;opened=np.asarray(g)>=.8
    return dict(close_ticks=(np.flatnonzero(~closed[:-1]&closed[1:])+1).tolist(),open_ticks=(np.flatnonzero(~opened[:-1]&opened[1:])+1).tolist())

def score(point):
    import resource,torch
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('NO_NEURAL_OR_OPTIMIZER_QUERY_IN_CPU_REPLAY')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden;assert not torch.cuda.is_available()
    base=ROOT.parent;inputs=OLD_RAW.parent/'PI0260/R2'
    sys.path[:0]=[str(inputs/'R3'),str(inputs/'runtime'),str(inputs/'qa'),str(base/'PI0250/runtime/reference-317a/src'),
        '/home/liuyu-nfs/Projects/isaac-gr00t',str(OLD/'project-src')]
    import pi0260_r2_codec as codec
    proc=codec.processor(read(ROOT/'contracts/TASK/source/dataset/statistics.json'))
    replay_source=base/'PI0268/R1/code/report.py';replay=function(replay_source,'replay',dict(np=np,sha=sha))
    metrics=function(OLD/'code/model_replay.py','metrics',dict(np=np,Rotation=Rotation))
    gate=function(OLD/'code/source_worker.py','pose_gate',dict(ROOT=OLD,np=np))
    complete=read(ROOT/'evidence'/f'FIXED_QUERIES_{point}.json');card=read(ROOT/'design/EVALUATION_CARD.json')
    originals={r['uid']:r for r in card['fit']+card['offline']};rows=[];proof=[]
    for record in complete['results']:
        path=Path(record['output']);assert sha(path)==record['sha256'];row=originals[record['uid']]
        receipt=read(path.with_suffix('.receipt.json'));assert receipt['output_sha256']==record['sha256'] and receipt['observed_query_dtypes']==['torch.bfloat16']*4
        proof.append(replay(path,proc,torch,codec))
        expected=packet(row)
        with np.load(path.with_name('input.npz'),allow_pickle=False) as z:assert set(z.files)==set(expected) and all(np.array_equal(z[k],v) for k,v in expected.items())
        with np.load(Path(row['episode'])/'commands.npz',allow_pickle=False) as z:target=z['action'][row['anchor']:row['anchor']+40]
        with np.load(path,allow_pickle=False) as z:
            pred=z['decoded'][0];raw=z['raw'][0];assert hashlib.sha256(z['noise'].tobytes()).hexdigest()==receipt['noise_sha256']
        finite=bool(np.isfinite(pred).all())
        if finite:
            assert np.max(np.abs((raw[:,9]+1)/2-pred[:,9]))<=1e-6,'AFFINE_GRIPPER_DECODE'
            try:decision=gate(pred,expected['state_v1'],row['anchor']*.05,row['uid'])
            except ValueError as e:decision=dict(decision='REJECT',error=str(e))
        else:decision=dict(decision='REJECT',reason='NONFINITE_OUTPUT')
        weight=np.r_[np.full(8,.8/8),np.full(32,.2/32)]
        aux=float(np.sum(weight*((pred[:,9].astype(np.float64)-target[:,9])/.1)**2)/weight.sum()) if finite else None
        extra=dict(generated_gripper_aux_unweighted=aux,close_target_ticks=int(np.sum(target[:,9]<=.2)),open_target_ticks=int(np.sum(target[:,9]>=.8)),
            close_threshold_mistakes=int(np.sum((target[:,9]<=.2)&(pred[:,9]>.2))) if finite else None,
            open_threshold_mistakes=int(np.sum((target[:,9]>=.8)&(pred[:,9]<.8))) if finite else None,
            target_events=json.dumps(transitions(target[:,9])),predicted_events=json.dumps(transitions(pred[:,9])) if finite else None)
        rows.append(dict(point=point,lane=record['lane'],uid=row['uid'],noise_index=record['noise_index'],seed=row['seed'],group=row['group'],
            split=row['split'],anchor=row['anchor'],finite=finite,gate=decision['decision'],gate_evidence=json.dumps(decision,sort_keys=True),
            output_sha256=record['sha256'],**metrics(pred,target),**extra))
    assert len(rows)==(128 if point=='TASK8192' else 320)
    destination=RAW/'delivery';destination.mkdir(exist_ok=True)
    csvpath=destination/f'FIXED_{point}.csv'
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for lane in sorted({r['lane'] for r in rows}):
        for group in range(4):
            selected=[r for r in rows if r['lane']==lane and r['group']==group]
            summary.append(dict(lane=lane,group=group,denominator=len(selected),REJECT=sum(r['gate']=='REJECT' for r in selected),
                nonfinite=sum(not r['finite'] for r in selected),metrics={key:dict(observed=sum(r[key] is not None for r in selected),
                mean=float(np.mean([r[key] for r in selected if r[key] is not None])) if any(r[key] is not None for r in selected) else None)
                for key in ('prefix8_XYZ_RMS_m','prefix8_SO3_RMS_rad','prefix8_gripper_MAE','gripper_MAE','generated_gripper_aux_unweighted')}))
    freeze(ROOT/'evidence'/f'FIXED_CPU_REPLAY_{point}.json',dict(status='PASS_SAVED_FIXED_S4_DECODE_AND_METRICS',point=point,predictions=proof,
        summary=summary,table=str(csvpath),table_sha256=sha(csvpath),replay_helper_sha256=sha(replay_source),
        metric_helper_sha256=sha(OLD/'code/model_replay.py'),shared_native_BF16_primitive_and_qualified_codec=True,new_model_queries=0,
        physics_steps=0,task_success='NOT_INFERRED_FROM_OFFLINE_FIT',selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(dict(point=point,rows=len(rows),summary=summary)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('TASK8192','CONTROL4096','GRIP4096'),required=True);score(p.parse_args().point)

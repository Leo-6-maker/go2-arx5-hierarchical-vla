"""Fixed DIAG48 actual inputs; saved-output CPU prefix metrics, no physics."""
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
from source import ROOT,reply,write,digest


def diagnostics(point):
    from cpu_inputs import FrozenDataset
    card=json.loads((ROOT/'design/SOURCE_CARD.json').read_text())
    data=FrozenDataset(ROOT/'data',ROOT/'data/DIAG48.jsonl',cache=False)
    folder=ROOT/'source/diagnostics'/point;folder.mkdir(parents=True,exist_ok=True)
    done=folder/'RESULT.json'
    if done.exists():return json.loads(done.read_text())
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=card['native_pythonpath'],HF_HOME=card['hf_home'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    rows=[];began=time.monotonic()
    with (folder/'worker.log').open('ab') as log:
        server=subprocess.Popen(card['policy_command']+['--point',point],env=environment,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        try:
            assert reply(server,'ready')['ready']
            for index,row in enumerate(data.rows):
                path=folder/f'{index:02d}';path.mkdir(exist_ok=True)
                source=path/'input.npz';output=path/'output.npz'
                if not output.exists():
                    base,wrist,state,commands=data.raw(row)
                    # Teacher target is saved separately; it never enters inference input.
                    np.savez_compressed(source,base_raw=base,wrist_raw=wrist,actual_state=state)
                    np.savez_compressed(path/'target.npz',commands=commands)
                    request=dict(input=str(source),output=str(output),condition='DIAG-'+row['uid'],query_index=0)
                    server.stdin.write(json.dumps(request)+'\n');server.stdin.flush();assert reply(server,'output')['output']==str(output)
                with np.load(output) as saved:assert str(saved['source_sha256'])==digest(source)
                rows.append(dict(index=index,uid=row['uid'],split=row['split'],phase=row['phase'],input_sha256=digest(source),output_sha256=digest(output),target_sha256=digest(path/'target.npz')))
                write(folder/'PROGRESS.json',dict(rows=rows))
        finally:
            server.stdin.close();code=server.wait(timeout=120);assert code==0
    result=dict(status='FIXED_DIAG48_EXECUTED',point=point,rows=rows,elapsed_s=time.monotonic()-began,model_requests=len(rows),physics_steps=0)
    write(done,result);return result


def recompute(root,point):
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,geodesic_angle
    root=Path(root);folder=root/'source/diagnostics'/point;source=json.loads((folder/'RESULT.json').read_text());rows=[]
    for row in source['rows']:
        path=folder/f"{row['index']:02d}"
        assert digest(path/'output.npz')==row['output_sha256'] and digest(path/'target.npz')==row['target_sha256']
        with np.load(path/'output.npz') as data:pred=data['decoded'][:8].copy();oob=float(data['normalized_OOB_fraction'])
        with np.load(path/'target.npz') as data:target=data['commands'][:8].copy()
        error='';passed=False;xyz=rotation=gripper=None
        try:
            assert np.isfinite(pred).all(),'NONFINITE_NATIVE_DECODE'
            xyz=float(np.sqrt(np.mean(np.sum((pred[:,:3]-target[:,:3])**2,axis=1))))
            angles=geodesic_angle(rot6d_rows_to_matrix(pred[:,3:9]),rot6d_rows_to_matrix(target[:,3:9]))
            rotation=float(np.sqrt(np.mean(angles**2)));gripper=float(np.mean(abs(pred[:,-1]-target[:,-1])))
            passed=xyz<=.01 and rotation<=.10 and gripper<=.10
        except (ValueError,AssertionError) as exc:error=str(exc)
        rows.append(dict(row,XYZ_RMS_m=xyz,SO3_RMS_rad=rotation,gripper_MAE=gripper,normalized_OOB_fraction=oob,fit_pass=passed,error=error))
    train=[r for r in rows if r['split']=='TRAIN'];phases={phase:[r for r in train if r['phase']==phase] for phase in range(4)}
    count=sum(r['fit_pass'] for r in train);adequate=len(train)==24 and count>=18 and all(v and sum(r['fit_pass'] for r in v)*2>=len(v) for v in phases.values())
    result=dict(status='PASS_CPU_FIXED_PREFIX_RECOMPUTE_NOT_TASK_QUALIFICATION',point=point,source_sha256=digest(folder/'RESULT.json'),rows=rows,TRAIN_fit_count=count,TRAIN_denominator=len(train),fit_adequate=adequate,
        phase_counts={k:dict(passed=sum(r['fit_pass'] for r in v),total=len(v)) for k,v in phases.items()},new_model_queries=0,selected='NONE',deployment='NONE')
    write(folder/'CPU_RECOMPUTE.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('diag','cpu'));parser.add_argument('--point',required=True);parser.add_argument('--root',default=str(ROOT));args=parser.parse_args()
    result=diagnostics(args.point) if args.mode=='diag' else recompute(args.root,args.point)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

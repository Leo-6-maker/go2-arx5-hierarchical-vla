"""Frozen six phase/event anchors per layout; first-valid K4 and CPU recomputation."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from contract import ROOT,INSTRUCTIONS,SPLITS,guard,sha,write


def anchors(rows):
    # ponytail: first eligible anchor in each of five phases, plus final full-H40 anchor.
    selected=[min((r for r in rows if r['phase']==p),key=lambda r:r['anchor']) for p in range(5)]
    selected.append(next(r for r in rows if r['anchor']==620))
    assert len({r['uid'] for r in selected})==6,'Six distinct anchors required; no score-driven replacements'
    return selected


def prepare():
    guard();assert os.environ['CUDA_VISIBLE_DEVICES']==''
    from data import Dataset,immutable
    from r2 import admission
    receipt=admission()
    for n,h in receipt['bindings'].items():assert sha(ROOT/'data'/n)==h,n
    windows=json.loads((ROOT/'data/WINDOW_INDEX.json').read_text())['rows']
    bank=json.loads((ROOT/'data/SOURCE_BANK.json').read_text());records=[]
    dataset=Dataset(ROOT/'data',ROOT/'data/CPU_INPUT_CHECK_PLAN.jsonl')
    for task in bank['tasks']:
        for split in ('TRAIN','DEV'):
            for seed in range(SPLITS[split][0],SPLITS[split][0]+8):
                group=[r for r in windows if (r['task'],r['split'],r['layout'])==(task,split,seed)]
                for row in anchors(group):
                    folder=ROOT/'diagnostics/inputs'/row['uid'];folder.mkdir(parents=True,exist_ok=True)
                    base,wrist,state,commands=dataset.raw(row)
                    for name,content in [('input.npz',dict(base_raw=base,wrist_raw=wrist,actual_state=state,instruction=np.asarray(INSTRUCTIONS[task]))),('target.npz',dict(commands=commands))]:
                        p=folder/name
                        if not p.exists():np.savez_compressed(p,**content)
                        with np.load(p) as f:assert set(f.files)==set(content) and all(np.array_equal(f[k],v) for k,v in content.items())
                    records.append(dict(row,input=str(folder/'input.npz'),input_sha256=sha(folder/'input.npz'),target=str(folder/'target.npz'),target_sha256=sha(folder/'target.npz')))
    assert all(sum(r['task']==t for r in records)==96 for t in bank['tasks'])
    value=dict(status='FROZEN_PRE_SCORE_DIAG96_PER_ADMITTED_TASK',rows=records,window_index_sha256=sha(ROOT/'data/WINDOW_INDEX.json'),anchor_rule='FIRST_VALID_H40_IN_PHASE_0_1_2_3_4_AND_LAST_H40_620',noise_rule='ONE_NATIVE_K4_PACKET_PER_UID_REUSED_ACROSS_ENDPOINTS',teacher_queries=0,physics=0,model_queries=0)
    immutable(ROOT/'design/DIAG96_PER_TASK.json',value);return value


def run(point):
    from policy import panel_gate,resident_worker,reply
    from budget import process
    selected,_=panel_gate(point,'TRAIN');assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    plan=ROOT/'design/DIAG96_PER_TASK.json';rows=[r for r in json.loads(plan.read_text())['rows'] if r['task'] in selected]
    assert len(rows)==96*len(selected)
    folder=ROOT/'diagnostics'/point;folder.mkdir(parents=True,exist_ok=True);done=folder/'RESULT.json'
    if done.exists():return json.loads(done.read_text())
    started=process(ROOT,'start');records=[]
    try:
        with resident_worker(point,folder/'worker.log') as server:
            for row in rows:
                assert sha(row['input'])==row['input_sha256'] and sha(row['target'])==row['target_sha256']
                output=folder/(row['uid']+'.npz')
                if not output.exists():
                    request=dict(input=row['input'],output=str(output),condition=dict(diagnostic_UID=row['uid']),query_index=0)
                    server.stdin.write(json.dumps(request)+'\n');server.stdin.flush();assert reply(server,'output')['output']==str(output)
                with np.load(output) as f:assert str(f['source_sha256'])==row['input_sha256'] and int(f['K'])==4
                records.append(dict(row,output=str(output),output_sha256=sha(output)))
                write(folder/'PROGRESS.json',dict(records=records))
        value=dict(status='FIXED_NATIVE_DIAG96_PER_TASK_COMPLETE_NOT_TASK_QUALIFICATION',point=point,plan_sha256=sha(plan),records=records,new_physics=0,teacher_queries=0,selected='NONE',deployment='NONE')
        write(done,value);return value
    finally:process(ROOT,'end',started)


def rotation(x):
    a=x[:,3:6].astype(float);norm=np.linalg.norm(a,axis=1);assert (norm>=1e-12).all(),'DEGENERATE_ROTATION';a/=norm[:,None]
    b=x[:,6:9].astype(float);b-=a*(a*b).sum(1)[:,None];norm=np.linalg.norm(b,axis=1);assert (norm>=1e-12).all(),'DEGENERATE_ROTATION';b/=norm[:,None]
    return np.stack((a,b,np.cross(a,b)),axis=1)


def metrics(x,y,independent=False):
    assert x.shape==y.shape and x.shape[1]==10 and np.isfinite(x).all() and np.isfinite(y).all(),'NONFINITE_OR_SHAPE'
    if independent:
        product=rotation(x)@rotation(y).transpose(0,2,1)
        angle=np.arccos(np.clip((np.trace(product,axis1=1,axis2=2)-1)/2,-1,1))
    else:
        from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,geodesic_angle
        angle=geodesic_angle(rot6d_rows_to_matrix(x[:,3:9]),rot6d_rows_to_matrix(y[:,3:9]))
    return dict(XYZ_RMS_m=float(np.sqrt(np.mean(np.sum((x[:,:3].astype(float)-y[:,:3])**2,axis=1)))),SO3_RMS_rad=float(np.sqrt(np.mean(angle**2))),gripper_MAE=float(np.mean(abs(x[:,9].astype(float)-y[:,9]))))


def cpu(point):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';guard()
    from data import codec
    from gate import original_gate
    from policy import disposition
    folder=ROOT/'diagnostics'/point;source=folder/'RESULT.json';saved=json.loads(source.read_text())
    assert saved['point']==point and saved['plan_sha256']==sha(ROOT/'design/DIAG96_PER_TASK.json')
    rows=[];encoder=codec(str(ROOT/'data'))
    for row in saved['records']:
        for name in ('input','output','target'):assert sha(row[name])==row[name+'_sha256']
        with np.load(row['input']) as f:state=f['actual_state'].copy();instruction=str(f['instruction'])
        with np.load(row['target']) as f:target=f['commands'].copy()
        with np.load(row['output']) as f:
            pred=f['decoded'].copy();raw=f['raw'].copy();error=str(f['decode_error']);noise=f['noise_uint16'].copy();K=int(f['K']);states=f['raw_states_uint16'].copy()
            assert str(f['source_sha256'])==row['input_sha256'] and np.array_equal(f['actual_state'],state) and str(f['instruction'])==instruction==INSTRUCTIONS[row['task']]
        key=json.dumps(dict(condition=dict(diagnostic_UID=row['uid']),query_index=0),sort_keys=True)
        assert np.array_equal(noise,np.load(ROOT/'runtime/noise'/(hashlib.sha256(key.encode()).hexdigest()+'.npy')))
        assert K==4 and states.shape==(5,1,40,132) and np.array_equal(states[0],noise)
        try:decoded=encoder.decode_actions(raw[0],state)
        except ValueError as exc:assert error==str(exc);decoded=np.full((40,10),np.nan,dtype=np.float32)
        assert np.array_equal(decoded,pred,equal_nan=True)
        rejection=disposition(pred,error);g=dict(decision='REJECT',reason=rejection);prefix=full=None;issue=error
        try:
            prefix=metrics(pred[:8],target[:8]);full=metrics(pred,target)
            for x,y,native in ((pred[:8],target[:8],prefix),(pred,target,full)):
                other=metrics(x,y,True)
                for k in native:assert np.isclose(native[k],other[k],rtol=1e-5,atol=1e-7),(row['uid'],k)
            if rejection is None:g=original_gate(pred,state,0.,'PI0285_DIAG-'+row['uid'])
        except (ValueError,AssertionError) as exc:
            # Only actual invalid outputs are valid failures; a CPU parity discrepancy stops finalization.
            if np.isfinite(pred).all():
                try:rotation(pred)
                except AssertionError:pass
                else:raise
            issue=str(exc)
        rows.append(dict(uid=row['uid'],task=row['task'],split=row['split'],layout=row['layout'],phase=row['phase'],anchor=row['anchor'],prefix=prefix,full_H40=full,gate=g,error=issue,raw_OOB_prefix_by_channel=((raw[0,:8,:10]<-1)|(raw[0,:8,:10]>1)).sum(0).tolist(),raw_OOB_H40_by_channel=((raw[0,:,:10]<-1)|(raw[0,:,:10]>1)).sum(0).tolist(),output_sha256=row['output_sha256']))
    value=dict(status='PASS_CPU_FIXED_DIAG_NATIVE_DECODE_NOISE_PREFIX_FULL_H40_AND_GATE',point=point,source_sha256=sha(source),rows=rows,new_model_queries=0,new_physics=0,diagnostic_only_not_task_qualification=True,selected='NONE',deployment='NONE')
    write(ROOT/'audit'/(point+'_DIAG_CPU.json'),value);return value


def self_check():
    rows=[dict(uid=str(i),anchor=i,phase=min(i//100,4)) for i in range(20,621)]
    assert [r['anchor'] for r in anchors(rows)]==[20,100,200,300,400,620]
    x=np.tile([0.,0.,0.,1.,0.,0.,0.,1.,0.,0.],(40,1));y=x.copy();y[:,0]=.01;y[:,9]=.2
    v=metrics(x,y,True);assert np.isclose(v['XYZ_RMS_m'],.01) and v['SO3_RMS_rad']==0 and np.isclose(v['gripper_MAE'],.2)
    y[:,3:9]=[0.,1.,0.,-1.,0.,0.];assert np.isclose(metrics(x,y,True)['SO3_RMS_rad'],np.pi/2)
    x[:,3:9]=[0.,0.,0.,0.,1.,0.]
    try:rotation(x)
    except AssertionError:pass
    else:raise AssertionError('Degenerate rotation became a metric')
    x[:,3:9]=[1.,0.,0.,0.,1.,0.]
    x[0,0]=np.nan
    try:metrics(x,y,True)
    except AssertionError:pass
    else:raise AssertionError('Nonfinite output became a metric')
    print('PASS_FIXED_96_ANCHORS_AND_INDEPENDENT_ROTATION_PREFIX_METRICS; model/physics=0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','cpu','self-check'));p.add_argument('--point');a=p.parse_args()
    if a.mode=='self-check':self_check()
    else:
        value=prepare() if a.mode=='prepare' else (run(a.point) if a.mode=='run' else cpu(a.point));print(json.dumps({k:v for k,v in value.items() if k not in ('rows','records')}))

"""Recorded-source reference errors, explicitly not Go2 task success."""
import hashlib
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .runtime_v1 import read_pin
from .checkpoint_v1 import sha
from scripts.pi0292.durable import write


def errors(prediction,target,state,statistics):
    p,t,s=map(np.asarray,(prediction,target,state));span=statistics['actions'].q99-statistics['actions'].q01+1e-6
    stationary=np.zeros_like(t);stationary[:,6]=s[6];stationary[:,13]=s[13]
    def measure(x):
        result=dict(normalized_active14_MAE=float(np.mean(abs(x-t)*2/span)))
        for k,time_s in((14,.5),(29,1.),(44,1.5)):
            arms=[]
            for offset in(0,7):
                rotation=Rotation.from_rotvec(x[k,offset+3:offset+6])*Rotation.from_rotvec(t[k,offset+3:offset+6]).inv()
                arms.append(dict(XYZ_native_error=float(np.linalg.norm(x[k,offset:offset+3]-t[k,offset:offset+3])),
                    SO3_error_rad=float(rotation.magnitude()),gripper_native_error=float(abs(x[k,offset+6]-t[k,offset+6]))))
            result[str(time_s)+'s']=arms
        result['normalized_motion_from_persistence']=float(np.mean(abs(x-stationary)*2/span))
        return result
    return dict(model=measure(p),persistence=measure(stationary),target_normalized_motion=float(np.mean(abs(t-stationary)*2/span)))


def summarize(rows):
    def group(selected):
        if not selected:return dict(count=0)
        out=dict(count=len(selected),normalized_reference_error=float(np.mean([r['error']['model']['normalized_active14_MAE']for r in selected])),
            persistence_error=float(np.mean([r['error']['persistence']['normalized_active14_MAE']for r in selected])),
            predicted_motion=float(np.mean([r['error']['model']['normalized_motion_from_persistence']for r in selected])),
            target_motion=float(np.mean([r['error']['target_normalized_motion']for r in selected])))
        out['better_than_persistence']=out['normalized_reference_error']<out['persistence_error']
        out['horizons']={h:{component:float(np.mean([arm[component]for r in selected for arm in r['error']['model'][h]]))
            for component in('XYZ_native_error','SO3_error_rad','gripper_native_error')}for h in('0.5s','1.0s','1.5s')}
        return out
    return dict(all=group(rows),dynamic=group([r for r in rows if r['window']['dynamic']]),
        low_change_anchor=group([r for r in rows if not r['window']['dynamic']]),
        tasks={t:group([r for r in rows if r['window']['dynamic']and r['window']['task']==t])for t in sorted({r['window']['task']for r in rows})},
        actual_task_success_rate=None,scope='SOURCE_FUTURE_LOGGED_REFERENCE_NOT_PHYSICAL_COMMAND_OR_TASK_SR')


def evaluate(engine,root,split,*,endpoint,dataset=None,windows=None):
    root=Path(root)
    if windows is None:
        if split not in('TRAIN','SOURCE_DEV'):raise ValueError('SOURCE_TEST_REQUIRES_EXPLICIT_POST_FREEZE_DATASET_AND_BANK')
        m=engine.dataset.manifest;bank=read_pin(dict(path=str(Path(engine.spec['dataset'])/m['evaluation_bank']['file']),sha256=m['evaluation_bank']['sha256']))
        dataset=engine.dataset if split=='TRAIN'else engine.dev;windows=bank['TRAIN96'if split=='TRAIN'else'SOURCE_DEV96']
    elif split!='SOURCE_TEST'or dataset is None or len(windows)!=96:
        raise ValueError('EXPLICIT_POST_FREEZE_TEST96_ONLY')
    lookup={key:i for i,key in enumerate(dataset.rows)};rows=[];restore=[]
    out=root/(endpoint+'-'+split);out.mkdir()
    for number,window in enumerate(windows):
        if split=='SOURCE_TEST' and (time.time()>=engine.spec['deadline_epoch']-60 or (root/'STOP.json').exists()):
            raise TimeoutError('FINITE_SOURCE_TEST_BEFORE_NEXT_GENERATION')
        index=lookup[window['episode'],window['anchor']];noise_id=int.from_bytes(hashlib.sha256(('V6_SOURCE/'+window['id']+'/2026111808').encode()).digest()[:4],'little')
        normalized,prediction=engine.generate(index,noise_id,dataset=dataset)
        arrays=dataset.arrays(window['episode']);anchor=window['anchor'];state=arrays['state'][anchor];target=arrays['actions'][anchor]
        path=out/('%03d.npz'%number);np.savez(path,normalized=normalized,reference=prediction)
        row=dict(window=window,index=index,noise_id=noise_id,array=dict(path=str(path),sha256=sha(path)),
            normalized_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),error=errors(prediction,target,state,dataset.statistics),
            future_span_dynamic=bool(arrays['future_span_dynamic'][anchor]),input_identity=dataset.identity(index))
        rows.append(row)
        if split=='TRAIN'and number<16:restore.append(row)
        write(root/'EVALUATION_PROGRESS.private.json',dict(endpoint=endpoint,split=split,completed=len(rows),actual=engine.counts,new_native_requests=0))
    value=dict(endpoint=endpoint,split=split,rows=rows,summary=summarize(rows),actual=engine.counts)
    write(root/(endpoint+'-'+split+'96.private.json'),value)
    if restore:write(root/(endpoint+'-RGB16.private.json'),restore)
    return value['summary']


def transfer_gate(s0,s1000,*,reloadable):
    a=s0['dynamic']['normalized_reference_error'];b=s1000['dynamic']['normalized_reference_error']
    gain=(a-b)/a if a>0 else None
    improved=[t for t in s0['tasks']if s1000['tasks'][t]['normalized_reference_error']<s0['tasks'][t]['normalized_reference_error']]
    finite=bool(np.isfinite([a,b]).all())
    passed=bool(reloadable and finite and gain is not None and gain>=.10 and len(improved)>=4 and s1000['dynamic']['better_than_persistence'])
    return dict(passed=passed,finite=finite,reloadable=reloadable,DEV_dynamic_relative_error_reduction=gain,improved_tasks=improved,
        no_stationary_collapse_evidence=s1000['dynamic']['better_than_persistence'],source_task_success_claim=False)

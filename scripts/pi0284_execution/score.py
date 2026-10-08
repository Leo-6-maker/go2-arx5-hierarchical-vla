"""Original decoding/gate and registered TRAIN-only sampler rule on fixed saved outputs."""
import json
from pathlib import Path
import numpy as np
from prepare_core import write,sha

def metrics(pred,target):
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,geodesic_angle
    assert np.isfinite(pred).all(),'NONFINITE'
    return dict(XYZ_RMS=float(np.sqrt(np.mean(np.sum((pred[:,:3]-target[:,:3])**2,axis=1)))),SO3_RMS=float(np.sqrt(np.mean(geodesic_angle(rot6d_rows_to_matrix(pred[:,3:9]),rot6d_rows_to_matrix(target[:,3:9]))**2))),gripper_MAE=float(np.mean(abs(pred[:,9]-target[:,9]))))

def gate(pred,state):
    from go2arx5_vla.contracts import TrajectoryChunkV1
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,matrix_to_quat
    c=TrajectoryChunkV1(chunk_id='PI0284_DIAG',source='PI0282_EEF10_NATIVE',observation_timestamp=0.,point_timestamps=np.arange(40)*.05,eef_position=pred[:,:3],eef_orientation=matrix_to_quat(rot6d_rows_to_matrix(pred[:,3:9])),source_frame='PI0276_MANISKILL_WORLD',tcp_semantic_id='PI0276_PANDA_HAND_TCP',validity_start=0.,validity_end=1.95,gripper=pred[:,-1])
    return TrajectorySafetyGateV1().evaluate(c,now=0.,achieved_position=state[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[None,3:9]))[0]).as_dict()

def run(root):
    source=json.loads((root/'diagnostics/RESULT.json').read_text());plan=json.loads((root/'design/PRE_SCORE_INPUTS.json').read_text())['rows'];rows=[]
    for record in source['records']:
        row=plan[record['index']];p=Path(record['path']);assert sha(p)==record['sha256']
        with np.load(p) as v:pred=v['decoded'].copy();raw=v['raw'][0,:,:10].copy();state=v['actual_state'].copy();latency=float(v['latency_s'])
        with np.load(row['target']) as v:target=v['commands'].copy()
        err='';prefix=full=None;g=dict(decision='REJECT');J=None
        try:
            prefix=metrics(pred[:8],target[:8]);full=metrics(pred,target);g=gate(pred,state)
            J=(prefix['XYZ_RMS']/.010)**2+(prefix['SO3_RMS']/.10)**2+(prefix['gripper_MAE']/.10)**2
        except (ValueError,AssertionError) as exc:err=str(exc)
        rows.append(dict(record,prefix=prefix,fullH40=full,gate=g,error=err,J=J,latency_s=latency,raw_OOB_prefix_by_channel=((raw[:8]<-1)|(raw[:8]>1)).sum(0).tolist(),raw_OOB_H40_by_channel=((raw<-1)|(raw>1)).sum(0).tolist(),predicted_open_events=np.flatnonzero(np.diff(pred[:,9])>.1).tolist(),expert_open_events=np.flatnonzero(np.diff(target[:,9])>.1).tolist()))
    train=[v for v in plan if v['split']=='TRAIN'];paired=[]
    for anchor in train:
        result={}
        for k in (4,16):
            draws=[v for v in rows if v['uid']==anchor['uid'] and v['K']==k];assert len(draws)==4
            result[str(k)]=dict(mean_J=float(np.mean([v['J'] if v['J'] is not None else np.inf for v in draws])),mean_XYZ=float(np.mean([v['prefix']['XYZ_RMS'] if v['prefix'] else np.inf for v in draws])),mean_gripper=float(np.mean([v['prefix']['gripper_MAE'] if v['prefix'] else np.inf for v in draws])))
        paired.append(dict(uid=anchor['uid'],seed=anchor['seed'],phase=anchor['phase'],anchor=anchor['anchor'],K=result))
    first_release=[min([v for v in paired if v['phase']==3 and v['seed']==seed],key=lambda x:x['anchor']) for seed in sorted({v['seed'] for v in paired})]
    approach=[v for v in paired if v['phase']==0]
    j={k:float(np.median([v['K'][k]['mean_J'] for v in paired])) for k in ('4','16')}
    improved=sum(v['K']['16']['mean_J']<v['K']['4']['mean_J'] for v in paired)
    def counts(k):
        v=[x for x in rows if x['split']=='TRAIN' and x['K']==k]
        return dict(invalid=sum(bool(x['error']) for x in v),REJECT=sum(x['gate']['decision']=='REJECT' for x in v))
    c4,c16=counts(4),counts(16)
    approach_delta=float(np.median([v['K']['16']['mean_XYZ'] for v in approach])-np.median([v['K']['4']['mean_XYZ'] for v in approach]))
    release_delta=float(np.median([v['K']['16']['mean_gripper'] for v in first_release])-np.median([v['K']['4']['mean_gripper'] for v in first_release]))
    checks=dict(median_J_at_most_75pct=j['16']<=.75*j['4'],at_least16_strict_improvements=improved>=16,no_added_invalid=c16['invalid']<=c4['invalid'],no_added_REJECT=c16['REJECT']<=c4['REJECT'],approach_regression_max2mm=approach_delta<=.002,release_regression_max02=release_delta<=.02)
    k=16 if all(checks.values()) else 4
    fit=[v for v in rows if v['split']=='TRAIN' and v['K']==k and v['packet']==0]
    good=lambda x:x['prefix'] is not None and x['prefix']['XYZ_RMS']<=.010 and x['prefix']['SO3_RMS']<=.10 and x['prefix']['gripper_MAE']<=.10
    phases={str(p):dict(passed=sum(good(x) for x in fit if x['phase']==p),total=sum(x['phase']==p for x in fit)) for p in range(4)}
    adequate=sum(good(x) for x in fit)>=18 and all(v['total'] and v['passed']*2>=v['total'] for v in phases.values())
    result=dict(status='PASS_FIXED_CPU_SAMPLER_CHOICE_NOT_TASK_QUALIFICATION',K=k,checks=checks,TRAIN_median_J=j,ratio_of_medians=j['16']/j['4'],strict_improvements=improved,TRAIN_counts={'4':c4,'16':c16},approach_delta_m=approach_delta,first_release_gripper_delta=release_delta,first_release_UIDs=[x['uid'] for x in first_release],packet0_fit=sum(good(x) for x in fit),fit_adequate=adequate,phase_fit=phases,paired=paired,rows=rows,selection_used_VAL=False,new_model_queries=0,new_physics=0)
    # Nonfinite outputs remain failures; JSON null records undefined aggregate values.
    def finite(v):
        if isinstance(v,float) and not np.isfinite(v):return None
        if isinstance(v,dict):return {k:finite(x) for k,x in v.items()}
        if isinstance(v,list):return [finite(x) for x in v]
        return v
    write(root/'design/SAMPLER_FROZEN.json',finite(result));return result

if __name__=='__main__':
    import sys
    v=run(Path(sys.argv[1]));print(json.dumps({k:v[k] for k in ('status','K','checks','ratio_of_medians','packet0_fit','fit_adequate')}))

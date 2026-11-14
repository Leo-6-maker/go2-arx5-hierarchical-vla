"""Read saved H40s: label slack, zero-end-rate boundaries and paired noise.

The original preview_metrics still decides legality. No policy, optimizer or
native world is run; all arrays and statistical identities are immutable.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np

from scripts.sla.audit_saved_reference_probes_v61 import preview_metrics, LIMITS
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.s16_offline_v7 import DATA_SHA, STATS_SHA
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix, matrix_to_quat
from go2arx5_vla.world_model.task_learning_grasp_v1 import WORKING_OPEN_M, WORKING_CLOSED_M

DT=.05


def series(physical):
    physical=np.asarray(physical,dtype=float)
    velocity=np.diff(physical[:,:3],axis=0)/DT
    acceleration=np.diff(np.vstack((np.zeros(3),velocity,np.zeros(3))),axis=0)/DT
    rotations=rot6d_rows_to_matrix(physical[:,3:9])
    q=matrix_to_quat(rotations[1:]@np.swapaxes(rotations[:-1],-1,-2))
    q=np.where(q[:,3:]<0,-q,q);sine=np.linalg.norm(q[:,:3],axis=1,keepdims=True)
    angular=q[:,:3]*(2*np.arctan2(sine,q[:,3:]))/np.where(sine>0,sine,1.)/DT
    angular_a=np.diff(np.vstack((np.zeros(3),angular,np.zeros(3))),axis=0)/DT
    return dict(maximum_translation_speed_m_s=np.linalg.norm(velocity,axis=1),
        maximum_angular_speed_rad_s=np.linalg.norm(angular,axis=1),
        maximum_finger_speed_m_s=(WORKING_OPEN_M-WORKING_CLOSED_M)*np.abs(np.diff(physical[:,9]))/DT,
        maximum_discrete_translation_acceleration_m_s2=np.linalg.norm(acceleration,axis=1),
        maximum_discrete_angular_acceleration_rad_s2=np.linalg.norm(angular_a,axis=1)),acceleration


def position(index,name):
    if 'acceleration' not in name:
        return dict(kind='INTERVAL',H40_knots=[index,index+1])
    if index==0:return dict(kind='START_ZERO_VELOCITY_BOUNDARY',H40_knots=[0,1])
    if index==39:return dict(kind='END_ZERO_VELOCITY_BOUNDARY',H40_knots=[38,39])
    return dict(kind='H40_INTERIOR',H40_knots=[index-1,index,index+1])


def compare(prediction,label,state):
    metrics,failed=preview_metrics(prediction,state);truth,label_failed=preview_metrics(label,state)
    predicted,a_pred=series(prediction);target,a_label=series(label)
    epsilon=float(np.max(np.linalg.norm(prediction[:,:3]-label[:,:3],axis=1)))
    rows={}
    for name,limit in LIMITS.items():
        p,t=predicted[name],target[name]
        assert np.isclose(p.max(),metrics[name],rtol=1e-12,atol=1e-12)
        assert np.isclose(t.max(),truth[name],rtol=1e-12,atol=1e-12)
        bad=np.flatnonzero(p>limit*(1+1e-10));peak=int(np.argmax(p))
        points=[]
        for k in map(int,bad):
            point=dict(index=k,**position(k,name),predicted=float(p[k]),label=float(t[k]),
                ratio=float(p[k]/limit),label_margin_at_same_index=float(limit-t[k]))
            if name=='maximum_discrete_translation_acceleration_m_s2':
                point.update(acceleration_error_norm_m_s2=float(np.linalg.norm(a_pred[k]-a_label[k])),
                    conservative_error_bound_m_s2=(2 if k in (0,39) else 4)*epsilon/DT**2)
            points.append(point)
        rows[name]=dict(limit=limit,predicted_peak=float(p.max()),predicted_peak_location=position(peak,name),
            label_peak=float(t.max()),label_peak_margin=float(limit-t.max()),
            label_peak_ratio=float(t.max()/limit),predicted_peak_ratio=float(p.max()/limit),
            exceeded_points=points,predicted_series=p.tolist(),label_series=t.tolist())
    return dict(EEF_pass=not failed,label_EEF_pass=not label_failed,failed_metrics=failed,
        epsilon_max_xyz_m=epsilon,xyz_RMSE_m=float(np.sqrt(np.mean(np.sum((prediction[:,:3]-label[:,:3])**2,axis=1)))),
        acceleration_error_max_m_s2=float(np.max(np.linalg.norm(a_pred-a_label,axis=1))),
        conservative_internal_error_bound_m_s2=4*epsilon/DT**2,metrics=rows)


def run(owner):
    owner=Path(owner);start=time.time();cpu=time.process_time()
    result=json.loads((owner/'RESULT.private.json').read_bytes());assert result['complete500'] and result['step']==500
    assert json.loads((owner/'control/LEASE.json').read_bytes())['status']=='CLOSED'
    spec=json.loads((owner/'EXECUTION_SPEC.json').read_bytes());dataset=Path(spec['dataset'])
    assert sha(dataset/'MANIFEST.private.json')==DATA_SHA
    m=json.loads((dataset/'MANIFEST.private.json').read_bytes());assert sha(dataset/m['statistics']['file'])==STATS_SHA
    rows=[(ep,k) for ep in m['episodes'] for k in range(61)]
    records=json.loads((owner/'candidate500/PREDICTIONS32.private.json').read_bytes());cache={};out=[]
    for row in records:
        ep,k=rows[row['index']]
        if ep['id'] not in cache:
            p=dataset/ep['labels']['file'];assert sha(p)==ep['labels']['sha256']
            with np.load(p,allow_pickle=False) as z:cache[ep['id']]={a:z[a].copy() for a in ('state_eef10','nominal_policy_H40')}
        path=owner/'candidate500/work/step500-RGB'/row['file']
        with np.load(path,allow_pickle=False) as z:
            prediction=z['physical_H40'].copy();assert __import__('hashlib').sha256(prediction.tobytes()).hexdigest()==row['physical_sha256']
        values=cache[ep['id']];a=compare(prediction,values['nominal_policy_H40'][k],values['state_eef10'][k])
        assert a['EEF_pass']==row['EEF_full_H40_pass'] and a['failed_metrics']==row['failed_metrics']
        out.append(dict(index=row['index'],episode=ep['id'],anchor=k,phase=row['phase'],noise=row['noise'],
            physical_sha256=row['physical_sha256'],nominal_grip_MAE=row['nominal_grip_MAE'],
            all_fit_targets_pass=bool(a['EEF_pass'] and row['xyz_RMSE_m']<=.002 and row['mean_SO3_rad']<=.01 and row['nominal_grip_MAE']<=.02),**a))
    pairs=[]
    for index in dict.fromkeys(r['index'] for r in out):
        group=[r for r in out if r['index']==index];assert sorted(r['noise'] for r in group)==[0,1]
        count=sum(r['EEF_pass'] for r in group)
        pairs.append(dict(index=index,episode=group[0]['episode'],anchor=group[0]['anchor'],phase=group[0]['phase'],
            EEF_pass_count=count,classification={0:'BOTH_REJECTED',1:'MIXED',2:'BOTH_PASSED'}[count],
            all_fit_pass_count=sum(r['all_fit_targets_pass'] for r in group)))
    return dict(schema='SLA_S16_LABEL_MARGIN_BOUNDARY_NOISE_ACTUAL_V7_1',records=out,pairs=pairs,
        predictions=len(out),EEF_pass=sum(r['EEF_pass'] for r in out),all_fit_pass=sum(r['all_fit_targets_pass'] for r in out),
        pair_classifications=dict(Counter(r['classification'] for r in pairs)),
        source_prediction_metrics_sha256=sha(owner/'candidate500/PREDICTIONS32.private.json'),
        data_manifest_sha256=DATA_SHA,statistics_sha256=STATS_SHA,
        index_convention='H40_ZERO_BASED; acceleration0 and39 join first/last interval to zero velocity; interior k uses k-1,k,k+1',
        additional_model_loads=0,new_model_generations=0,optimizer_updates=0,training_visits=0,
        physical_requests=0,physics_integrations=0,formal_gate_changed=False,
        full_WBC_state_restored=False,root_cause_selected=False,wall_s=time.time()-start,CPU_s=time.process_time()-cpu)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('owner',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    value=run(a.owner)
    with a.output.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in value.items() if k not in ('records','pairs')}))

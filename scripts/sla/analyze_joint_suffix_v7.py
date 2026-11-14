"""Partition the saved original clone certificate's H40 acceleration grid.

For two reference-history points, acceleration[k] uses H40 knots k-2,k-1,k.
Thus k=0,1 include history; k<10 and k>=10 partition by the rightmost knot.
This analysis does not grant prefix-only execution or change any certificate.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def partition(command_q, *, history_points=2, prefix_knots=10):
    q=np.asarray(command_q,float)
    if q.shape!=(42,6) or history_points!=2 or prefix_knots!=10 or not np.isfinite(q).all():
        raise ValueError('ORIGINAL_TWO_HISTORY_PLUS40_KNOTS_SIX_JOINTS_REQUIRED')
    velocity=np.diff(q,axis=0)/.05
    acceleration=np.diff(velocity,axis=0)/.05
    assert acceleration.shape==(40,6)
    maximum=lambda x:float(np.max(np.abs(x)))
    bad=np.argwhere(abs(acceleration)>1.*(1+1e-10))
    if not len(bad):raise ValueError('ORIGINAL_REJECTED_TRAJECTORIES_ONLY')
    k,j=map(int,bad[0])
    j3=np.flatnonzero(abs(acceleration[:,2])>1.*(1+1e-10))
    first3=int(j3[0])if len(j3)else None
    return dict(prefix_knots=10,suffix_knots=30,
        acceleration_k='RIGHTMOST_H40_KNOT; negative preceding knots are executed history',
        prefix_max_abs_acceleration_rad_s2=maximum(acceleration[:10]),
        suffix_max_abs_acceleration_rad_s2=maximum(acceleration[10:]),
        joint3_prefix_max_abs_acceleration_rad_s2=maximum(acceleration[:10,2]),
        joint3_suffix_max_abs_acceleration_rad_s2=maximum(acceleration[10:,2]),
        reference_join_max_abs_acceleration_rad_s2=maximum(acceleration[:2]),
        first_exceedance_k=k,first_exceedance_joint=j+1,
        first_exceedance_partition='PREFIX'if k<10 else'SUFFIX',
        first_joint3_exceedance_k=first3,
        joint3_first_IK_qpos_triple=q[first3:first3+3].tolist()if first3 is not None else None,
        first_IK_qpos_triple=q[k:k+3].tolist(),
        per_joint=[dict(joint=i+1,prefix_max_abs_acceleration_rad_s2=maximum(acceleration[:10,i]),
            suffix_max_abs_acceleration_rad_s2=maximum(acceleration[10:,i]),
            max_speed_rad_s=maximum(velocity[:,i]))for i in range(6)],
        original_acceleration_limit_rad_s2=1.,original_speed_limit_rad_s=.7,
        prefix_only_execution_admitted=False,full_WBC_restored=False)


def run(actual):
    rows=[]
    for r in actual['records']:
        if r['classification']!='JOINT_ACCELERATION_REJECT':continue
        joint=r['offline_joint_diagnostic']['joint_rates'];p=partition(joint['command_q'])
        assert p['first_exceedance_k']==joint['first_acceleration_exceedance']['H40_knot_triple'][-1]
        assert p['first_exceedance_joint']==joint['first_acceleration_exceedance']['joint']
        rows.append(dict(native_run=r['native_run'],layout=r['layout'],restart=r['restart'],
            query=r['query'],native_step=r['native_step'],actual_joint_qpos=r['actual_joint_qpos'],
            actual_joint_qvel=r['actual_joint_qvel'],state_sha256=r['pins']['state'],
            original_certifier_sha256=r['offline_joint_diagnostic']['original_certifier_sha256'],**p))
    assert len(rows)==16
    return dict(schema='SLA_ORIGINAL16_JOINT_PREFIX_SUFFIX_DIAGNOSTIC_V7',
        original_monitor_result_sha256=actual['monitor_result_sha256'],trajectories=16,
        first_exceedance_partitions=dict(Counter(r['first_exceedance_partition']for r in rows)),
        prefix_exceeds_original_joint_bound=sum(r['prefix_max_abs_acceleration_rad_s2']>1*(1+1e-10)for r in rows),
        suffix_exceeds_original_joint_bound=sum(r['suffix_max_abs_acceleration_rad_s2']>1*(1+1e-10)for r in rows),
        joint3_exceeds_in_prefix=sum(r['joint3_prefix_max_abs_acceleration_rad_s2']>1*(1+1e-10)for r in rows),
        joint3_exceeds_in_suffix=sum(r['joint3_suffix_max_abs_acceleration_rad_s2']>1*(1+1e-10)for r in rows),
        rows=rows,new_model_queries=0,optimizer_updates=0,training_visits=0,
        physical_requests=0,physics_integrations=0,safety_limits_changed=False,
        inference='VIOLATION_LOCATION_ONLY_NOT_A_CAUSAL_CLASSIFICATION',
        root_cause_selected=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('actual',type=Path)
    a=p.parse_args();raw=a.actual.read_bytes();result=run(json.loads(raw))
    result['input_sha256']=hashlib.sha256(raw).hexdigest()
    print(json.dumps(result,indent=2,allow_nan=False))

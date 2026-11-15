"""Read fixed64 PD4 predictions in physical units, without a model or physics.

Own frozen normalizers are decoded independently. Labels share actual query
state/clock, while their future command origins remain distinct. Four episodes
are never treated as32 independent scenes. No policy parameters are selected.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from go2arx5_vla.sla_adapter.pd4_config_v71 import MANIFESTS,STATISTICS,LAYOUTS
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles,pad32
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
from go2arx5_vla.sla_adapter.dense_dataset_v4 import member
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,geodesic_angle
from scripts.pi0292.eval_contract import matrix10
from scripts.sla.diagnose_eef_margin_v71 import compare,series


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def trajectory(value):
    values,_=series(value)
    crossing=np.flatnonzero(value[:,9]<=.5)
    return dict(path_length_m=float(np.linalg.norm(np.diff(value[:,:3],axis=0),axis=1).sum()),
        first_nominal_grip_at_or_below_half=int(crossing[0])if len(crossing)else None,
        nominal_grip_total_variation=float(np.abs(np.diff(value[:,9])).sum()),
        seam9_to10={k:float(v[9])for k,v in values.items()},
        seam10={k:float(v[10])for k,v in values.items()if 'acceleration'in k},
        zero_start_acceleration={k:float(v[0])for k,v in values.items()if 'acceleration'in k},
        zero_end_acceleration={k:float(v[39])for k,v in values.items()if 'acceleration'in k})


def classification(left,right):
    return 'BOTH_PASS'if left and right else'EXECUTED_ONLY'if left else'FINAL_ONLY'if right else'BOTH_FAIL'


def correlation(left,right):
    if np.std(left)<1e-12 or np.std(right)<1e-12:return None
    return float(np.corrcoef(left,right)[0,1])


def run(owner):
    owner=Path(owner);start=time.time();cpu=time.process_time();terminal=read(owner/'RESULT.private.json')
    lease=read(owner/'control/LEASE.json');assert lease['status']=='CLOSED'and lease['worker_exit_code']==terminal['exit_code']==0
    assert(terminal['actual_optimizer_updates'],terminal['actual_training_visits'])==(1000,32000)
    views={};allrows=[];contexts={};normalizer_pins={}
    for view in MANIFESTS:
        root=owner/'arms'/(owner.name+'-'+view.lower());candidate=root/'candidate500';s=read(root/'EXECUTION_SPEC.json')
        dataset=Path(s['dataset']);assert sha(dataset/'MANIFEST.private.json')==MANIFESTS[view]
        manifest=read(dataset/'MANIFEST.private.json');assert tuple(e['layout']for e in manifest['episodes'])==LAYOUTS
        p=member(dataset,manifest['statistics']['file']);assert sha(p)==STATISTICS[view];stats=read(p);normalizer_pins[view]=sha(p)
        quantiles=Quantiles(stats['actions']['q01'],stats['actions']['q99']);cache={}
        rows=read(candidate/'PREDICTIONS32.private.json');assert len(rows)==32
        lookup={};labels={}
        for record in rows:
            index,noise=record['index'],record['noise'];episode=manifest['episodes'][index//61];anchor=index%61
            assert record['anchor']==anchor and noise in (0,1)
            if episode['id']not in cache:
                p=member(dataset,episode['labels']['file']);assert sha(p)==episode['labels']['sha256']
                with np.load(p,allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
                source=Path(episode['source_root']);capture=source/'native-01/FINAL_PLAN_CAPTURE_ACTUAL.private.json'
                assert sha(capture)==episode['final_plan_summary_sha256']
                plans=read(capture)['plans'];spec=read(source/'EXECUTION_SPEC.json');assert sha(source/'EXECUTION_SPEC.json')==episode['source_spec_sha256']
                cache[episode['id']]=(data,plans,np.asarray(spec['task_phase_source_from_target']))
            data,plans,frame=cache[episode['id']];state=data['state_eef10'][anchor];target=data['nominal_policy_H40'][anchor]
            assert int(data['native_step'][anchor])==5703+250*anchor and data['supervision_mask'][anchor].all()
            plan=plans[anchor];assert plan['native_step']==int(data['native_step'][anchor])
            assert str(data['origin_query_ids'][anchor,0])==plan['query_id']
            context=dict(state=state.tolist(),native_step=int(data['native_step'][anchor]),query_id=plan['query_id'],
                mount=episode['mount'],source_spec_sha256=episode['source_spec_sha256'])
            if index in contexts:assert context==contexts[index],'ACTUAL_SAME_QUERY_STATE_CLOCK_AND_MOUNT_REQUIRED'
            else:contexts[index]=context
            p=candidate/'work/step500-RGB'/record['file']
            with np.load(p,allow_pickle=False)as z:normalized=z['normalized'].copy();physical=z['physical_H40'].copy()
            assert hashlib.sha256(physical.tobytes()).hexdigest()==record['physical_sha256']
            assert hashlib.sha256(normalized.tobytes()).hexdigest()==record['normalized_sha256']
            decoded=GripLogitCodec(episode['mount']).decode(state,pad32(quantiles.unnormalize(normalized[0,:,:7])))
            assert np.array_equal(decoded,physical),'OWN_FROZEN_NORMALIZER_PHYSICAL_DECODE_MUST_MATCH_PACKET'
            detail=compare(physical,target,state)
            assert detail['EEF_pass']==record['EEF_full_H40_pass']and detail['failed_metrics']==record['failed_metrics']
            reference=frame@np.asarray(plan['reference_before_query']);actual=matrix10(state)
            ref_error=float(np.linalg.norm(reference[:3,3]-actual[:3,3]))
            item=dict(index=index,noise=noise,layout=episode['layout'],anchor=anchor,phase=record['phase'],
                query_id=plan['query_id'],prediction_sha256=record['physical_sha256'],
                xyz_RMSE_m=record['xyz_RMSE_m'],mean_SO3_rad=record['mean_SO3_rad'],nominal_grip_MAE=record['nominal_grip_MAE'],
                all_fit=bool(detail['EEF_pass']and record['xyz_RMSE_m']<=.002 and record['mean_SO3_rad']<=.01 and record['nominal_grip_MAE']<=.02),
                reference_actual_position_error_m=ref_error,
                reference_actual_SO3_rad=float(geodesic_angle(reference[:3,:3],actual[:3,:3])),
                derivative_and_label_margin=detail,prediction_trajectory=trajectory(physical))
            lookup[index,noise]=item;allrows.append(dict(view=view,**item));labels[index]=target
        assert len(lookup)==32 and set(lookup)=={(i,n)for i in labels for n in(0,1)}
        views[view]=dict(records=lookup,labels=labels,prediction_metrics_sha256=sha(candidate/'PREDICTIONS32.private.json'))
    left,right=[views[v]for v in MANIFESTS];assert left['records'].keys()==right['records'].keys()
    assert len(left['labels'])==len(right['labels'])==16
    pairs=[];label_pairs=[]
    for key,a in left['records'].items():
        b=right['records'][key];assert(a['query_id'],a['phase'])==(b['query_id'],b['phase'])
        pairs.append(dict(index=key[0],noise=key[1],layout=a['layout'],phase=a['phase'],
            EEF=classification(a['derivative_and_label_margin']['EEF_pass'],b['derivative_and_label_margin']['EEF_pass']),
            all_fit=classification(a['all_fit'],b['all_fit']),
            final_minus_executed={k:b[k]-a[k]for k in('xyz_RMSE_m','mean_SO3_rad','nominal_grip_MAE')}))
    for index,a in left['labels'].items():
        b=right['labels'][index];difference=b[:,:3]-a[:,:3]
        label_pairs.append(dict(index=index,layout=LAYOUTS[index//61],anchor=index%61,
            last_xyz_delta_m=difference[-1].tolist(),maximum_xyz_difference_m=float(np.linalg.norm(difference,axis=1).max()),
            mean_SO3_difference_rad=float(np.mean(geodesic_angle(rot6d_rows_to_matrix(a[:,3:9]),rot6d_rows_to_matrix(b[:,3:9])))),
            maximum_nominal_grip_difference=float(np.abs(a[:,9]-b[:,9]).max()),executed=trajectory(a),final=trajectory(b)))
    grouped={}
    for layout in LAYOUTS:
        p=[r for r in pairs if r['layout']==layout]
        grouped[layout]=dict(paired_predictions=len(p),unique_windows=len({r['index']for r in p}),
            EEF=dict(Counter(r['EEF']for r in p)),all_fit=dict(Counter(r['all_fit']for r in p)))
    associations={}
    for view in MANIFESTS:
        rows=views[view]['records'];indices=list(views[view]['labels']);x=[rows[i,0]['reference_actual_position_error_m']for i in indices]
        associations[view]=dict(reference_error_range_m=[min(x),max(x)],
            Pearson_descriptive_by_window={metric:correlation(x,[np.mean([rows[i,n][metric]for n in(0,1)])for i in indices])for metric in('xyz_RMSE_m','mean_SO3_rad','nominal_grip_MAE')},
            independent_episodes=4,causal_or_multivalued_input_conclusion=False)
    return dict(schema='SLA_PD4_FIXED64_PHYSICAL_PAIRED_DIAGNOSTIC_V71',predictions=64,paired_predictions=32,
        windows=16,episodes=4,physical_decodes_verified_bit_exact=64,normalizer_sha256=normalizer_pins,
        source_prediction_metrics_sha256={v:views[v]['prediction_metrics_sha256']for v in MANIFESTS},
        records=allrows,pairs=pairs,label_pairs=label_pairs,by_episode=grouped,reference_error_associations=associations,
        EEF_pair_classification=dict(Counter(p['EEF']for p in pairs)),all_fit_pair_classification=dict(Counter(p['all_fit']for p in pairs)),
        model_loads=0,new_model_generations=0,optimizer_updates=0,training_visits=0,physical_requests=0,physics_integrations=0,
        formal_evaluation_changed=False,normalized_losses_compared=False,joint_certificate_or_dynamic_WBC_proof=False,
        wall_s=time.time()-start,CPU_s=time.process_time()-cpu)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('owner',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    result=run(args.owner)
    with args.output.open('x')as f:json.dump(result,f,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items()if k not in('records','pairs','label_pairs')},allow_nan=False))

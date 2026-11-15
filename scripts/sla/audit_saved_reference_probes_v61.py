"""Read the existing V6 step3000 RGB probes; no model load or new inference.

Reports policy-frame geometric rates and label errors, never joint/native
qualification. The original eight TRAIN probes are not the fixed S16 cohort.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles,pad32
from go2arx5_vla.sla_adapter.dense_dataset_v4 import member
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,geodesic_angle
from go2arx5_vla.world_model import fusion_trajectory_adapter as envelope
from go2arx5_vla.world_model.task_learning_grasp_v1 import WORKING_OPEN_M,WORKING_CLOSED_M,MECHANICAL_TRAVEL_M
from scripts.pi0292.eval_contract import matrix10

LIMITS=dict(maximum_translation_speed_m_s=envelope.MAX_TRANSLATION_SPEED_M_S,
    maximum_angular_speed_rad_s=envelope.MAX_ANGULAR_SPEED_RAD_S,
    maximum_finger_speed_m_s=envelope.MAX_FINGER_SPEED_M_S,
    maximum_discrete_translation_acceleration_m_s2=envelope.MAX_TRANSLATION_ACCELERATION_M_S2,
    maximum_discrete_angular_acceleration_rad_s2=envelope.MAX_ANGULAR_ACCELERATION_RAD_S2)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preview_metrics(physical,state10):
    physical=np.asarray(physical);desired=physical.copy()
    if physical.shape!=(40,10)or not np.isfinite(physical).all()or np.any((physical[:,9]<0)|(physical[:,9]>1)):
        raise ValueError('REAL_FINITE_PHYSICAL_H40_WITH_NOMINAL_GRIP_REQUIRED')
    desired[:,9]=(WORKING_CLOSED_M+(WORKING_OPEN_M-WORKING_CLOSED_M)*physical[:,9])/MECHANICAL_TRAVEL_M
    all_metrics=envelope._geometry(desired,matrix10(state10));metrics={k:all_metrics[k]for k in LIMITS}
    return metrics,[k for k,v in LIMITS.items()if metrics[k]>v*(1+1e-10)]


def run(owner):
    owner=Path(owner);started=time.time();cpu=time.process_time()
    closed=json.loads((owner/'control/LEASE.json').read_bytes());result=json.loads((owner/'RESULT.private.json').read_bytes())
    assert closed['status']=='CLOSED'and closed['worker_exit_code']==result['exit_code']==0 and result['schema']=='SLA_CORE_FRESH_REFERENCE_N20_FM3K_RGB_ACTUAL_V6'and result['step']==3000
    candidate=owner/'candidate3000';spec=json.loads((candidate/'EXECUTION_SPEC.json').read_bytes());root=Path(spec['dataset']);m_path=root/'MANIFEST.private.json';manifest=json.loads(m_path.read_bytes());assert digest(m_path)==spec['dense_manifest_sha256']==result['dataset_manifest_sha256']
    stats_path=member(root,manifest['statistics']['file']);assert digest(stats_path)==manifest['statistics']['sha256'];stats=json.loads(stats_path.read_bytes());quantiles=Quantiles(stats['actions']['q01'],stats['actions']['q99'])
    rows=[(e,k)for e in manifest['episodes']for k in range(61)];cache={};records=[]
    probe_root=candidate/'work/step3000-RGB';probe_path=probe_root/'MANIFEST.private.json';probes=json.loads(probe_path.read_bytes());assert len(probes)==16 and len({r['index']for r in probes})==8 and all(sorted(r['noise']for r in probes if r['index']==i)==[0,1]for i in {r['index']for r in probes})
    for row in probes:
        episode,anchor=rows[row['index']];labels=episode['labels'];p=member(root,labels['file'])
        if episode['id']not in cache:
            assert digest(p)==labels['sha256']
            with np.load(p,allow_pickle=False)as z:cache[episode['id']]={k:z[k].copy()for k in ('state_eef10','latent7','nominal_policy_H40','native_step','phase')}
        data=cache[episode['id']];assert data['native_step'][anchor]==5703+250*anchor
        p=member(probe_root,row['file'])
        with np.load(p,allow_pickle=False)as z:normalized=z['normalized'].copy();physical=z['physical_H40'].copy()
        assert hashlib.sha256(normalized.tobytes()).hexdigest()==row['normalized_sha256']and hashlib.sha256(physical.tobytes()).hexdigest()==row['physical_sha256']and normalized.shape==(1,40,32)
        truth=data['nominal_policy_H40'][anchor];target=pad32(quantiles.normalize(data['latent7'][anchor])).astype(np.float32)
        errors=(normalized[0].astype(float)-target)**2;metrics,failed=preview_metrics(physical,data['state_eef10'][anchor])
        records.append(dict(index=row['index'],noise=row['noise'],episode=episode['id'],anchor=anchor,phase=str(data['phase'][anchor]),
            xyz_RMSE_m=float(np.sqrt(np.mean(np.sum((physical[:,:3]-truth[:,:3])**2,axis=1)))),
            mean_SO3_rad=float(np.mean(geodesic_angle(rot6d_rows_to_matrix(physical[:,3:9]),rot6d_rows_to_matrix(truth[:,3:9])))),
            nominal_grip_MAE=float(np.mean(np.abs(physical[:,9]-truth[:,9]))),
            normalized_generation_MSE32=float(np.mean(errors)),normalized_generation_MSE7=float(np.mean(errors[:,:7])),normalized_generation_MSE25=float(np.mean(errors[:,7:])),
            policy_frame_geometric_H40_pass=not failed,failed_rate_or_acceleration_metrics=failed,geometric_metrics=metrics,
            physical_array_sha256=row['physical_sha256'],normalized_array_sha256=row['normalized_sha256']))
    return dict(schema='SLA_EXISTING_REFERENCE3K_TRAIN_RGB16_OFFLINE_PROBE_AUDIT_V6_1',passes=True,
        endpoint_step=3000,target_training_visits=96000,manifest_sha256=spec['dense_manifest_sha256'],statistics_sha256=manifest['statistics']['sha256'],
        original_probe_manifest_sha256=digest(probe_path),existing_predictions_read=16,unique_TRAIN_windows=8,
        episodes=len({r['episode']for r in records}),S16_is_this_population=False,model_loads=0,new_RGB_generations=0,
        optimizer_updates=0,training_visits=0,physical_requests=0,physics_integrations=0,
        full_native_or_joint_qualification=False,reference_history_restore=False,geometry_clipped=False,
        metric_scope='POLICY_FRAME_EEF_GEOMETRY_ONLY_SAME_50MS_DERIVATIVES_NO_NATIVE_CERTIFICATE',limits=LIMITS,
        geometric_pass=sum(r['policy_frame_geometric_H40_pass']for r in records),
        by_phase={p:dict(predictions=sum(r['phase']==p for r in records),geometric_pass=sum(r['phase']==p and r['policy_frame_geometric_H40_pass']for r in records))for p in sorted({r['phase']for r in records})},
        records=records,wall_s=time.time()-started,CPU_s=time.process_time()-cpu)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('owner',type=Path);args=p.parse_args();print(json.dumps(run(args.owner),allow_nan=False))

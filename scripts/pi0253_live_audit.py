#!/usr/bin/env python3
"""Zero-model D0 using original expert timestamps and actual same-tick TCP."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5')
OUT=ROOT/'PI0253'
OLD=ROOT/'PI0252/pre_full_sim'
sys.path[:0]=[str(ROOT/'PI0250/runtime/reference-317a/scripts'),str(ROOT/'PI0250/runtime/reference-317a/src')]
from pi0253_live_bridge import sha, read, write
import pi0242_train_infer_bridge as bridge
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix, quat_to_matrix
from stage21_native_receding_horizon import _pose_window


def stats(x):
    return dict(min=float(np.min(x)),median=float(np.median(x)),mean=float(np.mean(x)),max=float(np.max(x)),final=float(x[-1]))


def nearest(states,ids,live):
    xyz=np.linalg.norm(states[:,:3].astype(np.float64)-live[:3],axis=1)
    so3=bridge._geodesic(rot6d_rows_to_matrix(states[:,3:9].astype(np.float64)),
                        np.broadcast_to(rot6d_rows_to_matrix(live[None,3:9]),(len(states),3,3)))
    return dict(xyz_nearest=[dict(source_id=str(ids[i]),xyz_m=float(xyz[i]),so3_rad=float(so3[i])) for i in np.argsort(xyz,kind='stable')[:5]],
                so3_nearest=[dict(source_id=str(ids[i]),xyz_m=float(xyz[i]),so3_rad=float(so3[i])) for i in np.argsort(so3,kind='stable')[:5]],
                xyz_distribution=stats(xyz),so3_distribution=stats(so3))


def main():
    ep=OLD/'controls/A_attempt2/actual_expert_reference.npz'
    assert sha(ep)=='5df0c2c2e75d7204baebacfaa7f0415a41e972867ccb9d772fe36103d6dfb140'
    expert=dict(np.load(ep,allow_pickle=False))
    t,p,q=_pose_window(expert,start_time_s=0.,period_s=.05,points=40)
    assert np.allclose(t,np.arange(40)*.05,atol=1e-12,rtol=0)
    assert np.allclose(p,expert['position'],atol=1e-12,rtol=0)
    anchor=expert['anchor_world_m'].astype(np.float64);target=expert['target_world_m']
    import torch
    wbc=Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/extracted/WBCTrackingShare')
    sys.path.insert(0,str(wbc))
    from modules.trajectory import TrajectoryBatch, to_canonical, quat_to_mat
    gamma,tl=to_canonical(torch.tensor(p,dtype=torch.float32),quat_to_mat(torch.tensor(q,dtype=torch.float32)),.05)
    batch=TrajectoryBatch(1,len(gamma.s_grid),len(tl.t_grid),device='cpu')
    batch.load([0],[gamma],[tl]);batch.gamma_p[0]+=torch.tensor(anchor,dtype=torch.float32)
    sample_t=torch.tensor(t,dtype=torch.float32)[None,:]
    native_p=batch.p_at(batch.s_ref(sample_t))[0].numpy().astype(np.float64)-anchor
    native_q=batch.quat_at(batch.s_ref(sample_t))[0].numpy().astype(np.float64)
    request=OLD/'G/A/mailbox/requests/model-0000.npz'
    response=OLD/'G/A/mailbox/responses/model-0000.npz'
    assert sha(response)=='00a63737df8e59bb0850ac31cd4616f05eefe72ccfe9b741227f7739c2f09b4b'
    d=dict(np.load(request));model=dict(np.load(response));live=d['state'].astype(np.float64)
    decoded=model['decoded_physical_action'][0].astype(np.float64)
    assert np.array_equal(model['input_state'],d['state'].astype(np.float32))
    assert np.array_equal(target,d['target_position_world_m'])
    rows=[json.loads(x) for x in (OLD/'G/A/output/model_receding_rows.jsonl').read_text().splitlines()]
    desired=np.asarray([r['desired_pos'] for r in rows]);actual=np.asarray([r['achieved_pos'] for r in rows])
    tracking=np.linalg.norm(desired-actual,axis=1)
    ex_rows=[json.loads(x) for x in (OLD/'controls/A_attempt2/rows.jsonl').read_text().splitlines()]
    ex_t=torch.tensor([[r['reference_time_s'] for r in ex_rows]],dtype=torch.float32)
    reconstructed=batch.p_at(batch.s_ref(ex_t))[0].numpy()
    same_tick_reference_error=float(np.linalg.norm(reconstructed-np.asarray([r['desired_world_m'] for r in ex_rows]),axis=1).max())
    assert same_tick_reference_error<=2e-6, 'actual expert TrajectoryBatch reconstruction mismatch'
    ex_tracking=np.linalg.norm(np.asarray([r['desired_world_m'] for r in ex_rows])-np.asarray([r['achieved_world_m'] for r in ex_rows]),axis=1)
    y_max=float(anchor[1]+decoded[:,1].max())
    np.savez_compressed(OUT/'correct_expert_and_tracking.npz',expert_t=t,expert_local_xyz=p,expert_quaternion=q,
        expert_native_sample_xyz=native_p,expert_native_sample_quaternion=native_q,
        model_local_xyz=decoded[:,:3],anchor_world=anchor,target_world=target,
        native_time=np.asarray([r['native_reference_local_time'] for r in rows]),desired_world=desired,achieved_world=actual,
        same_tick_tracking_error=tracking)
    original64=ROOT/'PI0250/inputs/frozen_v1/source64'
    manifest=read(original64/'manifest.json');source_rows=manifest['files']
    states=[];ids=[];matches=[]
    for row in source_rows:
        path=original64/row['path'];assert sha(path)==row['sha256']
        z=dict(np.load(path));states.append(z['state']);ids.append(row['source_episode_id'])
        if all(np.array_equal(z[k],d[k]) for k in ('body_rgb','wrist_rgb','state')):matches.append(row['source_episode_id'])
    full_report=read(OUT/'full_support.json')
    full_path=OUT/'full_train_numeric_support.npz'
    assert sha(full_path)==full_report['numeric_artifact_sha256']
    full=dict(np.load(full_path));early=full['start_time']<=.4+1e-8
    reset=full['start_time']==0
    family_a=np.array(['persistent_train_a_' in str(x) for x in full['source_episode_id']])
    controls={k:read(OLD/('controls/A_attempt2' if k=='A' else 'controls/'+k+'_attempt1')/'control.json') for k in 'ABCD'}
    assert all(r['status']=='PASS' for r in controls.values())
    rot=bridge._geodesic(rot6d_rows_to_matrix(decoded[:,3:9]),quat_to_matrix(native_q))
    reference=decoded[:,:3]+anchor
    report=dict(schema='PI0253_LIVE_CONTRACT_AND_SUPPORT_AUDIT_V1',status='D0_NUMERIC_AUDIT_COMPLETE',
        selected_checkpoint='NONE',model_queries=0,optimizer_updates=0,
        observed=dict(expert_file_sha256=sha(ep),live_file_sha256=sha(request),response_sha256=sha(response),
            correct_sampling=dict(helper='original to_canonical + TrajectoryBatch.p_at(s_ref(real_t))',timestamps=t.tolist(),
                expert_raw_reference_min_target_m=float(np.linalg.norm(p+anchor-target,axis=1).min()),
                expert_native_sample_min_target_m=float(np.linalg.norm(native_p+anchor-target,axis=1).min()),
                original_native_loader_sha256=sha(ROOT/'PI0250/runtime/reference-317a/scripts/stage21_native_receding_horizon.py'),
                original_trajectory_sha256=sha(wbc/'modules/trajectory.py'),same_tick_expert_reference_reconstruction_max_m=same_tick_reference_error,
                model_expert_xyz_rmse_m=float(np.sqrt(np.mean(np.sum((decoded[:,:3]-native_p)**2,axis=1)))),model_expert_so3_rmse_rad=float(np.sqrt(np.mean(rot**2)))),
            model_reference_target_distance_m=stats(np.linalg.norm(reference-target,axis=1)),
            model_actual_target_distance_m=stats(np.linalg.norm(actual-target,axis=1)),
            model_same_tick_tracking_m=stats(tracking),expert_same_tick_tracking_m=stats(ex_tracking),
            geometric_y_bound=dict(anchor_y=float(anchor[1]),predicted_local_y_max=float(decoded[:,1].max()),world_y_max=y_max,target_y=float(target[1]),min_y_gap_m=float(target[1]-y_max)),
            live_state=live.tolist(),live_world_xyz=(live[:3]+anchor).tolist(),source64_exact_input_matches=matches,
            source64_support_counts={f:dict(windows=sum(r['target_family']==f for r in source_rows),early_phase0_windows=sum(r['target_family']==f and r['phase']==0 for r in source_rows),
                rollouts=len({r['source_episode_id'].split(':obs-')[0] for r in source_rows if r['target_family']==f})) for f in 'abcd'},
            source64_physical_nearest=nearest(np.stack(states),ids,live),full_support_sha256=sha(OUT/'full_support.json'),
            full_physical_nearest=nearest(full['state'],full['source_episode_id'],live),
            full_a_reset_physical_nearest=nearest(full['state'][reset&family_a],full['source_episode_id'][reset&family_a],live),
            full_a_early_physical_nearest=nearest(full['state'][early&family_a],full['source_episode_id'][early&family_a],live),
            full_a_window_start_to_desired_1s_displacement_quantiles_m=np.quantile((full['action_at_1s'][:,:3]-full['state'][:,:3])[family_a],[0,.5,1],axis=0).tolist(),
            original_controls={k:dict(status=v['status'],control_sha256=sha(OLD/('controls/A_attempt2' if k=='A' else 'controls/'+k+'_attempt1')/'control.json'),anchor_world=v['anchor_world_m'],initial_bind=v['initial_bind']) for k,v in controls.items()},
            bridge_card_sha256=sha(OUT/'bridge_card.json'),bridge_report_sha256=sha(OUT/'bridge_report.json')),
        inference=dict(support='Original TRAIN has all four tasks and reset/early desired H40 labels. Numeric proximity is not evidence of sufficient coverage.',
            mechanism='Q and S outputs agree exactly under controlled actual noise. Old live miss is not explained by an observed worker calculation divergence.',
            initial_condition='Same frozen runtime, seed, native parent reference and preserved bind fields. Control and model references differ; first physics ticks cannot prove byte-identical initial state.'),
        unobserved=['Old live four-step intermediate tensors and exact policy inventory; do not requery.',
                    'Expert-control zero-tick root/dof/history tensor bytes were not separately archived.',
                    'Per-rollout world anchor is absent from materialized TRAIN rows; do not treat current live anchor as each TRAIN rollout anchor.'],
        discarded='Deprecated oracle excluded; difference of independent minima is not tracking.',
        boundary_deviation=dict(historical_pi0237_label_alignment_report_peek=True,
            details='A lookup opened historical archived CALIB alignment rows while locating TRAIN provenance. No CALIB dataset, policy queries, weights, statistics or selector were run or used. Preserved explicitly; no selection claim.'))
    write(OUT/'live_contract_support_audit.json',report)
    print(json.dumps(dict(status=report['status'],geometric_y_bound=report['observed']['geometric_y_bound'],correct_sampling=report['observed']['correct_sampling'],tracking_m=stats(tracking))),flush=True)


if __name__=='__main__':main()

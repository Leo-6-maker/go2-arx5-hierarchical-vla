"""Saved-byte independent FD2 audit, no engine construction or simulation."""
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.pi0292.durable import read
from scripts.pi0294.fd1 import put,sha
from scripts.pi0294.scene_cards import digest


def audit(root,identifier):
    from .audit import independent_push_score
    from .e1.audit import video
    from .score import score
    root=Path(root);folder=root/'episodes'/identifier;out=root/'evidence/audits'/identifier
    out.mkdir(parents=True,exist_ok=False,mode=0o700)
    checks={};errors=[]
    def check(name,value):
        checks[name]=bool(value)
        if not value:errors.append(name)
    card,result,worker,reservation=[read(folder/n) for n in ['CARD.json','RESULT.json','WORKER_RECEIPT.json','RESERVATION.json']]
    raw=read(folder/'FILE_MANIFEST.json');before=sha(folder/'FILE_MANIFEST.json')
    check('all_raw_manifest_sizes_hashes_no_escape',all(not (folder/x['path']).is_symlink()
        and (folder/x['path']).resolve().is_relative_to(folder.resolve())
        and (folder/x['path']).stat().st_size==x['bytes'] and sha(folder/x['path'])==x['sha256'] for x in raw))
    check('actual_pinned_source',result['source_sha']==worker['source_sha']==reservation['source_sha'])
    check('fixed_card_and_reservation',card==reservation['card'] and digest(card)==reservation['card_sha256'])
    samples,refs=[read(folder/n) for n in ['samples.json','references.json']]
    check('all_actual_camera_hashes',all(sha(folder/s['policy_obs'][c]['path'])==s['policy_obs'][c]['sha256']
        for s in samples for c in ['body_rgb','wrist_rgb']))
    maximum=0.;clock_error=0.;lookup={r['ref_id']:r for r in refs}
    for i,s in enumerate(samples):
        label=s['supervision']
        if label is None:continue
        ref=lookup[label['ref_id']];p=label['point_index']
        for key in ['desired','accepted','installed','committed']:
            maximum=max(maximum,float(np.max(np.abs(np.asarray(label[key])-ref['planned_H40'][p]))))
        maximum=max(maximum,abs(label['committed_gripper']-ref['gripper_H40'][p]))
        clock_error=max(clock_error,abs(label['command_time_s']-i*.05))
    check('unchanged_reference_command_labels',maximum<=1e-12)
    check('original_command_clock',clock_error<=1e-8)
    with np.load(folder/'physics.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(folder/'highrate.npz',allow_pickle=False) as z:
        high=z['rows'].copy();forces=z['external_force_max'].copy()
    steps=result['physics_steps'];micro=data['contact_audit'];times=data['contact_time_s']
    check('complete_actual_live_tick_clock',len(high)==steps and micro.shape==(steps,5) and times.shape==(steps,)
        and np.isfinite(high).all() and np.isfinite(micro).all()
        and np.allclose(high[:,0],np.arange(1,steps+1)*.002,atol=1e-8,rtol=0)
        and np.allclose(times,high[:,0],atol=1e-8,rtol=0))
    check('zero_external_forces',forces.shape==(steps,2) and not np.any(forces))
    recomputed=score(data,card,complete=result['status']=='VALID_COMPLETE')
    recomputed.update(physics_sha256=sha(folder/'physics.npz'),raw_sample_count=len(data['time_s']))
    check('exact_original_saved_score',recomputed==read(folder/'SCORE.json'))
    independent=independent_push_score(data,card,result['status']=='VALID_COMPLETE',.44)
    if steps==16000:
        check('independent_push_predicates',all(independent[k]==recomputed[k] for k in independent))
    else:
        check('independent_partial_not_success',not independent['persistent_success'] and not recomputed['persistent_success'])
    s1=read(folder/'S1_RESULT.json') if (folder/'S1_RESULT.json').exists() else None
    forecasts=read(folder/'S1_PREFLIGHTS.json') if (folder/'S1_PREFLIGHTS.json').exists() else []
    if s1:
        check('original_S1_actions_unmodified',s1['actions_modified'] is False)
        check('original_S1_parity_and_nonmutation',s1['maximum_live_shadow_difference']<=1e-10
            and all(f['live_state_unchanged'] for f in forecasts))
        rejects=[f for f in forecasts if f['decision']=='REJECT']
        check('S1_reject_count',len(rejects)==s1['pre_execution_rejections'])
        check('rejected_prefix_zero_live',not rejects or len(rejects)==1 and steps==rejects[0]['sequence']*25)
        check('four_actual_negative_binding_checks',not refs or s1['negatives'] is not None
            and set(s1['negatives']['rejected'])=={'nonce','action','state','decision'})
    with np.load(folder/'DYNAMIC_CLEARANCE.npz',allow_pickle=False) as z:
        live=z['live'].copy();shadow=z['shadow'].copy()
    check('passive_live_measurement_every_tick',live.shape==(steps,19) and np.isfinite(live).all()
        and np.allclose(live[:,0],high[:,0],atol=1e-8,rtol=0))
    observer=read(folder/'DYNAMIC_OBSERVER.json')
    check('passive_observer_no_native_calls_or_mutations',observer['status']=='PASS_UNCHANGED_FINGERPRINT'
        and observer['extra_snapshot_forward_step_calls']==0)
    paths=[sorted((folder/c).glob('*.png')) for c in ['body_rgb','wrist_rgb']]
    n=len(paths[0]);check('actual_both_camera_counts',n==len(paths[1])==worker['rgb_pairs'] and n in [len(samples),len(samples)+1])
    for i in sorted({0,n-1}) if n else []:
        for c in range(2):
            with Image.open(paths[c][i]) as im:check(f'RGB_{c}_{i}',im.mode=='RGB' and im.size==(256,256) and any(b>a for a,b in im.getextrema()))
    movie=video(folder,out/'actual-body-wrist.mp4',n) if n else None
    if movie:movie['relative_path']=str((out/'actual-body-wrist.mp4').relative_to(root))
    check('raw_manifest_unchanged_during_audit',sha(folder/'FILE_MANIFEST.json')==before)
    events=read(folder/'TEACHER_EVENTS.json')
    metrics=dict(minimum_live_vertical_gap_m=None if not len(live) else live[:,1:3].min(axis=0).tolist(),
        minimum_live_signed_geom_distance_m=None if not len(live) else live[:,3:5].min(axis=0).tolist(),
        minimum_predicted_shadow_signed_geom_distance_m=None if not len(shadow) else shadow[:,3:5].min(axis=0).tolist(),
        live_force_contact_ticks=0 if not len(live) else int(np.count_nonzero((live[:,5:7]>0).any(axis=1))),
        maximum_live_XYZ_reference_error_m=None if not len(live) else np.abs(live[:,7:10]).max(axis=0).tolist(),
        maximum_live_SO3_reference_error_rad=None if not len(live) else float(live[:,10].max()),
        maximum_live_gripper_reference_error=None if not len(live) else float(np.abs(live[:,11]).max()),
        teacher_events=events,nominal_clearance_m=card['nominal_clearance_m'])
    value=dict(status='PASS_INDEPENDENT_SAVED_BYTE_AUDIT' if not errors else 'AUDIT_FAIL_PRESERVED',
        id=identifier,checks=checks,errors=errors,score=recomputed,independent_score=independent,
        geometry=metrics,video=movie,initial_state=read(folder/'INITIAL_STATE.json'),
        initial_camera_sha256={c:sha(paths[i][0]) for i,c in enumerate(['body_rgb','wrist_rgb'])} if n else {},
        raw_manifest_sha256=before,dynamic_measurement_sha256=sha(folder/'DYNAMIC_CLEARANCE.npz'),
        result_sha256=sha(folder/'RESULT.json'),S1=s1,integrations=0,engine_starts=0,
        independent_full32_predicates_qualified=steps==16000,
        observer_measurements_not_independent_hidden_state_replay=True,
        no_training_admission_or_new_dataset=True)
    put(out/'AUDIT.private.json',value)
    if errors:raise ValueError('FD2_AUDIT_GAP:'+','.join(errors))
    return value

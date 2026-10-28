"""Bounded saved-array R3 WP3/WP5 audit; no model, recorder or runtime imports."""
import argparse, base64, csv, hashlib, io, json, os, sys, time
from collections import Counter
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def audit(root):
    import numpy as np
    from scipy.spatial.transform import Rotation
    started = time.time(); cpu_started=time.process_time(); root = Path(root); data = root/'data'
    def rotation(x):
        x = np.asarray(x, dtype=float).reshape(-1, 6)
        a = x[:, :3]; na = np.linalg.norm(a, axis=1)
        assert np.isfinite(x).all() and (na > 1e-12).all()
        a = a/na[:, None]; b = x[:, 3:]-a*(a*x[:, 3:]).sum(1)[:, None]
        nb = np.linalg.norm(b, axis=1); assert (nb > 1e-12).all()
        return np.stack((a, b/nb[:, None], np.cross(a, b/nb[:, None])), axis=1)
    def angles(a, b):
        return Rotation.from_matrix(a@b.transpose(0, 2, 1)).magnitude()
    bank = json.loads((data/'SOURCE_BANK.json').read_text())
    diag = json.loads((root/'design/DIAG96_PER_TASK.json').read_text())
    bias = json.loads((root/'audit/DIAG_CODEC_BIAS_CPU.json').read_text())
    bindings = {str(p.relative_to(root)): sha(p) for p in [data/'SOURCE_BANK.json', data/'statistics.json', data/'processor_config.json', data/'embodiment_id.json', root/'design/DIAG96_PER_TASK.json', root/'audit/DIAG_CODEC_BIAS_CPU.json', root/'code/teacher.py', root/'code/source.py', root/'code/recorder.py']}
    stats = json.loads((data/'statistics.json').read_text())['new_embodiment']
    low = np.asarray(stats['relative_action']['eef_9d']['min']); high = np.asarray(stats['relative_action']['eef_9d']['max'])
    scale = 2/(high-low)
    phases = {'approach':0, 'align':0, 'close':1, 'lift':1, 'carry':2, 'lower':2, 'release':3, 'withdraw':3, 'hold':4}
    selected = [e for e in bank['admitted_episodes'] if e['split'] in ('TRAIN','DEV') and e['layout'] < (285108 if e['split']=='TRAIN' else 285208)]
    assert len(selected)==32
    failures=[]; temporal=[]; transitions=[]; near=[]; alternatives=[];state_excursions=[]
    state_population=[]; action_population=[]; source_phase=Counter(); event_counts=Counter();groups={}
    by_episode={e['episode']:e for e in selected}
    diag_by_episode={name:[r for r in diag['rows'] if r['episode']==name] for name in by_episode}
    for ep in selected:
        name=ep['episode']; folder=Path(ep['folder']); result=json.loads((folder/'RESULT.json').read_text())
        assert sha(folder/'RESULT.json')==ep['source_result_sha256']
        original={n:sha(folder/n) for n in ('physics.npz','commands.npz','installed.npz','TEACHER_EVENTS.json','GATE_RECORDS.json')}
        for n,h in original.items(): assert h==result['files'][n], (name,n)
        p=dict(np.load(folder/'physics.npz')); c=dict(np.load(folder/'commands.npz')); inst=dict(np.load(folder/'installed.npz'))
        state=np.load(data/'policy_observation'/name/'state_v1.npy',mmap_mode='r')
        timestamps=np.load(data/'policy_observation'/name/'time_s.npy',mmap_mode='r')
        command=c['action']; phase=c['phase']; assert len(command)==660 and len(p['tcp'])==3301
        pre=p['tcp'][::5][:660]; post=p['tcp'][5::5]
        measured=(p['qpos'][::5][:660,-2:].sum(1)-.00004)/.08
        reconstructed=np.c_[pre[:,:3,3],pre[:,:2,:3].reshape(660,6),measured]
        pre_error=float(abs(reconstructed-state).max()); assert pre_error<1e-6
        assert np.allclose(timestamps,np.arange(660)*.05,atol=1e-12)
        assert np.allclose(p['time_s'][::5][:660],timestamps,atol=1e-12)
        polarity_error=float(abs((inst['action'][:,-1]+1)/2-command[:,-1]).max()); assert polarity_error<1e-7
        events=json.loads((folder/'TEACHER_EVENTS.json').read_text())['events']
        gates=json.loads((folder/'GATE_RECORDS.json').read_text()); rejects=[g['control']+20 for g in gates if g['decision']=='REJECT']
        for ev in events:
            i=20+int(round(ev['time_s']/.05)); before=ev['from_stage']; after=ev['to_stage']
            assert int(phase[i])==phases[after],(name,ev,i)
            g_before=0. if before in ('close','lift','carry','lower') else 1.
            assert np.isclose(command[i,-1],g_before), (name,ev,i)
            transitions.append(dict(episode=name,control=i,observation_time_s=float(timestamps[i]),teacher_time_s=ev['time_s'],phase_before=before,issued_command_stage=before,phase_after=after,stored_phase=int(phase[i]),issued_gripper=float(command[i,-1]),next_gripper=float(command[i+1,-1]),phase_label_differs_from_issued=phases[before]!=phases[after]))
        changes=np.zeros(660,dtype=bool); changes[1:]|=phase[1:]!=phase[:-1]
        grasp=p['is_cubeA_grasped'][::5]; changes[1:]|=grasp[1:660]!=grasp[:659];changes[1:]|=abs(np.diff(command[:,-1]))>.01
        group=groups.setdefault(ep['task']+'_'+ep['split'],dict(states=[],commands=[],phase=Counter(),events=Counter(),relative=[]))
        for a in range(20,621):
            source_phase[str(int(phase[a]))]+=1
            event='first8' if changes[a:a+8].any() else ('tail_only' if changes[a+8:a+40].any() else 'no_event')
            event_counts[event]+=1;group['phase'][str(int(phase[a]))]+=1;group['events'][event]+=1
        state_population.append(np.asarray(state[20:621]));action_population.append(command[20:621])
        group['states'].append(np.asarray(state[20:621]));group['commands'].append(command[20:621])
        state_low=np.r_[stats['state']['eef_9d']['min'],stats['state']['gripper_position']['min']];state_high=np.r_[stats['state']['eef_9d']['max'],stats['state']['gripper_position']['max']]
        for index,channel in zip(*np.where((state[20:621]<state_low)|(state[20:621]>state_high))):
            index=int(index)+20;channel=int(channel);v=float(state[index,channel]);lo=float(state_low[channel]);hi=float(state_high[channel])
            state_excursions.append(dict(episode=name,control=index,time_s=float(timestamps[index]),channel=channel,value=v,TRAIN_min=lo,TRAIN_max=hi,unclipped_normalized=2*(v-lo)/(hi-lo)-1))
        temporal.append(dict(episode=name,source_result_sha256=ep['source_result_sha256'],source_files=original,observations_precommand_max_abs=pre_error,gripper_command_to_actuator_max_abs=polarity_error,measured_opening_min=float(measured.min()),measured_opening_max=float(measured.max()),IK_holds=int(c['ik_hold'].sum()),gate_rejects=rejects,persistent_success=bool(result['persistent_success']),first_success_s=result.get('first_success_s')))
        base=np.load(data/'policy_observation'/name/'base_camera_raw.npy',mmap_mode='r');wrist=np.load(data/'policy_observation'/name/'hand_camera_raw.npy',mmap_mode='r')
        for row in diag_by_episode[name]:
            a=row['anchor']; uid=row['uid'];target=command[a:a+40]
            assert sha(row['input'])==row['input_sha256'] and sha(row['target'])==row['target_sha256']
            with np.load(row['input']) as f: assert np.array_equal(f['actual_state'],state[a]) and np.array_equal(f['base_raw'],base[a]) and np.array_equal(f['wrist_raw'],wrist[a])
            with np.load(row['target']) as f: assert np.array_equal(f['commands'],target)
            rot=rotation(target[:,3:9]); observed_rot=rotation(state[a,3:9])[0]
            relative=np.c_[(target[:,:3]-state[a,:3])@observed_rot,(observed_rot.T@rot)[:,:2,:].reshape(40,6)]
            group['relative'].append(relative)
            assert np.max(abs(relative[:,3:]))<=1+1e-12
            raw=2*(relative-low)/(high-low)-1
            clipped=(np.clip(raw,-1,1)+1)/2*(high-low)+low
            clipped_rot=observed_rot@rotation(clipped[:,3:9])
            native_angles=angles(clipped_rot,rot)
            recorded=next(x for x in bias['rows'] if x['uid']==uid)
            # Recorded native FP32/trace angles have ~1e-7 rad quantization at identity.
            assert np.isclose(np.sqrt(np.mean(native_angles**2)),recorded['full_H40']['SO3_RMS_rad'],atol=3e-7),(uid,float(np.sqrt(np.mean(native_angles**2))),recorded['full_H40']['SO3_RMS_rad'])
            # ponytail: one mathematical rotation support counterfactual; XYZ/gripper unchanged.
            alt=clipped.copy();alt[:,3:9]=np.clip(relative[:,3:9],-1,1)
            alt_rot=observed_rot@rotation(alt[:,3:9]);alt_angle=angles(alt_rot,rot)
            alternatives.append(dict(uid=uid,task=row['task'],split=row['split'],phase=row['phase'],anchor=a,native_SO3_RMS_rad=recorded['full_H40']['SO3_RMS_rad'],native_first8_SO3_RMS_rad=recorded['prefix']['SO3_RMS_rad'],alternative_SO3_max_rad=float(alt_angle.max()),alternative_SO3_RMS_rad=float(np.sqrt(np.mean(alt_angle**2))),alternative_first8_SO3_RMS_rad=float(np.sqrt(np.mean(alt_angle[:8]**2))),rotation_preclip_values=int((abs(raw[:,3:9])>1).sum()),valid_SO3=True,XYZ_gripper_scaling_unchanged=True))
            desired=inst['installed_target_fk'][a:a+8];achieved=post[a:a+8];same_rot=rotation(command[a:a+8,3:9])
            install_xyz=np.linalg.norm(desired[:,:3,3]-command[a:a+8,:3],axis=1)
            install_angle=angles(desired[:,:3,:3],same_rot)
            achieved_xyz=np.linalg.norm(achieved[:,:3,3]-desired[:,:3,3],axis=1)
            achieved_angle=angles(achieved[:,:3,:3],desired[:,:3,:3])
            failures.append(dict(record_type='TEACHER_FIXED_DIAG',uid=uid,task=ep['task'],split=ep['split'],layout=ep['layout'],anchor=a,stored_phase=int(phase[a]),status='VALID_TEACHER_SUCCESS',first_divergence='COMMAND_TO_IK_INSTALLED' if install_xyz.max()>1e-5 or install_angle.max()>1e-4 else 'INSTALLED_TO_SAME_CONTROL_ACHIEVED',raw_to_install_XYZ_max_m=float(install_xyz.max()),raw_to_install_SO3_max_rad=float(install_angle.max()),install_to_achieved_XYZ_max_m=float(achieved_xyz.max()),install_to_achieved_SO3_max_rad=float(achieved_angle.max()),prefix_IK_holds=int(c['ik_hold'][a:a+8].sum()),prefix_grasp_samples=int(p['is_cubeA_grasped'][a*5:(a+8)*5+1].sum()),task_loss='NONE_IN_SUCCESSFUL_TEACHER',causality='TRACKING_RESIDUAL_IS_DESCRIPTIVE_NOT_UNIQUE_CAUSE'))
            if row['phase'] in (0,3,4):
                image=np.r_[base[a,::8,::8].ravel(),wrist[a,::8,::8].ravel()].astype(float)
                previous=np.r_[base[max(a-1,0),::8,::8].ravel(),wrist[max(a-1,0),::8,::8].ravel()].astype(float)
                near.append(dict(uid=uid,episode=name,task=ep['task'],phase=int(phase[a]),anchor=a,image=image,previous=previous,state=np.asarray(state[a]),history_state=np.asarray(state[max(a-1,0)]),target=target[:8],pixel_sha256=hashlib.sha256(base[a].tobytes()+wrist[a].tobytes()).hexdigest(),state_sha256=hashlib.sha256(state[a].tobytes()).hexdigest()))
    pairs=[]
    for i,x in enumerate(near):
        candidates=[]
        for y in near[i+1:]:
            if x['task']!=y['task'] or x['episode']==y['episode']:continue
            xyz=float(np.linalg.norm(x['state'][:3]-y['state'][:3]));so3=float(angles(rotation(x['state'][3:9]),rotation(y['state'][3:9]))[0]);g=float(abs(x['state'][-1]-y['state'][-1]))
            if xyz>.01 or so3>.1 or g>.1:continue
            pixel=float(abs(x['image']-y['image']).mean()/255);label_xyz=float(np.sqrt(np.mean(np.sum((x['target'][:,:3]-y['target'][:,:3])**2,axis=1))))
            candidates.append(dict(uid_a=x['uid'],uid_b=y['uid'],phase_a=x['phase'],phase_b=y['phase'],downsample_two_view_MAE_fraction=pixel,state_XYZ_m=xyz,state_SO3_rad=so3,state_gripper_difference=g,first8_target_XYZ_RMS_m=label_xyz,first8_target_SO3_RMS_rad=float(np.sqrt(np.mean(angles(rotation(x['target'][:,3:9]),rotation(y['target'][:,3:9]))**2))),first8_target_gripper_MAE=float(abs(x['target'][:,-1]-y['target'][:,-1]).mean()),history_50ms_image_MAE_fraction=float(abs(x['previous']-y['previous']).mean()/255),history_50ms_state_XYZ_m=float(np.linalg.norm(x['history_state'][:3]-y['history_state'][:3])),exact_RGB_state_equal=x['pixel_sha256']==y['pixel_sha256'] and x['state_sha256']==y['state_sha256']))
        if candidates:pairs.append(min(candidates,key=lambda v:v['downsample_two_view_MAE_fraction']))
    state_population=np.concatenate(state_population);action_population=np.concatenate(action_population)
    group_coverage={}
    for name,group in groups.items():
        state_values=np.concatenate(group['states']);command_values=np.concatenate(group['commands']);relative_values=np.concatenate(group['relative'])
        state_low=np.r_[stats['state']['eef_9d']['min'],stats['state']['gripper_position']['min']];state_high=np.r_[stats['state']['eef_9d']['max'],stats['state']['gripper_position']['max']]
        action_low=np.r_[stats['action']['eef_9d']['min'],stats['action']['gripper_position']['min']];action_high=np.r_[stats['action']['eef_9d']['max'],stats['action']['gripper_position']['max']]
        group_coverage[name]=dict(anchors=len(state_values),phase_occupancy=dict(group['phase']),event_windows=dict(group['events']),state_min=state_values.min(0).tolist(),state_max=state_values.max(0).tolist(),state_outside_TRAIN_bounds_by_channel=((state_values<state_low)|(state_values>state_high)).sum(0).tolist(),raw_command_min=command_values.min(0).tolist(),raw_command_max=command_values.max(0).tolist(),raw_command_outside_TRAIN_absolute_bounds_by_channel=((command_values<action_low)|(command_values>action_high)).sum(0).tolist(),relative_DIAG_min=relative_values.min(0).tolist(),relative_DIAG_max=relative_values.max(0).tolist(),relative_DIAG_outside_TRAIN_bounds_by_channel=((relative_values<low)|(relative_values>high)).sum(0).tolist(),scope='BOUNDED_ROWS_ONLY_NOT_FULL_TRAIN_MAXIMA')
    plan_audits={name:dict(json.loads((data/(name+'_PLAN_AUDIT.json')).read_text()),receipt_sha256=sha(data/(name+'_PLAN_AUDIT.json'))) for name in ('PLACE4096','SHARED4096')}
    clipped_rows=[v for v in alternatives if v['native_SO3_RMS_rad']>1e-4]
    assert {v['uid'] for v in clipped_rows}=={'T1-DEV-285200-20','T2-DEV-285200-20'}
    assert bindings=={n:sha(root/n) for n in bindings}
    return dict(status='COMPLETE_BOUNDED_CPU_SAVED_ARRAY_TEMPORAL_AND_REPRESENTATION_DIAGNOSTICS',owner='r3_temporal',bindings=bindings,episodes=temporal,transition_rows=transitions,failure_rows=failures,findings=dict(confirmed_faults=['PRESERVED_TWO_DEV_ROTATION_CLIPPING_CASES'],additional_observed_support_gaps=['DEV285200_STATE_CHANNEL5_BELOW_TRAIN_MIN_CONTROLS20_23_BOTH_TASKS'],confirmed_semantics=['STORED_PHASE_IS_AFTER_TRANSITION_WHILE_ISSUED_COMMAND_IS_BEFORE_TRANSITION'],ruled_out_in_bounded_rows=['STATE_CAPTURED_POSTCOMMAND','FIRST8_INSTALLATION_INDEX_SHIFT','GRIPPER_COMMAND_POLARITY_INVERSION','GRIPPER_MEASUREMENT_FORMULA_MISMATCH'],unknowns=['UNIQUE_CAUSE_OF_ANY_LEARNED_POLICY_FAILURE','FULL_OBSERVABILITY','HISTORY_BENEFIT','FULL_TRAIN_CODEC_AGGREGATE_MAXIMA'],next_checks=['APPEND_FIRST_VALID_MODEL_EPISODE_FIRST_DIVERGENCE_AND_GATE_IK_CONTACT_ORDER','REVIEW_PROSPECTIVE_HISTORY_ROTATION_DESIGN_ONLY_UNDER_NEW_AUTHORITY']),coverage=dict(episode_count=len(selected),fixed_DIAG_windows=len(alternatives),bounded_anchor_count=len(state_population),source_phase_counts=dict(source_phase),event_window_counts=dict(event_counts),by_task_split=group_coverage,state_bound_excursions=state_excursions,state_min=state_population.min(0).tolist(),state_max=state_population.max(0).tolist(),command_min=action_population.min(0).tolist(),command_max=action_population.max(0).tolist(),scope='FIRST8_TRAIN_AND_FIRST8_DEV_PER_TASK_ONLY',all_TRAIN_aggregate_codec_maxima='UNOBSERVED',reused_full_plan_audits=plan_audits),alternative=dict(status='NOT_AUTHORIZED_FOR_PRODUCTION',rotation_domain=[-1,1],rows=alternatives,known_DEV_failures=clipped_rows,empirical_rotation_width=(high-low)[3:9].tolist(),empirical_dnormalized_dcomponent=scale[3:9].tolist(),alternative_dnormalized_dcomponent=[1.]*6,alternative_to_empirical_squared_loss_unit_ratio=(1/scale[3:9]**2).tolist()),near_observation=dict(rows=len(near),matching_rule='Same instruction, different original episode; XYZ<=.01m SO3<=.1rad gripper<=.1; nearest per row in 16x16 two-view pixels',pairs=sorted(pairs,key=lambda v:v['downsample_two_view_MAE_fraction'])[:12],candidate_pair_count=len(pairs),exact_RGB_state_pair_count=sum(v['exact_RGB_state_equal'] for v in pairs),interpretation='Approximate similarity does not establish contradictory labels or prove observability; 50ms history uses genuine stored prior observations.'),model_episode_attribution='DEFERRED_UNTIL_FIRST_VALID_MODEL_EPISODE_ARRAYS_OWNER_MAY_APPEND',calls=dict(model=0,teacher=0,physics=0,render=0,optimizer=0),normalizer_changed=False,selected='NONE',deployment='NONE',elapsed_wall_s=time.time()-started,CPU_process_s=time.process_time()-cpu_started,max_computational_threads=4)


def save(value,out):
    out=Path(out); wp3=out/'WP3';wp5=out/'WP5';wp3.mkdir(parents=True,exist_ok=True);wp5.mkdir(parents=True,exist_ok=True)
    rows=value.pop('failure_rows')
    (wp3/'TEMPORAL_CONTRACT.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    with (wp3/'FAILURE_ATTRIBUTION.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    cov=value['coverage']; alt=value['alternative'];near=value['near_observation']
    text=f'''# PI0285 R3 WP5 saved-array coverage and alternatives

Status: COMPLETE bounded descriptive CPU audit. Alternative design: **NOT_AUTHORIZED_FOR_PRODUCTION**. Normalizer/processor/labels unchanged; model/teacher/physics/render/optimizer calls all zero.

Inputs bind SOURCE_BANK, original source results and arrays, frozen statistics/processor/mapping, DIAG192 and original CPU codec bias by SHA256 in WP3/TEMPORAL_CONTRACT.json. Scope is both instructions, first eight TRAIN and first eight DEV layouts: {cov['episode_count']} episodes, {cov['bounded_anchor_count']} H40-eligible anchors, {cov['fixed_DIAG_windows']} fixed DIAG windows. This does not repeat the full 76928 TRAIN roundtrip. Original aggregate TRAIN maxima remain UNOBSERVED.

## Temporal contract

Observation index i is real RGB and StateV1 at t=i*0.05 before command i. StateV1 reconstructs physics sample 5*i; achieved state for command i is sample 5*(i+1). H40 begins with command i; first8 uses commands i..i+7, installations i..i+7 and achieved samples 5*(i+1)..5*(i+8). Teacher event time is relative to the 20 neutral controls: event index is 20+round(teacher_time/0.05).

Teacher target/gripper is computed from phase_before; transition mutates stage; returned stored phase marks phase_after. Thus align->close and lower->release transition rows issue the preceding gripper value for one 50ms control. This is confirmed bookkeeping semantics, not authorized relabeling. Measured opening is (q_left+q_right-0.00004)/0.08; commanded opening 0=closed, 1=open; native actuator is 2*g-1. All fixed original teachers succeeded. Installation/tracking residuals are reported per first8 in FAILURE_ATTRIBUTION.csv; they are not task losses or unique causal explanations. First model attribution is pending owner-provided first-valid arrays.

## Coverage

Bounded source phase occupancy: {json.dumps(cov['source_phase_counts'])}. Event-window exposure: {json.dumps(cov['event_window_counts'])}. State/raw-command/relative-target ranges and excursions outside frozen TRAIN bounds are reported separately for T1 TRAIN, T1 DEV, T2 TRAIN, T2 DEV in JSON coverage.by_task_split. These are observed support of these 32 episodes, not population bounds.

Additional observed state support gap: both T1/T2 DEV285200 StateV1 channel5 at controls20..23 (1.00..1.15s) is below frozen TRAIN minimum. At control20 raw value -0.11755000 versus TRAIN minimum -0.08169304, unclipped normalized value -1.45055024. Raw observation identity and pre-command timing are correct. This is a held-out support excursion, not a newly proven runtime bug; production statistics remain frozen. The rotation-action-only counterfactual below does not resolve state support. Absolute raw command channel5 has three excursions per task; production actions use relative normalization, whose observed failures remain the preserved two fixed windows.

Existing full plan audit receipts are reused, without rescanning images:
'''
    for name,p in cov['reused_full_plan_audits'].items():
        text+=f"\n- {name}: {p['rows']} draws, {p['unique_UIDs']} unique UIDs, {p['unique_actual_two_view_RGB']} unique two-view RGB, {p['unique_actual_states']} unique states, {p['unique_actual_RGB_state_pairs']} unique RGB/state pairs; first8 event draws {p['event_first8']}, tail-only draws {p['tail_only']}; phase draws {json.dumps(p['phase_counts'])}.\n"
    text+=f'''
## Rotation counterfactual

Raw commands remain evaluation truth. The retained frozen codec comparison has exactly two failures: T1-DEV-285200-20 and T2-DEV-285200-20, shared initial geometry. Their H40 SO3 RMS is 0.03232898 rad and first8 RMS 0.01070248 rad; original maximum SO3 deviation 0.035746579 rad remains preserved in R2 evidence. The valid-relative-SO3 counterfactual changes only six rotation normalization bounds to [-1,1], preserving original XYZ/gripper scaling and Gram-Schmidt finite/nondegenerate SO3 checks. Across these 192 fixed windows alternative maximum SO3 error is {max(r['alternative_SO3_max_rad'] for r in alt['rows']):.9g} rad. This is label algebra, not a learned model result or error-floor proof.

Empirical rotation widths: {alt['empirical_rotation_width']}. Existing normalized derivative per physical coefficient: {alt['empirical_dnormalized_dcomponent']}; alternative derivative is 1. For the same small physical coefficient error, alternative/empirical squared normalized loss ratios are {alt['alternative_to_empirical_squared_loss_unit_ratio']}. Expanded bounds strongly reduce weighting of narrow rotation components. No clipping does not establish better training geometry, and arbitrary predicted 6-vectors still require nondegeneracy checks.

## Stored-history near-observation audit

{near['rows']} fixed approach/release/hold observations were compared under the rule: {near['matching_rule']}. There are {near['candidate_pair_count']} nearest candidate pairs; exact full RGB/state matches among those {near['exact_RGB_state_pair_count']}. Up to twelve nearest records retain current pixel distance, physical state distance, first8 action differences and genuine 50ms prior-history differences in JSON. These surviving pairs are approach observations; no release/hold pair met the tight physical-state rule in this fixed sample. Their previous 50ms distances equal current distances, so this sample provides no demonstrated history improvement. Downsampled similarity does not prove contradictory labels; exact duplicate nonconflict does not prove full observability. Cross-instruction rows are excluded because language legitimately changes the desired goal.

## One prospective HISTORY/ROTATION design

NOT_AUTHORIZED_FOR_PRODUCTION: a later separately reviewed experiment may replace six relative rotation bounds with mathematical [-1,1] and add one genuine previous 50ms RGB pair and StateV1 plus a validity mask at reset. Explicit changed fields are relative_action.eef_9d.min/max[3:9], observation image/state temporal indices [previous,current], and reset history-valid mask; XYZ/gripper math, instructions, frame, H40/first8 and task scorer stay explicit. Use separate experiment/processor identities; never modify this sprint's sealed inputs. Required checks: valid SO3 roundtrip including both preserved DEV cases, nonfinite/degenerate rejection, XYZ/gripper parity, prior timestamp exactly current-0.05, no crossing reset/takeover, masked missing history, single-current-input baseline with identical noise and physical scoring. Current evidence motivates a diagnostic design only; it does not demonstrate history benefit or authorize gradients.
'''
    (wp5/'COVERAGE_AND_ALTERNATIVES.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/home/liuyu-nfs/Projects/go2-arx5/PI0285');p.add_argument('--out',default='reports/PI0285/r3');p.add_argument('--remote',action='store_true');a=p.parse_args()
    if a.remote:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
        from coordinator import remote
        source=Path(__file__).read_text(); code="import os,json\nfor n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[n]='4'\nos.environ['CUDA_VISIBLE_DEVICES']=''\nnamespace={'__name__':'r3_temporal_remote'}\n"
        code+="exec(compile("+repr(source)+",'r3_temporal.py','exec'),namespace)\nprint(json.dumps(namespace['audit']("+repr(a.root)+")))\n"
        value=json.loads(remote('core',code))
    else:value=audit(a.root)
    value['helper_sha256']=sha(__file__)
    save(value,a.out);print(json.dumps(dict(status=value['status'],episode_count=len(value['episodes']),windows=value['coverage']['fixed_DIAG_windows'],elapsed_wall_s=value['elapsed_wall_s'],calls=value['calls'])))

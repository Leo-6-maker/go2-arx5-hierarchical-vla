"""Core-local analysis of sealed predictions and recorded contacts; no policy calls."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .storage import digest, write_json
from . import fusion_trajectory_adapter as limits
from .task_learning_grasp_v1 import FiftyMsProgram, B

LIMITS = {
    'maximum_translation_speed_m_s': limits.MAX_TRANSLATION_SPEED_M_S,
    'maximum_angular_speed_rad_s': limits.MAX_ANGULAR_SPEED_RAD_S,
    'maximum_finger_speed_m_s': limits.MAX_FINGER_SPEED_M_S,
    'maximum_discrete_translation_acceleration_m_s2': limits.MAX_TRANSLATION_ACCELERATION_M_S2,
    'maximum_discrete_angular_acceleration_rad_s2': limits.MAX_ANGULAR_ACCELERATION_RAD_S2,
}


def checked(pin):
    path = Path(pin['path'])
    if path.is_symlink() or digest(path) != pin['sha256']:
        raise ValueError('Changed sealed analysis input')
    return json.loads(path.read_bytes())


def geometry(raw, state):
    from scripts.pi0292.eval_contract import matrix10
    anchor = matrix10(state)
    desired = raw.astype(float).copy()
    desired[:, 9] = (limits.WORKING_CLOSED_M +
                    (limits.WORKING_OPEN_M-limits.WORKING_CLOSED_M)*raw[:, 9])/limits.MECHANICAL_TRAVEL_M
    metrics = limits._geometry(desired, anchor)
    reasons = [k for k, maximum in LIMITS.items() if metrics[k] > maximum*(1+1e-10)]
    if not np.all((raw[:, 9] >= 0) & (raw[:, 9] <= 1)):
        reasons.append('decoded_gripper_outside_0_1')
    error = None
    try:
        FiftyMsProgram(raw, anchor, anchor, np.array([0., 0., 0., 1., 0., 0., 0.]))
    except ValueError as exc:
        error = str(exc)
    return dict(metrics=metrics, failed_components=reasons, original_envelope_exception=error,
                original_envelope_pass=error is None,
                first_position_jump_m=float(np.linalg.norm(raw[0, :3]-state[:3])),
                first_gripper_jump=float(raw[0, 9]-state[9]),
                decoded_gripper_min=float(raw[:, 9].min()), decoded_gripper_max=float(raw[:, 9].max()),
                exact_closed_knots=int(np.count_nonzero(raw[:, 9] == 0)),
                exact_open_knots=int(np.count_nonzero(raw[:, 9] == 1)),
                first_point_not_included_in_original_discrete_rate_gate=True)


def prediction_audit(result_pin, samples, dataset_pin, output):
    result, manifest = checked(result_pin), checked(dataset_pin)
    if result['actual_VLA_queries'] != 40 or len(result['records']) != 40:
        raise ValueError('Original complete Core40 required')
    episodes = {e['id']: e for e in manifest['episodes']}
    output = Path(output); output.mkdir(exist_ok=False)
    rows = []; predictions = {}
    for record in result['records']:
        file = Path(samples)/record['sample_file']
        if file.name != record['sample_file'] or digest(file) != record['sha256']:
            raise ValueError('Changed prediction member')
        with np.load(file, allow_pickle=False) as z:
            raw, expert, state = [z[k].copy() for k in ('physical_H40', 'expert_policy_H40', 'actual_state10')]
        if raw.shape != expert.shape or raw.shape != (40, 10) or not np.isfinite(raw).all():
            raise ValueError('Full finite unchanged H40 required')
        episode = episodes[record['episode']]
        if record['anchor'] not in episode['supervised_anchors']:
            raise ValueError('Excluded intervention in sealed probes')
        row = {k:record[k] for k in ('episode','anchor','seed','phase_audit_only','genuine_recovery_audit_only')}
        row.update(prediction_sha256=record['sha256'], student=geometry(raw,state), expert=geometry(expert,state),
                   event_errors=record['errors'].get('close_events'), cross_chunk=None)
        predictions[record['episode'], record['anchor']] = raw
        rows.append(row)
    for row in rows:
        previous = predictions.get((row['episode'], row['anchor']-1))
        if previous is not None:
            current = predictions[row['episode'], row['anchor']]
            row['cross_chunk'] = dict(previous_anchor=row['anchor']-1,
                position_gap_m=float(np.linalg.norm(current[0,:3]-previous[-1,:3])),
                gripper_gap=float(current[0,9]-previous[-1,9]),
                interpretation='Counterfactual join of predictions at recorded Teacher states; neither prediction was executed')
    reasons = Counter(k for row in rows for k in row['student']['failed_components'])
    summary = dict(schema='TASK_LEARNING_CORE40_LOCAL_ENVELOPE_AUDIT_V1',
        input_result=result_pin, dataset=dataset_pin, requested=40, audited=len(rows),
        original_envelope_pass=sum(r['student']['original_envelope_pass'] for r in rows),
        original_envelope_reject=sum(not r['student']['original_envelope_pass'] for r in rows),
        expert_envelope_pass=sum(r['expert']['original_envelope_pass'] for r in rows),
        failures_by_component=dict(reasons), adjacent_prediction_pairs=sum(r['cross_chunk'] is not None for r in rows),
        missing_adjacent_prediction_pairs=sum(r['anchor']>0 and r['cross_chunk'] is None for r in rows),
        rows=rows, original_limits=LIMITS, new_model_loads=0, new_queries=0, optimizer_updates=0,
        native_integrations=0, policy_execution_authorized=False, native_contact_gate_not_evaluated=True,
        clipping_applied=False, Teacher_fallback=False, cross_hardware_reject_replaced=False)
    write_json(output/'RESULT.private.json', summary)
    return summary


def contact_events(dataset_pin, output, guard=lambda: None):
    """Stream actual native trace contacts; phase names never stand in for contacts."""
    from scripts.pi0292.durable import write
    manifest = checked(dataset_pin); output = Path(output); output.mkdir(exist_ok=False)
    episodes = [r for r in manifest['episodes'] if r['split']=='TRAIN']
    rows=[]; windows=[]
    for episode in episodes:
        guard(); checked(episode['proof']['spec'])
        root = Path(episode['proof']['native']['path']).parent
        trace = root/'physics.jsonl'; hasher = __import__('hashlib').sha256()
        first={}; last={}; counts=Counter(); samples=0; previous=None; contact_transitions=[]
        with trace.open('rb') as stream:
            for raw in stream:
                hasher.update(raw); value=json.loads(raw)
                if value['kind']=='model':
                    geoms=value['geoms']; continue
                if value['kind']!='sample': continue
                step=int(value['step']); samples+=1
                if (previous is None and step!=1) or (previous is not None and step!=previous+3):
                    raise ValueError('Changed recorded 6ms contact trace cadence')
                previous=step
                if step<B: continue
                active=set()
                for c in value['contacts']:
                    pair=tuple(sorted((int(c['geom1']),int(c['geom2']))))
                    names=[geoms[i]['name'] for i in pair]
                    if 'task_cube_geom' not in names: continue
                    if float(c['force'][0]) < .05: continue
                    label=' / '.join(n or '<unnamed>' for n in names)
                    active.add(label); counts[label]+=1; first.setdefault(label,step); last[label]=step
                # Preserve transitions, rather than conflating contact duration and contact onset.
                if not contact_transitions or set(contact_transitions[-1]['pairs'])!=active:
                    contact_transitions.append(dict(native_step=step,pairs=sorted(active)))
        if previous!=21202 or samples!=7068: raise ValueError('Incomplete sealed 6ms TRAIN contact trace')
        onsets=[t['native_step'] for i,t in enumerate(contact_transitions) if i>0 and set(t['pairs'])-set(contact_transitions[i-1]['pairs'])]
        for anchor in episode['supervised_anchors']:
            start=B+1000*anchor; end=start+1000
            events=[s for s in onsets if start<=s<end]
            windows.append(dict(episode=episode['id'],anchor=anchor,phase_audit_only=episode['expert_stage_audit_only'][anchor],
                                contact_onset_native_steps=events,actual_contact_onset=bool(events),
                                onset_intervals_native_steps=[[s-3,s]for s in events]))
        rows.append(dict(episode=episode['id'],samples=samples,trace_sha256=hasher.hexdigest(),
                         first_contact_native_step=first,last_contact_native_step=last,contact_sample_counts=dict(counts),
                         transitions=contact_transitions, native_grasp_not_inferred_from_contact_alone=True))
        write(output/'PROGRESS.private.json',dict(completed_episodes=len(rows),requested=len(episodes),windows=len(windows)))
    if len(episodes)!=100 or len(windows)!=1450: raise ValueError('Original TRAIN100 denominator changed')
    result=dict(schema='TASK_LEARNING_TRAIN100_RECORDED_CONTACT_EVENTS_V1',dataset=dataset_pin,
                requested_episodes=100,audited_episodes=len(rows),distinct_supervised_windows=len(windows),
                contact_onset_windows=sum(r['actual_contact_onset'] for r in windows),rows=rows,windows=windows,
                model_loads=0,queries=0,optimizer_updates=0,native_integrations=0,TEST_read=False,
                event_condition='Recorded task_cube_geom pair normal force >=0.05N; not a native-grasp certificate',
                recorded_contact_period_s=.006,onset_time_resolution_s=.006,
                unrecorded_between_sample_contacts_not_excluded=True,
                event_metadata_is_model_input=False, original_data_modified=False)
    write_json(output/'RESULT.private.json', result)
    return result


def reference_joins(result_pin, samples, dataset_pin, output):
    """Join each proposal to the actual preceding consumed references, offline."""
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0294.task_path_reference import rates
    result,manifest=checked(result_pin),checked(dataset_pin)
    episodes={e['id']:e for e in manifest['episodes']};frame=np.asarray(manifest['T_policy_from_target'],float)
    inverse=np.linalg.inv(frame);history={};rows=[]
    for row in result['records']:
        episode=episodes[row['episode']];anchor=row['anchor']
        if row['episode']not in history:
            commands=checked(episode['proof']['confirmed_commands'])['records']
            native=checked(episode['proof']['native'])['reference_qualification']
            prior=[w for w in native['windows']if w.get('native_end_step')in(B-25,B)]
            if len(prior)!=2:raise ValueError('Original preaction consumed reference pair missing')
            history[row['episode']]=(commands,prior)
        commands,prior=history[row['episode']]
        if anchor:
            pair=commands[40*anchor-2:40*anchor]
            previous=np.stack([matrix10(v['installed_consumed_EEF10'])for v in pair])
            if not all(v['solver_gripper_confirmed']for v in pair):raise ValueError('Unconfirmed previous references')
        else:previous=np.stack([matrix10(v['actual_installed_consumed_EEF10_BASE7'][:10])for v in prior])
        file=Path(samples)/row['sample_file']
        if digest(file)!=row['sha256']:raise ValueError('Changed saved prediction')
        with np.load(file,allow_pickle=False)as z:
            raw=z['physical_H40'].copy();state=z['actual_state10'].copy()
        actual=inverse@matrix10(state);desired=inverse@matrix10(raw);installed=desired.copy()
        installed[:,:3,3]=previous[-1,:3,3]+desired[:,:3,3]-actual[:3,3]
        installed[:,:3,:3]=previous[-1,:3,:3]@actual[:3,:3].T@desired[:,:3,:3]
        linear,angular=rates(np.concatenate([previous,installed]),np.arange(42)*.05)
        rows.append(dict(episode=row['episode'],anchor=anchor,phase_audit_only=row['phase_audit_only'],
            prior_references='ORIGINAL_PREACTION_CONSUMPTION'if anchor==0 else'ACTUAL_TEACHER_CONSUMPTION',
            original_join_linear_speed_m_s=float(linear),original_join_angular_speed_rad_s=float(angular),
            original_join_rate_pass=bool(linear<=.06+1e-10 and angular<=.3+1e-10),
            installed_first_point_position_gap_m=float(np.linalg.norm(installed[0,:3,3]-previous[-1,:3,3])),
            student_predictions_executed=False,contact_certificate_evaluated=False))
    if len(rows)!=40:raise ValueError('Full40 reference join denominator')
    value=dict(schema='TASK_LEARNING_CORE40_ACTUAL_REFERENCE_JOIN_ANALYSIS_V1',requested=40,audited=40,
               original_join_rate_pass=sum(r['original_join_rate_pass']for r in rows),rows=rows,
               model_loads=0,queries=0,optimizer_updates=0,native_integrations=0,
               meaning='Counterfactual student proposal at recorded state, joined to actual Teacher reference history; not student closed-loop evidence')
    write_json(output,value);return value


def veto_geometry(group, output):
    """Recompute the recorded planned configuration in a clone, never integrate."""
    import mujoco
    from scripts.pi0294.task_state import read_state
    from scripts.pi0294.mp_client import contact_permitted
    group=Path(group);children=list((group/'runs').iterdir())
    if len(children)!=1:raise ValueError('One original+3cm request required')
    root=children[0];spec=json.loads((root/'EXECUTION_SPEC.json').read_bytes())
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    if lease['status']!='CLOSED' or lease['worker_exit_code']!=1:raise ValueError('Original closed failure required')
    native=root/'native-01';events=[]
    with(native/'COMMANDS.private.jsonl').open()as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('phase')=='reference-certificate-rejected':events.append(row)
    if len(events)!=1 or events[0]['submitted_or_executed']is not False:raise ValueError('Original unexecuted veto required')
    event=events[0];detail=event['contact_details'];state_file=native/'state.bin';pin=digest(state_file)
    state=read_state(state_file)
    if state['step']!=event['native_step']:raise ValueError('Final state is not the actual veto boundary')
    load=(mujoco.MjModel.from_binary_path if Path(spec['model_path']).suffix=='.mjb' else mujoco.MjModel.from_xml_path)
    model=load(spec['model_path']);data=mujoco.MjData(model)
    if (model.nq,model.nv,model.nu)!=(34,32,20)or model.opt.timestep!=.002:raise ValueError('Original model ABI/dt')
    data.qpos[:]=state['qpos'];data.qvel[:]=state['qvel'];data.ctrl[:]=state['ctrl'];data.time=state['time_s']
    mujoco.mj_forward(model,data)
    actual_pairs=[dict(geom_ids=[int(c.geom1),int(c.geom2)],distance_m=float(c.dist))for c in data.contact
                  if set((c.geom1,c.geom2))==set(detail['geom_ids'])]
    joints=[model.joint('x5_joint'+str(i)).id for i in range(1,7)]
    data.qpos[model.jnt_qposadr[joints]]=detail['planned_arm_qpos']
    mujoco.mj_forward(model,data)
    pair=next((c for c in data.contact if set((c.geom1,c.geom2))==set(detail['geom_ids'])),None)
    cube=model.geom('task_cube_geom').id;site=model.site('x5_ee').id
    geoms=[]
    for g in detail['geom_ids']:
        geoms.append(dict(geom_id=g,geom_type=int(model.geom_type[g]),size=model.geom_size[g].tolist(),
            body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])),world_center=data.geom_xpos[g].tolist()))
    cloned_distance=float(pair.dist)if pair is not None else None
    value=dict(schema='TASK_LEARNING_PLUS3CM_RECORDED_VETO_GEOMETRY_V1',
        native_boundary_step=state['step'],planned_native_tick=detail['planned_native_tick'],stage=event['stage'],
        original_reason=event['reason'],original_signed_distance_m=detail['distance_m'],
        cloned_signed_distance_m=cloned_distance,
        recorded_penetration_reproduced=cloned_distance is not None and abs(cloned_distance-detail['distance_m'])<1e-9,
        original_contact_predicate_still_rejects=pair is not None and not contact_permitted(detail['geom_names'],pair.dist,event['stage']),
        cloned_contact_point_cube_frame=((pair.pos-data.geom_xpos[cube])@data.geom_xmat[cube].reshape(3,3)).tolist()if pair is not None else None,
        planned_TCP_FK_position_error_m=float(np.linalg.norm(data.site_xpos[site]-np.asarray(detail['planned_TCP'])[:3,3])),
        actual_boundary_pair_contacts=actual_pairs,geoms=geoms,MuJoCo_version=mujoco.__version__,
        model_sha256=digest(spec['model_path']),state_sha256=pin,actual_state_unchanged=digest(state_file)==pin,
        MuJoCo_model_loads=1,cloned_forward_calls=2,physics_integrations=0,GR00T_calls=0,optimizer_updates=0,
        proposal_executed=False,contact_threshold_changed=False,task_retried=False,
        interpretation='Static reconstruction of the planned link6/cube veto; not evidence of an executed illegal collision')
    write_json(output,value);return value


def recovery_preparation(dataset_pin, output):
    """Reuse sealed under-close evidence and state the unqualified Oracle boundary."""
    from .task_learning_recovery_v1 import contract
    manifest=checked(dataset_pin)
    train=[e for e in manifest['episodes']if e['split']=='TRAIN']
    recovery=[e for e in train if e.get('genuine_recovery_success')is True]
    rows=[]
    for e in recovery:
        lease,worker,score=[checked(e['proof'][k])for k in('lease','worker','score')]
        if lease['status']!='CLOSED'or worker['exit_code']!=0 or score['task_success']is not True:
            raise ValueError('Lost sealed under-close evidence')
        rows.append(dict(episode=e['id'],supervised_windows=len(e['supervised_anchors']),
                         excluded_intervention_anchors=sorted(set(range(15))-set(e['supervised_anchors'])),
                         full_task_success=True,recovery_from_student_rollin=False))
    if len(train)!=100 or len(rows)!=50:raise ValueError('Original TRAIN100/50 recovery population')
    value=dict(schema='TASK_LEARNING_RECOVERY_ORACLE_PREPARATION_V1',dataset=dataset_pin,
        reused_sealed_Teacher_recovery_episodes=50,rows=rows,original_under_close_contract=contract(),
        new_Oracle_qualification=False,new_Oracle_calls=0,new_student_rollin_states=0,new_DAgger_labels=0,
        eligibility=dict(new_population_split='TRAIN_ONLY_DISTINCT_VERSION',actual_student_executed_prefix_required=True,
            actual_current_observation_and_consumed_reference_required=True,privileged_Oracle_metadata_not_student_input=True,
            full_native_reference_contact_and_consumption_gates_required=True,unexecuted_Oracle_suffix_is_BC=False,
            interventions_and_takeover_marked_separately=True,original_DEV_TEST_never_BC=True),
        next_gate='New actual student roll-in state plus registered Oracle safety qualification; no synthetic DAgger or Teacher-success relabeling',
        GR00T_calls=0,optimizer_updates=0,physics_integrations=0)
    write_json(output,value);return value

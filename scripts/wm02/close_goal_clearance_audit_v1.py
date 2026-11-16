"""Observe one predeclared contact fence on frozen historical states only.

This is not an installed safety guard or a native admission record. In
particular, an alarm at the contact timestamp is not called prevention.
"""
import json
from pathlib import Path
import time
import traceback
import numpy as np

from scripts.pi0292.durable import write
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read,sha


def fence(clearance_m,actual_minus_receiver_z_m,overlaps_table_xy):
    return bool(overlaps_table_xy and clearance_m<=.003 and actual_minus_receiver_z_m<=-.005)


def summarize_case(samples,*,qualified,first_STOP_tick=None):
    alarms=[s for s in samples if s['candidate_alarm']]
    first=alarms[0]if alarms else None
    # Native contact payload is from t-2ms. Merely preceding STOP is insufficient.
    lead=None if first is None or first_STOP_tick is None else .002*(first_STOP_tick-1-first['step'])
    return dict(observed_samples=len(samples),candidate_alarm_samples=len(alarms),
        first_alarm=first,original_qualified=qualified,successful_episode_would_be_blocked=bool(qualified and alarms),
        first_STOP_tick=first_STOP_tick,lead_to_first_recorded_contact_s=lead,
        sampled_alarm_strictly_before_recorded_contact=lead is not None and lead>0,
        sampled_alarm_at_least10ms_before_contact=lead is not None and lead>=.01-1e-12,
        monitor_control_latency_validated=False,original_outcome_unchanged=True)


def pad_clearance(model,data):
    table=model.geom('task_table').id;position=data.geom_xpos[table];rotation=data.geom_xmat[table].reshape(3,3)
    table_half=model.geom_size[table];candidates=[]
    for name in ('task_left_pad','task_right_pad'):
        geom=model.geom(name).id
        if int(model.geom_type[geom])!=6:raise ValueError('RECORDED_BOX_PAD_GEOMETRY_REQUIRED')
        corner_sign=np.asarray([[x,y,z]for x in (-1,1)for y in (-1,1)for z in (-1,1)])
        world=(corner_sign*model.geom_size[geom])@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
        local=(world-position)@rotation
        overlaps=bool(np.all(local.max(axis=0)[:2]>=-table_half[:2])and
                      np.all(local.min(axis=0)[:2]<=table_half[:2]))
        candidates.append(dict(pad=name,clearance_m=float(local[:,2].min()-table_half[2]),
                               overlaps_table_xy=overlaps))
    return candidates


def run(spec):
    import mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration
    if mujoco.__version__!='3.2.7'or mujoco.get_mjcb_control()is not None:
        raise ValueError('FROZEN_PASSIVE_GEOMETRY_ONLY')
    root=Path(spec['registered_root']);previous=Path(spec['previous_campaign']);started=time.time()
    ledger=read(previous/'control/ATTEMPTS.private.json')
    good=[r for r in ledger['attempts']if r['state']=='QUALIFIED'];bad=[r for r in ledger['attempts']if r['state']=='UNSAFE']
    if len(good)!=38 or len(bad)!=2:raise ValueError('FIXED_CN38_AND_TWO_UNSAFE_EXPERTS')
    episodes=[dict(id=r['id'],role='EXPERT_SUCCESS'if r['state']=='QUALIFIED'else'EXPERT_UNSAFE',
                   case=str(Path(r['result']['path']).parent),result=r['result'],qualified=r['state']=='QUALIFIED')
              for r in good+bad]
    student=read(previous/'readonly-attribution-v3/COLLISION.private.json')[0]
    student_case=Path(spec['C20_root'])/'TRAIN8/native/runs'/student['case']
    episodes.append(dict(id=student_case.name,role='STUDENT_UNSAFE',case=str(student_case),
                         result=dict(path=str(student_case/'RESULT.private.json'),sha256=sha(student_case/'RESULT.private.json')),qualified=False))
    write(root/'FIXED_AUDIT_PLAN.private.json',dict(episodes=episodes,
        candidate=dict(clearance_m=.003,negative_tracking_error_m=-.005,table_xy_overlap=True),
        selected_before_results=True,threshold_search=False,native_requests=0,integration_calls=0,
        sampling_scope='EXISTING_50MS_METRIC_STATES_NOT_FULL_2MS_GUARD_QUALIFICATION'))
    results=[];models={};evaluations=0;source_pins={};minimum_lead=[]
    for ep in episodes:
        if time.time()>=spec['deadline_epoch']-60 or(root/'STOP.json').exists():raise TimeoutError('FINITE_READONLY_AUDIT')
        case=Path(ep['case']);native=case/'native-01';native_spec=read(case/'EXECUTION_SPEC.json')
        result=read(ep['result']['path'])
        if sha(ep['result']['path'])!=ep['result']['sha256']:raise ValueError('ORIGINAL_RESULT_CHANGED')
        if ep['qualified']and(not result['completed']or result['unsafe']or result['technical_failure']or not result['shutdown_clean']):
            raise ValueError('SUCCESS_COHORT_QUALIFICATION_CHANGED')
        commands=[json.loads(line)for line in(native/'COMMANDS.private.jsonl').read_text().splitlines()]
        boundary={c['native_step']:c for c in commands if c.get('phase')=='mobile-accepted-boundary-050'}
        metric_path=native/'MOBILE_SAMPLES.private.jsonl'
        samples=[json.loads(line)for line in metric_path.read_text().splitlines()
                 if json.loads(line)['native_end_step']>=5703]
        model_path=native_spec['model_path']
        if model_path not in models:
            pin=sha(model_path)
            if pin!=native_spec['frozen_files'][model_path]:raise ValueError('ORIGINAL_SCENE_IDENTITY_CHANGED')
            model=mujoco.MjModel.from_binary_path(model_path);models[model_path]=(model,mujoco.MjData(model),pin)
        model,data,pin=models[model_path];tcp=model.site('x5_ee').id;records=[]
        with forbid_integration(mujoco)as calls:
            for sample in samples:
                tick=sample['native_end_step'];witness=boundary.get(tick)
                if witness is None:raise ValueError('EXACT_RECEIVER_BOUNDARY_AT_SAVED_METRIC_STEP_REQUIRED')
                if not witness['exact_receiver_knot']:raise ValueError('NO_GUESSED_RECEIVER_INTERPOLATION')
                data.qpos[:]=sample['qpos'];data.qvel[:]=sample['qvel'];data.ctrl[:]=0;data.time=.002*tick
                mujoco.mj_forward(model,data);evaluations+=1
                error=float(np.linalg.norm(data.site_xpos[tcp]-np.asarray(sample['actual_TCP'])[:3,3]))
                if error>1e-10:raise ValueError('PASSIVE_FK_DISAGREES_WITH_ORIGINAL_RECORDED_TCP')
                pads=pad_clearance(model,data);delta=float(data.site_xpos[tcp,2]-witness['accepted_pose7'][2])
                alarms=[p for p in pads if fence(p['clearance_m'],delta,p['overlaps_table_xy'])]
                records.append(dict(step=tick,receiver_native_time_s=witness['native_time_s'],
                    actual_minus_receiver_z_m=delta,pads=pads,candidate_alarm=bool(alarms),
                    FK_vs_recorded_TCP_error_m=error,controller_selected_reference_known=False,
                    native_grasp=sample['native_grasp']))
            if calls:raise RuntimeError('PASSIVE_AUDIT_ATTEMPTED_INTEGRATION')
        stops=result['physics_stops'];stop_tick=stops[0]['steps']if stops else None
        summary=summarize_case(records,qualified=ep['qualified'],first_STOP_tick=stop_tick)
        value=dict(**ep,**summary,geometry_sha256=pin,samples=records,
                   metric_file=dict(path=str(metric_path),sha256=sha(metric_path)))
        target=root/'data'/(ep['id']+'.private.json');write(target,value)
        results.append({k:v for k,v in value.items()if k not in ('samples','case','result')})
        source_pins[ep['id']]=dict(path=str(target),sha256=sha(target))
        write(root/'PROGRESS.private.json',dict(epoch=time.time(),episodes=len(results),passive_FK=evaluations,
                                               new_physical_requests=0,new_integrations=0))
    successes=[r for r in results if r['qualified']];unsafe=[r for r in results if not r['qualified']]
    admissible=(not any(r['successful_episode_would_be_blocked']for r in successes)and
                all(r['sampled_alarm_at_least10ms_before_contact']for r in unsafe))
    outcome=dict(schema='SLA_CLOSE_CONTACT_FENCE_OBSERVATIONAL_AUDIT_V1',exit_code=0,
        successful_expert_episodes=len(successes),known_unsafe_episodes=len(unsafe),
        successful_episodes_candidate_would_block=sum(r['successful_episode_would_be_blocked']for r in successes),
        unsafe_episodes_with_any_sampled_alarm=sum(r['candidate_alarm_samples']>0 for r in unsafe),
        unsafe_episodes_with_10ms_sampled_precontact_alarm=sum(r['sampled_alarm_at_least10ms_before_contact']for r in unsafe),
        meets_observational_screen=admissible,native_safety_qualified=False,independent_admission=False,
        candidate_installed=False,original_safety_or_WBC_changed=False,
        sampled_50ms_states_not_2ms_runtime_validation=True,all_original_outcomes_preserved=True,
        episodes=results,passive_FK_calls=evaluations,model_generations=0,optimizer_updates=0,
        new_physical_requests=0,new_integrations=0,elapsed_wall_s=time.time()-started,CPU_s=time.process_time(),
        files=source_pins,next_action='REVIEW_NEW_PREVENTION_CONTRACT_AND_NATIVE_STOP_PATH'if admissible
            else'REJECT_THIS_MONITOR_CANDIDATE_NO_THRESHOLD_RETUNING_OR_NATIVE_SUBMISSION')
    write(root/'RESULT.private.json',outcome);return outcome


if __name__=='__main__':
    import sys
    s=read(sys.argv[2]);root=Path(s['registered_root'])
    try:run(s);code=0
    except BaseException:
        write(root/'RESULT.private.json',dict(exit_code=1,failure=traceback.format_exc(),new_physical_requests=0,
                                             new_integrations=0,candidate_installed=False));code=1
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(s,code);raise SystemExit(code)

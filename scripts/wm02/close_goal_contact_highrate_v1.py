"""Resolve sampling limitations on four fixed saved 2ms windows, never replay dynamics."""
import bisect
import json
from pathlib import Path
import struct
import time
import traceback
import numpy as np

from scripts.pi0292.durable import write
from scripts.pi0294.task_state import parse_state
from scripts.wm02.close_goal_clearance_audit_v1 import fence,pad_clearance,summarize_case
from go2arx5_vla.world_model.mfree_unattended_ledger_v1 import read,sha


def window_packets(path,lower,upper):
    """Scan length/step headers; decode only the already fixed narrow window."""
    with Path(path).open('rb')as handle:
        while size:=handle.read(4):
            if len(size)!=4:raise ValueError('TRUNCATED_ORIGINAL_JOURNAL_LENGTH')
            n=struct.unpack('<I',size)[0]
            if not 40<=n<=2*1024**2:raise ValueError('ORIGINAL_PACKET_LENGTH')
            raw=handle.read(n)
            if len(raw)!=n:raise ValueError('TRUNCATED_ORIGINAL_PACKET')
            step=struct.unpack_from('<Q',raw,8)[0]
            if step>upper:break
            if step>=lower:yield parse_state(raw)


def run(spec):
    import mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration
    root=Path(spec['registered_root']);screen=read(spec['screen_result']['path']);started=time.time()
    assert sha(spec['screen_result']['path'])==spec['screen_result']['sha256']
    assert mujoco.__version__=='3.2.7'and mujoco.get_mjcb_control()is None
    plan=read(Path(spec['screen_result']['path']).parent/'FIXED_AUDIT_PLAN.private.json')
    originals={e['id']:e for e in plan['episodes']};selected=[]
    for e in screen['episodes']:
        if not e['qualified']:
            selected.append(dict(id=e['id'],lower=e['first_STOP_tick']-300,upper=e['first_STOP_tick'],
                                 first_STOP_tick=e['first_STOP_tick'],original_qualified=False))
        elif e['successful_episode_would_be_blocked']:
            center=e['first_alarm']['step']
            selected.append(dict(id=e['id'],lower=center-150,upper=center+150,
                                 first_STOP_tick=None,original_qualified=True))
    assert len(selected)==4 and sum(e['original_qualified']for e in selected)==1
    write(root/'FIXED_WINDOWS.private.json',dict(windows=selected,
        selection='ALL_THREE_KNOWN_COLLISIONS_AND_THE_SINGLE_PREDECLARED_FENCE_FALSE_POSITIVE',
        threshold_unchanged=dict(clearance_m=.003,tracking_error_m=-.005),
        reference_signal='LATEST_EXACT_RECEIVER_KNOT_HELD_BETWEEN_50MS_WITNESSES_NOT_CONTROLLER_SELECTED',
        conditional_uncertainty_bound_m='original_50mm_s_reference_rate_times_knot_age_s',
        conditional_bound_is_not_a_controller_selected_reference_proof=True,
        new_physical_requests=0,model_generations=0))
    outcomes=[];calls=0
    for window in selected:
        if time.time()>=spec['deadline_epoch']-60:raise TimeoutError('FINITE_SAVED_WINDOW_AUDIT')
        original=originals[window['id']];case=Path(original['case']);native=case/'native-01'
        native_spec=read(case/'EXECUTION_SPEC.json');result=read(original['result']['path'])
        assert sha(original['result']['path'])==original['result']['sha256']
        model_path=native_spec['model_path'];model_sha=sha(model_path)
        assert model_sha==native_spec['frozen_files'][model_path]
        model=mujoco.MjModel.from_binary_path(model_path);data=mujoco.MjData(model);tcp=model.site('x5_ee').id
        commands=[json.loads(line)for line in(native/'COMMANDS.private.jsonl').read_text().splitlines()]
        boundary={c['native_step']:c for c in commands if c.get('phase')=='mobile-accepted-boundary-050'}
        ticks=sorted(boundary);states=[];bounded=[];first_saved_illegal=None
        journal=native/'state.bin.journal';before=(journal.stat().st_size,journal.stat().st_mtime_ns)
        with forbid_integration(mujoco)as forbidden:
            for packet in window_packets(journal,window['lower'],window['upper']):
                step=packet['step'];data.qpos[:]=packet['qpos'];data.qvel[:]=packet['qvel'];data.ctrl[:]=packet['ctrl'];data.time=packet['time_s']
                mujoco.mj_forward(model,data);calls+=1
                at=bisect.bisect_right(ticks,step)-1
                if at<0:raise ValueError('NO_RECEIVER_WITNESS_BEFORE_ORIGINAL_STATE')
                witness=boundary[ticks[at]];age=max(0.,packet['time_s']-witness['native_time_s'])
                delta=float(data.site_xpos[tcp,2]-witness['accepted_pose7'][2]);pads=pad_clearance(model,data)
                held=any(fence(p['clearance_m'],delta,p['overlaps_table_xy'])for p in pads)
                eligible=age<=.05+1e-8
                bound=any(fence(p['clearance_m'],delta+.05*age,p['overlaps_table_xy'])for p in pads)and eligible
                row=dict(step=step,time_s=packet['time_s'],receiver_knot_step=ticks[at],receiver_knot_age_s=age,
                    actual_minus_held_receiver_z_m=delta,conditional_upper_error_m=delta+.05*age,
                    pads=pads,candidate_alarm=held,conditional_rate_bound_alarm=bound,
                    native_grasp_not_recomputed=True,controller_selected_reference_known=False)
                states.append(row);bounded.append(dict(row,candidate_alarm=bound))
                for contact in packet['contacts']:
                    names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,contact[k])for k in ('geom0','geom1')}
                    if 'task_table'in names and names&{'task_left_pad','task_right_pad'} and first_saved_illegal is None:
                        first_saved_illegal=dict(step=step,contact_time_s=packet['contact_time_s'],pair=sorted(names),distance_m=contact['distance_m'])
            if forbidden:raise RuntimeError('UNEXPECTED_INTEGRATION_ATTEMPT')
        assert before==(journal.stat().st_size,journal.stat().st_mtime_ns)
        covered=[r['step']for r in states]
        assert covered==list(range(window['lower'],window['upper']+1)), 'ORIGINAL_2MS_WINDOW_MUST_BE_CONTIGUOUS'
        if not window['original_qualified']:
            assert first_saved_illegal is not None and first_saved_illegal['step']==window['first_STOP_tick']
        held_summary=summarize_case(states,qualified=window['original_qualified'],first_STOP_tick=window['first_STOP_tick'])
        bounded_summary=summarize_case(bounded,qualified=window['original_qualified'],first_STOP_tick=window['first_STOP_tick'])
        target=root/'data'/(window['id']+'.private.json');write(target,dict(window=window,states=states,
            held_reference_profile=held_summary,conditional_rate_bound_profile=bounded_summary,
            original_first_illegal_contact=first_saved_illegal,geometry_sha256=model_sha,
            journal=dict(path=str(journal),sha256=sha(journal)),original_result=original['result'],
            actual_new_integrations=0,full_control_state_restore=False))
        outcomes.append(dict(window=window,held_reference_profile=held_summary,
            conditional_rate_bound_profile=bounded_summary,first_saved_illegal=first_saved_illegal,
            artifact=dict(path=str(target),sha256=sha(target))))
        write(root/'PROGRESS.private.json',dict(epoch=time.time(),completed_windows=len(outcomes),passive_FK_calls=calls,
                                               new_physical_requests=0,new_integrations=0))
    output=dict(schema='SLA_CLOSE_FIXED_2MS_CONTACT_WINDOWS_ACTUAL_V1',exit_code=0,windows=outcomes,
        passive_FK_calls=calls,threshold_search=False,original_outcomes_preserved=True,
        native_guard_installed=False,native_stop_latency_proven=False,independent_admission=False,
        full_control_state_restore=False,reference_is_receiver_witness_not_controller_selected=True,
        new_model_generations=0,new_optimizer_updates=0,new_physical_requests=0,new_integrations=0,
        elapsed_wall_s=time.time()-started,CPU_s=time.process_time())
    write(root/'RESULT.private.json',output);return output


if __name__=='__main__':
    import sys
    s=read(sys.argv[2]);root=Path(s['registered_root'])
    try:run(s);code=0
    except BaseException:
        write(root/'RESULT.private.json',dict(exit_code=1,failure=traceback.format_exc(),new_physical_requests=0,
                                             new_integrations=0,native_guard_installed=False));code=1
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(s,code);raise SystemExit(code)

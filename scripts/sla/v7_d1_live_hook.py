"""Optional D1 Teacher decision at the unchanged completed Student P10 boundary."""
from pathlib import Path

from scripts.pi0292.durable import write
from go2arx5_vla.world_model.multitask_teacher_episode_v6 import confirmed_context
from v7_precontact_teacher import EarlyWarning,prepare_warning


def maybe_prepare(pilot,adapter,score,initial,query,consumed):
    if pilot.rejected:
        return None
    adapter.guard()
    sample=adapter.measured_task_state()
    if not hasattr(pilot,'d1_warning'):
        pilot.d1_warning=EarlyWarning(pilot.required_early_stage,adapter.actual_state.model,adapter.task.color)
    event=pilot.d1_warning.observe(sample,score.result(),consumed)
    if event is None:
        return None
    if sample['native_step']+250>adapter.spec['max_ticks'] or query+1>=adapter.spec['policy_queries']:
        return None
    previous=confirmed_context(adapter,sample['native_step'],sample['time_s'])
    try:
        teacher,program,evidence,margin=prepare_warning(initial,sample,score.result(),adapter.spec,
                                                      consumed,previous,adapter.base_pose_hold,event)
    except ValueError as error:
        pilot.rejected=True
        write(adapter.output/'D1-HANDOFF-REJECTED.private.json',dict(event,query=query,reason=str(error),first_install_sent=False))
        return None
    write(adapter.output/'D1-WARNING-BOUNDARY.private.json',dict(event,query=query,
          actual_previous_width_m=previous['width_m'],full_P10=True,
          teacher_input_only=True,student_input_additions=0))
    return dict(event=dict(event,actual_previous_width_m=previous['width_m'],holding_at_handoff=sample['native_grasp']),
        teacher=teacher,program=program,evidence=evidence,margin=margin,
        witness=consumed['confirmed_intervals'][-1]['accepted_boundary'],query=query)

"""50ms causal recorder; zero-physics rejection is a separate event."""
from copy import deepcopy
import numpy as np


def validate_record(row, *, native_dt):
    fields=('run_id','episode_id','parent_id','scene_id','split','executor_id','actor_version',
      't0','t1','physics_ticks','action_raw','reference_compiled','reference_installed',
      'reference_consumed','state_before','state_after','control_restore_ref',
      'terminal','truncated','safety_reason','valid_state_target')
    if any(k not in row for k in fields):raise ValueError('Missing actual transition provenance')
    ticks=row['physics_ticks'];dt=row['t1']-row['t0']
    if (type(ticks)is not int or ticks<=0 or not np.isfinite([dt,native_dt]).all()
        or native_dt<=0 or abs(dt-ticks*native_dt)>1e-8 or dt>.05+1e-8
        or(dt<.05-1e-8 and not(row['terminal']or row['truncated']))):
        raise ValueError('Actual positive physical interval required; rejection is not a transition')
    if row['split']not in('TRAIN','CALIB','DEV','TEST'):raise ValueError('Unknown parent split')
    if row['state_after'].get('reset')or row['state_before'].get('reset'):
        raise ValueError('Auto-reset observation is not the terminal successor')
    if abs(row['state_before']['time_s']-row['t0'])>1e-8 or abs(row['state_after']['time_s']-row['t1'])>1e-8:
        raise ValueError('State/physics clock mismatch')
    if not row['control_restore_ref']:raise ValueError('No complete control-state reference')
    if len(row['reference_consumed'])!=ticks:raise ValueError('Every tick needs actual command/progress evidence')
    return row


class TransitionRecorder:
    def __init__(self, *, native_dt):
        self.dt=native_dt;self.rows=[];self.events=[];self.parents={}

    def append(self,row):
        validate_record(row,native_dt=self.dt)
        parent=row['parent_id'];split=row['split']
        if parent in self.parents and self.parents[parent]!=split:raise ValueError('Parent split leakage')
        self.parents[parent]=split
        if self.rows:
            previous=self.rows[-1]
            if previous['episode_id']==row['episode_id']:
                if previous['terminal']or previous['truncated']:raise ValueError('Transition after terminal/reset')
                if any(previous[k]!=row[k]for k in('parent_id','scene_id','split','executor_id','actor_version')):
                    raise ValueError('Mid-episode identity changed')
                if abs(previous['t1']-row['t0'])>1e-8:raise ValueError('Missing interval')
        self.rows.append(deepcopy(row))

    def reject(self, *, episode_id, reason, state, action, physics_ticks=0):
        if physics_ticks!=0:raise ValueError('Executed failure must retain actual transitions')
        self.events.append(dict(episode_id=episode_id,reason=reason,state=deepcopy(state),
            action=deepcopy(action),physics_ticks=0,executed_expert_label=False))

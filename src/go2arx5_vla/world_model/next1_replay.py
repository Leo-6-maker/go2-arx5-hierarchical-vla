"""Select one real burn8/predict16 window after NEXT1's fixed HOLD prefix."""
from .target_replay import validate_windows

MANIFEST='WM02_NEXT1_LATE_DECISION_MANIFEST_V1'


def sample_view(diag,commands,*,prehold_atoms=40):
    if (prehold_atoms!=40 or diag.get('reference_rule')!='FIXED_HOLD40_BURN8_DECIDE_RAMP8_V1'
            or diag.get('complete_task_atoms')!=64 or diag.get('task_native_ticks')!=1600
            or diag.get('decision_atom')!=48 or not diag.get('late_decision_applied')
            or not diag.get('steady_prefix',{}).get('qualified')):
        raise ValueError('WHOLE_NEXT1_QUALIFIED_STEADY_DECISION_EPISODE_REQUIRED')
    original_start=diag['actual_collection_start_step'];validate_windows(diag['windows'],original_start,64)
    start=original_start+25*prehold_atoms
    selected=[p for p in commands if p.get('native_step',-1)>=start]
    windows=diag['windows'][prehold_atoms:]
    validate_windows(windows,start,24)
    return dict(diag,actual_collection_start_step=start,actual_collection_start_s=start*.002,task_native_ticks=600,complete_task_atoms=24,windows=windows),selected

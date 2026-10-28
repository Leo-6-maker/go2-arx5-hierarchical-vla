"""One source-bound clock/watchdog engineering adjudication; no FAIL relabeling."""
from pathlib import Path
from .task_capture_runtime import read,sha

EPISODE='TFNR1_T2_PC2_COUPLED_DEV_00'
SOURCE='ed681666bbd2e4b22f075d5a02220d66026396df'


def verified(root,episode):
    if episode in (EPISODE+'_E2',EPISODE+'_E3'):return verified_same_clock_pause(root,episode)
    if episode!=EPISODE:return False
    p=Path(root)/'factory/episodes'/episode
    try:
        r=read(p/'RESULT.json');proof=read(p/'CLOCK_RECOVERY_ADJUDICATION.json')
        return (r['source_sha']==SOURCE and r['status']=='ENGINEERING_OR_NATIVE_SAFETY_INCOMPLETE'
            and r['error']=='ValueError: ACTUAL_PAUSED_ROS_SOURCE_CLOCK_NOT_BOUND'
            and r['observations']==110 and r['submitted_intents']==14 and r['actual_plugin_steps']==5250
            and not r['native_stops'] and proof['result_sha256']==sha(p/'RESULT.json')
            and proof['ranges_sha256']==sha(p/'ALL_JOINT_RANGE_AUDIT.json')
            and read(p/'ALL_JOINT_RANGE_AUDIT.json')['all_raw_samples_within_original_ranges']
            and proof['illegal_contact_records']==0 and proof['planner_rejects']==0
            and proof['actual_native_watchdog_HOLD_observed'] is True
            and proof['same_actual_pause_history_preserved'] is True
            and proof['classification']=='VERIFIED_TASK_WRAPPER_RESUME_ENGINEERING_INCOMPLETE')
    except (FileNotFoundError,KeyError,ValueError):return False


def verified_same_clock_pause(root,episode):
    p=Path(root)/'factory/episodes'/episode
    try:
        r=read(p/'RESULT.json');proof=read(p/'CLOCK_RECOVERY_ADJUDICATION.json');ack=read(p/'pause_ack.json')
        case={EPISODE+'_E2':('fcba57a4612ad38bae77a603ff14b529f87a14b1',2,1,2550),EPISODE+'_E3':('9a51ddbb52408b39c244fcf64fe8892c8e7adee1',46,6,3650)}.get(episode)
        return (case is not None and r['source_sha']==case[0]
            and r['status']=='ENGINEERING_OR_NATIVE_SAFETY_INCOMPLETE' and r['error']=='RuntimeError: LIVE_PAUSE_ACK_MISSING'
            and r['observations']==case[1] and r['submitted_intents']==case[2] and r['actual_plugin_steps']==case[3]
            and not r['native_stops'] and proof['result_sha256']==sha(p/'RESULT.json')
            and proof['ranges_sha256']==sha(p/'ALL_JOINT_RANGE_AUDIT.json')
            and read(p/'ALL_JOINT_RANGE_AUDIT.json')['all_raw_samples_within_original_ranges']
            and proof['illegal_contact_records']==0 and proof['planner_rejects']==0
            and proof['classification']=='VERIFIED_SAME_SIMCLOCK_FRESH_COMMAND_BARRIER_INCOMPLETE'
            and 0<=ack['native_command_age_s']<=.45 and ack['native_arm_mode']=='WBC' and ack['native_command_valid'])
    except (FileNotFoundError,KeyError,ValueError):return False

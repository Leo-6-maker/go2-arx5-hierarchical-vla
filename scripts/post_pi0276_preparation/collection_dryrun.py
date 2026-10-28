"""Finite collection plan and mock state machine. No native execution at this release."""
import argparse,json
from pathlib import Path
from cpu_dataset import RELEASE

def jobs(cards):
    own=[r for r in cards if r['split']=='DEV'];result=[]
    for row in own:
        for executor in ('native_joint_expert','EEF_physical_replay'):
            result.append(dict(scene_id=row['scene_id'],parent_group=row['parent_group'],domain=row['domain'],executor=executor,
                native_reset_seed=row['native_reset_seed'],maximum_controls=640,initial_state_match_required=True,source_command_and_actual_state_separate=True,
                fresh_RGB_required=True,target_WBC_filter=False,registered_failure_retention=True))
    if len(result)!=48:raise ValueError('Fixed prospective pilot must have48 episodes')
    return result

def run(plan,executor,technical_retries=0):
    if not 0<=technical_retries<=1:raise ValueError('Finite draft retry bound')
    results=[]
    for job in plan:
        for attempt in range(technical_retries+1):
            value=executor(job,attempt);status=value['status']
            if status not in ('VALID_SUCCESS','VALID_QUALITY_FAIL','ENGINEERING_FAIL'):raise ValueError('Unknown failure class')
            results.append(dict(job=job,attempt=attempt,**value))
            if status!='ENGINEERING_FAIL':break  # Never retry a valid quality FAIL.
    return results

def command(release,execute,cards,executor=None):
    # Permission guard precedes any executor invocation, native import or construction.
    if execute:raise PermissionError('NO_COLLECTION_GO: preparation release '+release+' grants zero native calls')
    if release!=RELEASE:raise PermissionError('Unknown preparation authority')
    plan=jobs(cards)
    return dict(status='PLAN_ONLY_DEFAULT_DRY_RUN',authority=release,episodes=plan,prospective_episodes=48,
        prospective_control_calls=30720,prospective_physics_ticks=153600,setup_negative_and_retry_budgets='NOT_AUTHORIZED_NOT_YET_FIXED',
        launch_allowed=False,new_simulator_calls=0,new_teacher_calls=0,new_render_calls=0,new_model_calls=0,new_WBC_calls=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--release',default=RELEASE);p.add_argument('--cards',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:
        # Even input-file reading is deferred until after execution rejection.
        try:command(a.release,True,[])
        except PermissionError as error:
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='REJECTED_BEFORE_NATIVE_IMPORT',error=str(error),native_calls=0,model_calls=0),indent=2)+'\n');raise SystemExit(2)
    value=command(a.release,False,[json.loads(x) for x in a.cards.read_text().splitlines()]);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items() if k!='episodes'}))

if __name__=='__main__':main()

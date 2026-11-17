"""New V6 target screen over the original serial native lifecycle."""
import json
from pathlib import Path
from scripts.wm02.sim_only_campaign_v51 import run as original,carried_rows,interface_gate
from go2arx5_vla.world_model.target_scope_v6 import PLAN,SCOPE,validate,read


def preflight(spec):
    count=spec['new_physical_requests_budget']
    if(spec['plan_commit']!=PLAN or spec['scope']!=SCOPE or spec['new_optimizer_updates_budget']!=0
       or spec['evaluation_stage']not in('SCREEN','CONFIRM')or count!={'SCREEN':24,'CONFIRM':8}[spec['evaluation_stage']]
       or len(spec['cases'])!=count):raise PermissionError('FIXED_V6_TARGET_SCREEN_OR_UNIQUE_CONFIRMATION_ONLY')
    panel=read(spec['registered_panel'])
    if len(panel)!=count or len({r['id']for r in panel})!=count:raise ValueError('EXACT_NEW_V6_PANEL_REQUIRED')
    cases=[]
    for path,row in zip(spec['cases'],panel,strict=True):
        case=json.loads(Path(path).read_bytes());validate(case);template=read(row['template_spec'])
        if(case['panel_entry_id']!=row['id']or case['python']!=template['python']or case['mobile_world_lock']!=spec['mobile_world_lock']
           or case['ROS_DOMAIN_ID']!=spec['ROS_DOMAIN_ID']or case['mobile_group']!=spec['registered_root']
           or case['source_sha']!=spec['source_sha']or case['registered_panel']!=spec['registered_panel']):
            raise ValueError('EXACT_V6_WORLD_PROFILE_PANEL_AND_FROZEN_NATIVE_INTERPRETER_REQUIRED')
        cases.append(case)
    if spec['evaluation_stage']=='SCREEN':
        for family in('F1','F2'):
            for layout in('T0','T1','T2','T3'):
                matched=[r for r in panel if r['family']==family and r['layout']==layout]
                if {r['arm']for r in matched}!={'T-B1','T-FM','T-H40'}or len({(r['reset'],r['noise_seed'],r['scene']['sha256'])for r in matched})!=1:
                    raise ValueError('MATCHED_RESET_NOISE_SCENE_ALL_THREE_V6_ARMS_REQUIRED')
    else:
        decision=read(spec['screen_decision'])
        if not decision['qualified']or decision['selected_arm']!=spec['frozen_arm']or any(r['arm']!=spec['frozen_arm']for r in panel):
            raise PermissionError('ONE_ACTUALLY_QUALIFIED_FROZEN_TARGET_CANDIDATE_REQUIRED')
    prior=spec['prior_case_results']
    if len(prior)!=2 or not interface_gate(carried_rows(prior)):raise ValueError('REUSE_PASSED_FROZEN_NATIVE_INTERFACES_NO_REPLAY')
    return prior


def run(path):
    spec=json.loads(Path(path).read_bytes());count=spec['new_physical_requests_budget']
    return original(path,preflight=preflight,bootstrap='v6_bootstrap.py',expected_count=count+2,
        result_schema='SLA_V6_TARGET_'+spec['evaluation_stage']+'_ACTUAL_TERMINAL',pilot_evaluate=lambda rows:False)

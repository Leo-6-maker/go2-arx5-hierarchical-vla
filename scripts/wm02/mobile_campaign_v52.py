"""V5.2 preregistered E2 serial panel; no training or scientific replay."""
from pathlib import Path
from scripts.wm02.sim_only_campaign_v51 import run as original,carried_rows,interface_gate
from go2arx5_vla.world_model.mobile_scope_v52 import read,validate
from go2arx5_vla.sla_adapter.mobile_successor_v52 import PLAN


def preflight(spec):
    if (spec['plan_commit']!=PLAN or spec['scope']!='SLA_MOBILE_F2_V52'
            or spec['new_physical_requests_budget']!=24 or spec['new_optimizer_updates_budget']!=0
            or len(spec['cases'])!=24):raise PermissionError('FIXED_V52_E2_24_MATCHED_REQUESTS_ONLY')
    panel=read(spec['registered_panel'])
    if len(panel)!=24 or len({r['id']for r in panel})!=24:raise ValueError('EXACT_E2_PREREGISTRATION')
    panel_by_id={r['id']:r for r in panel}
    cases=[]
    for path in spec['cases']:
        import json
        case=json.loads(Path(path).read_bytes());validate(case)
        template=read(panel_by_id[case['panel_entry_id']]['template_spec'])
        if case['python']!=template['python']:raise ValueError('FROZEN_NATIVE_INTERPRETER_PROFILE_REQUIRED')
        if (case['mobile_world_lock']!=spec['mobile_world_lock']or case['ROS_DOMAIN_ID']!=spec['ROS_DOMAIN_ID']
                or case['mobile_group']!=spec['registered_root']or case['source_sha']!=spec['source_sha']
                or case['registered_panel']!=spec['registered_panel']):raise ValueError('EXACT_E2_PARENT_WORLD_OWNER_SOURCE_AND_PANEL')
        cases.append(case)
    if [c['panel_entry_id']for c in cases]!=[r['id']for r in panel]:raise ValueError('COUNTERBALANCED_FIXED_E2_ORDER')
    for family in('F1','F2'):
        for layout in('T0','T1','T2','T3'):
            matched=[r for r in panel if r['family']==family and r['layout']==layout]
            if {r['arm']for r in matched}!={'B0','B1','B2'}or len({(r['reset'],r['noise_seed'],r['scene']['sha256'])for r in matched})!=1:
                raise ValueError('SAME_RESET_SCENE_NOISE_ACROSS_ALL_THREE_ARMS')
    prior=spec['prior_case_results']
    if len(prior)!=2 or not interface_gate(carried_rows(prior)):raise ValueError('TWO_REAL_FROZEN_NATIVE_INTERFACES_REQUIRED')
    return prior


def run(path):
    return original(path,preflight=preflight,bootstrap='mobile_v52_bootstrap.py',expected_count=26,
        result_schema='SLA_MOBILE_V52_E2_ACTUAL_TERMINAL',pilot_evaluate=lambda rows:False)

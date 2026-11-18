from types import SimpleNamespace

import numpy as np
import pytest

from scripts.wm02.multitask_student_campaign_v6 import expected_conditions, summarize
from go2arx5_vla.sla_adapter.multitask_contract_v6 import ARMS, NATIVE_METHOD, NATIVE_VERSION
from go2arx5_vla.world_model.close_reference_input_v1 import invalidate, query_reference, remember
from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS


def test_g7_command_history_uses_only_one_confirmed_native_atom():
    spec = dict(mobile_experiment_version=NATIVE_VERSION,
                mobile_condition=dict(method=NATIVE_METHOD,mode='M_FREE'))
    adapter = SimpleNamespace(spec=spec)
    command = np.zeros(10)
    command[9] = .5
    source, step = 5678*.002, 5703
    packet = dict(step=step,time_s=step*.002)
    remember(adapter,command,dict(source_native_s=source),packet)
    context = query_reference(adapter,packet,dict(native_step=step))
    assert context == dict(width_m=.022,source_native_s=source,query_native_s=step*.002)
    with pytest.raises(ValueError,match='ADVANCE_ONE_ATOM'):
        remember(adapter,command,dict(source_native_s=(step+25)*.002),
                 dict(step=step+50,time_s=(step+50)*.002))
    invalidate(adapter)
    with pytest.raises(ValueError,match='MISSING_OR_INVALIDATED'):
        query_reference(adapter,packet,dict(native_step=step))


def screen_population():
    rows = []
    for family in ('F1','F2'):
        for reset in range(2):
            for task in INSTRUCTIONS:
                cluster = f'{family}-{reset}'
                order = ARMS if len(rows)%2==0 else ARMS[::-1]
                rows.append(dict(id=f'{cluster}-{task}',instruction=task,family=family,cluster=cluster,arm_order=order))
    return dict(conditions=dict(SCREEN=rows))


def test_promotion_requires_full_screen_and_generation_reconciliation():
    population = screen_population()
    planned = expected_conditions(population,'SCREEN',set(ARMS))
    assert [arm for _,arm in planned[:4]] == ['M-BASE','M-SRC','M-SRC','M-BASE']
    cases = [dict(instruction=row['instruction'],family=row['family'],cluster=row['cluster'],arm=arm,
        role='STUDENT',completed=True,technical_failure=False,hard_STOP=False,task_and_mobile_success=True,
        valid_policy_rejection=None,task_ticks=1000,native_ticks=6703,actual_model_generations=4,
        initial_native_state=dict(qpos=[0.],qvel=[0.],ctrl=[0.])) for row,arm in planned]
    spec = dict(evaluation_stage='SCREEN',compiled_population=dict(sha256='a'*64),
                new_physical_requests_budget=48,packages={arm:{'sha256':arm} for arm in ARMS})
    end = {arm:dict(TRAIN96=dict(predictions=96,original_H40_and_G7_seam_pass=96)) for arm in ARMS}
    worker = dict(exit_code=0,attempted_generations=192,completed_generations=192)
    result = dict(records=cases,exit_code=0,unresolved_case_launches=0,model_service=worker)
    selected = summarize(result,spec,end)
    assert selected['passed'] and selected['selected_arm']=='M-BASE' and len(selected['paired_conditions'])==24
    assert selected['new_native_ticks']==48*6703 and selected['attributed_case_generations']==192
    assert not summarize(dict(result,records=cases[:-1]),spec,end)['passed']
    assert not summarize(dict(result,model_service=dict(worker,attempted_generations=193)),spec,end)['passed']
    end['M-BASE']['TRAIN96']['original_H40_and_G7_seam_pass'] = 95
    assert summarize(result,spec,end)['selected_arm']=='M-SRC'

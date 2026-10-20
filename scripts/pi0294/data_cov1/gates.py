"""Count immutable task outcomes at the original cumulative DATA-COV1 gates."""
from collections import Counter

from .plan import capture_cards, protocol_card

TERMINAL = frozenset(('STATIC_REJECTED','NOT_RUN','VALID_TASK_FAIL','REFERENCE_REJECT',
    'S1_PREDICTED_REJECT','RAW_AUDIT_REJECT','CODEC_REJECT','ADMITTED',
    'ENGINEERING_INCOMPLETE','NATIVE_SAFETY_STOP'))


def _join(cards,rows):
    planned={c['id']:c for c in cards};actual={r['id']:r for r in rows}
    if len(planned)!=len(cards) or len(actual)!=len(rows) or not set(actual)<=set(planned):
        raise PermissionError('DATA_COV1_UNIQUE_PLANNED_FIRST_ATTEMPT_NO_REPLACEMENT')
    for identifier,r in actual.items():
        card=planned[identifier]
        if (r['scene_group']!=card['group_id'] or r['task']!=card['task'] or r['status'] not in TERMINAL
                or type(r['task_success']) is not bool or type(r['admitted']) is not bool
                or r['admitted']!=(r['status']=='ADMITTED')
                or r['admitted'] and (not r['task_success'] or not r['raw_admitted'] or not r['codec_admitted'])
                or r['native_faults']<0 or type(r['native_faults']) is not int):
            raise PermissionError('DATA_COV1_EXACT_CARD_AND_FULL_ADMISSION_NOT_STATUS_RELABEL')
        if r['native_faults'] and (r['status']!='NATIVE_SAFETY_STOP' or r['admitted']):
            raise PermissionError('DATA_COV1_NATIVE_FAULT_CANNOT_BE_TRAINING_SUCCESS')
    return actual


def first_four(groups,rows):
    cards=capture_cards(groups,'P24')[:4];actual=_join(cards,rows)
    checks=dict(all_four_terminal=len(actual)==4,
        all_four_identity_and_clock_audited=all(r.get('auditable_identity_and_clock') is True for r in actual.values()),
        known_effects=all(not r['unknown_effects'] for r in actual.values()),
        native_safety=all(r['native_faults']==0 for r in actual.values()),
        at_least_three_full_admissions=sum(r['admitted'] for r in actual.values())>=3,
        each_task_has_full_admission=all(any(r['task']==t and r['admitted'] for r in actual.values()) for t in ('T1','T2')))
    return dict(schema='PI0294_DATA_COV1_FIRST_FOUR_GATE_V1',checks=checks,
        count_gate_pass=all(checks.values()),attempt_ids=[c['id'] for c in cards],
        statuses=dict(Counter(r['status'] for r in actual.values())),
        missing=[c['id'] for c in cards if c['id'] not in actual],
        native_identity_and_fresh_resource_admission_still_required=True,
        effective_failures_must_not_be_retried=True)


def stage_gate(groups,stage,rows):
    config=next((r for r in protocol_card()['stages'] if r['id']==stage),None)
    if config is None:raise PermissionError('DATA_COV1_FIXED_STAGE_REQUIRED')
    cards=capture_cards(groups,stage);actual=_join(cards,rows)
    by_task={};by_stratum={};support={t:set() for t in ('T1','T2')}
    for task in ('T1','T2'):
        own=[r for r in actual.values() if r['task']==task]
        support[task]={r['scene_group'] for r in own if r['admitted']}
        by_task[task]=dict(planned=config['new_groups_cumulative'],reported=len(own),
            task_success=sum(r['task_success'] for r in own),admitted=len(support[task]),
            task_success_minimum=config['task_success_minimum_each'],
            admitted_minimum=config['fully_admitted_minimum_each'])
    for ap in ('AP0','AP1','AP2'):
        for direction in ('+Y','-Y'):
            for distance in ('NEAR','FAR'):
                key='/'.join((ap,direction,distance))
                expected=[c for c in cards if (c['appearance_profile'],c['direction'],c['distance_bin'])==(ap,direction,distance)]
                by_stratum[key]={t:dict(planned=config['groups_per_stratum'],
                    admitted=sum(c['id'] in actual and actual[c['id']]['admitted'] for c in expected if c['task']==t),
                    minimum=config['fully_admitted_each_stratum_task_minimum']) for t in ('T1','T2')}
    priority={ap:{t:dict(admitted=sum(r[t]['admitted'] for key,r in by_stratum.items() if key.startswith(ap+'/-Y/')),
        minimum=config['AP0_negative_Y_and_AP2_negative_Y_each_task_minimum']) for t in ('T1','T2')} for ap in ('AP0','AP2')}
    checks=dict(all_planned_cells_terminal=len(actual)==len(cards),
        native_safety=all(r['native_faults']==0 for r in actual.values()),
        known_effects=all(not r['unknown_effects'] for r in actual.values()),
        task_success_each=all(r['task_success']>=r['task_success_minimum'] for r in by_task.values()),
        full_admission_each=all(r['admitted']>=r['admitted_minimum'] for r in by_task.values()),
        paired_support=len(support['T1']&support['T2'])>=config['paired_admitted_groups_minimum'],
        each_stratum_task=all(r['admitted']>=r['minimum'] for own in by_stratum.values() for r in own.values()),
        priority_reverse_appearance=all(r['admitted']>=r['minimum'] for own in priority.values() for r in own.values()))
    return dict(schema='PI0294_DATA_COV1_CUMULATIVE_COUNT_GATE_V1',stage=stage,
        count_gate_pass=all(checks.values()),checks=checks,tasks=by_task,strata=by_stratum,
        priority_reverse_appearance=priority,paired_groups=len(support['T1']&support['T2']),
        paired_minimum=config['paired_admitted_groups_minimum'],
        missing=[c['id'] for c in cards if c['id'] not in actual],
        statuses=dict(Counter(r['status'] for r in actual.values())),
        native_identity_and_fresh_resource_admission_still_required=True,
        metric='TEACHER_DATA_QUALIFICATION_NOT_POLICY_SUCCESS',PV1_bank_changed=False)


def next_stage(stage,gate,approval_level,*,fresh_resources_pass,pv1_priority_clear):
    if gate.get('stage')!=stage or not gate.get('count_gate_pass'):
        return dict(next_stage=None,reason='STAGE_GATE_NOT_MET')
    if approval_level not in ('PILOT_ONLY','FULL_CONDITIONAL_CAMPAIGN'):
        return dict(next_stage=None,reason='MISSING_DATA_GO')
    if approval_level=='PILOT_ONLY' or stage=='P144':
        return dict(next_stage=None,reason='APPROVED_SCOPE_TERMINAL')
    if not pv1_priority_clear:return dict(next_stage=None,reason='PV1_PRIORITY')
    if not fresh_resources_pass:return dict(next_stage=None,reason='RESOURCE_OR_DEADLINE')
    return dict(next_stage={'P24':'P72','P72':'P144'}[stage],
        reason='COUNTS_PASS_NATIVE_ADMISSION_MUST_BIND_NEXT_STAGE')

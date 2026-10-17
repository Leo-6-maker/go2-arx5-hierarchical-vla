"""Original-budget replay of four already adjudicated S-BC512 failures."""
SCHEMA='MP1_SBC512_FAILURE_HISTORY_DIAGNOSTIC_V1'
SCOPE='MP1_SBC512_FAILURE_HISTORY_DIAGNOSTIC'
WALL_S=3600
CAPS=dict(main_starts=4,recovery_forks=0,live_and_recovery_integrations=0,
    shadow_integrations=28400,generation_queries=0,optimizer_updates=0)
CPU_CAPS=dict(core=3600,hopper=0)
ATTEMPTS={
    'PI0294-MP1-S-BC-512-TRAIN4-F0-000-T1':dict(last_sequence=4,main_steps=800,offset=64,
        result_sha256='9c5e4b23bd84493e85aa9dc055c8b553bb790cdde66890e538e510caef830431'),
    'PI0294-MP1-S-BC-512-TRAIN4-F0-000-T2':dict(last_sequence=52,main_steps=10400,offset=124,
        result_sha256='ef6400cccd05cb0392347acc92ee6e36673ea2b25241c768b4a232d798477441'),
    'PI0294-MP1-S-BC-512-TRAIN4-F1-000-T1':dict(last_sequence=26,main_steps=5200,offset=63,
        result_sha256='606f4693598a0bae2f992cd99731af593d7cc7e81d5dc3f24c7089d58fdbb4bc'),
    'PI0294-MP1-S-BC-512-TRAIN4-F1-000-T2':dict(last_sequence=56,main_steps=11200,offset=150,
        result_sha256='fa72634958f26ad70d2db8bf3a55dde035996dfd449f44df180a05b2dc249887'),
}


def is_diagnostic(stage):return stage.get('schema')==SCHEMA


def selected_sequences(identifier):
    value=ATTEMPTS[identifier]['last_sequence']
    return [value-1,value]


def validate_plan(plan):
    cases=plan['cases'];ids=[c['attempt_id'] for c in cases]
    if (len(cases)!=4 or len(set(ids))!=4 or set(ids)!=set(ATTEMPTS)
            or plan['new_queries']!=0 or plan['new_live_episodes']!=0
            or plan['selection']!='LAST_REJECTED_PREFIX_INPUT_AND_PREVIOUS_ACCEPTED_PREFIX_INPUT'):
        raise PermissionError('FOUR_FIXED_EXISTING_FAILURES_EIGHT_ANCHORS_ONLY')
    for case in cases:
        expected=ATTEMPTS[case['attempt_id']]
        if (case['sequences']!=selected_sequences(case['attempt_id'])
                or case['original_result_sha256']!=expected['result_sha256']
                or case['main_steps']!=expected['main_steps']
                or case['original_rejection_offset_ticks']!=expected['offset']):
            raise PermissionError('FIXED_ACTUAL_REJECTION_AND_PREVIOUS_ACK_ANCHORS')
    return cases

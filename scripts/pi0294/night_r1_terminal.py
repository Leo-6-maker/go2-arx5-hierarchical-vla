"""Account for a written gripper command whose final observation was safety-stopped."""


def gripper_tail(records, observations, native_end_step, start_step, safety_stopped):
    expected=[i for i in range(observations-1) if i%8]
    indices=[r['index'] for r in records]
    if indices==expected:return []
    last=observations-1
    if (not safety_stopped or last%8==0 or indices!=expected+[last]
            or not start_step+25*last<=native_end_step<=start_step+25*(last+1)):
        raise ValueError('UNEXPLAINED_GRIPPER_COMMAND_OR_OBSERVATION_GAP')
    return [dict(command_index=last,integrations_after_write=native_end_step-start_step-25*last,
        next_RGB_and_20Hz_observation_saved=False,full_episode_complete=False,
        raw_command_preserved=True,basis='REGISTERED_SAFETY_TERMINATION_AND_NATIVE_JOURNAL')]


def check():
    good=[{'index':i} for i in range(130) if i%8]
    assert gripper_tail(good,131,5750,2500,False)==[]
    actual=good+[{'index':130}]
    assert gripper_tail(actual,131,5775,2500,True)[0]['integrations_after_write']==25
    for args in [(actual,131,5775,2500,False),(actual,131,5776,2500,True),
                 (good[:-1],131,5775,2500,True),(actual+[{'index':131}],131,5775,2500,True)]:
        try:gripper_tail(*args)
        except ValueError:pass
        else:raise AssertionError('TERMINAL_NEGATIVE_NOT_REJECTED')
    return {'status':'PASS_SAFETY_TERMINAL_COMMAND_ACCOUNTING_CPU','new_physics':0}


if __name__=='__main__':print(check())

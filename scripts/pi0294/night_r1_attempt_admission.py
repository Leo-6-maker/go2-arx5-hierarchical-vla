"""Apply the direct user debug rule without resetting exposure or resource totals."""


def counts(attempts):
    rows=[x for x in attempts if x.get('role')=='DEVELOPMENT']
    engineering=[];scientific=[]
    for row in rows:
        r=row.get('result',{})
        # Only unambiguous pre-task engineering failures are exempt. Unknown
        # records or any observed task exposure remain conservatively counted.
        before_task=(r.get('status')=='ENGINEERING_OR_NATIVE_SAFETY_INCOMPLETE'
            and r.get('observations')==0 and not r.get('native_stops'))
        (engineering if before_task else scientific).append(row)
    return dict(raw=len(rows),scientific=len(scientific),pre_task_engineering=len(engineering))


def admit(attempts,debug,adoption):
    value=counts(attempts)
    approved=(debug.get('authorizes_evidence_bound_engineering_continuation') is True
        and debug.get('arbitrary_engineering_retry_and_startup_requalification_count_cap_removed') is True
        and debug.get('cumulative_resource_and_deadlines_unchanged') is True
        and debug.get('fixed_R1_release')==adoption['adopted_release']
        and debug.get('original_T0_epoch')==adoption['T0_epoch'])
    if (value['scientific'] if approved else value['raw'])>=6:
        raise PermissionError('R1_SCIENTIFIC_DEVELOPMENT_START_CAP' if approved else 'R1_DEVELOPMENT_START_CAP')
    return dict(value,direct_user_debug_rule_applied=approved,scientific_cap=6,total_resource_caps_unchanged=True)


def check():
    a={'adopted_release':'FIXTURE','T0_epoch':1}
    d=dict(authorizes_evidence_bound_engineering_continuation=True,
        arbitrary_engineering_retry_and_startup_requalification_count_cap_removed=True,
        cumulative_resource_and_deadlines_unchanged=True,fixed_R1_release='FIXTURE',original_T0_epoch=1)
    engineering=dict(role='DEVELOPMENT',result=dict(status='ENGINEERING_OR_NATIVE_SAFETY_INCOMPLETE',observations=0,native_stops=[]))
    valid=dict(role='DEVELOPMENT',result=dict(status='VALID_PLANNING_FAILURE_NO_RETRY',observations=121,native_stops=[]))
    rows=[valid]*4+[engineering]*2
    assert admit(rows,d,a)['scientific']==4
    assert admit(rows+[engineering]*100,d,a)['pre_task_engineering']==102
    for attempts,debug in [(rows,{}),([valid]*6,d),(rows,dict(d,original_T0_epoch=2))]:
        try:admit(attempts,debug,a)
        except PermissionError:pass
        else:raise AssertionError('DEVELOPMENT_NEGATIVE_NOT_REJECTED')
    assert counts([dict(role='DEVELOPMENT',result={})])['scientific']==1
    return dict(status='PASS_BOUND_DEBUG_AND_CONSERVATIVE_SCIENTIFIC_COUNTER_CPU',scientific_cap=6,new_physics=0)


if __name__=='__main__':print(check())

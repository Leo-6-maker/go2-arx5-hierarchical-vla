"""Preregistered E2 ranking and promotion, independent of model inference."""
import statistics

OFFLINE_GATES=('TRAIN96_all','CN38_180_all','saved_reject3_all','zero_CLOSE_omissions','CLOSE_within100ms_no_regression')
CONDITIONS={(f,'T%d'%n)for f in('F1','F2')for n in range(4)}


def summarize(rows,offline,*,order=('B0','B1','B2')):
    summaries={};ranks={};table=[]
    for arm in order:
        items=[r for r in rows if r.get('arm')==arm]
        keys=[(r['family'],r['condition']['layout'])for r in items]
        complete=len(items)==8 and len(set(keys))==8 and set(keys)==CONDITIONS
        identity_complete=complete and all(r['actual_native_requests']==1 and r['technical_failure']is False
            and type(r['hard_STOP'])is bool and type(r['completed'])is bool for r in items)
        successes=[r for r in items if r['task_and_mobile_success']]
        family={f:sum(r['family']==f for r in successes)for f in('F1','F2')}
        rejects=sum(bool(r.get('valid_policy_rejection'))for r in items)
        median=statistics.median(r['task_ticks']*.002 for r in successes)if successes else None
        checks=dict(all8=complete,all8_native_completed=complete and all(r['completed']for r in items),zero_technical=all(not r['technical_failure']for r in items),zero_hard_STOP=all(not r['hard_STOP']for r in items),success6=len(successes)>=6,each_family3=min(family.values())>=3,
            **{k:offline[arm][k]is True for k in OFFLINE_GATES})
        qualified=identity_complete and all(checks.values())
        summaries[arm]=dict(attempts=len(items),identity_complete=identity_complete,composite_successes=len(successes),by_family=family,H40_rejected_episodes=rejects,hard_STOP=sum(r['hard_STOP']for r in items),technical_failures=sum(r['technical_failure']for r in items),successful_task_time_median_s=median,promotion_checks=checks,qualified=qualified)
        if identity_complete:ranks[arm]=(-len(successes),-min(family.values()),rejects,float('inf')if median is None else median,order.index(arm))
    eligible=[a for a in ranks if summaries[a]['qualified']]
    winner=min(eligible,key=lambda a:ranks[a])if eligible else None
    development_parent=min(ranks,key=lambda a:ranks[a])if ranks else None
    for family,layout in sorted(CONDITIONS):
        line=dict(family=family,layout=layout,arms={})
        for arm in order:
            matching=[r for r in rows if r.get('arm')==arm and r['family']==family and r['condition']['layout']==layout]
            if len(matching)!=1:line['arms'][arm]=dict(status='MISSING_OR_DUPLICATE',success=None);continue
            r=matching[0]
            status='TECHNICAL_FAILURE'if r['technical_failure']else'HARD_STOP'if r['hard_STOP']else'COMPOSITE_SUCCESS'if r['task_and_mobile_success']else'H40_REJECT'if r.get('valid_policy_rejection')else'VALID_TASK_FAILURE'
            line['arms'][arm]=dict(status=status,success=bool(r['task_and_mobile_success']),actual_native_requests=r['actual_native_requests'],native_ticks=r['native_ticks'])
        if set(order)=={'B0','B1','B2'}and all(line['arms'][a]['success']is not None for a in order):
            line['B2_minus_B1']=int(line['arms']['B2']['success'])-int(line['arms']['B1']['success'])
            line['B1_minus_B0']=int(line['arms']['B1']['success'])-int(line['arms']['B0']['success'])
        table.append(line)
    return dict(schema='SLA_MOBILE_V52_FIXED_SELECTION',by_arm=summaries,paired_conditions=table,unique_confirmation_candidate=winner,development_parent=development_parent,development_parent_is_deployment_permission=False,old_V51_pilot_GO=False,TEST_permission=False)

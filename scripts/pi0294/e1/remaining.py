"""Select only unstarted or settled engineering-incomplete fixed conditions."""

BASE_LABELS={'M1-1024-CANONICAL','M2-BC','M2-CORR'}
CONDITIONS=tuple((f'PI0294/TRAIN/{family}/000',task) for family in ['F0','F1'] for task in ['T1','T2'])


def panel(rows,base_label,*,source_stopped=False):
    if base_label not in BASE_LABELS:raise PermissionError('E1_NO_256_OR_NEW_PANEL_IN_REMAINDER')
    aliases={base_label,base_label+'-REMAINDER-03'}
    own=[r for r in rows if r['model'] in aliases and not r.get('excluded_scope_deviation',False)]
    if any((r['scene_group'],r['task']) not in CONDITIONS for r in own):raise PermissionError('E1_REMAINDER_ONLY_ORIGINAL_TRAIN4')
    if source_stopped or any(r['live_native_failures'] or r['status']=='ACTUAL_SAFETY_FAIL' or r['failure_stage']=='S1_INTEGRITY' for r in own):
        return dict(base_label=base_label,status='NOT_RUN_SOURCE_PHYSICS_STOP',complete=[],remaining=[],new_engineering_recoveries=0)
    complete=[];remaining=[]
    for group,task in CONDITIONS:
        previous=[r for r in own if r['scene_group']==group and r['task']==task]
        valid=[r for r in previous if r['pipeline_trustworthy'] and r['status'] in ['SUCCESS','VALID_TASK_FAIL']]
        if len(valid)>1:raise PermissionError('E1_REPEATED_VALID_POLICY_OUTCOME_PRESERVE_SCOPE_DEVIATION')
        if valid:complete.append(dict(scene_group=group,task=task,actual_attempt_id=valid[0]['attempt_id'],status=valid[0]['status']));continue
        if any(r['status']!='ENGINEERING_INCOMPLETE' or r['pipeline_trustworthy'] for r in previous):raise PermissionError('E1_UNADJUDICATED_OUTCOME_NOT_ELIGIBLE_FOR_RETRY')
        remaining.append(dict(scene_group=group,task=task,reason='SETTLED_ENGINEERING_INCOMPLETE_NEW_ATTEMPT' if previous else 'NEVER_STARTED_FIRST_ATTEMPT',
            previous_attempt_ids=[r['attempt_id'] for r in previous]))
    return dict(base_label=base_label,status='ALL_PREDECLARED_CONDITIONS_ALREADY_VALID_NO_REPLAY' if not remaining else 'FIXED_REMAINDER_ONLY',
        complete=complete,remaining=remaining,new_engineering_recoveries=sum(bool(r['previous_attempt_ids']) for r in remaining))

"""Fixed source development coverage, never a new or hidden test split."""


def prior_recovery_rows(root):
    from pathlib import Path
    from scripts.pi0294.e1.control_read import read
    stage=read(Path(root)/'MP1_STAGE_BINDING.json')
    if 'prior_recovery_usage' not in stage:return []
    rows=read(Path(root)/'plans/PRIOR_RECOVERY_OUTCOMES.json')
    if any(r['status']!='ENGINEERING_INCOMPLETE' or r['pipeline_trustworthy'] or r['live_native_failures'] for r in rows):
        raise PermissionError('ONLY_TERMINAL_ENGINEERING_PRIORS_NO_VALID_RESULT_REPLAY')
    if len(rows)!=stage['prior_recovery_usage']['main_starts']:
        raise PermissionError('EVERY_PRIOR_RECOVERY_START_ACCOUNTED')
    return rows


def evaluation_plan(cards,parent_manifest_sha256):
    if len(parent_manifest_sha256)!=64 or any(c not in '0123456789abcdef' for c in parent_manifest_sha256):
        raise ValueError('EXACT_AUDITED_PARENT_MANIFEST_SHA256')
    if len(cards)!=12 or len({c['id'] for c in cards})!=12:
        raise ValueError('ORIGINAL_E0_TRAIN4_AND_EXPOSED_DEV8_ONLY')
    train=[c for c in cards if c['split']=='TRAIN'];dev=[c for c in cards if c['split']=='DEV']
    if (len(train)!=4 or len(dev)!=8 or {c['task'] for c in cards}!={'T1','T2'}
            or any(sum(c['task']==task for c in group)!=len(group)//2 for group in [train,dev] for task in ['T1','T2'])):
        raise ValueError('EXACT_BALANCED_FIXED_DEVELOPMENT_MEMBERSHIP')
    panels={}
    for branch in ['S-BC','S-CORR']:
        for step,split,group in [(512,'TRAIN',train),(2048,'TRAIN',train),(2048,'DEV',dev)]:
            label=branch+'-'+str(step)+'-'+split+str(len(group));rows=[]
            for original in sorted(group,key=lambda c:(c['scene_group'],c['task'])):
                card={k:v for k,v in original.items() if not k.startswith('relay_request_') and not k.startswith('e1_')}
                card.update(id='PI0294-MP1-'+label+'-'+card['scene_group'].split('/')[-2]+'-'+card['scene_group'].split('/')[-1]+'-'+card['task'],mp1_model_label=label)
                rows.append(card)
            panels[label]=dict(branch=branch,checkpoint_step=step,split=split,cards=rows,attempt_ids=[c['id'] for c in rows],
                teacher_takeover=False,original_S1=True,complete_task_duration_s=32,exposed_development=split=='DEV')
    return dict(schema='MP1_FIRST_SOURCE_PAIR_FIXED_DEVELOPMENT_V1',parent_checkpoint_manifest_sha256=parent_manifest_sha256,
        panels=panels,regular_starts=32,engineering_replacement_per_condition_max=1,valid_policy_outcomes_repeated=False,
        original_action_seed_namespace='PI0294',new_DEV_labels=False,independent_confirmation_included=False,
        candidate_checkpoints=dict(S_BC=2048,S_CORR=2048),checkpoint_selection='TERMINAL_2048_ONLY_NOT_BEST_SNAPSHOT')


def remaining_panel(rows,spec):
    """Keep every row; only an engineering interruption permits one replacement."""
    remaining=[]
    for card in spec['cards']:
        matches=[r for r in rows if r['scene_group']==card['scene_group'] and r['task']==card['task'] and r['model']==card['mp1_model_label']]
        if len(matches)>2:raise PermissionError('AT_MOST_ONE_ENGINEERING_REPLACEMENT')
        by_id={r['attempt_id']:r for r in matches}
        if len(by_id)!=len(matches) or any(i not in [card['id'],card['id']+'-REPLACEMENT-01'] for i in by_id):
            raise PermissionError('EXACT_ORIGINAL_OR_ONE_DECLARED_REPLACEMENT_ID')
        if any(r.get('live_native_failures',0) or r['status']=='ACTUAL_SAFETY_FAIL' or r['failure_stage']=='S1_INTEGRITY' for r in matches):
            raise PermissionError('SOURCE_DOMAIN_SAFETY_OR_INTEGRITY_STOP')
        valid=[r for r in matches if r['pipeline_trustworthy'] and r['status'] in ['SUCCESS','VALID_TASK_FAIL']]
        original=by_id.get(card['id'])
        if len(valid)>1 or (card['id']+'-REPLACEMENT-01' in by_id and (original is None or original['status']!='ENGINEERING_INCOMPLETE' or original['pipeline_trustworthy'])):
            raise PermissionError('VALID_POLICY_RESULT_MUST_NOT_BE_REPEATED')
        if valid or len(matches)==2:continue
        if matches and (matches[0]['status']!='ENGINEERING_INCOMPLETE' or matches[0]['pipeline_trustworthy']):
            raise PermissionError('ONLY_SETTLED_ENGINEERING_INTERRUPTION_CAN_BE_REPLACED')
        new=dict(card)
        if matches:new.update(id=card['id']+'-REPLACEMENT-01',mp1_replacement_of=matches[0]['attempt_id'])
        remaining.append(new)
    return remaining


def session_spec(plan,label,*,root=None):
    from .n1_limits import allocation_number,resolve
    n1=resolve(root) if root is not None else None
    maximum=5 if n1 is None else n1['confirmation']['PBS'] if 'confirmation' in n1 else n1['source']['PBS']
    base,_=allocation_number(label,maximum)
    if base not in plan['panels']:
        raise PermissionError('EXACT_FIXED_PANEL_ONLY')
    return plan['panels'][base]

"""One frozen source candidate and new paired scenes, never development replay."""
from copy import deepcopy

from scripts.pi0294.scene_cards import scene_group,task_card,normalise_layouts,near_duplicate_report,digest
from .endpoint_anchor_ni import CHECKPOINTS

SCHEMA='MP1_SOURCE_CONFIRM_V1'
CHILD='source-confirm-01'
SPLIT='SOURCE_CONFIRM_V1'
CANDIDATE='S-CORR-2048'
PARENT='7dba1c4c8abacdb57e2faa0e6d7563ed5dcef2e5473cb13c281eb0e57d5a5150'
BANKS=(29201,29202)
FAMILIES=(('F0',2),('F1',4),('F2',4))
MAX_CANDIDATES=40
ORIGINAL_GENERATOR_INDEX_OFFSET=100


def original_group(family,index,candidate):
    if family not in dict(FAMILIES) or not 0<=index<dict(FAMILIES)[family] or not 0<=candidate<1024:
        raise PermissionError('ONLY_PREDECLARED_ORIGINAL_GENERATOR_SOURCE_GROUPS')
    group=scene_group('CONFIRM_CORE',family,ORIGINAL_GENERATOR_INDEX_OFFSET+index,candidate_index=candidate)
    original_identity=group['group_id'];identity='PI0294/'+SPLIT+'/'+family+'/'+str(index).zfill(3)
    group.update(group_id=identity,scene_group=identity,split=SPLIT,layout_candidate_index=candidate,
        original_frozen_generator_identity=original_identity,
        original_frozen_generator_index=ORIGINAL_GENERATOR_INDEX_OFFSET+index)
    return group


def rank_candidate(matrix):
    """Original worst-task, macro, then lower-cost ranking; no new rate gate."""
    if matrix['actual_safety_stop_present']:raise PermissionError('NO_CONFIRMATION_AFTER_NATIVE_SAFETY_STOP')
    result=[]
    for index,model in enumerate(['S-BC-2048','S-CORR-2048']):
        rows=[r for r in matrix['cells'] if r['model']==model+'-DEV8']
        if (len(rows)!=8 or any(not r['valid_policy_outcome'] for r in rows)
                or matrix['checkpoint_manifest_identities'].get(model)!=CHECKPOINTS[model]):
            raise PermissionError('COMPLETE_AUDITED_FIXED_ENDPOINT_DEV_ONLY')
        rates=[]
        for task in ['T1','T2']:
            selected=[r for r in rows if r['task']==task]
            if len(selected)!=4:raise PermissionError('FOUR_PAIRED_DEV_LAYOUTS_PER_TASK')
            rates.append(sum(r['status']=='SUCCESS' for r in selected)/4)
        result.append(dict(model=model,worst_task=min(rates),macro=sum(rates)/2,
            lower_complexity_tiebreak=index,task_rates=dict(zip(['T1','T2'],rates))))
    ordered=sorted(result,key=lambda r:(-r['worst_task'],-r['macro'],r['lower_complexity_tiebreak']))
    if ordered[0]['model']!=CANDIDATE:raise PermissionError('ONLY_CURRENT_EVIDENCE_CHOSEN_SINGLE_CORR2048_CANDIDATE')
    return dict(candidate=CANDIDATE,checkpoint_manifest_sha256=CHECKPOINTS[CANDIDATE],parent_checkpoint_manifest_sha256=PARENT,
        ranking=ordered,criterion='WORST_TASK_THEN_MACRO_THEN_LOWER_COMPLEXITY',new_DEV_success_threshold=None,
        selected_for_deployment=False,confirmation_opened=False)


def candidate_groups(exclusions):
    """No physics/model feedback: order is fixed before static qualification."""
    exclusions=normalise_layouts(exclusions);groups=[];rejected=[]
    for family,count in FAMILIES:
        for index in range(count):
            for candidate in range(1024):
                group=original_group(family,index,candidate)
                check=near_duplicate_report([group],[*exclusions,*groups])
                if check['conflicts']:
                    rejected.append(dict(group_id=group['group_id'],candidate_index=candidate,conflicts=check['conflicts']))
                    continue
                group['layout_candidate_index']=candidate;groups.append(group);break
            else:raise PermissionError('FINITE_PREDECLARED_GEOMETRIC_DEDUP_CANDIDATES_EXHAUSTED')
    if len(groups)!=10:raise PermissionError('TEN_NEW_SOURCE_GROUPS_ONLY')
    return dict(groups=groups,prefreeze_geometric_rejections=rejected,exclusion_population_sha256=digest(exclusions),
        qualification='NOT_STATIC_OR_FOV_QUALIFIED_NOT_OPEN_FOR_POLICY',new_physics=0,new_queries=0,
        distribution='SOURCE_F0_2_F1_4_F2_4_NO_F3_OR_TASK_PARAMETER_CHANGE')


def evaluation_plan(groups,candidate,static_receipts,exclusions):
    """Freeze only after original static/FOV rules, never Teacher success."""
    groups=deepcopy(groups);exclusions=normalise_layouts(exclusions)
    if (candidate['candidate']!=CANDIDATE or candidate['checkpoint_manifest_sha256']!=CHECKPOINTS[CANDIDATE]
            or candidate['parent_checkpoint_manifest_sha256']!=PARENT or candidate['confirmation_opened']
            or candidate['selected_for_deployment'] or len(groups)!=10
            or len({g['group_id'] for g in groups})!=10
            or {(g['family'],int(g['group_id'].rsplit('/',1)[1])) for g in groups}
                !={(family,n) for family,count in FAMILIES for n in range(count)}
            or any(g['split']!=SPLIT for g in groups)
            or near_duplicate_report(groups,exclusions)['conflicts']):
        raise PermissionError('ONE_FIXED_CANDIDATE_TEN_NEW_SOURCE_GROUPS_NO_LEAKAGE')
    wanted={(g['group_id'],t) for g in groups for t in ['T1','T2']}
    for group in groups:
        if group!=original_group(group['family'],int(group['group_id'].rsplit('/',1)[1]),group['layout_candidate_index']):
            raise PermissionError('FROZEN_ORIGINAL_GENERATOR_NO_TASK_GEOMETRY_ADJUSTMENT')
    indexed={(r['scene_group'],r['task']):r for r in static_receipts}
    if len(static_receipts)!=20 or set(indexed)!=wanted:raise PermissionError('ALL20_ORIGINAL_STATIC_TASK_QUALIFICATIONS_REQUIRED')
    for key,r in indexed.items():
        group=next(g for g in groups if g['group_id']==key[0])
        if (r['status']!='STATIC_PASS_PENDING_LIVE_CONTACT' or not r['geometry_passed']
                or not r['initial_safety_valid'] or not r['field_of_view']['passed']
                or r['integration_steps']!=0 or r['task_success_claim'] or r['object_pose_modified']
                or r['card_sha256']!=digest(task_card(group,key[1]))):
            raise PermissionError('ORIGINAL_STATIC_IK_COLLISION_FOV_NOT_TEACHER_SUCCESS_FILTER')
    bank_cards={}
    for bank in BANKS:
        cards=[]
        for group in groups:
            for task in ['T1','T2']:
                card=task_card(group,task)
                card.update(confirmation_noise_bank=bank,
                    confirmation_candidate=CANDIDATE,confirmation_condition_id=group['group_id']+'/'+task+'/N'+str(bank))
                cards.append(card)
        bank_cards[bank]=cards
    # Small fixed sessions retain the mature metadata-reader's packet/time bounds.
    panels={}
    for block in range(4):
        for bank in BANKS:
            label='SOURCE-CONFIRM-S-CORR-2048-N'+str(bank)+'-B'+str(block)
            cards=deepcopy(bank_cards[bank][block*5:(block+1)*5])
            for card in cards:
                card.update(id='PI0294-MP1-'+label+'-'+card['family']+'-'+card['scene_group'].rsplit('/',1)[1]+'-'+card['task'],mp1_model_label=label)
            panels[label]=dict(branch='S-CORR',checkpoint_step=2048,split=SPLIT,cards=cards,
                attempt_ids=[c['id'] for c in cards],teacher_takeover=False,original_S1=True,complete_task_duration_s=32,
                confirmation_noise_bank=bank,noise_namespace='PI0294/SOURCE_CONFIRM_V1/N'+str(bank),
                checkpoint_manifest_sha256=CHECKPOINTS[CANDIDATE],independent_confirmation=True,
                fixed_block=block,packet_directories_max_files=400)
    return dict(schema=SCHEMA,panels=panels,candidate=candidate,scene_groups=groups,static_receipts=static_receipts,
        exclusion_population_sha256=digest(exclusions),conditions=40,independent_scene_groups=10,
        tasks=['T1','T2'],fixed_noise_banks=list(BANKS),engineering_replacement_per_condition_max=1,
        valid_policy_outcomes_repeated=False,Teacher_success_screening=False,confirmation_used_for_training=False,
        task_goal_per20=14,task_goal_is_deployment_proof=False,second_candidate_after_results=False)

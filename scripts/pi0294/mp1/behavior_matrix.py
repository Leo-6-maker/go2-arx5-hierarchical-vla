"""Report every fixed source condition; never authorize retries or select weights."""
from collections import Counter
import hashlib
import json

from scripts.pi0294.e1.analysis import audited_outcome
from scripts.pi0294.public_checkpoint import reject_private
from .probe_compare import PRECISION

PANELS={f'{branch}-{step}-{split}{count}':dict(branch=branch,step=step,split=split,count=count)
    for branch in ['S-BC','S-CORR'] for step,split,count in [(512,'TRAIN',4),(2048,'TRAIN',4),(2048,'DEV',8)]}
VIDEO_FIELDS=['sha256','bytes','decoded_frames','actual_paired_frames','fps',
    'playback_duration_s','native_duration_s','initial_frame_included','padding_or_interpolation','logical_id']


def semantic_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fixed_cards(plan):
    if (plan['schema']!='MP1_FIRST_SOURCE_PAIR_FIXED_DEVELOPMENT_V1'
            or set(plan['panels'])!=set(PANELS) or plan['regular_starts']!=32
            or plan['valid_policy_outcomes_repeated'] or plan['new_DEV_labels']
            or plan['engineering_replacement_per_condition_max']!=1):
        raise PermissionError('ORIGINAL_SIX_PANELS_32_CONDITIONS_ONLY')
    cards={}
    for name,spec in plan['panels'].items():
        expected=PANELS[name]
        if (spec['branch']!=expected['branch'] or spec['checkpoint_step']!=expected['step']
                or spec['split']!=expected['split'] or len(spec['cards'])!=expected['count']
                or spec['attempt_ids']!=[c['id'] for c in spec['cards']]
                or spec['teacher_takeover'] or not spec['original_S1'] or spec['complete_task_duration_s']!=32):
            raise PermissionError('UNCHANGED_FIXED_PANEL_TASK_SAFETY_AND_DURATION')
        for card in spec['cards']:
            if (card['id'] in cards or card['mp1_model_label']!=name or card['split']!=expected['split']
                    or card['task'] not in ['T1','T2']
                    or card['id']!='PI0294-MP1-'+name+'-'+'-'.join(card['scene_group'].split('/')[-2:])+'-'+card['task']):
                raise PermissionError('EXACT_UNIQUE_SOURCE_CONDITION_IDENTITY')
            cards[card['id']]=card
        if any(sum(c['task']==t for c in spec['cards'])!=expected['count']//2 for t in ['T1','T2']):
            raise PermissionError('ORIGINAL_BALANCED_FIXED_TASK_MEMBERSHIP')
        groups=[('F0','000'),('F1','000')] if expected['split']=='TRAIN' else [('F0','000'),('F1','000'),('F1','001'),('F2','000')]
        if {(c['scene_group'],c['task']) for c in spec['cards']}!={(f"PI0294/{expected['split']}/{f}/{n}",t) for f,n in groups for t in ['T1','T2']}:
            raise PermissionError('EXACT_ORIGINAL_LAYOUT_GROUPS_NOT_NEW_DEVELOPMENT_SCENES')
    return cards


def chain_verified(value,actual,panel,checkpoint,registry):
    if value is None:return False
    if value.get('status')!='PASS_COMPLETE_SEALED_CONDITION_FRESH_DUAL_SSH_CHAINS':return False
    if registry is None:return False
    core,hopper=value['core'],value['hopper'];steps=actual['main_integrations'];rejected=actual['S1_rejections']
    from .n1_limits import allocation_number,registry_cap
    base,_=allocation_number(hopper['weight_binding']['label'],registry_cap(registry.get('n1_authority_binding')))
    if base!=panel:raise ValueError('ACTUAL_MODEL_PANEL_MUST_MATCH_NATIVE_CHAIN')
    reference_reject=actual['failure_stage']=='ORIGINAL_REFERENCE_GATE_REJECT'
    if reference_reject and core.get('original_reference_gate_rejections')!=1:return False
    if (type(steps) is not int or not 0<=steps<=16000 or steps%200
            or type(rejected) is not int or rejected not in [0,1]):raise ValueError('EXACT_VALID_PREFIX_PROGRESS_AND_REJECTION')
    if reference_reject and rejected:raise ValueError('REFERENCE_GATE_AND_S1_REJECTIONS_ARE_DISTINCT')
    blocked=int(bool(rejected or reference_reject))
    count=steps//200+blocked;weight=hopper['weight_binding'];step=PANELS[panel]['step'];case=registry['cases'].get(core['episode_uuid'])
    if (not 0<count<=80 or core['integrations']!=steps or core['executed_ACKs']!=steps//200
            or core['status']!='PASS_ACTUAL_FRESH_OBSERVATION_ACTION_EXECUTED_ACK_AND_REJECTED_ZERO_LIVE_PREFIX'
            or hopper['status']!='PASS_MATCHING_NATIVE_GENERATIONS_AND_UNMODIFIED_ACTION_RECEIPTS'
            or value['read_only'] is not True or value['scientific_inputs_modified'] is not False
            or value['new_physics']!=0 or value['new_model_queries']!=0
            or core['rejected_unexecuted_prefixes']!=blocked or core['distinct_snapshots']!=count
            or core['raw_camera_hashes_matched']!=2*count or hopper['draws']!=count
            or not hopper['all_arrays_sha256_pass'] or len(core['rows'])!=count
            or [r['sequence'] for r in core['rows']]!=list(range(count))
            or len({r['query_uuid'] for r in core['rows']})!=count
            or checkpoint is None or weight['checkpoint_manifest_sha256']!=checkpoint
            or weight['global_step']!=step
            or registry['run_uuid']!='source-pair-01' or registry['checkpoint_identity']!=weight or case is None
            or case['attempt_id']!=actual['attempt_id'] or case['card']['id']!=actual['attempt_id']
            or any(case['card'][k]!=actual[k] for k in ['task','split','scene_group'])
            or not hopper['precision_modes']
            or any(any(p.get(k)!=v for k,v in PRECISION.items()) for p in hopper['precision_modes'])):
        raise ValueError('ACTUAL_CHAIN_PROGRESS_INPUTS_MODEL_AND_PRECISION_IDENTITY')
    if rejected and actual['failure_stage']!='S1_PRE_EXECUTION_REJECT':
        raise ValueError('REJECTED_PREFIX_STAGE_MUST_REMAIN_S1_NOT_NATIVE_COLLISION')
    if not blocked and steps!=16000:raise ValueError('NO_PARTIAL_SAFE_PREFIX_CALLED_VALID_TASK_OUTCOME')
    return True


def assemble(plan,adjudications,audits,chains,checkpoints,registries,*,closed=False,geometries=None,public_videos=None):
    cards=fixed_cards(plan);geometries=geometries or {};public_videos=public_videos or {}
    if any(k not in cards or cards[k]['mp1_model_label']!='S-BC-512-TRAIN4' for k in geometries):
        raise PermissionError('CURRENT512_GEOMETRY_CANNOT_FILL_OTHER_CHECKPOINTS_OR_DEV')
    if type(closed) is not bool:raise ValueError('EXPLICIT_OPEN_OR_NATIVE_CLOSED_REPORTING_SCOPE')
    by_id={a['attempt_id']:a for a in adjudications}
    if len(by_id)!=len(adjudications):raise ValueError('UNIQUE_ACTUAL_ATTEMPT_RECEIPTS')
    grouped={i:[] for i in cards};attempts=[]
    for identifier,a in by_id.items():
        base=identifier.removesuffix('-REPLACEMENT-01')
        if base not in cards or identifier not in [base,base+'-REPLACEMENT-01']:
            raise PermissionError('NO_OLD_MODEL_OR_UNDECLARED_CONDITION_FILL')
        card=cards[base];panel=card['mp1_model_label']
        if (any(a[k]!=card[k] for k in ['task','split','scene_group']) or a['model']!=panel
                or a['teacher_takeover'] is not False):
            raise PermissionError('ACTUAL_MODEL_SPLIT_SCENE_TASK_AND_AUTONOMY')
        audit=audits.get(identifier);outcome=audited_outcome(a,audit,autonomous=True)
        policy=a['pipeline_trustworthy'] and a['status'] in ['SUCCESS','VALID_TASK_FAIL']
        key=PANELS[panel]['branch']+'-'+str(PANELS[panel]['step'])
        chain=chains.get(identifier)
        session=((chain or {}).get('hopper') or {}).get('weight_binding',{}).get('label')
        registry=registries.get(session)
        verified_chain=chain_verified(chain,a,panel,checkpoints.get(key),registry) if policy else False
        if outcome['audit_verified'] and policy and a['status']=='VALID_TASK_FAIL' and outcome['persistent_success'] is not False:
            raise ValueError('VALID_NEGATIVE_MUST_REMAIN_A_REAL_NEGATIVE_TASK_SCORE')
        video=(audit or {}).get('video') or {}
        if outcome['audit_verified'] and policy and (outcome['native_safety_flag_rows']!=0
                or video.get('decoded_frames')!=a['RGB_pairs'] or video.get('actual_paired_frames')!=a['RGB_pairs']):
            raise ValueError('VALID_POLICY_AUDIT_REQUIRES_ZERO_NATIVE_FAULTS_AND_FULL_VIDEO_DECODE')
        qualified=policy and outcome['audit_verified'] and verified_chain and not a['live_native_failures']
        public=public_videos.get(identifier)
        if public is not None and (public['sha256']!=video.get('sha256') or public['bytes']!=video.get('bytes')):
            raise ValueError('PUBLIC_VIDEO_MUST_MATCH_THIS_EXACT_PRIVATE_AUDITED_CLIP')
        row={k:a[k] for k in ['attempt_id','model','task','split','scene_group','status','failure_stage',
            'duration_s','main_integrations','RGB_pairs','S1_rejections','live_native_failures','teacher_takeover']}
        row.update(adjudication_pipeline_trustworthy=a['pipeline_trustworthy'],raw_audit_verified=outcome['audit_verified'],
            cross_host_chain_verified=verified_chain,qualified_policy_outcome=qualified,
            persistent_success=outcome['persistent_success'],raw_audit_status=outcome['evidence_audit_status'],
            actual_execution_source_sha=(audit or {}).get('execution_source_sha'),
            video={k:video[k] for k in VIDEO_FIELDS if k in video},
            public_video_url=None if public is None else public['url'],
            first_illegal_geometry=geometries.get(identifier,'NOT_APPLICABLE_NO_S1_OR_NATIVE_UNSAFE_PREFIX' if not a['S1_rejections'] and not a['live_native_failures'] else 'NOT_INDEPENDENTLY_RECONSTRUCTED_FOR_THIS_EXACT_ATTEMPT'),
            geometry_from_another_model_imputed=False,
            registry_session_label=session,
            semantic_raw_audit_sha256=semantic_sha(audit) if audit is not None else None,
            semantic_chain_sha256=semantic_sha(chains[identifier]) if identifier in chains else None,
            raw_result_sha256=a.get('result_sha256'),missing_audit_is_retry_authority=False)
        grouped[base].append(row);attempts.append(row)
    cells=[]
    for base,card in cards.items():
        rows=grouped[base];raw={r['attempt_id']:r for r in rows}
        original=raw.get(base);replacement=raw.get(base+'-REPLACEMENT-01')
        if len(rows)>2 or replacement and (original is None or original['status']!='ENGINEERING_INCOMPLETE'
                or original['adjudication_pipeline_trustworthy'] or original['live_native_failures']):
            raise PermissionError('ONLY_ONE_ENGINEERING_REPLACEMENT_NO_VALID_RESULT_REPLAY')
        chosen=[r for r in rows if r['qualified_policy_outcome']]
        if len(chosen)>1:raise PermissionError('NO_BEST_OF_MULTIPLE_VALID_ATTEMPTS')
        safety=any(r['live_native_failures'] or r['status']=='ACTUAL_SAFETY_FAIL' for r in rows)
        status=('ACTUAL_SAFETY_FAIL' if safety else chosen[0]['status'] if chosen else
            ('NOT_RUN' if closed else 'PENDING_OR_IN_PROGRESS') if not rows else
            'ENGINEERING_INCOMPLETE' if all(r['status']=='ENGINEERING_INCOMPLETE' for r in rows) else 'UNVERIFIED_POLICY_OUTCOME')
        cells.append(dict(condition_id=base,model=card['mp1_model_label'],split=card['split'],task=card['task'],
            scene_group=card['scene_group'],status=status,valid_policy_outcome=bool(chosen),
            actual_attempt_ids=[r['attempt_id'] for r in rows],actual_attempts=len(rows),
            actual_native_safety_fault_present=safety))
    summaries={}
    for panel in PANELS:
        selected=[c for c in cells if c['model']==panel];task_summary={}
        for task in ['T1','T2']:
            chosen=[c for c in selected if c['task']==task];complete=all(c['valid_policy_outcome'] for c in chosen)
            task_summary[task]=dict(planned_conditions=len(chosen),layout_groups=len({c['scene_group'] for c in chosen}),
                actual_attempts=sum(c['actual_attempts'] for c in chosen),status_counts=dict(Counter(c['status'] for c in chosen)),
                failure_stages_by_actual_attempt=dict(Counter(r['failure_stage'] for r in attempts if r['model']==panel and r['task']==task)),
                successes=sum(c['status']=='SUCCESS' for c in chosen),valid_results=sum(c['valid_policy_outcome'] for c in chosen),
                complete=complete,success_fraction_only_if_complete=sum(c['status']=='SUCCESS' for c in chosen)/len(chosen) if complete else None)
        summaries[panel]=dict(conditions=len(selected),layout_groups=len({c['scene_group'] for c in selected}),
            all_conditions_have_verified_policy_results=all(c['valid_policy_outcome'] for c in selected),tasks=task_summary,
            exposed_development=PANELS[panel]['split']=='DEV',independent_hidden_confirmation=False)
    value=dict(schema='MP1_ALL_FIXED_SOURCE_BEHAVIOR_AND_FAILURE_MATRIX_V1',
        status='COMPLETE_ALL32_VERIFIED_POLICY_RESULTS' if all(c['valid_policy_outcome'] for c in cells) else 'PARTIAL_ALL32_CONDITIONS_PRESERVED',
        source_execution_scope_closed=closed,original_fixed_conditions=32,cells=cells,actual_attempts=attempts,
        panels=summaries,original_plan_semantic_sha256=semantic_sha(plan),checkpoint_manifest_identities=checkpoints,
        native_registry_semantic_sha256={k:semantic_sha(v) for k,v in registries.items()},
        actual_safety_stop_present=any(r['live_native_failures'] or r['status']=='ACTUAL_SAFETY_FAIL' for r in attempts),
        native_adjudication_not_relabelled_by_missing_audit=True,missing_evidence_does_not_authorize_a_retry=True,
        original_S1_rejection_and_safe_complete_object_failure_are_separate=True,
        training_updates_exposure_and_model_parent_proofs_are_separate=True,
        candidate_selection='NOT_PERFORMED_BY_THIS_REPORT_AGGREGATOR',selected='NONE',deployment='NONE')
    reject_private(value);return value

"""Pure confirmation evidence joins; no model, physics, scoring or file writes."""
import hashlib
import re

from . import confirmation_plan as config
from .plans import remaining_panel

AUDIT_PASS={'PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT','PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE'}


def canonical_row(row):
    return {k:v for k,v in row.items() if k!='route_failure'}


def coverage(plan,rows,audits,*,unresolved=()):
    if (plan['schema']!=config.SCHEMA or plan['candidate']['candidate']!=config.CANDIDATE
            or plan['candidate']['checkpoint_manifest_sha256']!=config.CHECKPOINTS[config.CANDIDATE]
            or plan['candidate']['parent_checkpoint_manifest_sha256']!=config.PARENT
            or plan['conditions']!=40 or plan['independent_scene_groups']!=10 or len(plan['panels'])!=8):
        raise ValueError('ONLY_ORIGINAL_FROZEN_SINGLE_CANDIDATE_CONFIRM40')
    indexed={};cards=[]
    for label,spec in plan['panels'].items():
        for card in spec['cards']:
            if (card['confirmation_noise_bank']!=spec['confirmation_noise_bank']
                    or spec['noise_namespace']!='PI0294/SOURCE_CONFIRM_V1/N'+str(card['confirmation_noise_bank'])):
                raise ValueError('NOISE_BANK_AND_ACTUAL_PANEL_NAMESPACE_MUST_MATCH')
            key=(card['scene_group'],card['task'],label)
            if key in indexed:raise ValueError('UNIQUE_FROZEN_CONDITION')
            indexed[key]=(spec,card);cards.append((key,card))
    if len(cards)!=40 or len({c['confirmation_condition_id'] for _,c in cards})!=40:
        raise ValueError('EXACT_FORTY_FROZEN_CONDITIONS')
    expected={(g['group_id'],t,b) for g in plan['scene_groups'] for t in ['T1','T2'] for b in config.BANKS}
    if len(expected)!=40 or {(c['scene_group'],c['task'],c['confirmation_noise_bank']) for _,c in cards}!=expected:
        raise ValueError('TEN_GROUPS_TWO_TASKS_TWO_FIXED_BANKS_EXACT_CARTESIAN_MEMBERSHIP')
    grouped={key:[] for key in indexed};seen=set()
    for original in rows:
        row=canonical_row(original);key=(row['scene_group'],row['task'],row['model'])
        if key not in grouped or row['attempt_id'] in seen:raise ValueError('NO_FOREIGN_OR_DUPLICATE_ACTUAL_OUTCOME')
        seen.add(row['attempt_id']);grouped[key].append(row)
    unresolved=list(unresolved)
    if len(set(unresolved))!=len(unresolved):raise ValueError('DUPLICATE_UNRESOLVED_ATTEMPT_ID')
    unresolved=set(unresolved)
    allowed={c['id']+suffix for _,c in cards for suffix in ['', '-REPLACEMENT-01']}
    if not unresolved<=allowed or unresolved&seen:raise ValueError('UNRESOLVED_ATTEMPT_MUST_BE_KNOWN_AND_NOT_ADJUDICATED')
    cells=[];safety_stop=False
    for key,card in cards:
        attempts=sorted(grouped[key],key=lambda r:r['attempt_id']!=card['id']);ids={r['attempt_id'] for r in attempts}
        if len(attempts)>2 or not ids<={card['id'],card['id']+'-REPLACEMENT-01'}:
            raise ValueError('ONE_ORIGINAL_AND_AT_MOST_ONE_ENGINEERING_REPLACEMENT')
        stopped=any(r['live_native_failures'] or r['status']=='ACTUAL_SAFETY_FAIL' or r['failure_stage']=='S1_INTEGRITY' for r in attempts)
        safety_stop|=stopped
        if not stopped:remaining_panel(attempts,indexed[key][0])
        elif len(attempts)>1:raise ValueError('REAL_SAFETY_STOP_MUST_NOT_HAVE_A_REPLACEMENT')
        valid=[r for r in attempts if r['pipeline_trustworthy'] and r['status'] in ['SUCCESS','VALID_TASK_FAIL']]
        if len(valid)>1:raise ValueError('VALID_CONDITION_MUST_NEVER_BE_REPEATED')
        if valid and {card['id'],card['id']+'-REPLACEMENT-01'}&unresolved:
            raise ValueError('VALID_OUTCOME_CANNOT_HAVE_AN_UNRESOLVED_REPEAT')
        for row in attempts:
            if row['teacher_takeover'] or row['split']!=config.SPLIT:raise ValueError('ONLY_ACTUAL_AUTONOMOUS_CONFIRMATION_OUTCOMES')
            if row['status']=='SUCCESS' and (row['main_integrations']!=16000 or row['duration_s']!=32
                    or row['RGB_pairs']!=641 or row['live_native_failures'] or row['S1_rejections']):
                raise ValueError('NO_SHORT_UNSAFE_OR_REJECTED_ROLLOUT_MAY_CLAIM_FULL_TASK_SUCCESS')
        selected=valid[0] if valid else attempts[-1] if attempts else None
        audit=None if selected is None else audits.get(selected['attempt_id'])
        verified=bool(audit and audit['status'] in AUDIT_PASS and audit.get('adjudication')==selected)
        unknown=bool({card['id'],card['id']+'-REPLACEMENT-01'}&unresolved)
        status=('UNRESOLVED_PHYSICAL_OUTCOME' if unknown else 'NOT_RUN' if selected is None
            else selected['status'] if verified else 'PENDING_INDEPENDENT_AUDIT')
        qualified=bool(valid and verified and not stopped and not unknown)
        cells.append(dict(condition_id=card['confirmation_condition_id'],scene_group=card['scene_group'],task=card['task'],
            noise_bank=card['confirmation_noise_bank'],status=status,raw_status=None if selected is None else selected['status'],
            valid_policy_outcome=qualified,success=qualified and selected['status']=='SUCCESS',
            actual_attempts=len(attempts),attempt_ids=[r['attempt_id'] for r in attempts],
            engineering_replacement_used=card['id']+'-REPLACEMENT-01' in ids,
            failure_stage=None if selected is None else selected['failure_stage'],
            S1_rejections=sum(r['S1_rejections'] for r in attempts),live_native_failures=sum(r['live_native_failures'] for r in attempts),
            independently_audited=verified,video=None if not verified else audit.get('video')))
    tasks={task:dict(planned_conditions=20,independent_groups=10,
        attempted_conditions=sum(c['actual_attempts']>0 for c in cells if c['task']==task),
        qualified_policy_outcomes=sum(c['valid_policy_outcome'] for c in cells if c['task']==task),
        successes=sum(c['success'] for c in cells if c['task']==task),
        S1_rejections=sum(c['S1_rejections'] for c in cells if c['task']==task),
        live_native_failures=sum(c['live_native_failures'] for c in cells if c['task']==task)) for task in ['T1','T2']}
    return dict(status='COMPLETE_ACTUAL_AUDITED_CONFIRM40' if all(c['valid_policy_outcome'] for c in cells) else 'PARTIAL_ACTUAL_COVERAGE_NO_FABRICATED_RESULTS',
        cells=cells,tasks=tasks,actual_attempts=sum(c['actual_attempts'] for c in cells),independent_scene_groups=10,
        paired_noise_trials_are_independent_scenes=False,actual_safety_stop=safety_stop,unresolved_attempts=sorted(unresolved),
        new_training_updates=0,selected='NONE',deployment='NONE')


def draw_receipt(spec,card,sequence,receipt,*,expected_noise_format):
    if type(sequence) is not int or not 0<=sequence<80:raise ValueError('ORIGINAL_EIGHTY_PREFIX_SEQUENCE_BOUND')
    bank=card['confirmation_noise_bank'];namespace='PI0294/SOURCE_CONFIRM_V1/N'+str(bank)
    if bank not in config.BANKS or spec['noise_namespace']!=namespace:
        raise ValueError('ONLY_FROZEN_NOISE_BANK_NAMESPACE')
    seed=int.from_bytes(hashlib.sha256((namespace+'/'+card['scene_group']+'/'+str(sequence)).encode()).digest()[:8],'big')%(2**63)
    precision=receipt['inference_precision'];noise=receipt['noise']
    if (type(receipt['seed']) is not int or receipt['seed']!=seed or precision['mode']!='ORIGINAL_E0_BF16_AUTOCAST_FP32_MATMUL_HIGHEST'
            or precision['cuda_matmul_allow_tf32'] is not False or precision['bf16_autocast'] is not True
            or {k:noise[k] for k in ['shape','dtype']}!=expected_noise_format
            or any(re.fullmatch('[0-9a-f]{64}',receipt[k]) is None for k in ['action_sha256','arrays_sha256'])
            or re.fullmatch('[0-9a-f]{64}',noise['sha256']) is None):
        raise ValueError('ACTUAL_FIXED_SEED_PRECISION_NOISE_FORMAT_AND_ACTION_ARRAY_FINGERPRINTS')
    return dict(seed=seed,noise_sha256=noise['sha256'],namespace=namespace,sequence=sequence,
        action_sha256=receipt['action_sha256'],arrays_sha256=receipt['arrays_sha256'],
        GPU_RNG_reexecuted=False,new_model_queries=0)


def paired_noise(records):
    by_seed={}
    for row in records:
        if by_seed.setdefault(row['seed'],row['noise_sha256'])!=row['noise_sha256']:
            raise ValueError('ACTUAL_SAME_SEED_MUST_HAVE_THE_SAME_ORIGINAL_NOISE_TENSOR')
    return dict(actual_draw_receipts=len(records),unique_seeds=len(by_seed),same_seed_actual_noise_hashes_match=True,
        observations_claimed_identical=False,new_model_queries=0)

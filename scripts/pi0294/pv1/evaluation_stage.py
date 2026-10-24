"""V4A stage 1: exactly A/B on the original bank294101, no six-panel dispatch."""
from copy import deepcopy
import hashlib,json
from . import evaluation_platform as e
from .evaluation_work import resource_upper

RELEASE='e78133c5d8ebd77ed5a618ccf72cd1fbe3731bc7'
VERSION='V4A_MATCHED_AB_N294101_STAGE1'
PLAN='plans/V4A_EVAL_FIRST_EXECUTION_PLAN.md'
PLAN_SHA='137142763c3e89063122a9159f56cf1159a51058c0d5d8d598c335d08b9d9f3e'
CARD='plans/V4A_EVAL_FIRST_EXECUTION_CARD.json'
CARD_SHA='4dd8cee6891cda32ea3f92b343fdd7bccaca76a22571c12939e412408f8c4967'
CANDIDATES=('A_CONTROL','B_COVERAGE')


def need(ok,why):
    if not ok:raise PermissionError('PV1_V4A_STAGE_'+why)


def request(manifest,r):
    need(r.get('evaluation_execution_stage')==VERSION and r.get('evaluation_stage_plan_sha256')==PLAN_SHA
         and r.get('evaluation_stage_card_sha256')==CARD_SHA
         and r.get('evaluation_candidate') in CANDIDATES and r.get('noise_bank')==294101,
         'ONLY_TWO_MATCHED_ORIGINAL_AB_FIRST_BANK_NO_BASELINE_OR_SECOND_BANK')
    for value in (manifest['execution'],r):
        need(value['sealed_inputs'].get(PLAN)==PLAN_SHA and value['sealed_inputs'].get(CARD)==CARD_SHA,
             'EXACT_V4A_PLAN_AND_CARD_SEALED_IN_BOTH_INPUTS')


def sealed(root,manifest,r):
    from .future_admission import _file
    request(manifest,r)
    for name,pin in ((PLAN,PLAN_SHA),(CARD,CARD_SHA)):
        need(hashlib.sha256(_file(root,name)).hexdigest()==pin,'ORIGINAL_IMMUTABLE_STAGE_BYTES')


def first_stage(work,*,manifest,current_request,global_ledger,purpose):
    request(manifest,current_request)
    stage=deepcopy(work);original_jobs=work['jobs']
    stage['jobs']=[j for j in original_jobs if j['candidate'] in CANDIDATES and j['noise_bank']==294101]
    stage['resources']=resource_upper(stage['jobs'])
    rows=[v for v in global_ledger['PBS_submissions'] if v.get('platform_evaluation_version')==e.VERSION]
    need(all(v.get('evaluation_execution_stage')==VERSION and v.get('noise_bank')==294101
         and v.get('evaluation_candidate') in CANDIDATES for v in rows),
         'NO_UNCLASSIFIED_OR_AUTOMATIC_OTHER_STAGE_EVALUATION')
    counts={c:sum(v['evaluation_candidate']==c for v in rows) for c in CANDIDATES}
    need(all(n<=1 for n in counts.values()) and len(rows)<=2,'ONE_ORIGINAL_PANEL_JOB_PER_ARM_MAX_TWO')
    already={v['request_id'] for v in rows}
    if purpose!='NATIVE_PREMODEL':
        need(current_request['request_id'] not in already
             and counts[current_request['evaluation_candidate']]==0,'NO_REPEATED_ARM_OR_EXISTING_UNKNOWN_REQUEST')
    else:need(current_request['request_id'] in already,'ACTUAL_ALREADY_BOUND_STAGE_REQUEST')
    # An unfinished arm whose only PBS was terminal cannot acquire another job
    # silently. Missing outcomes remain engineering partial rather than FAIL.
    for job in stage['jobs']:
        same=[v for v in rows if v['evaluation_candidate']==job['candidate']]
        need(not same or same[0]['status']!='NATIVE_TERMINAL', 'CLOSED_INCOMPLETE_STAGE_ARM_REQUIRES_RESULT_REVIEW_NO_NEW_POOL')
    need(len(stage['jobs'])+sum(v['status']=='NATIVE_TERMINAL' for v in rows)<=2,
         'BOTH_ARMS_STILL_COVERED_INSIDE_TWO_STAGE_SUBMISSIONS')
    stage.update(execution_stage=VERSION,original_full_DEV48_unchanged=True,
        full_remaining_DEV_jobs=len(original_jobs),full_remaining_DEV_conditions=sum(j['conditions'] for j in original_jobs),
        later_panels='NOT_SCHEDULED_REQUIRES_FIRST_STAGE_RESULT_DECISION',
        first_stage_total_original_conditions=48,first_stage_PBS_limit=2,
        stage_plan_sha256=PLAN_SHA,stage_card_sha256=CARD_SHA)
    return stage


def clock(work,*,now_epoch,current_native_facts=None):
    walls=[j['walltime_s']+j['accounting_pad_s'] for j in work['jobs']]
    elapsed=0
    if current_native_facts is not None:
        elapsed=max(0,now_epoch-current_native_facts['allocation_start_epoch'])
        need(elapsed<=28800,'CURRENT_NATIVE_ELAPSED_INSIDE_ORIGINAL_ALLOCATION')
    remaining_wall=max(0,sum(walls)-elapsed)
    last_new_submit_wall=max(0,sum(walls[:-1])-elapsed)
    return dict(first_stage_remaining_wall_upper_s=remaining_wall,
        both_stage_panels_fit_original_physics_window=now_epoch+remaining_wall<=e.STOP_PHYSICS,
        last_stage_submission_fits_original_stop_new=(True if current_native_facts is not None and len(walls)<=1
            else now_epoch+last_new_submit_wall<e.STOP_NEW),
        original_matrix_and_final_resources_protected=True,
        all_protected_future_jobs_wall_completion_promised=False)

"""Exact administrative closure of failed PBS startup, without a process final.

The original typed arithmetic is reused in a separate function whose proof
schema explicitly records that the scientific process never produced a final.
All21 scheduler attempts and queue/hold time are conservatively covered by the
single native request's complete lifetime. AMS credits are separate evidence.
"""
from copy import deepcopy
import datetime
import inspect
import math
from . import a40_accounting as original

JOB='1480412.stdct-mgmt-02'
REQUEST='9895f812-13f5-4fb0-9dcb-0c27ebb97d41'
REQUEST_SHA='4455e91b4f49c7f627d82dd9dacab0ec939a23e11cbf1034468d73d43f4523f9'
SOURCE='07937041d7b330354dfe004ed6bf5a179588147e'
KEY='VAP1_B_RECOVERY1_B2_CONTROL_CPU1800_LOCAL180_TRANSFER256M_OUTPUT64M_V1'
SCHEMA='VAP1_NATIVE_A40_ADMINISTRATIVE_STARTUP_TERMINAL_V1'


def fail(why):raise PermissionError('A40_STARTUP_FAILURE_'+why)


def proof(evidence,*,now_epoch):
    v=deepcopy(evidence);rows=v.get('rows',{});g=rows.get('vanda',{});h=rows.get('hopper',{});j=g.get('raw_native',{})
    if (v.get('status')!='INDEPENDENT_NATIVE_STARTUP_F21_NO_MODEL_EFFECTS_ALL_INPUTS_AND_ENTIRE_LIFETIME_COST_BOUNDS'
            or any(r.get('source')!=SOURCE or r.get('all_sealed_SHA_match') is not True for r in (g,h))
            or g.get('sealed_members')!=972 or h.get('sealed_members')!=939
            or g.get('request_sha256')!=REQUEST_SHA or g.get('process_final_absent') is not True
            or g.get('all_model_and_effect_intents_absent') is not True
            or g.get('recorded_process_terminal') is not False or g.get('scientific_completion') is not False):
        fail('EXACT_READONLY_SEALS_AND_ABSENT_PROCESS_OR_MODEL_EFFECTS')
    if (j.get('job_state')!='F' or str(j.get('Exit_status'))!='-3' or j.get('run_count')!=21
            or j.get('Hold_Types')!='s' or j.get('Rerunable') not in (False,'False')
            or j.get('Job_Owner','').split('@')[0]!='e1192548' or j.get('project')!='CFP04-CF-007'
            or j.get('Job_Name')!='pi0294_dec1_a40_train'
            or j.get('Variable_List',{}).get('PV1_REQUEST_ID')!=REQUEST
            or j.get('Variable_List',{}).get('PV1_SUBMISSION_NONCE')!='886b1d63-2119-4ee0-85b2-0d26d3482c9d'
            or j.get('comment')!='job held, too many failed attempts to run and terminated'):
        fail('EXACT_WITHDRAWN_SYSTEM_HELD21_NATIVE_F_MINUS3')
    def epoch(s):
        return datetime.datetime.strptime(s,'%a %b %d %H:%M:%S %Y').replace(
            tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
    elapsed=j['history_timestamp']-epoch(j['ctime'])
    if (type(now_epoch) not in (int,float) or not math.isfinite(now_epoch)
            or not 0<=now_epoch-v['observed_epoch']<=120
            or not 0<=now_epoch-g['observed_epoch']<=120 or not 0<=now_epoch-h['observed_epoch']<=120
            or not 0<=elapsed<22500 or j['Resource_List'].get('ngpus')!=1
            or j['Resource_List'].get('ncpus')!=36):fail('FRESH_NATIVE_SINGLE_GPU_WHOLE_LIFETIME_BOUND')
    gpu=elapsed+60;cpu=36*elapsed+120
    if (g.get('entire_single_request_lifetime_s')!=elapsed or g.get('conservative_all_attempts_GPU_s')!=gpu
            or g.get('conservative_all_allocated_CPU_s')!=cpu or gpu>22560 or cpu>54000):
        fail('ALL_ATTEMPTS_CONSERVATIVE_COSTS_NOT_AMS_UNITS_OR_ZERO_ASSUMPTION')
    output=g.get('owned_unique_inode_output_upper_bytes',-1)+h.get('owned_unique_inode_output_upper_bytes',-1)
    if (any(type(r.get('owned_unique_inode_output_upper_bytes')) is not int
            or r['owned_unique_inode_output_upper_bytes']<0 for r in (g,h))
            or v.get('combined_CPU_GPU_roots_actual_output_upper_bytes')!=output
            or output>64*1024**2 or v.get('already_prepaid_small_output_upper_bytes')!=64*1024**2
            or v.get('additional_output_charge_needed_bytes')!=0):fail('BOTH_ROOTS_COVERED_BY_ALREADY_CHARGED64M')
    n=dict(schema=SCHEMA,job_id=JOB,job_state='F',project='CFP04-CF-007',job_name='pi0294_dec1_a40_train',
        owner='e1192548',ngpus=1,request_id=REQUEST,
        submission_nonce='886b1d63-2119-4ee0-85b2-0d26d3482c9d',recorded_process_terminal=False,
        observed_epoch=now_epoch,conservative_GPU_s=gpu,conservative_CPU_s=cpu,Exit_status=-3,
        new_resident_allocated_upper_bytes=0,administrative_failure=True,scientific_completion=False,
        all21_scheduler_attempts_retained=True,whole_request_lifetime_s=elapsed,
        actual_process_final_created=False,AMS_pending_is_not_physical_GPU_usage=True)
    e=dict(schema='VAP1_A40_NATIVE_SOURCE_EFFECT_SUMMARY_V1',job_id=JOB,request_id=REQUEST,
        request_sha256=REQUEST_SHA,source_sha=SOURCE,arm='B_COVERAGE',start_native_step=2048,
        last_native_step=2048,optimizer_updates_committed=0,query_attempts=0,model_load_attempts=0,
        pending_updates=[],unknown_query_intents=[],new_physics=0,new_episodes=0,effect_journal_complete=True,
        proof_basis='EXACT_NEGATIVE_PBS_STARTUP_TERMINAL_ALL_SEALED_CODE_AND_NO_READY_MODEL_RELEASE_OR_EFFECT_INTENTS')
    return n,e


def administrative_arithmetic():
    """Create a separate adapter; leave the original process-final guard intact."""
    text=inspect.getsource(original.settle)
    if text.count("schema='VAP1_NATIVE_A40_SOURCE_TERMINAL_V1'")!=1 or text.count('recorded_process_terminal=True')!=1:
        fail('EXACT_MATURE_TYPED_ARITHMETIC_SEAM')
    text=text.replace('def settle(','def _administrative_settle(',1)
    text=text.replace("schema='VAP1_NATIVE_A40_SOURCE_TERMINAL_V1'",'schema='+repr(SCHEMA),1)
    text=text.replace('recorded_process_terminal=True','recorded_process_terminal=False',1)
    ns=dict(original.settle.__globals__);exec(compile(text,'<separate-administrative-startup-arithmetic>','exec'),ns)
    return ns['_administrative_settle']


def settle(value,DEC1_value,*,evidence,now_epoch):
    n,e=proof(evidence,now_epoch=now_epoch)
    cost=value.get('engineering_CPU_upper_settlements',{}).get(KEY)
    if (not isinstance(cost,dict) or DEC1_value.get('CPU_upper_prepayments',{}).get(KEY)!=cost
            or cost.get('GPU_site_CPU_upper_s')!=1800 or cost.get('local_CPU_upper_s')!=180):
        fail('EXACT_EXISTING_CONTROL_PREPAY_NOT_NEW_CHARGE')
    rows=[r for r in value['PBS_submissions'] if r.get('request_id')==REQUEST]
    if (len(rows)!=1 or rows[0].get('job_id')!=JOB or rows[0].get('request_sha256')!=REQUEST_SHA
            or rows[0].get('source_sha')!=SOURCE or value['optimizer_updates']!=8744
            or DEC1_value['optimizer_updates']!=8744):fail('EXACT_ACTUAL_BOUND_FAILED_REQUEST_AND_ALL_PRIOR_COSTS')
    nh,nd=administrative_arithmetic()(value,DEC1_value,request_id=REQUEST,native_proof=n,effects=e,now_epoch=now_epoch)
    return dict(ledger=nh,DEC1=nd,native_proof=n,effects=e,
        status='ADMINISTRATIVE_STARTUP_FAILURE_SETTLED_WITH_ALL_ATTEMPTS_CONSERVATIVE_COSTS_NOT_TRAINING_SUCCESS')

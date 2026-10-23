"""Exact current token-stop/HeartbeatLost evidence; never a success or retry GO.

No IO or dispatch. The independent auditor must stream both original full
states and all original probes through the original immutable validator. The
922 paid updates after saved2048 remain spent, even though not recoverable.
"""
from . import b_retry4096_completion as prior
from .b_recovery_completion import identity,complete_states

JOB=prior.JOB
SOURCE=prior.SOURCE
REQUEST=prior.REQUEST
REQUEST_SHA=prior.REQUEST_SHA
LAST_NATIVE=2970
PAID_UPDATES=922
CUMULATIVE_UPDATES=9666
HEARTBEAT_ERROR=dict(type='HeartbeatLost',message='MP1_SINGLE_OWNER_HEARTBEAT_LOST_AFTER_SERVER_REVALIDATION')


def need(ok,why):
    if not ok:raise PermissionError('B_RETRY2970_FAILURE_'+why)


def expected(value):return prior.expected(value)


def owner(state,stop,rpc):
    need(state==dict(MainPID='0',ActiveState='failed',ExecMainStatus='1')
         and stop==dict(epoch=1792755974.8454535,type='PermissionError',message='VANDA_PROJECT_TOKEN_EXPIRED',automatic_mutation_retry=False)
         and rpc.get('status')=='CONFIRMED' and rpc.get('intent',{}).get('effect')=='HEARTBEAT',
         'ACTUAL_FAILED_OWNER_EXACT_TOKEN_STOP_NO_UNKNOWN_HEARTBEAT_WRITE')


def native(job,value):
    expected(value);identity(job,value)
    need(job.get('job_state')=='F' and job.get('Exit_status')==1,'REAL_SAME_JOB_FAILED_TERMINAL_NOT_TIMEOUT_OR_STALE_R')


def effects(e,value):
    expected(value)
    want=dict(job_id=JOB,request_id=REQUEST,request_sha256=REQUEST_SHA,source_sha=SOURCE,arm='B_COVERAGE',
        start_native_step=2048,last_native_step=LAST_NATIVE,optimizer_updates_committed=PAID_UPDATES,
        query_attempts=17,model_load_attempts=2,recipe_sha256=prior.b.RECIPE,pending_updates=[],
        unknown_query_intents=[],effect_journal_complete=True,new_physics=0,new_episodes=0)
    need(all(e.get(k)==v for k,v in want.items()) and all(type(e.get(k)) is int for k in
        ('start_native_step','last_native_step','optimizer_updates_committed','query_attempts','model_load_attempts','new_physics','new_episodes')),
        'ACTUAL922_PAID_UNSAVED_UPDATES17_QUERIES2_LOADS_NO_UNKNOWN_EFFECTS')


def origin(step,complete,value):
    need(step in (1024,2048),'ONLY_ORIGINAL_COMPLETE1024_AND2048_NOT_PARTIAL2970_OR4096')
    return prior.origin(step,complete,value)


def audit(v,value):
    req=expected(value);native(v['raw_native'],value);effects(v['effects'],value)
    f=v['process_final'];w=v['wrapper_final'];p=v['native_proof']
    need(f.get('status')=='FAILED_PRESERVED' and f.get('global_step')==LAST_NATIVE
         and f.get('last_returned_training_step')==2048 and f.get('error')==HEARTBEAT_ERROR
         and f.get('job_id')==w.get('job_id')==JOB and w.get('returncode')==1
         and f.get('source_sha')==SOURCE and f.get('request_sha256')==REQUEST_SHA
         and {k:z for k,z in f.get('effects',{}).items() if k!='finished_epoch'}=={k:z for k,z in v['effects'].items() if k!='finished_epoch'},
         'FAILED_PROCESS_AND_WRAPPER_NOT_B4096_COMPLETION')
    need(p.get('job_id')==JOB and p.get('request_id')==REQUEST and p.get('job_state')=='F'
         and p.get('Exit_status')==1 and p.get('recorded_process_terminal') is True
         and p.get('new_resident_allocated_upper_bytes')==req['additional_output_upper_bytes'],
         'ORIGINAL_FULL70GB_BOUND_NO_UNKNOWN_REFUND')
    complete_states(v)
    need(v['checkpoints']['2048']['complete_sha256']==prior.saved.CHECKPOINT
         and v.get('actual_complete_step_inventory')==[1024,2048]
         and v.get('new_1024_probe_queries')==0 and v.get('imported1024_and2048_origin_preserved') is True,
         'EXACT_ORIGINAL2048_SEED_AND_FULL_DIRECTORY_INVENTORY_NO_LATER_COMPLETE_STATE')
    for step in ('1024','2048'):
        need(all(isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
                 for s in v['checkpoint_members'][step].values()),'EVERY_FULL_MEMBER_SHA256_REQUIRED')
    return v

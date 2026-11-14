"""Retain the actual R0/R1 prefix and execute only its never-started suffix."""
import json
from pathlib import Path
import subprocess

from . import task_learning_grasp_v1 as task
from .stable_task_scene_v1 import pinned
from .storage import digest

SCHEMA='SLA_R1_COLD_UNSTARTED_TRAIN8_SUFFIX_INPUT_V5'


def joined(panel,result,rows,lease,inp,old):
    task.need(all(panel[k]==old[k] for k in ('schema','task_learning_role','protocol','population',
        'B1_global_step','policies','preregistration','assignments')),
        'R1_SUFFIX_RETAINS_FULL_FIXED_POPULATION_SEEDS_POLICIES_AND_SCORING')
    task.need(lease['status']=='CLOSED'and lease['worker_exit_code']==result['exit_code']==1
        and result['unresolved_pair']is None and result['completed_in_group']==len(rows)==13
        and result['summary']['requested']==16 and not result['summary']['complete']
        and inp['assignment_slice']==[0,16]
        and old['source_sha']==inp['source_sha']==result['source_sha']
        and all(all(a[k]==b[k]for k in ('native_run','method','layout','restart','query_seeds'))
                for a,b in zip(rows,old['assignments'][:13],strict=True))
        and not any(r['hard_safety_stop']for r in rows)
        and rows[-1]['native_run']==old['assignments'][12]['native_run']
        and rows[-1]['layout']=='TRAIN13'and rows[-1]['restart']==3
        and not rows[-1]['technical_valid']and rows[-1]['actual_SLA_queries']==15
        and rows[-1]['actual_native_integrations']==20128
        and rows[-1]['first_failure']=='TimeoutError: AUTO1_FINITE_ENDPOINT_OR_OPERATOR_STOP',
        'R1_EXACT_TIMEOUT_PREFIX_RETAINED_NOT_RETRIED_OR_DROPPED')
    return rows


def scope_failure_prefix(panel,retained):
    proof=panel.get('r1_wall_scope_predecessor_v5')
    if proof is None:return retained
    result,rows,lease,inp,old=[pinned(proof[k])for k in('result','outcomes','lease','input','panel')]
    group=Path(proof['result']['path']).parent
    task.need(all(panel[k]==old[k]for k in('schema','task_learning_role','protocol','population',
        'B1_global_step','policies','preregistration','assignments'))
        and inp['panel']==proof['panel']['path']and inp['panel_sha256']==proof['panel']['sha256']
        and lease['input_sha256']==proof['input']['sha256']==digest(group/'INPUT.private.json')
        and lease['status']=='CLOSED'and lease['worker_exit_code']==result['exit_code']==1
        and result['completed_in_group']==1 and result['retained_prefix_conditions']==13
        and result['unresolved_pair']is None and len(rows)==14 and rows[:13]==retained
        and inp['assignment_slice']==[13,16]
        and rows[-1]['native_run']==old['assignments'][13]['native_run']
        and not rows[-1]['technical_valid']and not rows[-1]['native_world_attempted']
        and rows[-1]['actual_native_integrations']==rows[-1]['actual_SLA_queries']==0
        and not any(r['hard_safety_stop']for r in rows),
        'R1_ZERO_INTEGRATION_SCOPE_FAILURE_RETAINED_NOT_RETRIED')
    task.need(subprocess.check_output(['systemctl','--user','show','wm02-'+group.name,
        '--property=MainPID','--value'],text=True,timeout=10).strip()=='0','R1_SCOPE_FAILURE_ACTUALLY_COLD')
    for suffix in('','-gpu'):
        root=group/'runs'/(rows[-1]['native_run']+suffix)
        terminal=json.loads((root/'control/LEASE.json').read_bytes());worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
        task.need(terminal['status']=='CLOSED'and terminal['worker_exit_code']==worker['exit_code']==1
            and terminal['spec_sha256']==digest(root/'EXECUTION_SPEC.json'),'R1_ZERO_TICK_PARTNER_TERMINALS_REQUIRED')
        if not suffix:
            task.need(worker['failure']=='PermissionError: AUTO1_REGISTERED_SCOPE_HOST_LEASE_OR_TIME_INVALID',
                'R1_ACTUAL_SECOND_WALL_ENTRY_GUARD_FAILURE_REQUIRED')
    task.need(all(not(group/'runs'/(r['native_run']+tail)).exists()
        for r in old['assignments'][14:]for tail in('','-gpu')),'R1_LAST_TWO_ROWS_NEVER_STARTED')
    return rows


def prefix(panel):
    proof=panel['r1_cold_suffix_v5'];result,rows,lease,inp,old=[pinned(proof[k])for k in ('result','outcomes','lease','input','panel')]
    group=Path(proof['result']['path']).parent
    task.need(proof['schema']==SCHEMA and proof['result']['path']==str(group/'RESULT.private.json')
        and proof['outcomes']['path']==str(group/'OUTCOMES.private.json')
        and proof['input']['path']==str(group/'INPUT.private.json')
        and proof['lease']['path']==str(group/'control/LEASE.json')
        and lease['input_sha256']==proof['input']['sha256']==digest(group/'INPUT.private.json')
        and inp['panel']==proof['panel']['path']and inp['panel_sha256']==proof['panel']['sha256'],
        'R1_EXACT_CLOSED_GROUP_PROOF_JOIN')
    joined(panel,result,rows,lease,inp,old)
    task.need(subprocess.check_output(['systemctl','--user','show','wm02-'+group.name,
                  '--property=MainPID','--value'],text=True,timeout=10).strip()=='0',
        'R1_ORIGINAL_COORDINATOR_ACTUALLY_COLD')
    for row in rows:
        for suffix in ('','-gpu'):
            root=group/'runs'/(row['native_run']+suffix)
            terminal=json.loads((root/'control/LEASE.json').read_bytes())
            worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
            task.need(terminal['status']=='CLOSED'and terminal['worker_exit_code']==worker['exit_code']
                and type(worker['exit_code'])is int and worker['exit_code']>=0
                and terminal['spec_sha256']==digest(root/'EXECUTION_SPEC.json'),
                'R1_EVERY_PREFIX_PARTNER_ACTUALLY_TERMINAL')
    for row in old['assignments'][13:]:
        task.need(not (group/'runs'/row['native_run']).exists()
            and not(group/'runs'/(row['native_run']+'-gpu')).exists(),'R1_SUFFIX_WAS_NEVER_STARTED')
    return scope_failure_prefix(panel,rows)


def window(panel):
    task.need(panel['r1_cold_suffix_v5']['schema']==SCHEMA
        and panel['registration']['condition_wall_window_v5']==dict(
            original_s=1200,new_s=3600,reason='ACTUAL_20128_OF21203_TICKS_AT1200S_UNDER_PARALLEL_LOAD',
            physics_clock_changed=False,model_changed=False,score_changed=False),
        'R1_ONLY_WALL_ALLOWANCE_REPAIR_NATIVE_CLOCK_UNCHANGED')
    return 3600


def selection(inp,panel,rows):
    retained=prefix(panel)
    task.need(inp['schema']==SCHEMA and inp['source_sha']==panel['source_sha']
        and inp['assignment_slice']==[len(retained),16]
        and inp['population_predecessor']==panel['r1_cold_suffix_v5']['result'],
        'R1_ONLY_NEVER_STARTED_CONDITIONS_NO_FAILED_CASE_REPLAY')
    window(panel)
    return rows[len(retained):]

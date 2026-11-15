"""Keep N20's three terminal rows; run only its five unstarted conditions.

The observed2450-tick ACK arrived after the35s reader deadline. This cold
successor changes only the existing sealed90s read allowance,never ENTER
count,physics quota,policy bytes,action compiler or scoring.
"""
import json
from pathlib import Path
import subprocess

from . import task_learning_grasp_v1 as task
from .stable_task_scene_v1 import pinned
from .storage import digest

KEY='named_cold_suffix_v5'
SCHEMA='SLA_N20_KEEP_THREE_RUN_FIVE_UNSTARTED_SUFFIX_V5'
OLD_SOURCE='75e703f5c412ddc725e9145e12c27b02c3390e6e'


def joined(panel,result,rows,lease,inp,old):
    task.need(all(panel[k]==old[k]for k in('schema','task_learning_role','protocol',
        'population','B1_global_step','training_seed','method','policies','preregistration','assignments'))
        and panel['schema']=='SLA_FRESH_N20_RAW_P10_TRAIN8_PANEL_V5'
        and panel['method']=='N20_FM3K_RAW_P10'
        and old['source_sha']==inp['source_sha']==result['source_sha']==OLD_SOURCE
        and lease['status']=='CLOSED'and lease['worker_exit_code']==result['exit_code']==1
        and result['unresolved_pair']is None and inp['assignment_slice']==[0,8]
        and result['completed_in_group']==result['cumulative_completed']==len(rows)==3
        and result['summary']['requested']==8 and not result['summary']['complete']
        and all(all(a[k]==b[k]for k in('native_run','method','layout','restart','query_seeds'))
            for a,b in zip(rows,old['assignments'][:3],strict=True))
        and [(r['layout'],r['restart'])for r in rows]==[('TRAIN01',3),('TRAIN01',4),('TRAIN07',3)]
        and all(r['technical_valid']and r['policy_proposal_rejected']and not r['task_success']
            and r['actual_native_integrations']==5703 and r['actual_SLA_queries']==1 for r in rows[:2])
        and not rows[2]['technical_valid']and rows[2]['actual_native_integrations']==2450
        and rows[2]['actual_SLA_queries']==0
        and rows[2]['first_failure']=='RuntimeError: EXACT_NATIVE_PAUSE_NOT_REACHED'
        and not any(r['hard_safety_stop']for r in rows),
        'N20_EXACT_THREE_TERMINALS_RETAINED_NO_FAILURE_RETRY_OR_SELECTION')
    return rows


def clock_evidence(failure,ack,requests):
    task.need(failure['requested_step']==2450 and failure['pause_ack']['step']==2425
        and failure['control']['running']is True and failure['automatic_ENTER_retries']==0
        and ack['step']==2450 and ack['clock_paused']is True and ack['running']is False
        and abs(ack['time_s']-2450*.002)<1e-8
        and len(requests)==98 and requests[-1]['requested_step']==2450
        and sum(r['requested_step']==2450 for r in requests)==1
        and all(r['publication_count']==r['matched_subscriptions']==1 for r in requests),
        'N20_SAME_SINGLE_GRANT_ACK_ACTUALLY_ARRIVED_AFTER_ORIGINAL_READ_FAILURE')


def prefix(panel):
    proof=panel[KEY];task.need(proof['schema']==SCHEMA,'N20_TYPED_COLD_SUFFIX')
    result,rows,lease,inp,old=[pinned(proof[k])for k in('result','outcomes','lease','input','panel')]
    group=Path(proof['result']['path']).parent
    task.need(all(proof[k]['path']==str(group/name)for k,name in(
        ('result','RESULT.private.json'),('outcomes','OUTCOMES.private.json'),
        ('lease','control/LEASE.json'),('input','INPUT.private.json')))
        and lease['input_sha256']==proof['input']['sha256']==digest(group/'INPUT.private.json')
        and inp['panel']==proof['panel']['path']and inp['panel_sha256']==proof['panel']['sha256'],
        'N20_CLOSED_PREFIX_PROOF_JOIN')
    joined(panel,result,rows,lease,inp,old)
    roots=[group]+[group/'runs'/(r['native_run']+tail)for r in rows for tail in('','-gpu')]
    states=subprocess.check_output(['systemctl','--user','show','--property=MainPID','--value',
        *['wm02-'+r.name for r in roots]],text=True,timeout=20)
    task.need(len(states.split())==len(roots)and set(states.split())=={'0'},'N20_ALL_ORIGINAL_OWNERS_ACTUALLY_COLD')
    for root in roots[1:]:
        closed=json.loads((root/'control/LEASE.json').read_bytes());worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
        task.need(closed['status']=='CLOSED'and closed['worker_exit_code']==worker['exit_code']
            and type(worker['exit_code'])is int and worker['exit_code']>=0
            and closed['spec_sha256']==digest(root/'EXECUTION_SPEC.json'),'N20_BOTH_PARTNERS_TERMINAL')
    task.need(all(not(group/'runs'/(r['native_run']+tail)).exists()
        for r in old['assignments'][3:]for tail in('','-gpu')),'N20_FIVE_SUFFIX_ROWS_NEVER_STARTED')
    failed=group/'runs'/rows[2]['native_run']/'native-01'
    task.need(proof['clock_failure']['path']==str(failed/'CLOCK_RESUME_FAILURE.json')
        and proof['late_pause_ack']['path']==str(failed/'pause_ack.json')
        and proof['clock_requests']['path']==str(failed/'CLOCK_RESUME_REQUESTS.jsonl')
        and digest(proof['clock_requests']['path'])==proof['clock_requests']['sha256'],
        'N20_EXACT_FAILED_NATIVE_CLOCK_EVIDENCE')
    requests=[json.loads(line)for line in Path(proof['clock_requests']['path']).read_text().splitlines()]
    clock_evidence(pinned(proof['clock_failure']),pinned(proof['late_pause_ack']),requests)
    from .sla_native_continuation_v2 import wait_seconds
    task.need(wait_seconds(dict(task_learning_role=task.SLA_DENSE_ROLE,
        native_pause_read_wait=panel['registration']['native_pause_read_wait']))==90,
        'N20_REUSES_EXISTING_EXACT_ACK_READ_ALLOWANCE')
    return rows


def selection(inp,panel,rows):
    retained=prefix(panel)
    task.need(inp['schema']=='SLA_FRESH_N20_RAW_P10_TRAIN8_GROUP_INPUT_V5'
        and inp['source_sha']==panel['source_sha']and inp['assignment_slice']==[3,8]
        and inp['population_predecessor']==panel[KEY]['result'],
        'N20_ONLY_FIVE_UNSTARTED_ROWS_NO_TERMINAL_REPLAY')
    return rows[len(retained):]

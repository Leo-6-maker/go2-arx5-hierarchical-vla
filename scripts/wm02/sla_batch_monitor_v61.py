"""Finite, model-free observer for one existing SLA training/evaluation batch.

Existing sealed workers own all compute, dispatch and safety. This monitor
never launches/restarts experiments. It seals a completion/attention event for
a separate local notification bridge, so neither physics nor evaluation waits
for a foreground agent or the local computer's connectivity.
"""
import argparse,hashlib,json,os,socket,subprocess,time
from pathlib import Path
from scripts.pi0292.durable import write

SCHEMA='SLA_EXISTING_BATCH_FINITE_COMPLETION_MONITOR_V6_1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(root):
    p=subprocess.run(['systemctl','--user','show','wm02-'+Path(root).name,
        '--property=LoadState,MainPID,Result,SubState'],capture_output=True,text=True,timeout=15,check=True)
    return dict(x.split('=',1)for x in p.stdout.splitlines()if'='in x)


def load(path):
    try:return json.loads(Path(path).read_bytes())
    except FileNotFoundError:return None


def training_state(row,state,lease,result,worker):
    """A lease/result file alone is never proof of terminal process state."""
    if state.get('LoadState')!='loaded':return dict(status='UNKNOWN_HANDLE',complete=False)
    if state.get('MainPID')!='0':return dict(status='RUNNING',complete=False)
    if not lease or lease.get('status')!='CLOSED'or not result or not worker:
        return dict(status='TERMINAL_UNRECONCILED',complete=False)
    if lease.get('worker_exit_code')!=worker.get('exit_code')or result.get('exit_code')!=worker.get('exit_code'):
        return dict(status='TERMINAL_INCONSISTENT',complete=False)
    if lease.get('spec_sha256')!=row['spec_sha256']:
        return dict(status='TERMINAL_INCONSISTENT',complete=False)
    if worker['exit_code']!=0:return dict(status='FAILED',complete=False)
    updates=result.get('candidate_updates',result.get('step'))
    actual=result.get('actual',{})
    if (result.get('schema')!=row['result_schema']or updates!=row['updates']
            or actual.get('optimizer_updates')!=row['updates']
            or result.get('target_visits')!=row['target_visits']or result.get('source_visits')!=row['source_visits']
            or result.get('fresh_RGB16_bit_exact')is not True):
        return dict(status='ENDPOINT_NOT_QUALIFIED',complete=False)
    return dict(status='CLOSED0_ACTUAL_ENDPOINT',complete=True,updates=updates,
        target_visits=result['target_visits'],source_visits=result['source_visits'],
        completed_gradient_visits=actual.get('completed_training_visits'),fresh_RGB16_bit_exact=True)


def panel_snapshot(root):
    root=Path(root);outcomes=[];started=0;seen=set()
    if not root.exists():return dict(planned=None,started=0,terminal=0,successes=0,SLA_queries_closed=0,native_integrations_closed=0)
    p=load(root/'PANEL.private.json');planned=len(p['assignments'])if p else None
    for group in root.glob('*-group-*'):
        rows=load(group/'OUTCOMES.private.json')
        if rows is not None and len(rows)>len(outcomes):outcomes=rows
        runs=group/'runs'
        if runs.exists():
            for run in runs.iterdir():
                if not run.is_dir()or run.name.endswith('-gpu'):continue
                if (run/'SUPERVISOR.private.json').exists():seen.add(run.name)
    started=len(seen)
    return dict(planned=planned,started=started,terminal=len(outcomes),successes=sum(r['task_success']for r in outcomes),
        technical_failures=sum(not r['technical_valid']for r in outcomes),hard_safety_stops=sum(r['hard_safety_stop']for r in outcomes),
        SLA_queries_closed=sum(r['actual_SLA_queries']for r in outcomes),native_integrations_closed=sum(r['actual_native_integrations']for r in outcomes),
        outcomes=[{k:r.get(k)for k in('native_run','layout','restart','task_success','technical_valid','hard_safety_stop',
            'actual_SLA_queries','actual_native_integrations','first_failure','result_sha256','GPU_result_sha256')}for r in outcomes])


def evaluation_state(row,state,lease,result,worker):
    if state.get('LoadState')!='loaded':return 'UNKNOWN_HANDLE'
    if state.get('MainPID')!='0':return 'RUNNING'
    if not lease or lease.get('status')!='CLOSED'or not result or not worker:return 'TERMINAL_UNRECONCILED'
    if lease.get('worker_exit_code')!=worker.get('exit_code')or result.get('exit_code')!=worker.get('exit_code'):return 'TERMINAL_INCONSISTENT'
    if lease.get('spec_sha256')!=row['spec_sha256']:return 'TERMINAL_INCONSISTENT'
    return 'CLOSED0'if worker['exit_code']==0 else'CLOSED_FAILURE_RETAINED'


def run(spec):
    root=Path(spec['registered_root']);source=Path(__file__).resolve().parents[2]
    if (spec['schema']!=SCHEMA or os.getuid()!=2009 or socket.gethostname()!='corelab-4080s'
            or Path(spec['source_root'])!=source or spec['poll_s']!=120):raise ValueError('FINITE_EXISTING_CORE_OBSERVER_ONLY')
    # Identity verification is at this version boundary,not on every poll.
    for row in spec['training_jobs']+[spec['evaluation']]:
        p=Path(row['root'])/'EXECUTION_SPEC.json';v=load(p)
        if sha(p)!=row['spec_sha256']or v['source_sha']!=row['source_sha']:
            raise ValueError('EXISTING_JOB_IDENTITY_CHANGED_NO_DISPATCH')
    started=time.time();polls=0;barrier=False
    while True:
        now=time.time();polls+=1;training=[]
        for row in spec['training_jobs']:
            r=Path(row['root']);value=training_state(row,unit(r),load(r/'control/LEASE.json'),load(r/'RESULT.private.json'),load(r/'WORKER_EXIT.private.json'))
            training.append(dict(run_id=r.name,**value))
        all_closed=all(v['complete']for v in training)
        if all_closed and not barrier:
            write(root/'TRAINING_BARRIER.private.json',dict(epoch=now,training=training,all_current_batch_training_closed=True,
                scientific_matched_comparison=False,planned_unadmitted_jobs=spec['planned_unadmitted_jobs']))
            barrier=True
        e=Path(spec['evaluation']['root']);handle=unit(e);lease=load(e/'control/LEASE.json');result=load(e/'RESULT.private.json');worker=load(e/'WORKER_EXIT.private.json')
        status=evaluation_state(spec['evaluation'],handle,lease,result,worker)
        panels={k:panel_snapshot(v)for k,v in spec['evaluation']['panels'].items()}
        value=dict(schema=SCHEMA,epoch=now,polls=polls,training=training,all_current_batch_training_closed=all_closed,
            evaluation_status=status,evaluation_handle=handle,panels=panels,
            planned_unadmitted_jobs=spec['planned_unadmitted_jobs'],goal_complete=False,
            observer_model_loads=0,optimizer_updates=0,training_visits=0,physical_requests=0)
        write(root/'STATE.private.json',value)
        attention=any(v['status']not in('RUNNING','CLOSED0_ACTUAL_ENDPOINT')for v in training)
        stopped=(root/'STOP.json').exists()or Path(spec['campaign_root'],'STOP.json').exists()
        if status!='RUNNING'or attention or stopped or now>=spec['deadline_epoch']:
            panel_complete=all(v['planned']is not None and v['terminal']==v['planned']for v in panels.values())
            final=dict(value,status='EXISTING_EVALUATION_CLOSED'if status in('CLOSED0','CLOSED_FAILURE_RETAINED')else'MONITOR_NEEDS_ATTENTION',
                evaluation_result=result,monitor_wall_s=now-started,monitor_stopped=stopped,
                monitor_deadline_reached=now>=spec['deadline_epoch'],automatic_experiment_retries=False,
                requested_evaluation_complete=panel_complete,
                automatic_new_experiment_dispatch=False,exit_code=0 if status=='CLOSED0'and all_closed and panel_complete else 1)
            write(root/'RESULT.private.json',final)
            # A completion/attention request is written only after actual
            # handle observation. It is not an assertion of task success.
            write(root/'WAKE_REQUEST.private.json',dict(schema='SLA_BATCH_TERMINAL_OR_ATTENTION_WAKE_V6_1',
                epoch=now,result_sha256=sha(root/'RESULT.private.json'),status=final['status'],
                evaluation_status=status,all_current_batch_training_closed=all_closed,goal_complete=False))
            return final
        time.sleep(min(spec['poll_s'],max(0,spec['deadline_epoch']-time.time())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);args=p.parse_args()
    print(json.dumps(run(json.loads(args.spec.read_bytes())),allow_nan=False))

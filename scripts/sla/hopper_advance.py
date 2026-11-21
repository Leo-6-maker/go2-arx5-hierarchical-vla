"""Advance a single fenced Hopper lineage from actual, terminal PBS and checkpoint receipts."""
import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.sla.hopper_checkpoint import BOUNDARIES
from scripts.sla.hopper_course_decision import decide
from scripts.sla.hopper_engineering_results import select
from scripts.sla.hopper_inputs import PLAN, PARENT, STEPS
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import DEADLINE, pinned
from hopper_pbs import native, pin, previous_cost, register

POPULATION = '9e483c85eb90cc2e9d1da66b9f3f8dff6606c6173b66755bcffcc4c7e5f91279'


def segment(line, selection, population, view, science, seconds):
    chosen = pinned(selection)
    backend = pinned(chosen['candidates'][chosen['winner']])
    if not backend['qualified'] or backend['source'] != science or chosen['source'] != science:
        raise ValueError('Selected backend and source changed')
    trigger_file = line/'curriculum/FIRST_CONTROL_TRIGGER.json'
    trigger = pin(trigger_file) if trigger_file.exists() else None
    main_done = (line/'MAIN/sealed'/f'phase{STEPS:05d}.json').exists()
    role = 'CONTROL2000' if main_done else 'MAIN'
    if main_done and not trigger:
        return None
    pair = pinned(trigger) if trigger else None
    origin = pair['boundary'] if role=='CONTROL2000' else 0
    end = origin+2000 if role=='CONTROL2000' else STEPS
    events = sorted((line/role/'sealed').glob('phase[0-9][0-9][0-9][0-9][0-9].json'))
    prior = pair['common_parent'] if origin else dict(path=str(view/'parent.verified'),sha256=PARENT)
    start = origin
    course = pair['control_course'] if origin else None
    decision = None
    if events:
        event = read(events[-1])
        start, prior, course = event['phase'], event['checkpoint'], event['course']
        if (events[-1].name != f'phase{start:05d}.json' or event['role']!=role or event['plan']!=PLAN
                or event['binding']['source']!=science or event['binding']['role_start']!=origin
                or digest(Path(prior['path'])/'COMPLETE.json')!=prior['sha256']):
            raise ValueError('Last training seal does not match the role')
        if start==end:
            return None
        if start in BOUNDARIES and role=='MAIN':
            decision_row=decide(line,start,pin(events[-1]),view/'train-ready',population)
            decision=pin(line/'curriculum'/f'phase{start:05d}.json')
            course,trigger=decision_row['next_course'],decision_row['first_control_trigger']
    if not origin<=start<end:
        raise ValueError('Unexpected start of training segment')
    # Close the allocation at a planned step with spare time for cold restore, compilation and saving.
    cold = max(1200.,2*backend['first_update_s'])
    warm = max(backend['warm_p90_s'],backend['warm_p50_s']*1.25)
    count = math.floor((seconds-900-cold)*0.8/warm)
    if count < 1:
        raise TimeoutError('Remaining allocation cannot fit a complete update with save margin')
    limit = min([end]+[k for k in BOUNDARIES if k>start and role=='MAIN'])
    return dict(plan=PLAN,parent=PARENT,source=science,role=role,role_start=origin,start=start,stop=min(start+count,limit),
                previous=prior,course=course,first_control_trigger=trigger,boundary_decision=decision,
                selection=selection,population_file=str(population))


def advance(line,view,source,population):
    if digest(population)!=POPULATION:
        raise ValueError('Population changed')
    if time.time()>=DEADLINE-900 or any((p/'STOP.json').exists() for p in (view,line)):
        return dict(status='STOP')
    with closing(OwnerLock(line/'advance.lock')):
        alloc=line/'allocations'
        current=max((p for p in alloc.iterdir() if p.name.isdigit()),key=lambda p:int(p.name))
        if not (current/'RELEASE.json').exists():
            return dict(status='RECONCILE_REGISTRATION',allocation=current.name)
        prior=read(current/'RELEASE.json')['job']
        row=native(prior)
        if row['job_state']!='F':
            return dict(status='WAIT_PBS',job=prior,state=row['job_state'])
        total,eng=previous_cost(line,None)
        if int(row['Exit_status'])!=0 or not (current/'RESULT.private.json').exists():
            return dict(status='RECONCILE_TERMINAL',job=prior,exit=row['Exit_status'])
        recovery=view/'TECHNICAL_RECOVERY.json'
        engineering=Path(read(recovery)['engineering_root'])
        science=read(source/'SOURCE_READY.json')['source']
        result=read(current/'RESULT.private.json')
        kind=read(current/'RESERVATION.json')['group']
        if kind=='training' and result['status'] not in ('SEGMENT_COMPLETE','SAVED_AT_ALLOCATION_OR_STOP'):
            raise ValueError('Unrecognized completed training segment')
        seconds=min(21600,320*3600-total,math.floor(DEADLINE-time.time()))
        if seconds<=900:
            return dict(status='BUDGET_OR_DEADLINE')
        group='training'
        record=None
        selection=engineering/'SELECTION.json'
        if not selection.exists():
            if kind!='engineering' or result['status']!='SAVED_BEFORE_NEXT_CANDIDATE':
                raise RuntimeError('Unresolved engineering selection')
            seconds=min(seconds,8*3600-eng)
            if seconds>900:
                group='engineering'
            else:
                for n in range(1,7):
                    role=f'E{n}'
                    outcome=engineering/(role+'.outcome.json')
                    if not outcome.exists():
                        if (engineering/role).exists():
                            raise RuntimeError('Unresolved candidate at budget exhaustion')
                        write_once(outcome,dict(plan=PLAN,parent=PARENT,source=science,role=role,
                                   qualified=False,status='NO_ENGINEERING_BUDGET',optimizer_updates=0,visits=0))
                select(engineering,science)
                return dict(status='SELECTION_RECONCILED')
        if group=='training':
            record=segment(line,pin(selection),population,view,science,seconds)
            if record is None:
                return dict(status='TRAINING_COMPLETE')
        number=int(current.name)+1
        request=None
        if record:
            requests=line/'segments'
            requests.mkdir(exist_ok=True)
            request=requests/f'{number:03d}.json'
            write_once(request,record)
        job=register(line,view,source,number,group,seconds,request,recovery)
        return dict(status='SUBMITTED',job=job,group=group,allocation=number)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    for k in ('line','view','source','population'):
        cli.add_argument(k,type=Path)
    args=cli.parse_args()
    print(json.dumps(advance(args.line,args.view,args.source,args.population)))

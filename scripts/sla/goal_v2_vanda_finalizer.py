"""Close the known owned PBS graph at the Goal V2 boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


PLAN = '1b2a25b7c1720051a2627d30831fb18130462374'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pin(pin):
    path = Path(pin['path'])
    if sha(path) != pin['sha256']:
        raise ValueError('VANDA_GOAL_FINALIZER_PIN_CHANGED')
    return json.loads(path.read_bytes())


def once(path, value):
    with path.open('x') as output:
        json.dump(value,output,indent=2,allow_nan=False)
        output.flush()
        os.fsync(output.fileno())


def pbs(operation, *args):
    return subprocess.run(['/opt/pbs/bin/'+operation,*args],capture_output=True,text=True,timeout=20)


def verify_job(job, row, config, name):
    if (not re.fullmatch(r'[0-9]+\.stdct-mgmt-[0-9]+',job) or row['project'] != config['project']
            or row['Job_Owner'].split('@')[0] != config['user'] or row['Job_Name'] != name):
        raise PermissionError('VANDA_GOAL_FINALIZER_JOB_IDENTITY_CHANGED')


def snapshot(job, config, name):
    result = pbs('qstat','-xf','-F','json',job)
    if result.returncode:
        raise RuntimeError('PBS_STATUS_UNAVAILABLE')
    row = json.loads(result.stdout)['Jobs'][job]
    verify_job(job,row,config,name)
    return row


def observe(entry, config, base, *, audit):
    job, name = entry['job'],entry['name']
    card_pin = entry['spec']
    card_path = Path(card_pin['path']).resolve(strict=True)
    if not card_path.is_relative_to(base) or card_path.stat().st_uid != config['uid']:
        raise PermissionError('VANDA_GOAL_FINALIZER_CARD_OUTSIDE_OWNED_SCOPE')
    card = read_pin(card_pin)
    if card['job_id'] != job or Path(card['root']).resolve(strict=True) != card_path.parent:
        raise PermissionError('VANDA_GOAL_FINALIZER_CARD_JOB_MISMATCH')
    row = snapshot(job,config,name)
    observed = dict(job=job,name=name,state=row['job_state'],exit_status=row.get('Exit_status'),
                    start_seen='stime' in row,stop_written=False,deleted=False)
    if audit or row['job_state'] in ('F','X'):
        return observed
    stop = Path(entry['stop']).resolve()
    if (not stop.is_relative_to(base) or stop.name != 'STOP.json'
            or stop.parent.stat().st_uid != config['uid']):
        raise PermissionError('VANDA_GOAL_FINALIZER_STOP_OUTSIDE_OWNED_SCOPE')
    if not stop.exists():
        try:
            once(stop,dict(reason='SLA_GOAL_V2_CAMPAIGN_BOUNDARY',epoch=time.time(),plan_sha=PLAN))
            observed['stop_written'] = True
        except FileExistsError:
            pass
    if row['job_state'] in ('Q','H','W') and 'stime' not in row:
        hold = pbs('qhold','-h','u',job)
        if hold.returncode:
            observed['stop_reconciliation'] = 'COULD_NOT_HOLD_UNSTARTED_PBS_JOB'
            return observed
        held = snapshot(job,config,name)
        if held['job_state'] == 'H' and 'u' in held.get('Hold_Types','') and 'stime' not in held:
            deletion = pbs('qdel',job)
            if not deletion.returncode:
                observed['deleted'] = True
            else:
                observed['stop_reconciliation'] = 'USER_HOLD_WITH_STOP_DELETE_UNCONFIRMED'
        else:
            observed['stop_reconciliation'] = 'STARTED_AT_BOUNDARY_STOP_ONLY'
    return observed


def run(config, *, audit):
    campaign = read_pin(config['campaign'])
    root = Path(config['root']).resolve(strict=True)
    base = Path(config['base']).resolve(strict=True)
    local_stop = root/'STOP.json'
    if (os.getuid() != config['uid'] or os.environ.get('USER') != config['user']
            or not root.is_relative_to(base) or root.stat().st_uid != os.getuid()
            or campaign['plan_sha'] != PLAN or campaign['deadline_epoch'] != campaign['t0_epoch']+14*86400):
        raise PermissionError('VANDA_GOAL_FINALIZER_SCOPE_MISMATCH')
    epoch = time.time()
    deadline = epoch >= campaign['deadline_epoch']
    if not audit:
        job = config['job_id']
        if os.environ.get('PBS_JOBID') != job or (not deadline and not local_stop.exists()):
            raise PermissionError('VANDA_FINALIZER_NEEDS_OWN_LIVE_PBS_AND_ACTUAL_BOUNDARY')
        own = snapshot(job,config,config['job_name'])
        if own['job_state'] != 'R' or int(own['Resource_List'].get('ngpus',0)):
            raise PermissionError('VANDA_FINALIZER_REQUIRES_CPU_ONLY_ALLOCATED_JOB')
    result = root/'FINALIZER_ACTUAL.private.json'
    if not audit and result.exists():
        raise FileExistsError(result)
    entries = config['members']
    if len({x['job'] for x in entries}) != len(entries):
        raise ValueError('VANDA_GOAL_FINALIZER_DUPLICATE_JOB')
    states=[]
    for entry in entries:
        try:
            states.append(observe(entry,config,base,audit=audit))
        except (OSError,ValueError,RuntimeError,KeyError) as error:
            states.append(dict(job=entry['job'],name=entry['name'],state='REQUIRES_RECONCILIATION',
                               reason=f'{type(error).__name__}: {error}'))
    actual = dict(schema='SLA_MULTISOURCE_GOAL_V2_VANDA_FINALIZER_V1',site='VANDA',
                  observed_epoch=epoch,deadline_epoch=campaign['deadline_epoch'],audit_only=audit,states=states,
                  reason='DEADLINE' if deadline else 'OPERATOR_STOP' if local_stop.exists() else 'READ_ONLY',
                  optimizer_updates=0,model_generations=0,native_requests=0,native_ticks=0)
    if not audit:
        once(result,actual)
    return actual


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    parser.add_argument('--audit',action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(run(read_pin(dict(path=args.spec,sha256=args.sha256)),audit=args.audit),allow_nan=False))

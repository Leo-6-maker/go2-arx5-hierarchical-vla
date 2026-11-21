"""Verify the named project's site-issued PBS allocation without retaining Budgets credentials."""
import argparse
import ast
import csv
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from scripts.sla.hopper_parent_custody import PLAN, digest, write_once
from scripts.sla.hopper_site import DEADLINE, T0, pinned, site_epoch, wall_seconds

PROJECT = 'CFP04-CF-007'
CLUSTER = 'hopper-m-02'
SELECT = '1:ngpus=1:gpu_model=H200:ncpus=12:mem=225GB'
AMGR = '/opt/am/python/bin/amgr'


def site_prechecks():
    settings = Path('/etc/am.conf').read_text().splitlines()
    matches = [line.split('=', 1)[1].strip() for line in settings
               if re.match(r'^\s*AM_BALANCE_PRECHECK\s*=', line)]
    if matches != ['True']:
        raise PermissionError('The site balance precheck could not be confirmed')


def allocation(row, account, wall, state):
    resources = row['Resource_List']
    if (row['job_state'] != state or row['Job_Owner'].split('@')[0] != account
            or row['project'] != PROJECT or resources['select'] != SELECT
            or int(resources['ngpus']) != 1 or int(resources['ncpus']) != 12
            or wall_seconds(resources['walltime']) != wall):
        raise PermissionError('PBS project, allocation, owner or state differs')
    rate = float(resources['queue_charge_rate'])
    if not math.isfinite(rate) or rate <= 0:
        raise PermissionError('Missing positive site queue charge rate')
    if state == 'R':
        cache = resources['am_job_cache']
        if isinstance(cache, str):
            try:
                cache = json.loads(cache)
            except json.JSONDecodeError:
                cache = ast.literal_eval(cache)
        expected = {'job.ngpus': 1, 'job.ncpus': 12, 'job.walltime': wall, 'job.queue_charge_rate': rate}
        if any(float(cache[k]) != value for k, value in expected.items()):
            raise PermissionError('Site billing hook and assigned resources differ')
    return rate


def capture(line, number, dest):
    from hopper_pbs import native, project_grant
    if os.uname().nodename.split('.')[0] != 'hopper-l-01':
        raise PermissionError('Project evidence requires the Hopper login node')
    site_prechecks()
    folder = line/'allocations'/number
    mount_file = folder/'MOUNT.private.json'
    mount = json.loads(mount_file.read_bytes())
    reservation = pinned(mount['gpu_reservation'])
    job = mount['job']
    if (mount['plan'] != PLAN or mount['T0_epoch'] != T0 or mount['deadline_epoch'] != DEADLINE
            or Path(mount['lineage']).resolve() != line.resolve()
            or reservation['job'] != job or mount['uid'] != os.getuid()):
        raise ValueError('Unexpected project evidence allocation')
    row = native(job)
    wall = reservation['reserved_seconds']
    grant = project_grant(wall)
    rate = allocation(row, grant['account'], wall, 'R')
    raw = subprocess.check_output([AMGR,'report','transaction','-i',job,'-l','-r'],text=True,timeout=30)
    acquired = [v for v in csv.DictReader(StringIO(raw)) if v['transaction_id']==job and
                v['account']==PROJECT and v['transaction_type']=='acquired' and v['type']=='debit'
                and v['service_unit'] in ('gpu_hrs','su')]
    required = wall / 3600 * rate * grant['units_per_gpu_hour_at_queue_rate_one']
    if (sorted(v['service_unit'] for v in acquired) != ['gpu_hrs','su']
            or any(not math.isclose(float(v['amount']),required) for v in acquired)):
        raise PermissionError('Site transaction did not reserve this allocation against the project')
    # This records a real membership period and one charged job. Each later job seeks its own PBS award.
    write_once(dest,dict(plan=PLAN,T0=T0,deadline=DEADLINE,cluster=CLUSTER,line=str(line.resolve()),
                         project=PROJECT,uid=os.getuid(),project_observation=grant,job=job,wall=wall,
                         queue_rate=rate,acquired=acquired,mount_sha256=digest(mount_file)))


def project_period(basis, line, seconds):
    site_prechecks()
    claim = pinned(basis)
    grant = claim['project_observation']
    now = time.time()
    if (claim['plan'] != PLAN or claim['T0'] != T0 or claim['deadline'] != DEADLINE
            or claim['uid'] != os.getuid() or claim['line'] != str(line.resolve())
            or claim['cluster'] != CLUSTER or claim['project'] != PROJECT
            or grant['project'] != PROJECT or grant['account'] != os.environ['USER']
            or not T0 <= grant['observed_epoch'] <= now < min(grant['expiry_epoch'], DEADLINE)
            or grant['valid_from_epoch'] > now or not 0 < seconds <= 21600):
        raise PermissionError('Project period is not valid for this campaign allocation')
    return dict(project=PROJECT,account=grant['account'],valid_from_epoch=grant['valid_from_epoch'],
                expiry_epoch=grant['expiry_epoch'],period_observed_epoch=grant['observed_epoch'],
                source='SITE_PBS_PROJECT_AWARD_REQUIRED',basis=basis,
                units_per_gpu_hour_at_queue_rate_one=grant['units_per_gpu_hour_at_queue_rate_one'])


def running(folder):
    mount = json.loads((folder/'MOUNT.private.json').read_bytes())
    job = os.environ['PBS_JOBID']
    basis = mount['pbs_project_basis']
    claim = pinned(basis)
    grant = pinned(mount['site_grant'])
    reservation = pinned(mount['gpu_reservation'])
    if (job != mount['job'] or job != reservation['job'] or mount['plan'] != PLAN
            or claim['line'] != str(Path(mount['lineage']).resolve()) or mount['uid'] != os.getuid()
            or grant['project'] != PROJECT or grant['account'] != os.environ['USER']
            or claim['project_observation']['account'] != grant['account'] or claim['cluster'] != CLUSTER
            or time.time() >= min(DEADLINE,grant['expiry_epoch'],claim['project_observation']['expiry_epoch'])):
        raise PermissionError('Current compute process differs from the registered project and period')
    rows = json.loads(subprocess.check_output([mount['qstat'],'-f','-F','json',job],text=True,timeout=45))['Jobs']
    row = rows[job]
    wall = reservation['reserved_seconds']
    rate = allocation(row,grant['account'],wall,'R')
    host = os.uname().nodename.split('.')[0]
    if host != row['exec_host'].split('/')[0].split('.')[0] or time.time() >= site_epoch(row['stime']) + wall:
        raise PermissionError('Current process is outside the assigned node or walltime')
    write_once(folder/'PBS_PROJECT_AWARD.json',dict(job=job,project=PROJECT,host=host,wall_seconds=wall,
               queue_rate=rate,site_start=row['stime'],basis_sha=basis['sha256'],site_billing_cache_verified=True))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    commands=p.add_subparsers(dest='operation',required=True)
    c=commands.add_parser('capture')
    c.add_argument('line',type=Path)
    c.add_argument('allocation')
    c.add_argument('output',type=Path)
    commands.add_parser('running').add_argument('folder',type=Path)
    args=p.parse_args()
    os.umask(0o077)
    if args.operation=='capture':
        capture(args.line,args.allocation,args.output)
    else:
        running(args.folder)

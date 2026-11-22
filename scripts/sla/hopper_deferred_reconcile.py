"""Reconcile one timed-out deferred PBS submission without replacing its intent."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import re
import subprocess
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.pi0294.e1.journal import Journal, digest as payload_digest
from scripts.sla.hopper_parent_custody import PLAN, write_once
from scripts.sla.hopper_site import DEADLINE, pinned, site_epoch, wall_seconds

from hopper_deferred_segment import old_state
from hopper_pbs import PBS, PROJECT, native, pin
from hopper_project_admission import CLUSTER, allocation, project_period

JOB_ID = re.compile(r'(\d+)\.hopper-m-02')


def pbs(*arguments, timeout=55):
    return subprocess.check_output([str(PBS / arguments[0]), *arguments[1:]],
                                   text=True, timeout=timeout)


def history(intent, created):
    server = dict(line.strip().split(' = ', 1) for line in pbs('qstat', '-Bf', CLUSTER).splitlines()
                  if ' = ' in line)
    if (server['server_state'] != 'Active' or server['job_history_enable'] != 'True'
            or wall_seconds(server['job_history_duration']) <= time.time() - created):
        raise PermissionError('PBS history does not cover the timed-out request')
    ids = pbs('qselect', '-x', '-u', os.environ['USER'], '-q', '@' + CLUSTER).split()
    if not all(JOB_ID.fullmatch(job) for job in ids) or intent['predecessor'] not in ids:
        raise ValueError('Incomplete project-owner job history')
    minimum = int(JOB_ID.fullmatch(intent['predecessor'])[1])
    candidates = [job for job in ids if int(JOB_ID.fullmatch(job)[1]) >= minimum]
    rows = json.loads(pbs('qstat', '-xf', '-F', 'json', *(job + '@' + CLUSTER for job in candidates)))['Jobs']
    if (set(rows) != set(candidates) or any(row['Job_Owner'].split('@')[0] != os.environ['USER']
                                          for row in rows.values())):
        raise ValueError('Unreconciled PBS job history')
    same = [job for job, row in rows.items() if row['Job_Name'] == intent['name']
            or intent['pbs']['path'] in str(row.get('Submit_arguments', ''))]
    if same:
        raise RuntimeError('Matching job already in PBS history: ' + ', '.join(same))
    # Also ask the scheduler directly by exact name, independent of the job-id range.
    if pbs('qselect', '-x', '-u', os.environ['USER'], '-N', intent['name'], '-q', '@' + CLUSTER).strip():
        raise RuntimeError('Matching name already in PBS history')
    return dict(scheduler=CLUSTER, history_enabled=True,
                history_duration=server['job_history_duration'], queried_jobs=len(ids),
                latest_jobs=candidates, matching_jobs=[], observed_epoch=time.time())


def reconcile(folder, submit):
    if (os.uname().nodename.split('.')[0] != 'hopper-l-01' or os.environ.get('PBS_JOBID')
            or os.environ.get('PBS_SERVER') != CLUSTER):
        raise PermissionError('Requires the original Hopper login owner and server')
    folder = folder.resolve(strict=True)
    line = folder.parents[1]
    with closing(OwnerLock(line/'advance.lock')), closing(OwnerLock(line/'submit.lock')):
        intent_file = folder/'INTENT.json'
        intent = read(intent_file)
        journal = Journal(folder/'journal', folder.name)
        failed = read(journal.path('qsub'))
        previous, mount, request, reserved = old_state(line, intent['ordinal'])
        seconds, pred = intent['wall_seconds'], intent['predecessor']
        if (intent['plan'] != PLAN or intent['project'] != PROJECT or intent['group'] != 'training'
                or folder != line/'allocations'/f"{intent['ordinal']+1:03d}"
                or intent['name'] != f"slachv1t{intent['ordinal']+1:02d}"
                or intent['pbs'] != pin(folder/'job.pbs') or intent['previous_mount'] != pin(previous/'MOUNT.private.json')
                or intent['grant'] != pin(folder/'GRANT.json') or intent['source'] != mount['source_revision']
                or intent['project_basis'] != mount['pbs_project_basis'] or intent['deadline'] != DEADLINE
                or pred != mount['job'] or not 900 < seconds <= 21600
                or failed['status'] != 'INTENT' or failed['payload_sha256'] != payload_digest(intent)
                or any(pin(Path(p['path'])) != p for p in [intent['deferred_worker'], *intent['modules']])
                or any((folder/name).exists() for name in ('JOB.json','RESERVATION.json','RELEASE.json'))
                or any((p/'STOP.json').exists() for p in (line, previous, folder, Path(mount['custody'])))
                or journal.path('qsub-reconciled').exists()):
            raise PermissionError('Unresolved submission differs from its pinned predecessor or authority')
        prior = []
        for p in sorted(folder.parent.iterdir()):
            if p in (folder, previous) or not (p/'INTENT.json').is_file():
                continue
            reservation = read(p/'RESERVATION.json')
            settled = read(p/'PBS_FINISHED.json')
            if (settled['job'] != reservation['job'] or settled['group'] != reservation['group']
                    or settled['GPU_seconds'] < 0):
                raise ValueError('Prior site GPU accounting changed')
            prior.append(settled)
        upper = sum(v['GPU_seconds'] for v in prior) + reserved['reserved_seconds']
        engineering = sum(v['GPU_seconds'] for v in prior if v['group'] == 'engineering')
        period = project_period(intent['project_basis'], line, seconds)
        grant = pinned(intent['grant'])
        row = native(pred)
        allocation(row, grant['account'], reserved['reserved_seconds'], 'R')
        if (upper != intent['conservative_prior_GPU_seconds'] or upper+seconds > 320*3600
                or grant['project'] != PROJECT or grant['account'] != os.environ['USER']
                or site_epoch(row['stime']) + reserved['reserved_seconds'] + seconds >= min(
                    period['expiry_epoch'], grant['expiry_epoch'], DEADLINE)):
            raise PermissionError('Original reservation, wall time or project term no longer fits')
        evidence = dict(intent=pin(intent_file), unknown=pin(journal.path('qsub')),
                        predecessor=pred, projected_previous_stop=request['stop'], gpu_upper_seconds=upper,
                        history=history(intent, failed['created_epoch']), queue='auto@' + CLUSTER)
        if not submit:
            return dict(status='RECONCILED_NO_PBS_JOB', new_PBS_jobs=0, **evidence)
        write_once(folder/'RECONCILE.private.json', evidence)

        def launch():
            jid = pbs('qsub', '-h', '-q', evidence['queue'], '-W', 'depend=afterok:' + pred,
                      str(folder/'job.pbs'), timeout=75).strip()
            if not JOB_ID.fullmatch(jid):
                raise RuntimeError('Second submission outcome unknown, preserve both intents')
            write_once(folder/'JOB.json', dict(job=jid, name=intent['name']))
            return jid

        job = journal.once('qsub-reconciled', evidence, launch)
        row = native(job)
        rate = allocation(row, grant['account'], seconds, 'H')
        if (row['Job_Name'] != intent['name'] or not row.get('depend', '').startswith('afterok:' + pred)
                or ('balance_gpu_units' in grant and rate * grant['required_units_at_queue_rate_one'] >
                    min(grant['balance_gpu_units'], grant['balance_su']))):
            raise PermissionError('Reconciled PBS job does not match the original identity and reservation')
        write_once(folder/'RESERVATION.json', dict(plan=PLAN, job=job, group='training', reserved_gpu_count=1,
            reserved_seconds=seconds, all_prior_seconds=upper, engineering_prior_seconds=engineering,
            accounting='CONSERVATIVE_ACTIVE_UPPER_BOUND'))
        release = dict(job=job, pbs_name=intent['name'], dependency='afterok:' + pred, epoch=time.time())

        def release_job():
            subprocess.run([str(PBS/'qrls'), job], check=True, timeout=60)
            return release

        got = journal.once('release', dict(job=job, intent=evidence['intent']), release_job)
        write_once(folder/'RELEASE.json', got)
        observed = native(job)
        return dict(status='RECONCILED_PBS_DEPENDENT_TRAINING_REGISTERED', **got,
                    observed_state=observed['job_state'], observed_dependency=observed.get('depend'),
                    reserved_seconds=seconds)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('folder', type=Path)
    cli.add_argument('--submit', action='store_true')
    args = cli.parse_args()
    os.umask(0o077)
    print(json.dumps(reconcile(args.folder, args.submit)))

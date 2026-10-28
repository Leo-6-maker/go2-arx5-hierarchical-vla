"""Prepare or explicitly activate stopped premodel record handoff; never qsub.

After explicit user amendment, owner calls --activate, then original submit.py.
Both original accepted job and closed allocation remain exact audit evidence.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def prepare(activate=False):
    script='ACTIVATE='+repr(activate)+'\n'+r'''
import fcntl,hashlib,json,shutil,subprocess,time
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286');old_job='655971.hopper-m-02'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
amendment=r/'design/ENGINEERING_LEASE_AMENDMENT.json'
a=json.loads(amendment.read_text()) if amendment.exists() else {}
authorized=(a.get('status')=='USER_APPROVED_SEQUENTIAL_PREMODEL_ENGINEERING_REPLACEMENT' and a.get('launch_allowed') is True)
if not authorized:
 assert not ACTIVATE,'Explicit user amendment is pending; no record handoff or submission'
 value=dict(status='NOT_AUTHORIZED_ONE_ACCEPTED_LEASE_CAP_REMAINS',launch_allowed=False,previous_PBS_JOBID=old_job,technical_recovery_ordinal=1,old_budget_transfer=False,new_PBS_submissions=0)
 (r/'evidence/ENGINEERING_RECOVERY_NOT_AUTHORIZED.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value));raise SystemExit(0)
assert a['previous_PBS_JOBID']==old_job and a['maximum_accepted_total']==2 and a['replacement_leases_allowed']==1
assert isinstance(a['explicit_user_approval'],str) and a['explicit_user_approval'].strip() and 0<a['approval_received_epoch']<=time.time()
assert a['canonical']==6072813602 and a['release']=='eae6dbc1f9281b204a4e28e5f7c100d705837318'
assert time.time()<a['arm_start_stop_epoch']
with (r/'runtime/SUBMISSION.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 handoff=r/'runtime/ENGINEERING_RECOVERY_1_HANDOFF.json'
 if handoff.exists():
  value=json.loads(handoff.read_text());print(json.dumps(dict(status='ADOPT_EXISTING_HANDOFF_NO_SECOND_HANDOFF',receipt=value)));raise SystemExit(0)
 result=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qstat','-xf',old_job],capture_output=True,text=True)
 assert result.returncode==0 and 'job_state = F' in result.stdout and 'Exit_status = 1' in result.stdout
 failure=json.loads((r/'evidence/655971_MOUNT_STAMP_FAILURE.json').read_text())
 assert failure['completed_updates']==0 and not failure['checkpoint_dirs'] and not failure['quality_ledger']
 ledger=r/'learning/completed_updates.jsonl';assert not ledger.exists() or not ledger.read_text().strip()
 assert not list((r/'learning').glob('*/**/checkpoint-*'))
 proof=r/'evidence/655971_BYTE_REPAIR_CPU_READBACK.json';repair=json.loads(proof.read_text())
 assert repair['status']=='PASS_STOPPED_HOPPER_MOUNT_INDEPENDENT_FULL_BYTE_REPAIR_CPU_READBACK' and repair['old_allocation_unchanged']
 allocation=r/'runtime/TRAINING_ALLOCATION.json';submission=r/'runtime/JOB_SUBMISSION.json'
 assert json.loads(allocation.read_text())['PBS_JOBID']==old_job and json.loads(submission.read_text())['PBS_JOBID']==old_job
 assert (r/'runtime'/('ALLOCATION_CLOSED_'+old_job+'.json')).exists()
 processes=r/'runtime/GPU_PROCESSES.jsonl';rows=[json.loads(x) for x in processes.read_text().splitlines()]
 ended={(v['pid'],v['start_epoch']) for v in rows if v['event']=='end'}
 assert all((v['pid'],v['start_epoch']) in ended for v in rows if v['event']=='start')
 value=dict(status='PREPARED_EXPLICITLY_AUTHORIZED_HANDOFF_NOT_SUBMITTED',previous_PBS_JOBID=old_job,technical_recovery_ordinal=1,
  previous_reported_scheduler_occupancy_s=17,previous_updates=0,previous_queries=0,amendment_sha256=sha(amendment),CPU_repair_sha256=sha(proof),
  historical_records={str(p.relative_to(r)):sha(p) for p in (allocation,submission,processes)},formal_cap=2048,all_updates_cap=2112,canary_cap=32,technical_rollback_cap=32,new_PBS_submissions=0)
 if ACTIVATE:
  capacity=json.loads((r/'design/CAPACITY_ADMISSION.json').read_text())
  assert capacity['status']=='OWNER_ADMITTED_PI0286_OUTPUT_HEADROOM' and capacity['remaining_headroom_bytes']>=capacity['required_peak_bytes'] and time.time()<=capacity['valid_until_epoch']
  archive=r/'runtime/failed-engineering655971';archive.mkdir(exist_ok=True)
  for p in (allocation,submission):
   dst=archive/p.name
   if dst.exists():assert sha(dst)==sha(p)
   else:shutil.copyfile(p,dst)
   assert sha(dst)==value['historical_records'][str(p.relative_to(r))];p.unlink()
  value.update(status='ACTIVATED_EXPLICIT_USER_AMENDMENT_STOPPED_RECORD_HANDOFF_OWNER_SUBMIT_READY',historical_directory=str(archive),GPU_process_ledger_unchanged=True,
   occupancy_accounting='Retain17s reported oldscheduler occupancy in total PI0286 reconciliation; never reset closed process intervals',owner_next_command='py -3.12 scripts/pi0286_execution/submit.py')
  handoff.write_text(json.dumps(value,indent=2)+'\n')
 (r/'evidence/ENGINEERING_RECOVERY_PREPARATION.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''
    value=json.loads(remote('hop',script));p=Path('reports/PI0286/preparation/ENGINEERING_RECOVERY_PREPARATION.json');p.write_text(json.dumps(value,indent=2)+'\n');return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--activate',action='store_true');a=p.parse_args();print(json.dumps(prepare(a.activate)))

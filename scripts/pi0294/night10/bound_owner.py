"""One existing owner, user-supplied finite DAG and original native workers."""
import fcntl,json,os,sys,time
from pathlib import Path
P=Path(__file__).parent
sys.path[:0]=[str(P/'source'),str(P/'source/src')]
from scripts.pi0294.night10.supervisor import Supervisor,atomic,check_plan
import native_tasks as n
def main():
 plan=json.loads((P/'BOUND_PLAN.json').read_bytes());config=n.CONFIG
 if os.getuid()!=2009 or os.uname().nodename!='corelab-4080s':raise PermissionError('EXACT_AUTHORIZED_NATIVE_OWNER')
 for name,pin in json.loads((P/'CAPSULE_SEAL.json').read_bytes())['files'].items():
  if n.sha(P/name)!=pin:raise PermissionError('EXACT_NIGHT10_BOUND_CAPSULE:'+name)
 if int(n.unit(n.M)['MainPID']):raise PermissionError('OLD_MANAGER_HAS_NOT_RELINQUISHED_CUSTODY')
 lock=(Path(config['legacy_root'])/'OWNER.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  n.preflight();runner=Supervisor(plan,P/'runtime')
  atomic(P/'ACTUAL_OWNER_STARTED.json',dict(PID=os.getpid(),proc_start=int(Path('/proc/self/stat').read_text().split()[21]),plan_sha256=n.sha(P/'BOUND_PLAN.json'),epoch=time.time(),source='30979b4edddbf2355b46a46dbb5d7db7e1546bb6',legacy_global_owner_lock_held=True,old_manager_mainpid=0,no_SSH_KRB_Vanda_dependency=True))
  code=runner.run(poll_s=3)
  # The user requires a review even when the dispatch window leaves branches unrun.
  # This reads only completed original receipts, never starts late GPU work.
  if not(P/'receipts/policy_review.json').exists():n.policy()
  summary=json.loads((P/'runtime/SUMMARY.json').read_bytes());summary['policy_review']=json.loads((P/'receipts/policy_review.json').read_bytes())
  atomic(P/'SUMMARY.json',summary)
  (P/'SUMMARY.txt').open('x').write('\n'.join(['Core Night10 actual results',str({k:v['state']for k,v in summary['tasks'].items()}),'provisional_policy='+str(summary['policy_review']['provisional_policy']),'Original Phase4096:1 success/4 valid/6 unrun. Shared/target success is not implied.'])+'\n')
  return code
 finally:lock.close()
if __name__=='__main__':raise SystemExit(main())

"""Register one sealed target FIT4 bundle and submit one held PBS A40 job.

Uses the existing registry, canonical ledger, Journal and onceIO. Lost qsub
or release replies require independent PBS readback; this entry never retries.
"""
import argparse,ast,json,os,re,subprocess,time,uuid
from pathlib import Path
from contextlib import closing


def project():
    def query(args):
        q=subprocess.run(['/opt/am/python/bin/amgr']+args,capture_output=True,text=True,timeout=30)
        if q.returncode:raise PermissionError('TARGET_ACTUAL_AMGR_ENTITLEMENT:'+q.stderr[-800:])
        return q.stdout
    rows=json.loads(query(['ls','project','-n','CFP04-CF-007','-j']))
    balance=ast.literal_eval(query(['checkbalance','project','-n','CFP04-CF-007']))
    if len(rows)!=1:raise PermissionError('TARGET_ONE_ACTUAL_PROJECT')
    from datetime import datetime
    from zoneinfo import ZoneInfo
    day=datetime.now(ZoneInfo('Asia/Singapore')).date().isoformat();p=rows[0]
    cpu=float(balance.get('cpu_hrs',0));gpu=float(balance.get('gpu_hrs',0))
    if not (p['name']=='CFP04-CF-007'and p['active']and os.environ['USER']in p['users']
            and p['start_date']<=day<=p['end_date']and cpu>0 and gpu>0):
        raise PermissionError('TARGET_ACTUAL_PROJECT_DATE_MEMBER_AND_BALANCE')
    return dict(project=p,cpu_hrs=cpu,gpu_hrs=gpu,observed_epoch=time.time(),
                source='ACTUAL_AMGR_PROJECT_AND_BALANCE')


def settled_predecessor(owner):
    """Accept the terminal status actually written by the existing reconciler."""
    if owner is None or owner.get('status')=='CLOSED_INDEPENDENT_PBS_TERMINAL':
        return True
    if owner.get('status')!='CLOSED':return False
    from go2arx5_vla.world_model.storage import digest
    root=Path(owner['root']);receipt=root/'PBS_TERMINAL_ACTUAL.json'
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    terminal=json.loads(receipt.read_bytes())
    return (root.resolve()==root and lease.get('status')=='CLOSED'
        and owner.get('job_id')==lease.get('job_id')==terminal.get('job_id')
        and lease.get('actual_PBS_terminal')==terminal.get('PBS_state')=='F'
        and type(terminal.get('PBS_exit'))is int
        and owner.get('PBS_exit_code')==lease.get('PBS_exit_code')==terminal['PBS_exit']
        ==lease.get('worker_exit_code')==terminal.get('native_exit')
        and owner.get('PBS_terminal_receipt_sha256')
        ==lease.get('PBS_terminal_receipt_sha256')==digest(receipt)
        and lease.get('result_sha256')==digest(root/'ALLOCATION_NATIVE_FINAL.json'))


def submit(request):
    from go2arx5_vla.world_model import target_gr00t_profile_v1 as profile
    from go2arx5_vla.world_model.target_gr00t_allocation_v1 import pbs_script
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot,once
    from scripts.pi0294.multitask.shared_site import PROJECT_SCRATCH,project_quota
    need=profile.need;c=dict(request);root=Path(c['root']);data=Path(c['bindings']['dataset'])
    need(os.getuid()==22182 and not os.environ.get('PBS_JOBID') and not root.exists()
         and root.parent.resolve()==root.parent and root.is_relative_to(Path(PROJECT_SCRATCH)),
         'FRESH_PROJECT_ROOT_LOGIN_SUBMISSION_ONLY')
    need(c['profile']==profile.PROFILE and c['absolute_deadline_epoch']>time.time()+3600,
         'INDEPENDENT_FINITE_NONZERO_TRAINING_SCOPE')
    manifest=json.loads((data/'DATASET_MANIFEST.json').read_bytes())
    admission=json.loads((data/'PROCESSOR_TRAINING_ADMISSION.json').read_bytes())
    reconstructed=json.loads((data/'PROCESSOR_VANDA_RECONSTRUCTION.json').read_bytes())
    mode=c.get('allocation_mode','QUALIFY_FIT4')
    need(mode in ('QUALIFY_FIT4','FORMAL_SLICE','RESTORE_ONLY'),'DECLARED_ALLOCATION_STAGE')
    if mode=='FORMAL_SLICE':
        from go2arx5_vla.world_model.target_gr00t_formal_v1 import validate_request
        validate_request(c,manifest)
    need(digest(data/'DATASET_MANIFEST.json')==c['dataset_sha256']
         and digest(data/'statistics.json')==c['statistics_sha256']
         and (manifest['bundle_kind']=='FORMAL' if mode=='FORMAL_SLICE' else
              manifest['bundle_kind']=='FIT4_DIAGNOSTIC' and len([r for r in manifest['episodes']if r['split']=='TRAIN'])==4)
         and manifest['WM_or_HOLD_or_diagnostic_relabelled']is False
         and admission['status']=='PASS_ACTUAL_OFFICIAL_PROCESSOR_ROUNDTRIP'
         and admission['manifest_sha256']==c['dataset_sha256']
         and admission['statistics_sha256']==c['statistics_sha256']
         and reconstructed['status']=='PASS_ACTUAL_CROSS_SITE_TARGET_PROCESSOR_RECONSTRUCTION'
         and reconstructed['manifest_sha256']==c['dataset_sha256']
         and reconstructed['statistics_sha256']==c['statistics_sha256']
         and reconstructed['original_Core_admission_sha256']==digest(data/'PROCESSOR_TRAINING_ADMISSION.json'),
         'REAL_SEALED_FIT4_AND_ORIGINAL_PROCESSOR')
    for name,pin in c['dataset_files'].items():need(digest(data/name)==pin,'EXACT_PRIVATE_DATA_TRANSFER:'+name)
    for name,pin in c['source_files'].items():need(digest(Path(c['source_root'])/name)==pin,'EXACT_SOURCE_TRANSFER:'+name)
    need(all((data/name).is_file()and c['dataset_files'].get(name)==digest(data/name)
             for name in ('DATASET_MANIFEST.json','statistics.json','PROCESSOR_TRAINING_ADMISSION.json','PROCESSOR_VANDA_RECONSTRUCTION.json')),
         'ALL_MANDATORY_DATA_TRANSFER_PINS')
    qpath=Path('/home/svu/vanda/quotas/CFP04-CF-007.proj_scratch.txt')
    quota=project_quota(qpath.read_text(),user=os.environ['USER'],report_epoch=qpath.stat().st_mtime,observed_epoch=time.time())
    need(quota['hard_bytes']-quota['used_bytes']>c['predicted_new_peak_bytes']+20*1024**3,
         'ACTUAL_PROJECT_QUOTA_HEADROOM')
    q=subprocess.run(['/opt/pbs/bin/qselect','-u',os.environ['USER'],'-N','wm02_target_gr00t'],
                     text=True,capture_output=True,timeout=20)
    need(q.returncode==0 and not q.stdout.strip(),'NO_EXISTING_TARGET_PBS_OR_UNKNOWN_QUEUE')
    receipt=project();need(receipt['gpu_hrs']>=c['wall_s']/3600
        and receipt['cpu_hrs']>=8*c['wall_s']/3600,'ACTUAL_ENTITLEMENT_FOR_REQUEST')
    with closing(OwnerLock(c['ledger_lock'])):
        ledger,before=snapshot(c['ledger']);owner=ledger.get('TARGET_GR00T_V1_owner')
        need(settled_predecessor(owner),
             'EXISTING_UNIQUE_TARGET_PRODUCER_MUST_BE_SETTLED')
        reg,rpin=snapshot(c['registry'])
        for e in reg['entries']:
            if e.get('scope')==profile.PROFILE:
                need(json.loads(Path(e['local_lease']).read_bytes())['status']=='CLOSED',
                     'NO_UNSETTLED_TARGET_LEASE')
        need(not any(e.get('path')==str(root)for e in reg['entries']),'FRESH_TARGET_REGISTRY_ROW')
        root.mkdir();(root/'control').mkdir()
        c.update(project_receipt=receipt,submission_epoch=time.time(),submission_nonce=uuid.uuid4().hex)
        write(root/'ALLOCATION_CONFIG.json',c)
        script=root/'job.pbs';script.write_text(pbs_script(root,c['source_root'],c['python'],c['wall_s']))
        write(root/'control/LEASE.json',dict(status='REGISTERED_NOT_SUBMITTED',profile=profile.PROFILE,
            config_sha256=digest(root/'ALLOCATION_CONFIG.json'),deadline_epoch=c['absolute_deadline_epoch'],
            GR00T_updates=('FORMAL_TO_'+str(c['formal_target_endpoint'])+'_FROM_'+str(c['start_step'])if mode=='FORMAL_SLICE'
                else 'FULL_ACTION_CAPACITY2_PLUS_INDEPENDENT_RESUME1_ONLY'if mode=='RESTORE_ONLY'
                else 'CAPACITY2_PLUS_INDEPENDENT_RESUME1_PLUS_FIT4_200_OR500')))
        reg['entries'].append(dict(role='TARGET-GR00T-A40-ALLOCATION',path=str(root),scope=profile.PROFILE,
            local_lease=str(root/'control/LEASE.json'),owner='EXISTING_LOCAL_PC',executor_node='VANDA_A40',source_sha=c['source_sha']))
        once(Path(c['registry']),reg,root.name+'-REGISTER',expected_before_sha256=rpin)
        ledger['TARGET_GR00T_V1_owner']=dict(root=str(root),generation_id=c['generation_id'],
            submission_nonce=c['submission_nonce'],status='REGISTERED_UNIQUE_TARGET_CANDIDATE',
            dataset_sha256=c['dataset_sha256'],source_sha=c['source_sha'],profile=profile.PROFILE)
        once(Path(c['ledger']),ledger,root.name+'-OWNER-REGISTER',expected_before_sha256=before)
    journal=Journal(root/'control/journal',root.name)
    def qsub():
        args=['/opt/pbs/bin/qsub','-h','-v','PV1_REQUEST_ID='+c['generation_id']+
              ',PV1_SUBMISSION_NONCE='+c['submission_nonce'],str(script)]
        write(root/'PBS_SUBMIT_INTENT.json',dict(command=args,epoch=time.time(),automatic_retry=False))
        q=subprocess.run(args,capture_output=True,text=True,timeout=30)
        write(root/'PBS_QSUB_ACTUAL.json',dict(exit_code=q.returncode,stdout=q.stdout,stderr=q.stderr))
        need(q.returncode==0 and re.fullmatch(r'[0-9]+\.[A-Za-z0-9_.-]+',q.stdout.strip()),
             'QSUB_OUTCOME_REQUIRES_INDEPENDENT_READBACK_NEVER_RESEND')
        return dict(job_id=q.stdout.strip(),submitted_epoch=time.time(),held=True)
    submitted=journal.once('SUBMIT',dict(config_sha256=digest(root/'ALLOCATION_CONFIG.json')),qsub)
    jid=submitted['job_id']
    with closing(OwnerLock(c['ledger_lock'])):
        lease,pin=snapshot(root/'control/LEASE.json');lease.update(status='PBS_SUBMITTED',job_id=jid)
        once(root/'control/LEASE.json',lease,root.name+'-PBS-BIND',expected_before_sha256=pin)
        ledger,pin=snapshot(c['ledger']);ledger['TARGET_GR00T_V1_owner'].update(status='PBS_SUBMITTED',job_id=jid)
        once(Path(c['ledger']),ledger,root.name+'-PBS-OWNER-BIND',expected_before_sha256=pin)
    def release():
        q=subprocess.run(['/opt/pbs/bin/qrls',jid],text=True,capture_output=True,timeout=20)
        write(root/'PBS_RELEASE_ACTUAL.json',dict(exit_code=q.returncode,stdout=q.stdout,stderr=q.stderr))
        need(q.returncode==0,'RELEASE_UNKNOWN_READ_BACK_DONT_RETRY')
        return dict(job_id=jid,release_returncode=0)
    journal.once('RELEASE',dict(job_id=jid),release)
    return dict(status='ONE_TARGET_A40_JOB_SUBMITTED_AND_RELEASED',**submitted,
        config_sha256=digest(root/'ALLOCATION_CONFIG.json'),dataset_sha256=c['dataset_sha256'],
        source_sha=c['source_sha'],actual_updates_observed=0,model_loads_observed=0,
        automatic_resubmit=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True)
    a=p.parse_args(argv);print(json.dumps(submit(json.loads(a.request.read_bytes()))));return 0

if __name__=='__main__':raise SystemExit(main())

"""One fresh Vanda account view after the already-adopted authority handoff.

Unavailable Hopper is a fenced historical origin, never a fresh empty queue.
This is a campaign view, not a budget grant, scheduler admission or task PASS.
"""
from copy import deepcopy
import hashlib
import json
import math
import re
import subprocess
import time

from . import dual_platform as dual, vanda_control_handoff as handoff
from .alternate_accounting import digest
from .owner_transport import ReadOnlyFailure

SCHEMA='WDR1_VANDA_ONLY_CURRENT_CAMPAIGN_VIEW_V2'
PASS='PASS_VANDA_ONLY_CAMPAIGN_VIEW'
AUTHORITY='WDR1_VANDA_CURRENT_AUTHORITY_READBACK_V1'
QUEUE='WDR1_VANDA_COMPLETE_ACCOUNT_QUEUE_READBACK_V1'
QUEUE_FAILURE='WDR1_VANDA_NATIVE_QUEUE_READ_FAILURE_V1'
TERMINAL_HISTORY='WDR1_VANDA_FRESH_TERMINAL_HISTORY_READBACK_V1'
PLATFORM='VANDA_A40'

def need(ok,why):
    if not ok:raise PermissionError('VANDA_CAMPAIGN_'+why)

def fresh(epoch,now):
    need(all(type(v) in (int,float) and math.isfinite(v) for v in (epoch,now))
         and 0<=now-epoch<=120,'FRESH_REAL_CURRENT_VANDA_OBSERVATION')

def sha(raw):return hashlib.sha256(raw).hexdigest()

def authority(h,d,proof,*,now_epoch):
    fresh(proof.get('observed_epoch'),now_epoch)
    marker=h.get(handoff.FIELD,{})
    need(proof.get('schema')==AUTHORITY and proof.get('physical_platform')==PLATFORM
         and proof.get('user')=='e1192548' and proof.get('active_registered_control_lease') is True
         and proof.get('parent_STOP') is False and proof.get('control_STOP') is False
         and proof.get('single_mutation_owner') is True,'REAL_REGISTERED_VANDA_AUTHORITY')
    need(marker.get('schema')=='WDR1_VANDA_SINGLE_CONTROLLER_HANDOFF_V1'
         and marker.get('authority_release')==handoff.RELEASE
         and marker.get('grant_sha256')==handoff.GRANT_SHA
         and marker.get('origin_global_sha256')==handoff.GLOBAL_SHA
         and marker.get('origin_DEC1_sha256')==handoff.DEC1_SHA
         and marker.get('origin_access')=='UNAVAILABLE_FENCED_BY_HUMAN_DIRECTION'
         and marker.get('origin_is_fresh_native_read') is False
         and marker.get('old_parent_mutations_permitted') is False
         and marker.get('old_parent_reconciliation_required_when_available') is True
         and marker.get('new_control_platform')==PLATFORM and marker.get('one_mutation_owner') is True,
         'ALREADY_ADOPTED_EXACT_VANDA_HANDOFF_NOT_UNAVAILABLE_HOPPER_SNAPSHOT')
    need(marker.get('journal')==dict(read_calls=542,failed_reads_preserved=1,heartbeat_calls=255,
         Vanda_release_calls=2,inferred_Hopper_writes_after_snapshot=0,unresolved_effect_calls=0),
         'COMPLETE_STOPPED_OWNER_JOURNAL_NO_UNRESOLVED_WRITE')
    need(h.get('unknown_effects')==h.get('pending_updates')==[],'NO_UNKNOWN_GLOBAL_EFFECT')
    if d is not None:
        need(d.get(handoff.FIELD)==marker and d.get('unknown_effects')==[]
             and h.get('campaign_run_uuid')==d.get('campaign_run_uuid'),'SAME_CURRENT_PAIRED_AUTHORITY')
    failed=[r for r in h['PBS_submissions'] if r.get('job_id')==handoff.failed.JOB]
    need(len(failed)==1 and failed[0].get('request_sha256')==handoff.failed.REQUEST_SHA
         and failed[0].get('source_sha')==handoff.failed.SOURCE and failed[0].get('status')=='NATIVE_TERMINAL'
         and failed[0].get('native_exit_status')==1 and failed[0].get('committed_optimizer_updates')==922
         and failed[0].get('query_attempts')==17 and failed[0].get('conservative_H200_s')==8481
         and failed[0].get('conservative_CPU_s')==11023,'EXACT_MIGRATED_FAILED2970_COSTS_ALREADY_ONCE_SETTLED')
    # These are the old origin's actual registered terminal classifications.
    # They do not report its present account queue or unrelated account jobs.
    history=[];seen=set()
    for item in h['PBS_submissions']:
        jid=item.get('job_id')
        if not jid:
            need(item['status'] in dual.TERMINAL,'NO_UNBOUND_ACTIVE_LEDGER_INTENT')
            continue
        need(jid not in seen,'UNIQUE_GLOBAL_NATIVE_HANDLES');seen.add(jid)
        if re.fullmatch(r'[0-9]+\.hopper-m-02',jid):
            need(item.get('native_platform') in (None,'HOPPER_H200')
                 and item['status'] in dual.TERMINAL and type(item.get('native_exit_status')) is int,
                 'ALL_OLD_HOPPER_CAMPAIGN_JOBS_ALREADY_TERMINAL')
            history.append(dict(job_id=jid,status=item['status'],native_exit_status=item['native_exit_status']))
        else:
            need(re.fullmatch(r'[0-9]+\.stdct-mgmt-02',jid) is not None
                 and item.get('native_platform')==PLATFORM,'KNOWN_VANDA_NATIVE_HANDLE_AND_PLATFORM')
    return history

def collect_queue(*,user='e1192548',runner=subprocess.run,clock=time.time):
    """Two bounded own-account reads; failures carry complete diagnostics."""
    need(user=='e1192548','EXACT_REGISTERED_ACCOUNT')
    def call(argv,timeout):
        try:r=runner(argv,capture_output=True,text=True,timeout=timeout)
        except (subprocess.TimeoutExpired,TimeoutError,ConnectionError) as exc:
            error=ReadOnlyFailure('VANDA_QUEUE_READ_TIMEOUT_NOT_EMPTY_OR_TERMINAL')
            error.diagnostic=dict(argv=argv,error_type=type(exc).__name__,
                stdout=getattr(exc,'stdout',None),stderr=getattr(exc,'stderr',None));raise error from exc
        if r.returncode or r.stderr:
            # Actual PBS clients can return0 with "cannot connect" on stderr
            # and no IDs. That is UNKNOWN, never a valid empty queue.
            error=ReadOnlyFailure('VANDA_QUEUE_READ_DIAGNOSTIC_NOT_EMPTY_OR_TERMINAL')
            error.diagnostic=dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);raise error
        return dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    select=call(['/opt/pbs/bin/qselect','-u',user],10)
    ids=select['stdout'].split()
    need(len(ids)<=4096 and len(ids)==len(set(ids))
         and all(re.fullmatch(r'[0-9]+\.stdct-mgmt-02',v) for v in ids),'COMPLETE_OWN_VANDA_QSELECT_HANDLES')
    stat=call(['/opt/pbs/bin/qstat','-f','-F','json',*ids],25) if ids else None
    return dict(schema=QUEUE,user=user,observed_epoch=clock(),qselect=select,qstat=stat)

def queue_entry():
    """Registered native readonly subprocess entry; never prints an empty view on error."""
    import sys
    try:value=collect_queue()
    except ReadOnlyFailure as error:
        diagnostic=deepcopy(error.diagnostic)
        for key in ('stdout','stderr'):
            if isinstance(diagnostic.get(key),bytes):diagnostic[key]=diagnostic[key].decode(errors='replace')
        print(json.dumps(dict(schema=QUEUE_FAILURE,queue_known=False,diagnostic=diagnostic)),file=sys.stderr,flush=True)
        raise SystemExit(75)
    print(json.dumps(value))

def queue_rows(proof,*,now_epoch):
    fresh(proof.get('observed_epoch'),now_epoch)
    need(proof.get('schema')==QUEUE and proof.get('user')=='e1192548','COMPLETE_OWN_ACCOUNT_READBACK')
    s=proof['qselect'];need(s.get('argv')==['/opt/pbs/bin/qselect','-u','e1192548']
         and type(s.get('returncode')) is int and s['returncode']==0
         and s.get('stderr','')=='','ACTUAL_QSELECT_SUCCESS_WITHOUT_NATIVE_DIAGNOSTIC')
    ids=s['stdout'].split();need(len(ids)<=4096 and len(ids)==len(set(ids)),'NO_DROPPED_OR_DUPLICATE_HANDLES')
    if ids:
        q=proof['qstat'];need(q.get('argv')==['/opt/pbs/bin/qstat','-f','-F','json',*ids]
             and type(q.get('returncode')) is int and q['returncode']==0
             and q.get('stderr','')=='','EXACT_FULL_QSTAT_ALL_SELECTED_HANDLES')
        jobs=json.loads(q['stdout'])['Jobs'];need(set(jobs)==set(ids),'NO_FILTERED_OR_INCOMPLETE_ACCOUNT_QUEUE')
    else:
        need(proof.get('qstat') is None,'GENUINE_EMPTY_QSELECT_NOT_FAILED_QSTAT');jobs={}
    rows=[]
    for jid,j in sorted(jobs.items()):
        need(re.fullmatch(r'[0-9]+\.stdct-mgmt-02',jid) is not None
             and j.get('Job_Owner','').split('@')[0]=='e1192548','OWN_CURRENT_NATIVE_JOB')
        rows.append(dict(platform=PLATFORM,job_id=jid,state=j.get('job_state'),project=j.get('project'),
            name=j.get('Job_Name'),ngpus=j.get('Resource_List',{}).get('ngpus',0)))
    return rows

def classify(h,rows):
    registered={x['job_id']:x for x in h['PBS_submissions'] if x.get('job_id')}
    campaign=[];outside=[];unknown=[];seen=set()
    for row in rows:
        jid=row['job_id'];need(jid not in seen,'NO_DUPLICATE_ACCOUNT_ROW');seen.add(jid)
        need(row['platform']==PLATFORM and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',jid) is not None,'VANDA_ONLY_CURRENT_ROWS')
        item=registered.get(jid);short={k:v for k,v in row.items() if k!='ngpus'}
        if item:
            need(item.get('native_platform')==PLATFORM and item.get('project')==row['project']
                 and item.get('job_name')==row['name']
                 and not(item['status'] in dual.TERMINAL and row['state']!='F'),
                 'NO_NATIVE_IDENTITY_OR_PRIOR_TERMINAL_CONTRADICTION')
            if row['state']!='F':
                need(type(row['ngpus']) in (int,str) and str(row['ngpus'])=='1','ONE_ACTUAL_CAMPAIGN_GPU')
                campaign.append(short)
                if row['state'] not in ('Q','R'):unknown.append(short)
        elif row['name'] and row['name'].startswith('pi0294_'):unknown.append(short)
        else:outside.append(dict(short,classification='OUTSIDE_REGISTERED_CAMPAIGN_NOT_ADOPTED',user_confirmed_independent=False))
    for item in h['PBS_submissions']:
        if item['status'] not in dual.TERMINAL and item.get('job_id') not in seen:
            unknown.append(dict(job_id=item.get('job_id'),reason='ACTIVE_LEDGER_WITHOUT_CURRENT_NATIVE_JOB_PROOF'))
    need(len(campaign)<=1,'ONE_VANDA_ACTIVE_CHAIN_NO_DUPLICATE_Q_OR_R')
    return campaign,outside,unknown

def terminal_rows(proofs,*,ledger,current_rows,now_epoch):
    """A separate exact current qstat-history read, never a fabricated queue row."""
    need(type(proofs) is list and len(proofs)<=1,'AT_MOST_ONE_CURRENT_REGISTERED_TERMINAL_HISTORY')
    out=[];current={v['job_id'] for v in current_rows}
    for proof in proofs:
        need(type(proof) is dict,'EXACT_NATIVE_TERMINAL_READBACK_OBJECT')
        fresh(proof.get('observed_epoch'),now_epoch)
        need(proof.get('schema')==TERMINAL_HISTORY and proof.get('user')=='e1192548'
             and type(proof.get('returncode')) is int and proof['returncode']==0
             and proof.get('stderr','')=='','REAL_SUCCESSFUL_TERMINAL_HISTORY_READ')
        argv=proof.get('argv',[])
        need(len(argv)==6 and argv[:5]==['/opt/pbs/bin/qstat','-x','-f','-F','json']
             and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',argv[5]) is not None,'EXACT_SINGLE_VANDA_HISTORY_HANDLE')
        jid=argv[5];registered=[v for v in ledger['PBS_submissions'] if v.get('job_id')==jid]
        jobs=json.loads(proof['stdout'])['Jobs']
        need(jid not in current and set(jobs)=={jid} and len(registered)==1,'MISSING_CURRENT_JOB_WITH_OWN_DISTINCT_NATIVE_HISTORY')
        item=registered[0];job=jobs[jid];variables=job.get('Variable_List',{})
        need(item.get('native_platform')==PLATFORM and item['status'] not in dual.TERMINAL
             and job.get('job_state')=='F' and type(job.get('Exit_status')) is int
             and job.get('Job_Owner','').split('@')[0]=='e1192548'
             and job.get('project')==item.get('project')=='CFP04-CF-007'
             and job.get('Job_Name')==item.get('job_name')=='pi0294_dec1_a40_train'
             and isinstance(item.get('request_id'),str) and isinstance(item.get('submission_nonce'),str)
             and variables.get('PV1_REQUEST_ID')==item['request_id']
             and variables.get('PV1_SUBMISSION_NONCE')==item['submission_nonce'],
             'KNOWN_F_EXIT_AND_EXACT_ORIGINAL_REQUEST_NONCE_NOT_INFERRED_TERMINAL')
        out.append(dict(platform=PLATFORM,job_id=jid,state='F',project=job['project'],name=job['Job_Name'],
            ngpus=job.get('Resource_List',{}).get('ngpus',0)))
    return out

def observe(*,grant_bytes,global_ledger_bytes,DEC1_ledger_bytes,authority_readback,queue_readback,now_epoch,terminal_readbacks=None):
    handoff.grant(grant_bytes)
    need(type(global_ledger_bytes) is bytes and type(DEC1_ledger_bytes) is bytes,'ACTUAL_CURRENT_PAIRED_BYTES')
    h,d=json.loads(global_ledger_bytes),json.loads(DEC1_ledger_bytes)
    history=authority(h,d,authority_readback,now_epoch=now_epoch)
    need(authority_readback.get('global_sha256')==sha(global_ledger_bytes)
         and authority_readback.get('DEC1_sha256')==sha(DEC1_ledger_bytes),'FRESH_AUTHORITY_BINDS_FULL_CURRENT_PAIRED_BYTES')
    rows=queue_rows(queue_readback,now_epoch=now_epoch)
    terminal_readbacks=[] if terminal_readbacks is None else terminal_readbacks
    terminals=terminal_rows(terminal_readbacks,ledger=h,current_rows=rows,now_epoch=now_epoch)
    campaign,outside,unknown=classify(h,rows+terminals)
    return dict(schema=SCHEMA,status=PASS if not unknown else 'STOP_UNKNOWN_VANDA_CAMPAIGN_STATE',
        observed_epoch=min(authority_readback['observed_epoch'],queue_readback['observed_epoch']),
        grant_sha256=handoff.GRANT_SHA,authority=deepcopy(authority_readback),
        global_ledger_canonical_sha256=digest(h),DEC1_ledger_canonical_sha256=digest(d),migration_marker_sha256=digest(h[handoff.FIELD]),
        complete_Vanda_account_queue_sha256=digest(queue_readback),queue_observed_epoch=queue_readback['observed_epoch'],
        native_terminal_history_readbacks=deepcopy(terminal_readbacks),native_terminal_history_sha256=digest(terminal_readbacks),
        current_Vanda_account_owned_jobs=rows,account_owned_job_count=len(rows),campaign_jobs=campaign,
        campaign_running_by_platform=({PLATFORM:1} if campaign and campaign[0]['state']=='R' else {}),
        outside_registered_campaign_jobs=outside,user_confirmed_independent_jobs=[],unknown_campaign_jobs=unknown,
        historical_Hopper_registered_terminal_jobs=history,Hopper_current_queue_observed=False,
        unavailable_Hopper_is_fresh_or_empty=False,account_view_filtered_or_forged_empty=False,
        other_jobs_modified=False,maximum_active_campaign_GPUs=1,new_resource_or_science_grant=False)

def validate(view,*,ledger,DEC1_ledger=None,now_epoch):
    """Use the actual current ledger again at each reserve/release/heartbeat."""
    need(view.get('schema')==SCHEMA and view.get('status')==PASS and view.get('grant_sha256')==handoff.GRANT_SHA,
         'EXPLICIT_QUALIFIED_SINGLE_PLATFORM_VIEW_NOT_RELABELED_DUAL_VIEW')
    fresh(view.get('observed_epoch'),now_epoch);fresh(view.get('queue_observed_epoch'),now_epoch)
    history=authority(ledger,DEC1_ledger,view['authority'],now_epoch=now_epoch)
    need(view.get('global_ledger_canonical_sha256')==digest(ledger)
         and (DEC1_ledger is None or view.get('DEC1_ledger_canonical_sha256')==digest(DEC1_ledger))
         and view.get('migration_marker_sha256')==digest(ledger[handoff.FIELD]),'SAME_CURRENT_AUTHORITY_NOT_STALE_ACCOUNTING')
    need(isinstance(view.get('complete_Vanda_account_queue_sha256'),str)
         and re.fullmatch('[0-9a-f]{64}',view['complete_Vanda_account_queue_sha256']) is not None,
         'RETAINED_COMPLETE_VANDA_QUEUE_READBACK_DIGEST')
    terminal_proofs=view.get('native_terminal_history_readbacks')
    terminals=terminal_rows(terminal_proofs,ledger=ledger,current_rows=view['current_Vanda_account_owned_jobs'],now_epoch=now_epoch)
    need(view.get('native_terminal_history_sha256')==digest(terminal_proofs),'UNCHANGED_SEPARATE_TERMINAL_HISTORY_BYTES')
    campaign,outside,unknown=classify(ledger,view['current_Vanda_account_owned_jobs']+terminals)
    need(not unknown and view.get('campaign_jobs')==campaign and view.get('outside_registered_campaign_jobs')==outside
         and view.get('unknown_campaign_jobs')==[] and view.get('account_owned_job_count')==len(view['current_Vanda_account_owned_jobs'])
         and view.get('historical_Hopper_registered_terminal_jobs')==history
         and view.get('campaign_running_by_platform')==({PLATFORM:1} if campaign and campaign[0]['state']=='R' else {}),
         'WHOLE_ACCOUNT_VIEW_AND_ALL_ORIGINAL_TERMINAL_HISTORY')
    need(view.get('Hopper_current_queue_observed') is False and view.get('unavailable_Hopper_is_fresh_or_empty') is False
         and view.get('account_view_filtered_or_forged_empty') is False and view.get('other_jobs_modified') is False
         and view.get('maximum_active_campaign_GPUs')==1 and view.get('new_resource_or_science_grant') is False,
         'NO_FORGED_FRESH_HOPPER_OR_NEW_BUDGET_OR_CONCURRENCY')
    return view

def validate_heartbeat_receipt(value,intent,*,now_epoch):
    """A new-kind native receipt cannot pass by renaming a legacy heartbeat."""
    v=value.get('vanda_campaign_view',{})
    need(v.get('schema')==SCHEMA and v.get('status')==PASS and v.get('grant_sha256')==handoff.GRANT_SHA
         and value.get('campaign_view_status')==PASS and value.get('campaign_view_observed_epoch')==v.get('observed_epoch'),
         'EXACT_VANDA_NATIVE_HEARTBEAT_VIEW_KIND')
    fresh(v.get('observed_epoch'),now_epoch);fresh(v.get('queue_observed_epoch'),now_epoch)
    need(v.get('Hopper_current_queue_observed') is False and v.get('unavailable_Hopper_is_fresh_or_empty') is False
         and v.get('unknown_campaign_jobs')==[] and v.get('account_view_filtered_or_forged_empty') is False
         and v.get('maximum_active_campaign_GPUs')==1 and v.get('new_resource_or_science_grant') is False,
         'VANDA_HEARTBEAT_IS_NOT_UNAVAILABLE_HOPPER_OR_NEW_QUOTA')
    own=[j for j in v.get('campaign_jobs',[]) if j['job_id']==intent['identity']['job_id']]
    need(len(own)==len(v['campaign_jobs'])==1 and own[0]['platform']==PLATFORM and own[0]['state'] in ('Q','R')
         and v['authority'].get('physical_platform')==PLATFORM and v['authority'].get('active_registered_control_lease') is True
         and v['authority'].get('single_mutation_owner') is True,'ONE_CURRENT_VANDA_OWNER_AND_JOB')
    for key in ('global_ledger_canonical_sha256','DEC1_ledger_canonical_sha256','migration_marker_sha256','complete_Vanda_account_queue_sha256'):
        need(isinstance(v.get(key),str) and re.fullmatch('[0-9a-f]{64}',v[key]) is not None,'FULL_VANDA_HEARTBEAT_PROVENANCE_DIGESTS')
    beat=value['heartbeat']
    need(beat.get('campaign_view_schema')==SCHEMA and beat.get('control_execution_platform')==PLATFORM
         and beat.get('agent_liveness_required') is False and beat.get('migration_grant_sha256')==handoff.GRANT_SHA
         and beat.get('grant_sha256')==dual.GRANT_SHA
         and beat.get('current_authority_global_canonical_sha256')==v['global_ledger_canonical_sha256']
         and beat.get('migration_marker_sha256')==v['migration_marker_sha256']
         and beat.get('control_authority_proof')==v['authority']
         and beat.get('queue_observed_epoch')==v['queue_observed_epoch'],
         'NATIVE_HEARTBEAT_EXPLICIT_RESIDENT_PRODUCER_AND_MIGRATED_AUTHORITY')
    return value

def validate_owner_heartbeat(beat,*,ledger,now_epoch):
    """Original300s heartbeat; require fresh queue/authority at publication."""
    need(type(beat.get('observed_epoch')) in (int,float) and 0<=now_epoch-beat['observed_epoch']<=300,
         'ORIGINAL300_SECOND_WATCHDOG_NOT_EXTENDED')
    fresh(beat.get('queue_observed_epoch'),beat['observed_epoch'])
    authority(ledger,None,beat['control_authority_proof'],now_epoch=beat['observed_epoch'])
    need(beat.get('campaign_view_schema')==SCHEMA and beat.get('control_execution_platform')==PLATFORM
         and beat.get('agent_liveness_required') is False and beat.get('migration_grant_sha256')==handoff.GRANT_SHA
         and beat.get('grant_sha256')==dual.GRANT_SHA
         # The full digest attests publication's exact read, not a lock on
         # subsequent charged queries/updates. Runtime guards below and the
         # existing request/lease checks bind immutable authority instead.
         and isinstance(beat.get('current_authority_global_canonical_sha256'),str)
         and re.fullmatch('[0-9a-f]{64}',beat['current_authority_global_canonical_sha256']) is not None
         and beat.get('migration_marker_sha256')==digest(ledger[handoff.FIELD]),'SAME_CURRENT_AUTHORITY_AND_VANDA_PRODUCER')
    return beat

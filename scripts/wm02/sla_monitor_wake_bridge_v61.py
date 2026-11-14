"""One finite completion bridge, with no model or experiment dispatcher.

It polls a pinned read-only probe and queues one human-readable completion or
attention message into the user's existing thread. --resume requests wakeup;
queue acknowledgement alone is not proof that the app actually resumed.
"""
import argparse,hashlib,json,re,subprocess,time
from contextlib import closing
from pathlib import Path
from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.journal import Journal,MutationOutcomeUnknown

SCHEMA='SLA_LOCAL_FINITE_JOB_COMPLETION_BRIDGE_V6_1'


def queue_once(root,spec,event,*,runner=subprocess.run):
    journal=Journal(root/'journal',root.name);key='QUEUE-TERMINAL-EVENT'
    payload=dict(thread=spec['thread_id'],event_result_sha256=event['result_sha256'])
    message=('后台 SLA monitor 已产生终态或需处理事件。请读取本机 '+str(root/'REMOTE_EVENT.private.json')+
        ' 与其中固定的 Core 结果证据，核对真实训练更新、访问、原生分母、失败原因和剩余计划，再汇报。'
        '这不是任务成功声明。不要重投、重训、替换有效失败条件或改安全门；现役作业和旧 RP8 失败保持。')
    argv=[spec['codex'],'queue','--thread',spec['thread_id'],'--message',message,'--resume']
    def deliver():
        p=runner(argv,capture_output=True,text=True,timeout=45)
        match=re.search(r'Queued message (\S+) for thread '+re.escape(spec['thread_id'])+r'\.',p.stdout)
        value=dict(exit_code=p.returncode,queue_ack=p.returncode==0 and match is not None,
            message_id=match.group(1)if match else None,stdout=p.stdout,stderr=p.stderr,
            automatic_resume='REQUESTED_NOT_INDEPENDENTLY_VERIFIED')
        if p.stderr.strip():value['automatic_resume']='UNAVAILABLE_OR_WARNING_RETAINED'
        write(root/'QUEUE_ACTUAL.private.json',value)
        return value
    return journal.once(key,payload,deliver)


def run(spec):
    root=Path(spec['registered_root']);start=time.time();poll=0
    if spec['schema']!=SCHEMA or spec['poll_s']!=180:raise ValueError('FINITE_READONLY_LOCAL_COMPLETION_BRIDGE_ONLY')
    for key in ('bridge','probe'):
        p=spec[key]
        if hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('PINNED_BRIDGE_OR_PROBE_CHANGED')
    with closing(OwnerLock(root/'OWNER.lock')):
        while time.time()<spec['deadline_epoch']:
            poll+=1
            try:
                # core_rpc records each read independently;unique read IDs
                # allow retrying observations without replaying mutations.
                name=root.name+'-READ-'+str(poll).zfill(5)
                argv=[spec['python'],spec['core_rpc'],spec['probe']['path'],name,'60']
                p=subprocess.run(argv,capture_output=True,text=True,timeout=75)
                outer=json.loads(p.stdout)
                if p.returncode or outer['exit']:raise RuntimeError('READ_ONLY_PROBE_FAILED:'+outer.get('err','')[-500:])
                value=json.loads(outer['out'])
                write(root/'LAST_PROBE.private.json',value)
                if value.get('event'):
                    event=value['event'];write(root/'REMOTE_EVENT.private.json',value)
                    try:queued=queue_once(root,spec,event)
                    except MutationOutcomeUnknown:
                        final=dict(status='QUEUE_INTENT_UNKNOWN_NEVER_RESEND',exit_code=1,goal_complete=False)
                    else:
                        final=dict(status='EVENT_QUEUED_RESUME_NOT_PROVEN'if queued['queue_ack']else'EVENT_QUEUE_FAILED_NO_AUTOMATIC_RESEND',
                            queue=queued,exit_code=0 if queued['queue_ack']else 1,goal_complete=False)
                    write(root/'RESULT.private.json',dict(final,wall_s=time.time()-start));return final
                if value.get('unit',{}).get('LoadState')=='loaded'and value['unit'].get('MainPID')=='0':
                    write(root/'REMOTE_EVENT.private.json',value)
                    event=dict(result_sha256=value.get('monitor_spec_sha256'),status='MONITOR_PROCESS_TERMINAL_WITHOUT_EVENT_NOT_JOB_TERMINAL')
                    queued=queue_once(root,spec,event)
                    final=dict(status='MONITOR_STOP_REQUIRES_RECONCILIATION',queue=queued,exit_code=1,goal_complete=False)
                    write(root/'RESULT.private.json',final);return final
                write(root/'STATE.private.json',dict(status='WAIT_SERVER_EVENT',epoch=time.time(),poll=poll,
                    model_loads=0,optimizer_updates=0,physical_requests=0))
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError)as error:
                write(root/'STATE.private.json',dict(status='READ_ONLY_OBSERVATION_FAILED_RETRY_SAME_MONITOR',epoch=time.time(),
                    poll=poll,error=str(error)[-1000:],experiments_restarted=False))
            time.sleep(min(spec['poll_s'],max(0,spec['deadline_epoch']-time.time())))
        final=dict(status='FINITE_BRIDGE_DEADLINE_NO_JOB_TERMINAL_INFERRED',exit_code=75,goal_complete=False)
        write(root/'RESULT.private.json',final);return final


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec',type=Path);p.add_argument('--sha256',required=True);a=p.parse_args()
    if hashlib.sha256(a.spec.read_bytes()).hexdigest()!=a.sha256:raise ValueError('BRIDGE_SPEC_CHANGED')
    value=run(json.loads(a.spec.read_bytes()));raise SystemExit(value['exit_code'])

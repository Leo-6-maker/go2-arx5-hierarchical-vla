"""Finite existing-entry Core metadata worker; no model import or physics.

The original model-only HF upload/download protocol is retained. Configuration
contains private paths, never credentials. One Journal owns source launch; an
unknown launch/commit is preserved for exact readback, never automatically sent
again. Actual CPU/IO witnesses stay inside the already reserved Eval stage.
"""
import base64, hashlib, json, os, resource, shlex, subprocess, sys, time
import urllib.request
from pathlib import Path
from scripts.pi0292.durable import OwnerLock
from scripts.pi0294.dec1.owner_transport import Journal
from . import mix_accounting as a, mix_delivery_control as guard
from . import mix_evaluation_transfer as transfer, successor_cd as scope


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with path.open('xb') as f:
        f.write(a.raw(value));f.flush();os.fsync(f.fileno())
    path.chmod(0o400)
    return value


START=r'''
import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
assert os.getuid()==22182 and not os.environ.get('PBS_JOBID')
r=Path(P['control_root']);m=json.loads((r/'CONTROL_RUN_MANIFEST.json').read_bytes());assert m['source_sha']==P['source_sha']and m['model_transfer_only']is True
code=r/'code';assert hashlib.sha256((code/'scripts/pi0294/pv1/mix_model_upload.py').read_bytes()).hexdigest()==m['worker_sha256']
root=r/'upload';assert not root.exists()and not(r/'UPLOAD_START_INTENT.json').exists()
def put(name,value):
 with(r/name).open('x')as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 (r/name).chmod(0o400)
put('UPLOAD_START_INTENT.json',dict(candidate=P['candidate'],source=P['source_sha'],epoch=time.time(),automatic_retry=False))
wrapper='import runpy,sys,json,time,os,signal;from pathlib import Path;signal.alarm(7200);sys.path[:0]='+repr([str(code),str(code/'src')])+';rc=0\ntry:runpy.run_module("scripts.pi0294.pv1.mix_model_upload",run_name="__main__")\nexcept BaseException as exc:rc=exc.code if isinstance(exc,SystemExit)and isinstance(exc.code,int)else 1\nfinally:\n p=Path('+repr(str(r/'UPLOAD_OS_EXIT.json'))+');p.write_text(json.dumps(dict(returncode=rc,finished_epoch=time.time(),pid=os.getpid()))+"\\n");p.chmod(0o400)\nraise SystemExit(rc)'
log=(r/'UPLOAD.log').open('xb');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
process=subprocess.Popen([sys.executable,'-u','-c',wrapper],stdin=subprocess.PIPE,stdout=log,stderr=log,env=env,start_new_session=True)
stat=Path('/proc',str(process.pid),'stat').read_text().rsplit(')',1)[1].split();record=dict(pid=process.pid,proc_start=int(stat[19]),candidate=P['candidate'],source_sha=P['source_sha'],started_epoch=time.time(),new_GPU_model_queries_physics=0)
put('UPLOAD_NATIVE_STARTED.json',record)
if P['candidate']not in('E1_ADAPTER_ONLY','E2_SHARED_ACTION_HEAD'):P.pop('control_root')
process.stdin.write(json.dumps(P).encode());process.stdin.close();log.close();print(json.dumps(record))
'''
POLL=r'''
import json,os,time
from pathlib import Path
r=Path(P['root']);start=json.loads((r/'UPLOAD_NATIVE_STARTED.json').read_bytes());p=Path('/proc',str(start['pid']),'stat');f=p.read_text().rsplit(')',1)[1].split()if p.exists()else None
result={}
for n in ['UPLOAD_OS_EXIT.json','upload/COMPLETED.json','upload/FAILED_OR_UNKNOWN.json','upload/COMMIT_RETURNED.json']:
 if(r/n).is_file():result[n]=json.loads((r/n).read_bytes())
print(json.dumps(dict(active=bool(f and f[0]not in('Z','X')and int(f[19])==start['proc_start']),native=start,receipts=result,observed_epoch=time.time())))
'''


def candidate(cfg):
    """Legacy C50 config stays C50; D25 must carry its own explicit identity."""
    arm=cfg.get('candidate','C50')
    guard.need(arm in ('C50','D25')and cfg['compiled_plan']['public_plan']['candidate']==arm
        and cfg['compiled_plan']['public_plan']['checkpoint_step']==1024,
        'CONFIG_EXPLICIT_OWN_CLOSED_CANDIDATE_NOT_C50_ALIAS')
    return arm


def delivery_journal(root,cfg,configuration_bytes):
    # Journal accepts an existing registered directory, never creates it.
    directory=Path(root)/'rpc';directory.mkdir(mode=0o700)
    return Journal(directory,dict(run_uuid=cfg['control_run_uuid'],job_id='MODEL_ONLY_'+candidate(cfg)+'_NO_GPU',
        source_sha=cfg['source_sha'],request_sha256=a.sha(configuration_bytes)))


def full_remaining(g,l,*,Core_storage_upper_bytes,LOCAL_storage_upper_bytes):
    """Use the adopted whole campaign forecast, retaining legacy semantics.

    The old successor-only forecast predates the actually adopted first512
    seven caps and whole package. Its old PBS36/output limits must not replace
    the real authority, or omit the new package, in this finite IO worker.
    """
    from scripts.pi0294.generalization import minimal_package,minimal_scope
    if minimal_package.FIELD not in g:
        return scope.recompute(g,l,Core_storage_upper_bytes=Core_storage_upper_bytes,
            local_storage_upper_bytes=LOCAL_storage_upper_bytes)
    minimal_package.validate(g)
    proof=minimal_scope.first512_forecast(a.raw(g),a.raw(l))
    from copy import deepcopy
    rows=deepcopy(proof['dimensions'])
    for key,upper in [('Core_storage_bytes',Core_storage_upper_bytes),
            ('local_storage_bytes',LOCAL_storage_upper_bytes)]:
        guard.need(type(upper)is int and upper>=rows[key]['used'],
            'RETAIN_ALL_HISTORICAL_STORAGE_IN_FULL_CAMPAIGN')
        rows[key]['projected']+=upper-rows[key]['used'];rows[key]['used']=upper
        rows[key]['shortfall']=max(0,rows[key]['projected']-rows[key]['cap'])
    return dict(schema='V5_CD_MODEL_DELIVERY_ACTUAL_ADOPTED_FULL_CAMPAIGN_FORECAST_V1',
        full_campaign=proof,dimensions=rows,
        shortfalls={k:v['shortfall']for k,v in rows.items()if v['shortfall']},
        native_model_or_physics_admission=False,new_reservation={})


def main(config_path):
    began=time.process_time();started=time.time();cfg=json.loads(Path(config_path).read_bytes());root=Path(config_path).parent
    assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s' and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not os.environ.get('PBS_JOBID')
    arm=candidate(cfg);prior=cfg.get('Core_metadata_prior_CPU_upper_s',0);assert type(prior)is int and 0<=prior<300
    cpu_limit=300-prior
    resource.setrlimit(resource.RLIMIT_CPU,(cpu_limit,cpu_limit));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.sched_setaffinity(0,{7})
    manifest=json.loads((root/'CONTROL_RUN_MANIFEST.json').read_bytes());assert manifest['configuration_sha256']==a.sha(Path(config_path).read_bytes())
    code=root/'code';seal=json.loads((root/'SOURCE_SEAL.json').read_bytes());assert seal['source_sha']==cfg['source_sha']and all(a.sha((code/n).read_bytes())==h for n,h in seal['files'].items())
    lock=OwnerLock(root/'MODEL_DELIVERY_OWNER.lock');status='FAILED_PRESERVED';phase='REGISTERED_CONTROL_ADMISSION';source_terminal=None
    put(root/'MODEL_DELIVERY_NATIVE_STARTED.json',dict(pid=os.getpid(),proc_start=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),source_sha=cfg['source_sha'],observed_epoch=started,new_GPU_model_queries_physics=0))
    journal=delivery_journal(root,cfg,Path(config_path).read_bytes())
    b=cfg['remote_binding'];args=['ssh','-T','-S',b['master_socket'],'-o','ControlMaster=no','-o','ProxyCommand=false','-o','CanonicalizeHostname=no','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-l',b['remote_user'],b['remote_host'],shlex.join([b['remote_python'],'-'])]
    def remote(phase,effect,body,payload):
        def invoke(intent,limit):
            reply=subprocess.run(args,input='P='+repr(payload)+'\n'+body,text=True,capture_output=True,timeout=limit)
            # Remote code emits bounded sanitized JSON only; token-bearing stdin
            # is never retained, including SDK/transport exceptions.
            if reply.returncode:
                put(root/(intent['operation_id']+'.transport.json'),dict(returncode=reply.returncode,operation=phase))
                raise ConnectionError('SANITIZED_SOURCE_DELIVERY_TRANSPORT_FAILURE')
            return json.loads(reply.stdout)
        return journal.call(phase,effect,invoke)
    try:
        assert started<cfg['deadline_epoch']-28860
        local=Path(cfg['canonical_LOCAL_ledger']).read_bytes();assert a.sha(local)==cfg['expected_LOCAL_sha256']
        g=json.loads(base64.b64decode(cfg['GLOBAL_base64'],validate=True));l=json.loads(local)
        guard.c.paired(g,l);plan=cfg['compiled_plan'];cp=base64.b64decode(plan['private_restoration']['original_complete_base64'],validate=True)
        guard.upload_admission(g,plan,now_epoch=started,checkpoint_complete_bytes=cp)
        proof=full_remaining(g,l,Core_storage_upper_bytes=cfg['Core_storage_upper_bytes'],LOCAL_storage_upper_bytes=cfg['LOCAL_storage_upper_bytes']);assert not proof['shortfalls'];put(root/'FULL_REMAINING_NATIVE_CONTROL_CHECK.json',proof)
        notices=cfg['notices'];assert all(a.sha(base64.b64decode(v['base64'],validate=True))==v['sha256']for v in notices.values())
        token=Path(cfg['HF_core_token_path']).read_text().strip()
        payload=dict(source_sha=cfg['source_sha'],canonical_parent=b['canonical_parent'],control_root=cfg['source_control_root'],upload_root=cfg['source_control_root']+'/upload',candidate=arm,checkpoint=cfg['checkpoint_root'],checkpoint_sha256=plan['public_plan']['checkpoint_manifest_sha256'],public_plan=plan['public_plan'],compiled_plan=plan,rewritten=plan['public_rewritten_metadata_base64'],notices=notices,repo_id='leoyuer/pi0294-public-artifacts',prefix=cfg['HF_prefix'],stop_new_epoch=cfg['deadline_epoch']-28860,token=token)
        phase='SOURCE_UPLOAD_ONCE';put(root/'SOURCE_UPLOAD_STARTED.json',remote('START_'+arm+'_MODEL_ONLY_UPLOAD','MUTATION',START,payload));token=None;payload=None
        while time.time()<min(started+7000,cfg['deadline_epoch']-28860):
            state=remote('READ_'+arm+'_MODEL_ONLY_UPLOAD','READ_ONLY',POLL,dict(root=cfg['source_control_root']))
            if not state['active']:
                put(root/'SOURCE_UPLOAD_TERMINAL.json',state);r=state['receipts'];assert r['UPLOAD_OS_EXIT.json']['returncode']==0 and 'upload/FAILED_OR_UNKNOWN.json'not in r
                source_terminal=r['upload/COMPLETED.json'];break
            time.sleep(30)
        assert source_terminal is not None,'SOURCE_UPLOAD_NOT_TERMINAL_NO_RESTART'
        phase='FIXED_REVISION_CORE_DOWNLOAD';dest=root/'model_view';dest.mkdir(mode=0o700);files={};downloaded={}
        for name,expected in dict(plan['public_plan']['files'],**notices).items():
            assert time.time()<cfg['deadline_epoch']-28860
            path=dest/name;path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);tmp=path.with_name(path.name+'.inflight');h=hashlib.sha256();total=0
            url='https://huggingface.co/datasets/'+source_terminal['repo_id']+'/resolve/'+source_terminal['revision']+'/'+source_terminal['prefix']+'/'+name
            with urllib.request.urlopen(url,timeout=30)as response,tmp.open('xb')as f:
                while True:
                    chunk=response.read(8*1024**2)
                    if not chunk:break
                    total+=len(chunk);assert total<=expected['bytes'];h.update(chunk);f.write(chunk)
                f.flush();os.fsync(f.fileno())
            assert(total,h.hexdigest())==(expected['bytes'],expected['sha256']);os.rename(tmp,path);path.chmod(0o400);downloaded[name]=dict(bytes=total,sha256=h.hexdigest());put(root/'download-receipts'/(name.replace('/','_')+'.json'),downloaded[name])
            if name in plan['public_plan']['files']:files[name]=path
        phase='EXACT_PRIVATE_METADATA_RESTORATION';restored=transfer.restore_original(files,plan['private_restoration'],arm=arm)
        for name,value in restored.items():
            if isinstance(value,bytes):
                path=dest/name;os.rename(path,path.with_name(path.name+'.public-transfer-original'))
                with path.open('xb')as f:f.write(value);f.flush();os.fsync(f.fileno())
                path.chmod(0o400)
        with(dest/'CHECKPOINT_COMPLETE.json').open('xb')as f:f.write(cp);f.flush();os.fsync(f.fileno())
        (dest/'CHECKPOINT_COMPLETE.json').chmod(0o400)
        status='ACTUAL_CLOSED_'+arm+'_1024_INFERENCE_VIEW_DELIVERED_NO_MODEL_LOADED'
        put(root/'ACTUAL_CORE_MODEL_DELIVERED.json',dict(status=status,model_root=str(dest),checkpoint_manifest_sha256=a.sha(cp),HF_revision=source_terminal['revision'],repo_id=source_terminal['repo_id'],prefix=source_terminal['prefix'],files=downloaded,full_resume_files_present=False,source_CPU_s=source_terminal['CPU_s'],Core_CPU_s=time.process_time()-began,IO_accounting='original_'+arm+'_Eval_stage_terminal_once_not_additional_reserve',new_GPU_model_queries_physics=0,completed_epoch=time.time()))
    except BaseException as exc:
        put(root/'MODEL_DELIVERY_PRESERVED_EXCEPTION.json',dict(status='FAILED_OR_UNKNOWN_NO_AUTOMATIC_REPLAY',phase=phase,error_type=type(exc).__name__,source_terminal_known=source_terminal is not None,observed_epoch=time.time()))
        raise RuntimeError('FINITE_'+arm+'_MODEL_DELIVERY_STOPPED_'+type(exc).__name__)from None
    finally:
        put(root/'MODEL_DELIVERY_NATIVE_FINAL.json',dict(status=status,phase=phase,CPU_s=time.process_time()-began,started_epoch=started,completed_epoch=time.time(),new_GPU_models_queries_physics=0));lock.close()


if __name__=='__main__':main(sys.argv[1])

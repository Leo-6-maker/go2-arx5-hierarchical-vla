"""Run pinned existing stage/evaluation/audit/close programs after GPU admission.

The private capsule supplies host bindings and the original native programs.
No model, physics, scoring or resource ledger is implemented here.
"""
import base64
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_bytes())


def preflight(root):
    root=Path(root);config=read(root/'BINDINGS.json')
    for name,pin in read(root/'CAPSULE_SEAL.json')['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=pin:
            raise PermissionError('QUEUED_CAPSULE_CHANGED:'+name)
    for path,pin in config['required_closed_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=pin:
            raise PermissionError('QUEUED_NATIVE_INPUT_CHANGED')
    for name in config['terminal_units']:
        q=subprocess.run(['systemctl','--user','show',name,'-p','MainPID'],capture_output=True,text=True,timeout=5)
        if q.stdout.strip()!='MainPID=0':
            raise PermissionError('PRIOR_NATIVE_OWNER_NOT_TERMINAL:'+name)
    return dict(status='PASS_PINNED_COLD_QUEUE_PREFLIGHT',epoch=time.time(),new_models_physics=0)


def run(root):
    root=Path(root);preflight(root);config=read(root/'BINDINGS.json');programs=read(root/'NATIVE_PROGRAMS.json')
    folder=root/'jobs/evaluation'
    spec=importlib.util.spec_from_file_location('_queued_existing_native_control',folder/'resident_common.py')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    def save(name,value):
        b.exclusive(folder/(name+'.json'),b.raw(value))
    def load(name):
        return read(folder/(name+'.json'))
    def call(name,code,args,*,effect='READ_ONLY',CPU=15,reply=22000):
        value=b.call('QUEUED_D0_'+name,code,args,effect=effect,CPU=CPU,reply=reply)
        save(name,value);return value
    staged=None
    try:
        staged=call('STAGED',programs['PREPARE_STAGE'],dict(config['prepare_payload'],stage_code=programs['STAGE']),effect='MUTATION',CPU=45)
        prepared=read(config['prepared_path'])
        a=dict(staged,endpoint=4096,LOCAL=prepared['LOCAL'],registry=prepared['registry'])
        independent=call('INDEPENDENT',programs['READ'],a,CPU=22)
        a=dict(staged,independent_read_epoch=independent['observed_epoch'])
        call('STARTED',programs['START'],a,effect='MUTATION',CPU=12)
        native=Path(staged['root']);started=time.monotonic()
        while True:
            q=subprocess.run(['systemctl','--user','show',native.name+'.service','-p','MainPID'],capture_output=True,text=True,timeout=5)
            if q.stdout.strip()=='MainPID=0':break
            if q.returncode or time.monotonic()-started>7500:
                raise RuntimeError('NATIVE_UNIT_STATE_OR_TERMINAL_UNKNOWN')
            time.sleep(20)
        result=read(native/'RESULT.json')
        if (result.get('error')=='PermissionError: BOOK_DEV10_EXPLICIT_NATIVE_GPU_CAPACITY_ENTRY'
                and result['model_load_attempts']==result['model_queries']==result['integrations']==0
                and result['conditions']==[]):
            closed=call('ENTRY_REFUSAL_CLOSED',programs['CLOSE_ENTRY_REFUSAL'],dict(staged=staged,code=programs['CLOSE']),effect='MUTATION',CPU=18)
            closed['LOCAL']=prepared['LOCAL'];call('FINAL_INDEPENDENT',programs['CLOSED_READ'],closed,CPU=8)
            save('TASK_RECEIPT',dict(status='FAILED_RESOURCE_ENTRY_CLOSED',native_effects_closed=True,actual=closed['actual']))
            return 1
        header=call('RAW_HEADER',programs['HEADER'],staged,CPU=24,reply=2*1024**2)
        result=header['files']['RESULT.json']['value']
        for i,row in enumerate(result['conditions']):
            a=dict(staged,case=row,physics_python=config['python'])
            v=call('CASE_WIRE'+str(i),programs['CASE'],a,CPU=48,reply=8*1024**2)
            body=gzip.decompress(base64.b64decode(v['gzip_base64'],validate=True))
            if b.sha(body)!=v['sha256']:raise PermissionError('ACTUAL_CASE_TRANSPORT_CHANGED')
            save('RAW_CASE'+str(i),json.loads(body))
        from types import SimpleNamespace
        d=SimpleNamespace(load=load,save=save,path=lambda name:folder/(name+'.json'))
        ns=dict(b=b,d=d,math=math,io=io,json=json,base64=base64,sys=sys,time=time,Path=Path)
        exec(compile(programs['AUDIT_SOURCE'],'original-native-independent-case-audit','exec'),ns)
        ns['audit']();audit=load('INDEPENDENT_AUDIT')
        journal=header['journal'];unit=native.name+'.service'
        cpu=[int(x['CPU_USAGE_NSEC'])/1e9 for x in journal if 'CPU_USAGE_NSEC'in x]
        begin=[int(x['__REALTIME_TIMESTAMP'])for x in journal if x.get('MESSAGE','').startswith('Started '+unit)]
        end=[int(x['__REALTIME_TIMESTAMP'])for x in journal if 'CPU_USAGE_NSEC'in x]
        if len(cpu)!=1 or len(begin)!=1 or len(end)!=1:raise PermissionError('EXACT_NATIVE_USAGE')
        files={name:v['sha256']for name,v in header['files'].items()}
        actual=dict(model_loads=result['model_load_attempts'],generation_queries=result['model_queries'],
            valid_DEV_conditions=audit['valid_conditions'],DEV_successes=audit['successes'],
            unrun_DEV_conditions=audit['remaining_unrun'],hard_native_safety_conditions=audit['hard_native_safety_conditions'],
            physics_integrations=result['integrations'],source_episode_starts=len(result['conditions']),source_constructors=len(result['conditions']),
            RGB_pairs=result['RGB_pairs'],optimizer_updates=0,CPU_s=cpu[0],GPU_allocated_s_upper=math.ceil((end[0]-begin[0])/1e6),
            own_allocated_bytes_before_closing=header['own_allocated_bytes'],result_sha256=files['RESULT.json'],
            checkpoint_manifest_sha256=result['evaluated_checkpoint_manifest_sha256'],
            independent_audit_sha256=b.sha((folder/'INDEPENDENT_AUDIT.json').read_bytes()),all_costs_retained=True)
        if 'STOP.json'in files:actual['STOP_sha256']=files['STOP.json']
        code="import fcntl,hashlib;from pathlib import Path;lock=Path(P['owner_lock']).open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);a=dict(P['args']);a['LOCAL_before_sha256']=hashlib.sha256(Path(a['LOCAL']).read_bytes()).hexdigest();exec(compile(P['code'],'existing-native-close','exec'),{'__name__':'_close','P':a})"
        a=dict(staged,LOCAL=prepared['LOCAL'],registry=prepared['registry'],actual=actual,files=files)
        closed=call('COMPLETED_CLOSED',code,dict(args=a,code=programs['CLOSE'],owner_lock=config['native_owner_lock']),effect='MUTATION',CPU=18)
        closed['LOCAL']=prepared['LOCAL'];call('FINAL_INDEPENDENT',programs['CLOSED_READ'],closed,CPU=8)
        save('TASK_RECEIPT',dict(status='COMPLETE_DEV_CLOSED',native_effects_closed=True,actual=actual))
        return 0
    except BaseException as exc:
        # A stage refusal before native registration is provably zero compute.
        # Preserve its intent, failed transport, metadata and full control costs.
        closed=False
        if staged is None and Path(config['prepared_path']).exists():
            a=read(config['prepared_path']);native=Path(a['parent'])/a['run']
            code="import hashlib,json;from pathlib import Path;a=P;r=Path(a['parent'])/a['run'];local=Path(a['LOCAL']).read_bytes();registry=Path(a['registry']).read_bytes();v=json.loads(local);assert not r.exists()and hashlib.sha256(local).hexdigest()==a['LOCAL_before_sha256']and hashlib.sha256(registry).hexdigest()==a['registry_before_sha256']and r.name not in v.get('V7_book_learning_runs',{});print(json.dumps(dict(native_effects_closed=True,new_models_physics=0,prepared_metadata_retained=True)))"
            try:
                proof=call('UNREGISTERED_STAGE_EFFECT_READ',code,a,CPU=8)
                closed=proof['native_effects_closed']
            except BaseException:
                pass
        if not(folder/'TASK_RECEIPT.json').exists():
            save('TASK_RECEIPT',dict(status='FAILED',native_effects_closed=closed,error=type(exc).__name__+': '+str(exc),epoch=time.time()))
        raise


def main():
    action,root=sys.argv[1:]
    if action=='preflight':print(json.dumps(preflight(root)));return 0
    if action=='run':return run(root)
    raise ValueError(action)


if __name__=='__main__':raise SystemExit(main())

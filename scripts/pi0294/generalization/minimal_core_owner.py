"""Cold adapters of the exact existing Core owner; original driver untouched."""
import hashlib
import json
import os
from pathlib import Path
import time
from scripts.pi0294.pv1.node_local_owner_entry import derive_owner_environment,LOCAL_ENVIRONMENT

PINS={'v5_core_resident_owner.py':'15b636f2e6f05545c7e3b7d22c3de57616d3f500ce8daf8681180fa446244177',
 'v5_core_resident_methods.py':'040fcac62470c43eca2773b58c584e4d2b61d6e81cac35fa42d3e93bd107cbdc',
 'fixed_file_rpc_client.py':'59a4156d8212a1c16839b223cfa192b9d6b4283ff7d19609a8e8b94fbb78ece0',
 'ORIGINAL_RESIDENT_KERNEL_HELPERS.py':'b2d48f4a385f427daef3d87adeb15d7aea3c7fa30394ea302f1637e9a1d7e501'}
ENVIRONMENT=dict(LOCAL_ENVIRONMENT,HUGGINGFACE_HUB_CACHE='cache/huggingface/hub')


def replace(text,old,new):
    if text.count(old)!=1:raise PermissionError('FIXED512_EXACT_ORIGINAL_OWNER_SEAM:'+old[:60])
    return text.replace(old,new)


def derive(files):
    if set(files)!=set(PINS) or any(hashlib.sha256(b).hexdigest()!=PINS[n]for n,b in files.items()):
        raise PermissionError('FIXED512_ALL_FOUR_ORIGINAL_OWNER_BYTES')
    out=dict(files);owner,_=derive_owner_environment(files['v5_core_resident_owner.py']);text=owner.decode()
    text=replace(text,
        'from scripts.pi0294.pv1 import mix_core_accounting as c,mix_accounting as a,mix_core_settlement as terminal,mix_core_admission as admission,mix_core_identity as identity,successor_cd as successor',
        'from scripts.pi0294.pv1 import mix_accounting as a,mix_core_settlement as terminal\nfrom scripts.pi0294.generalization import minimal_core_accounting as c,minimal_core_native as admission,minimal_core_identity as identity,minimal_core_accounting as successor')
    text=replace(text,"assert self.arm=='D25'","assert self.arm=='E1'")
    text=replace(text,
        "from scripts.pi0294.pv1.pre_model_qualification import require_current\n   require_current(Path(self.sites['hopper']['root']),proof['allocation'])",
        "from scripts.pi0294.pv1.pre_model_qualification import require_resident\n   require_resident(Path(self.sites['hopper']['root']),self.local_ledger)")
    text=replace(text,'from scripts.pi0294.pv1.mix_core_d25_local_successor import finite_control_scope',
        'from scripts.pi0294.generalization.minimal_core_accounting import finite_control_scope')
    text=replace(text,'_NODE_LOCAL_ENVIRONMENT='+repr(LOCAL_ENVIRONMENT),'_NODE_LOCAL_ENVIRONMENT='+repr(ENVIRONMENT))
    out['v5_core_resident_owner.py']=text.encode()
    methods=files['v5_core_resident_methods.py'].decode()
    anchor="  code=info['code'];bootstrap="
    methods=replace(methods,anchor,"  local_env={k:str(Path(info['root'])/'control'/v)for k,v in "+repr(ENVIRONMENT)+".items()}\n"+anchor)
    anchor="  profile='4080-simulation' if role=='core' else '4080-neural-evaluation'"
    methods=replace(methods,anchor,"  bootstrap='import os;os.environ.update('+repr(local_env)+');'+bootstrap\n"+anchor)
    anchor='        return drive(sessions,faults,plan,registry,emit,certify_fault,certify_nominal,publish,heartbeat,'
    methods=replace(methods,anchor,
        '        from scripts.pi0294.generalization.minimal_core_owner import await_fixture_sessions\n'
        '        try:await_fixture_sessions([*sessions.values(),*faults.values()],self.sites,heartbeat,job,deadline=min(ready["deadline_epoch"],time.time()+180))\n'
        '        except BaseException:\n'
        '            for session in [*sessions.values(),*faults.values()]:session.close()\n'
        '            raise\n'+anchor)
    out['v5_core_resident_methods.py']=methods.encode()
    helper=files['ORIGINAL_RESIDENT_KERNEL_HELPERS.py'].decode()
    helper=replace(helper,"GPU_uuid=r['GPU_uuid'],own_pid=pid",
        "GPU_uuid=r['GPU_uuid'],shared_request=r if 'shared_GPU_grant_sha256'in r else None,own_pid=pid")
    out['ORIGINAL_RESIDENT_KERNEL_HELPERS.py']=helper.encode()
    for n,b in out.items():compile(b,'<new-cold-fixed512-'+n+'>','exec')
    return out,dict(schema='ARX5_FIXED512_EXACT_ORIGINAL_CORE_OWNER_ADAPTER_V1',
        original_sha256=PINS,derived_sha256={n:hashlib.sha256(b).hexdigest()for n,b in out.items()},
        local_environment=ENVIRONMENT,original_driver_controller_safety_and_score_unchanged=True,
        native_admission=False)


def await_fixture_sessions(sessions,sites,heartbeat,job,*,deadline):
    """Wait for original local server initialization before any RPC clock.

    The existing server writes its process record only after Endpoint has
    validated its immutable inputs and ledger. No request is sent or retried
    here, and the original per-call twenty-second watchdog stays unchanged.
    """
    from scripts.pi0294.dec1.source_cpu_once_io import _file
    pending=list(sessions)
    while pending:
        heartbeat()
        if time.time()>=deadline:
            raise TimeoutError('FIXED512_LOCAL_RPC_INITIALIZATION_NOT_READY_NO_REQUEST_SENT')
        for session in pending[:]:
            if session.process.poll() is not None:
                raise ChildProcessError('FIXED512_LOCAL_RPC_EXITED_BEFORE_FIRST_REQUEST')
            info=sites[session.role];root=Path(info['root'])
            if (root/'STOP.json').exists():
                raise PermissionError('FIXED512_LOCAL_RPC_STOP_BEFORE_FIRST_REQUEST')
            path=root/'aux_worker_processes'/('fixture-rpc-'+session.role+'-'+session.label+'.json')
            try:record=json.loads(_file(path))
            except FileNotFoundError:continue
            fields=Path('/proc',str(session.process.pid),'stat').read_text().rsplit(')',1)[1].split()
            if not (record['pid']==session.process.pid and record['proc_start']==fields[19]
                    and fields[0]not in ('Z','X') and record['source_sha']==info['source_sha']
                    and record['job_id']==job and record['source_live_worker']is False
                    and record['work_type']=='PV1_ISOLATED_TRANSPORT_NO_MODEL_NO_PHYSICS'
                    and record['maximum_CPU_s']==180):
                raise PermissionError('FIXED512_EXACT_EXISTING_LOCAL_RPC_READY_PROCESS')
            pending.remove(session)
        if pending:time.sleep(.05)

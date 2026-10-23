"""Cold server-side transport for the existing single-owner lifecycle.

The mature watch/release/runtime are reused. This class neither starts an
owner nor submits a job. Its bounded local subprocesses require no SSH/agent
connection and keep the original durable unknown-write latch and deadlines.
"""
import base64
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from .cold_owner import ColdRPC
from .owner_transport import ReadOnlyFailure
from . import vanda_campaign as campaign

def local_invoke(argv,program,*,timeout_s,runner=subprocess.run):
    """A bounded local Python process; the legacy SSH adapter stays unchanged."""
    if (tuple(argv)!=(sys.executable,'-') or type(timeout_s) not in (int,float)
            or not 0<timeout_s<=40 or type(program) is not str):
        raise ValueError('VANDA_REGISTERED_LOCAL_PYTHON_AND_ORIGINAL_CONTROL_DEADLINE')
    r=runner(list(argv),input=program,capture_output=True,text=True,timeout=timeout_s)
    if r.returncode==75 and not r.stdout.strip():
        try:failure=json.loads(r.stderr)
        except (ValueError,TypeError):failure={}
        diagnostic=failure.get('diagnostic',{});command=diagnostic.get('argv',[])
        if (failure.get('schema')==campaign.QUEUE_FAILURE and failure.get('queue_known') is False
                and command and command[0] in ('/opt/pbs/bin/qselect','/opt/pbs/bin/qstat')
                and (type(diagnostic.get('returncode')) is int and diagnostic['returncode']!=0
                     or diagnostic.get('error_type') in ('TimeoutExpired','TimeoutError','ConnectionError'))):
            error=ReadOnlyFailure('VANDA_CURRENT_QUEUE_READ_FAILED_NEVER_EMPTY_OR_TERMINAL')
            error.diagnostic=failure;raise error
    if r.returncode:
        error=RuntimeError('VANDA_LOCAL_CONTROL_PROGRAM_FAILED:'+str(r.returncode))
        error.diagnostic=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        raise error
    return json.loads(r.stdout)

class VandaResidentRPC(ColdRPC):
    def __init__(self,journal,*,remote_root,request_directory,code_root,control_root,
                 control_manifest_sha256,python,transport=local_invoke,clock=time.time):
        root,control,directory,code=map(Path,(remote_root,control_root,request_directory,code_root))
        if not re.fullmatch(r'(?:stdct-login-[0-9]+|gn-a40-[0-9]+)(?:\..*)?',socket.gethostname().lower()):
            raise PermissionError('VANDA_RESIDENT_REQUIRES_REAL_VANDA_HOST_NOT_AGENT')
        if str(python)!=sys.executable or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
            raise PermissionError('VANDA_REGISTERED_CONTROL_PYTHON_AND_NO_LOGIN_CUDA')
        cm=read(control/'RUN_MANIFEST.json');cb=read(control/'OUTPUT_BINDING.private.json');cl=read(cb['lease'])
        gm=read(root/'RUN_MANIFEST.json');request=directory/'REQUEST.json'
        if (sha(control/'RUN_MANIFEST.json')!=control_manifest_sha256
                or cb.get('run_manifest_sha256')!=control_manifest_sha256 or cb.get('uid')!=os.getuid()
                or cl.get('status')!='ACTIVE' or cl.get('run_uuid')!=control.name
                or cm['code']['vla_commit']!=journal.identity['source_sha']
                or gm['code']['vla_commit']!=journal.identity['source_sha']
                or gm.get('execution',{}).get('control_execution_platform')!='VANDA_A40'
                or cm['execution'].get('physical_site')!='VANDA'
                or cm['execution'].get('execution_class')!='CPU_CONTROL_ONLY_NO_CUDA'
                or cm['execution'].get('agent_liveness_required') is not False
                or root.name!=journal.identity['run_uuid'] or sha(request)!=journal.identity['request_sha256']
                or not directory.resolve().is_relative_to(root.resolve())
                or code.resolve()!=(root/gm['code']['capsule_relative']).resolve()):
            raise PermissionError('VANDA_EXACT_NEW_REGISTERED_CONTROL_AND_TRAINING_CAPSULES')
        if any(p.is_symlink() for p in (root,control,directory,code,request)):
            raise PermissionError('VANDA_REGISTERED_LOCAL_PATHS_NOT_SYMLINKS')
        gb=read(root/'OUTPUT_BINDING.private.json');gl=read(gb['lease'])
        for folder,binding in ((control,cb),(root,gb)):
            registry=read(binding['registry']);entries=[x for x in registry['entries'] if x.get('role')==binding['role']]
            if (len(entries)!=1 or entries[0].get('path')!=str(folder)
                    or binding.get('uid')!=os.getuid() or binding.get('run_manifest_sha256')!=sha(folder/'RUN_MANIFEST.json')):
                raise PermissionError('VANDA_CURRENT_REGISTRY_ROLE_PATH_UID_AND_MANIFEST')
        if gl.get('status')!='ACTIVE' or gl.get('run_uuid')!=root.name or root.parent.parent!=control.parent.parent:
            raise PermissionError('VANDA_ONE_ACTIVE_REGISTERED_CAMPAIGN_CONTROL_AND_JOB')
        self.journal=journal;self.argv={'vanda':(str(python),'-')}
        self.root,self.directory,self.code_root=map(str,(root,directory,code))
        self.transport=transport;self.observation=None;self.resident_only=True;self.clock=clock

    def bind_observation(self,state):
        a=state['authority'];h=json.loads(base64.b64decode(a['global_ledger_base64'],validate=True))
        d=json.loads(base64.b64decode(a['DEC1_ledger_base64'],validate=True))
        campaign.fresh(state['observed_epoch'],self.clock())
        campaign.validate(state['view'],ledger=h,DEC1_ledger=d,now_epoch=self.clock())
        # Preserve full native authority for heartbeat publication. The old
        # ColdRPC's two-field ledger summary cannot prove the migrated origin.
        self.observation=json.loads(json.dumps(dict(view=state['view'],ledger=h)))
        return state

    def __call__(self,site,code,values=None,label='UNCLASSIFIED'):
        if site!='vanda' or 'hopper' in label.lower():
            raise PermissionError('VANDA_ONLY_RESIDENT_NEVER_DISPATCHES_OR_RELABELS_HOPPER')
        return super().__call__(site,code,values,label)

def bind_resident_owner(owner,rpc,*,validate_new_registration):
    """Bind only a new cold owner; original failed actors are never resumed."""
    proof=validate_new_registration()
    if (not isinstance(rpc,VandaResidentRPC) or getattr(owner,'cold_rpc_bound',False)
            or proof.get('status')!='PASS_NEW_VANDA_REGISTERED_RESIDENT_OWNER'
            or proof.get('identity')!=rpc.journal.identity
            or proof.get('physical_control_platform')!='VANDA_A40'
            or proof.get('prior_owner_process_confirmed_absent') is not True
            or proof.get('prior_GPU_lease_closed_and_financially_settled') is not True
            or proof.get('old_Hopper_origin_fenced_not_claimed_closed') is not True
            or proof.get('new_recovery_authority_adopted') is not True
            or proof.get('agent_connection_required') is not False):
        raise PermissionError('VANDA_NEW_REGISTERED_AUTHORIZED_RESIDENT_OWNER_ONLY')
    previous=owner.fresh
    def fresh(*args,**kw):return rpc.bind_observation(previous(*args,**kw))
    owner.previous.rpc=rpc;owner.fresh=fresh;owner.cold_rpc_bound=True
    owner.resident_control_platform='VANDA_A40'
    return owner

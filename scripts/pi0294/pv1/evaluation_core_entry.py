"""One additive profile for the existing Core entry; no scientific GO.

The native machine configuration remains private. The exact observed original
bytes are required so this cannot replace other profiles or infer a new env.
The registered evaluator must still validate its immutable capsule, manifest,
lease, native GPU process, full resources and original communication proof.
"""
from copy import deepcopy
import hashlib
import json

from .decision import AUTHORITY
from .evaluation_platform import T0,STOP_NEW,STOP_PHYSICS,HARD_CLOSE

PROFILE='4080-neural-evaluation'
ORIGINAL_ENV_SHA='741615588da7699e1cd3b17617c332ea30a607887376f54ad0849a34c5f3708b'


def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_ENTRY_'+why)


def proposal(before_bytes,*,now_epoch):
    need(type(before_bytes) is bytes and hashlib.sha256(before_bytes).hexdigest()==ORIGINAL_ENV_SHA,
         'EXACT_OBSERVED_EXISTING_ENVIRONMENT_CONFIGURATION')
    need(type(now_epoch) in (int,float) and T0<=now_epoch<STOP_NEW,'ORIGINAL_COLD_REGISTRATION_WINDOW')
    before=json.loads(before_bytes);after=deepcopy(before);rows=before['profiles']
    need(type(rows) is list and len({r['name'] for r in rows})==len(rows)
         and not any(r['name']==PROFILE for r in rows),'ONE_NEW_PROFILE_NO_REPLAY')
    existing=[r for r in rows if r['name']=='4080-cpu-data']
    need(len(existing)==1 and existing[0]['hostnames']==['corelab-4080s']
         and type(existing[0]['python']) is str and existing[0]['python'].startswith('/')
         and type(existing[0]['source']) is str and existing[0]['source'].startswith('/'),
         'EXISTING_ACTUAL_CORE_NEURAL_ENV_NO_NEW_INSTALL_OR_PATH_GUESS')
    old=existing[0]
    new=dict(name=PROFILE,hostnames=deepcopy(old['hostnames']),python=old['python'],source=old['source'],
        role='EXISTING_NEURAL_ENV_NATIVE_CORE_GPU_ALLOCATION_REQUIRED',
        no_model_or_physics_in_doctor=True,native_GPU_allocation_required=True,
        run_manifest_and_active_lease_required=True,
        source_policy='IMMUTABLE_REGISTERED_PV1_CAPSULE_REQUIRED',
        authority_release=AUTHORITY,original_T0_epoch=T0,stop_new_epoch=STOP_NEW,
        stop_physics_epoch=STOP_PHYSICS,hard_close_epoch=HARD_CLOSE,
        new_install=False,scientific_GO=False,resource_budget_created=False)
    after['profiles'].append(new)
    encoded=(json.dumps(after,indent=2,allow_nan=False)+'\n').encode()
    need(after['profiles'][:-1]==rows and all(after[k]==v for k,v in before.items() if k!='profiles'),
         'EVERY_EXISTING_PROFILE_AND_SETTING_PRESERVED')
    return dict(proposed_bytes=encoded,before_sha256=ORIGINAL_ENV_SHA,
        after_sha256=hashlib.sha256(encoded).hexdigest(),new_profile=new,
        entry_shell_or_doctor_modified=False,new_scientific_authority=False,
        native_GPU_allocation=False,new_model_or_physics=False,IO_performed=False)
